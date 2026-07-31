import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv
from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.data import Batch
from torch_geometric.utils import dropout_edge
from typing import Tuple, List, Optional
from enum import Enum

class DeltaSampling(Enum):
    NONE = 1
    CENTERLINES = 2
    UNIFORM = 3
    BETA = 4
    DOUBLE_PC = 5

class VANTAGE(nn.Module):
    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        deg: torch.Tensor,
        num_pts: int = 2048,
        latent_dim: int = 384,
        enc_channels: List[int] = [384, 384, 384, 384, 384, 384],
        enc_templ_fusion: List[int] = [2, 4],
        dec_channels: List[int] = [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
        dec_dropout: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        dec_norm_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        dec_latent_in: List[int] = [],
        dropout_prob: float = 0.1,
        pc_norm_offset: bool = True,
        pt_dim: int = 2
    ) -> None:
        super().__init__()

        # Only 2D point clouds are supported for learning and computing k-nearest neighbor normal vector offsets
        if pc_norm_offset and pt_dim != 2:
            raise ValueError(f'Point cloud must be 2D to compute norm offsets, but got {pt_dim}D instead.')

        # Additional model attributes
        self.num_pts = num_pts
        self.latent_dim = latent_dim
        self.pt_dim = pt_dim
        self.dropout_prob = dropout_prob
        self.pc_norm_offset = pc_norm_offset

        # Initialize template-aligned Principle Neighborhood Aggregation (PNA) graph encoder
        self.encoder = PNAEncoder(
            node_channels,
            edge_channels,
            deg,
            latent_dim=latent_dim,
            channels=enc_channels,
            templ_fusion=enc_templ_fusion
        )

        # Initialize point cloud deformation decoder 
        self.decoder = DeformDecoder(
            latent_dim,
            dims=dec_channels,
            dropout=dec_dropout,
            norm_layers=dec_norm_layers,
            latent_in=dec_latent_in,
            dropout_prob=dropout_prob,
            pt_dim=self.pt_dim,
            output_dim=self.pt_dim + (1 if self.pc_norm_offset else 0)
        )   # DeformDecoder, DeformResidualDecoder

        # Learnable prior (atlas)
        self.atlas_pc = nn.Parameter(torch.zeros((1, self.num_pts, self.pt_dim)))
        torch.nn.init.uniform_(self.atlas_pc, a=-1.0, b=1.0)

        # Learable k-nearest neighbor normal vector offset prior (atlas)
        if self.pc_norm_offset:
            self.atlas_delta = nn.Parameter(torch.zeros((1, self.num_pts)))
            torch.nn.init.normal_(self.atlas_delta, mean=-5.0, std=1.0)

    def get_atlas(self, sampling: DeltaSampling = DeltaSampling.DOUBLE_PC) -> torch.Tensor:
        """
        Returns the learned atlas prior, sampling k-nearest neighbor normal vector
        offsets based on the specified sampling method.
        """
        if self.pc_norm_offset:
            # Construct the learned atlas prior by sampling k-nearest neighbor normal vector offsets
            return self.sample_norm_offset_pc(self.atlas_pc, self.atlas_delta.exp(), sampling=sampling)[0]
        else:
            # Return the learned atlas prior
            return self.atlas_pc[0]

    def forward(
        self,
        data: Batch,
        n_templates: int = 2,
        compute_jacob: bool = False,
        sampling: DeltaSampling = DeltaSampling.DOUBLE_PC
    ) -> torch.Tensor:
        """
        Given a batch of B * (n_templates + 1) graphs, where node embeddings
        across batches are spatially aligned, predicts a point cloud diffeomorphism 
        of the learned atlas prior that best matches the query graph. Leverages
        the framework of a diffeomorphic transport model to construct an atlas prior
        and estimates the point cloud representation of the query graphs most 
        consistent with the learned atlas. Uses a template-aligned Principle
        Neighborhood Aggregation (PNA) encoder architecture and a multilayer 
        perceptron (MLP) decoder architecture, which represents deformation fields
        as an implicit representation with low spectral bias.
        
        The first sample of each batch is assumed to be the query graph, while 
        the subsequent n_templates samples are the template graph(s). The number
        of templates n_templates must be >= 0. Outputs a B x N x d matrix, where 
        d is the dimension of each point of the point cloud.
        """
        # Forward pass through PNA encoder and deformation decoder
        z = self.encoder(data, n_templates)
        output = self.decode(z, compute_jacob, sampling)

        return output

    def decode(
        self,
        z: torch.Tensor,
        compute_jacob: bool = False,
        sampling: DeltaSampling = DeltaSampling.DOUBLE_PC
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the multilayer perceptron (MLP) deformation
        deocder network and applying transformation onto the learned atlas
        prior point cloud.
        """
        # Repeat embeddings across all query points in the atlas point cloud
        z = z.view(-1, 1, self.latent_dim).repeat(1, self.num_pts, 1)
        atlas = self.atlas_pc.repeat(z.size(0), 1, 1)

        # Forward pass through MLP implicit representation decoder for deformation field
        x = torch.cat([z, atlas], dim=-1)
        transf = self.decoder(x)
        outputs = []

        # Compute the jacobian of the deformation field if necessary
        if compute_jacob:
            # Find jacobian of the spatial deformation using autograd
            jacobian = torch.zeros(transf.size(0), self.num_pts, self.pt_dim, self.pt_dim, device=transf.device)
            for i in range(self.pt_dim):
                grad = torch.autograd.grad(
                    transf[...,i], x,
                    torch.ones_like(transf[...,i]),
                    create_graph=True,
                    retain_graph=True
                )[0]
                jacobian[...,i] = grad[...,-self.pt_dim:]
            outputs = [jacobian]

            # Find jacobian of the radial deformation using autograd if necessary
            if self.pc_norm_offset:
                grad = torch.autograd.grad(
                    transf[...,-1], x,
                    torch.ones_like(transf[...,-1]),
                    create_graph=True,
                    retain_graph=True
                )[0]
                outputs.append(grad[...,-self.pt_dim:])

        # Apply deformation and sample k-nearest neighbor normal vector offsets if necessary
        if self.pc_norm_offset:
            atlas_delta = self.atlas_delta.repeat(z.size(0), 1)
            deform, deform_delta = transf[:,:,:-1], transf[:,:,-1]
            log_delta = atlas_delta + deform_delta
            pred_pc = self.sample_norm_offset_pc(atlas + deform, log_delta.exp(), sampling=sampling)
            outputs = [pred_pc, deform, log_delta, deform_delta] + outputs
        else:
            deform = transf
            pred_pc = atlas + deform
            outputs = [pred_pc, deform] + outputs
        
        return tuple(outputs)

    def sample_norm_offset_pc(
        self,
        pc: torch.Tensor,
        delta: torch.Tensor,
        k: int = 8,
        sampling: DeltaSampling = DeltaSampling.DOUBLE_PC,
        detach_normals: bool = True,
        eps: float = 1e-12
    ) -> torch.Tensor:
        """
        Samples k-nearest neighbor normal vector offsets given a 2D point cloud
        as a B x N x 2 tensor, and their corresponding vector offset magnitudes
        delta as a B x N matrix. For every point in the point cloud, finds the
        k-nearest neighbors and uses principal component analysis (PCA) performed
        using singular value decomposition (SVD) to find the tangent vector. 
        The normal vector is subsequently computed from the tangent. Normal vector 
        offsets are computed based one of the following sampling strategies:

        -  DeltaSampling.NONE: Uses provided delta as the normal vector offset 
                               magnitudes.
        -  DeltaSampling.CENTERLINES: No normal vector offsets used, simply
                                      keep the centerline points.
        -  DeltaSampling.UNIFORM: Samples normal vector offset magnitudes from
                                  a uniform distribution U(-delta, delta).
        -  DeltaSampling.BETA: Samples normal vector offset magnitudes from a
                               beta distribution B(0.25, 0.25), which is then
                               normalized to the domain [-delta, delta].
        -  DeltaSampling.DOUBLE_PC: Uses both the provided delta and -delta
                                    as the normal vector offset, resulting in
                                    an output point cloud containing 2N points.

        Outputs a 2D point cloud as a B x N x 2 tensor (or 2N points if using
        DeltaSampling.DOUBLE_PC) with normal vector offsets applied.
        """
        # No normal vector offsets to compute if sampling method is to use centerlines only
        if sampling == DeltaSampling.CENTERLINES:
            return pc

        # Extract shape of point cloud tensor and standardize shape of deltas
        B, N, d = pc.shape
        delta = delta.view(B, N, 1)

        # Compute pairwise euclidean distances
        diff = pc.unsqueeze(2) - pc.unsqueeze(1)    # [B, N, N, d]
        dist = torch.norm(diff, dim=-1) # [B, N, N]

        # Find k-nearest neighbors (exclude self)
        _, nn_idx = torch.topk(dist, k + 1, dim=-1, largest=False)
        nn_idx = nn_idx[:,:,1:] # [B, N, k]

        # Retrieve k-nearest neighboring points
        idx_expand = nn_idx.unsqueeze(-1).expand(-1, -1, -1, d)
        knn = torch.gather(pc.unsqueeze(1).expand(-1, N, -1, -1), 2, idx_expand)    # [B, N, k, d]

        if d == 2:
            # Zero-center and perform PCA with SVD to retrieve tangent vector (eigenvector with largest eigenvalue)
            centered = knn - knn.mean(dim=2, keepdim=True)  # [B, N, k, d]
            U, S, Vh = torch.linalg.svd(centered.reshape(B*N, k, 2), full_matrices=False)
            tangent = Vh[:,0].reshape(B, N, 2)  # [B, N, 2]

            # Compute the unit normal vector from the tangent
            normal = torch.stack([-tangent[...,1], tangent[...,0]], dim=-1)
            normal = F.normalize(normal, p=2, dim=-1, eps=eps)

            # Detach normal vector from the computation graph if necessary
            if detach_normals:
                normal = normal.detach()

            # Compute normal vector offsets (delta)
            if sampling == DeltaSampling.NONE:
                # Deterministically use raw deltas
                sampled_delta = delta

            elif sampling == DeltaSampling.UNIFORM:
                # Sample delta from a uniform distribution
                sampled_delta = delta * (torch.rand_like(delta) - 0.5) / 0.5

            elif sampling == DeltaSampling.BETA:
                # Sample delta from a beta distribution
                sampled_delta = torch.distributions.Beta(0.25, 0.25).sample(delta.shape).to(delta.device)
                sampled_delta = delta * (sampled_delta - 0.5) / 0.5

            elif sampling == DeltaSampling.DOUBLE_PC:
                # Deterministically offset point cloud by ± delta * normal, doubling the number of points
                norm_offset_pc = torch.cat([pc + delta * normal, pc - delta * normal], dim=1)   # [B, 2N, 2]
                return norm_offset_pc

            else:
                raise ValueError(f'Expected delta sampling method to be "DeltaSampling.NONE", "DeltaSampling.UNIFORM", "DeltaSampling.BETA", or "DeltaSampling.DOUBLE_PC", but got "{sampling}" instead!')

            # Offset point cloud normal vectors by the computed deltas
            norm_offset_pc = pc + sampled_delta * normal     # [B, N, 2]
            return norm_offset_pc
        else:
            raise ValueError(f"Point cloud must be 2D to compute norm offsets, but got {d}D instead.")

class PNAEncoder(nn.Module):
    def __init__(
        self,
        node_channels: int,
        edge_channels: int,
        deg: torch.Tensor,
        latent_dim: int = 384,
        channels: List[int] = [384, 384, 384, 384, 384, 384],
        templ_fusion: List[int] = [2, 5],
        dropout_prob: float = 0.1
    ) -> None:
        super().__init__()

        # Aggregators and scaling methods for PNA layers
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        # Lists for constructing VANTAGE model layers
        self.convs = nn.ModuleList()    # PNA layers
        self.norms = nn.ModuleList()    # LayerNorm and activation layers

        # Graph-level latent vector dimension and dropout probability
        self.latent_dim = latent_dim
        self.dropout_prob = dropout_prob
        channels = [node_channels] + channels

        # Construct encoder layers
        for i in range(len(channels) - 1):
            # Add PNA convolution layer
            self.convs.append(PNAConv(channels[i], channels[i+1], aggregators=aggregators,
                                      scalers=scalers, deg=deg, edge_dim=edge_channels,
                                      pre_layers=1, post_layers=1, divide_input=False))
            
            # Add LayerNorm and activation layer
            self.norms.append(nn.Sequential(
                nn.LayerNorm(channels[i+1]),
                nn.ReLU()
            ))

            # Add template information fusion layer if necessary
            if i in templ_fusion:
                setattr(self, f'templ_fuse{i}', TemplateNodeFusion(channels[i+1]))

        # Attention pooling layer MLP to learn attention weights for computing graph-level embeddings
        self.pool = AttentionalAggregation(nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], 1)
        ))

        # Final encoder non-linear probing layer
        self.fc = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Linear(channels[-1], self.latent_dim)
        )

    def forward(self, data: Batch, n_templates: int = 2) -> torch.Tensor:
        """
        Given a batch of B * (n_templates + 1) graphs, where node embeddings
        across batches are spatially aligned, computes graph-level embeddings
        of all the query graphs using a template-aligned Principle Neighborhood 
        Aggregation (PNA) graph encoder architecture.
        
        The first sample of each batch is assumed to be the query graph, while 
        the subsequent n_templates samples are the template graph(s). The number
        of templates n_templates must be >= 0. Outputs a B x d matrix, where d 
        is the dimension of the graph-level latent vectors.
        """
        # Extract graph attributes
        input, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = input

        # Apply edge dropout
        edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout_prob, training=self.training)
        edge_attr = edge_attr[edge_mask]

        # Forward pass through the encoder layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Apply convolution and normalization layers
            res = conv(x, edge_index, edge_attr)
            res = norm(res)

            # Add residual connection
            if i >= 1:
                x = x + res
            else:
                x = res

            # Fuse template alignment information if necessary
            if n_templates > 0 and hasattr(self, f'templ_fuse{i}'):
                data.x = x
                templ_fuse = getattr(self, f'templ_fuse{i}')
                x = templ_fuse(data, n_templates)
                x, batch = x.x, x.batch
            
            # Apply dropout
            x = F.dropout(x, p=self.dropout_prob, training=self.training)
            data.x = x
        
        # Reconstruct batch with only query graphs
        query_graphs = data.to_data_list()
        query_graphs = query_graphs[::n_templates+1]
        data_queries = Batch.from_data_list(query_graphs)

        # Restore the original state of the input graph data object
        data.x = input

        # Apply attention pooling and project graph-level to target embedding dimension
        z = self.pool(data_queries.x, data_queries.batch)
        z = self.fc(z)
        return z

class TemplateNodeFusion(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        # Input embedding dimension
        self.input_dim = input_dim

        # Key, query, and value projection layers
        self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, input_dim, bias=False)

        # Projection MLP to learn template update embeddings
        self.templ_update_mlp = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(self, data: Batch, n_templates: int = 2):
        """
        Given a batch of B * (n_templates + 1) graphs, where node embeddings
        across batches are spatially aligned, the query graph node embeddings
        are updated using sigmoid-gated cross attention based on computed keys,
        queries, and values.
        
        The first sample of each batch is assumed to be the query graph, while
        the subsequent n_templates samples are the template graph(s). The number
        of templates n_templates must be > 0. Outputs a batch of B * (n_templates + 1)
        graphs, where only the node embeddings of the query graphs are updated.
        """
        # The number of template graphs must be > 0
        if n_templates <= 0:
            raise ValueError(f'Number of template graphs must be > 0, but got {n_templates} instead!')
        
        # Compute the dimensions of the graph node embedding tensor
        graphs_per_sample = n_templates + 1
        B = data.num_graphs // graphs_per_sample
        N = data.ptr[1] - data.ptr[0]
        d = data.x.size(-1)

        # Number of graphs in the batch should be evenly divisible by the specified graphs_per_sample
        if data.num_graphs % graphs_per_sample != 0:
            raise ValueError(f'Number of samples in the batch of graphs should be divisible by the specified graphs_per_sample of {graphs_per_sample} (1 query graph + n_templates).')

        # Extract node embeddings
        x = data.x.view(B, graphs_per_sample, N, d)

        # Extract query and template graphs
        x_q = x[:,0]
        x_t = x[:,1:].view(B, n_templates, N, d)

        # Compute key, query, and values
        q = self.q_proj(x_q)
        k = self.k_proj(x_t)
        v = self.v_proj(x_t)

        # Compute cross attention query-template graph similarity gate
        sim = torch.sum(q.unsqueeze(1) * k, dim=-1) / (d ** 0.5)
        gate = torch.sigmoid(sim).unsqueeze(-1)

        # Compute template update embedding
        update = self.templ_update_mlp(torch.cat([x_q.unsqueeze(1).expand_as(v), v], dim=-1)) 
        template_update = torch.sum(gate * update, dim=1)

        # Update query graph node embeddings
        x_q_updated = x_q + template_update

        # Reconstruct batch with updated query graph node embeddings
        query_graphs = data.to_data_list()
        for i in range(0, data.num_graphs, graphs_per_sample):
            query_graphs[i].x = x_q_updated[i // graphs_per_sample]
        
        return Batch.from_data_list(query_graphs)

class DeformDecoder(nn.Module):
    # DeepSDF decoder implementation from https://github.com/facebookresearch/DeepSDF/blob/main/networks/deep_sdf_decoder.py
    def __init__(
        self,
        latent_size,
        dims: List[int] = [512, 512, 512, 512, 512, 512, 512, 512],
        dropout: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob: float = 0.2,
        norm_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        latent_in: List[int] = [4],
        weight_norm: bool = True,
        xyz_in_all: bool = False,
        use_tanh: bool = False,
        latent_dropout: bool = False,
        pt_dim: int = 2,
        output_dim: Optional[int] = None
    ) -> None:
        super().__init__()

        # Create list of intermediate dimensions
        dims = [latent_size + pt_dim] + dims + [pt_dim if output_dim is None else output_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.pt_dim = pt_dim

        # Construct MLP
        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.pt_dim

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.parametrizations.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "ln" + str(layer), nn.LayerNorm(out_dim))

            if layer < self.num_layers - 2:
                setattr(self, "act" + str(layer), nn.ReLU())

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        # self.activation = nn.ReLU() # nn.LeakyReLU(negative_slope=0.05)

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: B x N x (L+2)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, N, C = input.size()
        input = input.view(-1, C)

        xyz = input[:, -self.pt_dim:]

        if input.shape[1] > self.pt_dim and self.latent_dropout:
            latent_vecs = input[:, :-self.pt_dim]
            latent_vecs = F.dropout(latent_vecs, p=self.dropout_prob, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    ln = getattr(self, "ln" + str(layer))
                    x = ln(x)
                activation = getattr(self, "act" + str(layer))
                x = activation(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x.view(B, N, -1)

class DeformResidualDecoder(nn.Module):
    # Based on DeepSDF decoder implementation from https://github.com/facebookresearch/DeepSDF/blob/main/networks/deep_sdf_decoder.py
    def __init__(
        self,
        latent_size,
        dims: List[int] = [512, 512, 512, 512, 512, 512, 512, 512],
        dropout: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob: float = 0.1,
        norm_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
        latent_in: List[int] = [4],
        weight_norm: bool = True,
        xyz_in_all: bool = False,
        use_tanh: bool = False,
        latent_dropout: bool = False,
        pt_dim: int = 2,
        output_dim: Optional[int] = None
    ) -> None:
        super().__init__()

        dims = [latent_size + pt_dim] + dims + [pt_dim if output_dim is None else output_dim]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm
        self.pt_dim = pt_dim
        self.layer_lin = [0, self.num_layers - 2] + latent_in + [max(0, l-1) for l in latent_in]

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= self.pt_dim
            
            if layer in self.layer_lin:
                lin = nn.utils.parametrizations.weight_norm(nn.Linear(dims[layer], out_dim)) if weight_norm and layer in self.norm_layers else nn.Linear(dims[layer], out_dim)
                # print(f'lin {dims[layer]}, {out_dim}')
            else:
                wn = weight_norm and layer in self.norm_layers
                ln = (not weight_norm) and self.norm_layers is not None and layer in self.norm_layers
                drop = dropout_prob if layer in dropout else None
                lin = ResidualMLP(dims[layer], out_dim, norm_layer=ln, weight_norm=wn, dropout=drop)
                # print(f'res {dims[layer]}, {out_dim}: {wn}, {ln}, {drop}')

            setattr(self, "lin" + str(layer), lin)

            if (
                layer in self.layer_lin and layer != self.num_layers - 2
                and not weight_norm
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "ln" + str(layer), nn.LayerNorm(out_dim))
                # print('ln')

            if layer in self.layer_lin and layer != self.num_layers - 2:
                setattr(self, "act" + str(layer), nn.ReLU())
                # print('act')
        # print()
        # assert False
        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: B x N x (L+2)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, N, C = input.size()
        input = input.view(-1, C)

        xyz = input[:, -self.pt_dim:]

        if input.shape[1] > self.pt_dim and self.latent_dropout:
            latent_vecs = input[:, :-self.pt_dim]
            latent_vecs = F.dropout(latent_vecs, p=self.dropout_prob, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer in self.layer_lin and layer != self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    ln = getattr(self, "ln" + str(layer))
                    x = ln(x)
                activation = getattr(self, "act" + str(layer))
                x = activation(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x.view(B, N, -1)

class ResidualMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        norm_layer: bool = False,
        weight_norm: bool = True,
        dropout: Optional[float] = None
    ) -> None:
        super().__init__()
        
        if weight_norm:
            # Create weight-normalized linear projection layer if input and output dimensions are mismatched
            self.proj = nn.Identity() if input_dim == output_dim else nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, output_dim))
            
            # Create ResNet MLP block with weight normalization
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim) if norm_layer else nn.Identity(),
                nn.ReLU(),
                nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, output_dim)),
                nn.LayerNorm(output_dim) if norm_layer else nn.Identity(),
                nn.ReLU(),
                nn.utils.parametrizations.weight_norm(nn.Linear(output_dim, output_dim))
            )
        else:
            # Create linear projection layer if input and output dimensions are mismatched
            self.proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)

            # Create ResNet MLP block
            self.mlp = nn.Sequential(
                nn.LayerNorm(input_dim) if norm_layer else nn.Identity(),
                nn.ReLU(),
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim) if norm_layer else nn.Identity(),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )

        # Initialize dropout layer if necessary
        self.dropout = nn.Identity() if dropout is None else nn.Dropout(dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Identity skip connection
        identity = self.proj(input)

        # Compute and apply residual
        x = identity + self.mlp(input)
        x = self.dropout(x)
        return x