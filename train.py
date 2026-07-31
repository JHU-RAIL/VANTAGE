import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.utils as utils
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import os
import json
from typing import Any
import random
from tqdm import tqdm

from utils.apml import AdaptiveProbabilisticMatchingLoss
from VANTAGE.fundus_dataset import FundusVesselDataset
from VANTAGE.vantage import VANTAGE, DeltaSampling

def parse_args():

    # Sample command line call:
    # >>> python3 train.py --fundus_train ../datasets/FIVES/train/Original/*_N.png --vessel_train ../datasets/FIVES/train/Ground\ truth/*_N.png --gpu 0

    parser = argparse.ArgumentParser()

    parser.add_argument('--fundus_train', type=str, nargs='+', required=True, help='Path to training retinal fundus images.')
    parser.add_argument('--vessel_train', type=str, nargs='+', required=True, help='Path to the corresponding vessel segmentation masks.')
    parser.add_argument('--output_dir', type=str, default='./results/train/', help='Number of vessel points to sample.')
    parser.add_argument('--loader_cache', type=str, default='./data_prepr/fundus_vasc_pc_deform/', help='Dataloader file cache.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for training.')
    parser.add_argument('--gpu', type=int, default=None, help='Inference on a GPU device.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader.')

    parser.add_argument('--rand_atlas', action='store_true', help='Train with a randomly selected sample serving as the atlas.')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs.')
    parser.add_argument('--log_interval', type=int, default=100, help='Epoch interval for logging output and saving model weights.')
    parser.add_argument('--latent_dim', type=int, default=384, help='VANTAGE PNA encoder latent vector dimension.')
    parser.add_argument('--n_pc', type=int, default=2048, help='Number of points in the learned vessel atlas point cloud.')
    parser.add_argument('--n_nodes', type=int, default=4096, help='Number of nodes in the vessel graph grid.')

    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='AdamW optimizer learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='AdamW optimizer weight decay')
    parser.add_argument('--lambda_pc', type=float, default=1.0, help='Point cloud APML loss term weight.')
    parser.add_argument('--lambda_mag', type=float, default=1.0, help='Deformation magnitude regularization loss term weight.')
    parser.add_argument('--lambda_jac', type=float, default=2e-1, help='Jacobian norm loss term weight.')
    parser.add_argument('--lambda_topo', type=float, default=1.0, help='Topological consistency loss term weight.')
    parser.add_argument('--lambda_radii', type=float, default=1.0, help='Radii loss term weight.')
    parser.add_argument('--n_templ', type=int, default=2, help='Number of VANTAGE vessel graph templates alignments for training.')
    
    opt = parser.parse_args()
    return opt

def estimate_vessel_directions(pc: torch.Tensor, k: int = 10) -> torch.Tensor:
    """
    Estimate local vessel direction at each point with PCA.
    """
   # Extract shape of point cloud tensor
    B, N, d = pc.shape

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
    else:
        raise ValueError(f'Point cloud must be 2D to compute norm offsets, but got {d}D instead.')
    
    return tangent  # [B, N, D]

def direction_alignment_weight(pc: torch.Tensor, knn_idx: torch.Tensor, directions: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """
    Compute alignment weight between each point and its KNN neighbors.
    High weight = neighbor is along the vessel.
    Low weight = neighbor is across vessels.
    """
    B, N, D = pc.shape
    k = knn_idx.shape[2]
    
    # Vector from point i to neighbor j
    knn_idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    neighbor_pos = torch.gather(
        pc.unsqueeze(1).expand(-1, N, -1, -1),
        2, knn_idx_exp
    )  # [B, N, k, D]
    
    to_neighbor = neighbor_pos - pc.unsqueeze(2)          # [B, N, k, D]
    to_neighbor = to_neighbor / (
        to_neighbor.norm(dim=-1, keepdim=True) + 1e-8
    )  # [B, N, k, D] unit vectors
    
    # Alignment with local vessel direction
    dir_i = directions.unsqueeze(2)                        # [B, N, 1, D]
    alignment = (to_neighbor * dir_i).sum(-1).abs()        # [B, N, k] in [0, 1]
    return alignment

def logdelta_smoothness_loss(pc: torch.Tensor, log_delta: torch.Tensor, k: int = 8, alpha: float = 10.0) -> torch.Tensor:
    """
    Direction-aware radius smoothness loss.
    Only penalizes radius differences between along-vessel neighbors.
    """
    B, N, D = pc.shape

    # Pairwise distances
    diff = pc[:, :, None, :] - pc[:, None, :, :]
    dist2 = (diff ** 2).sum(-1)
    eye = torch.eye(N, device=pc.device)[None]
    dist2 = dist2 + eye * 1e8

    knn_idx = dist2.topk(k, largest=False).indices

    # Estimate vessel directions
    directions = estimate_vessel_directions(pc, k=min(k+4, N-1))

    # Alignment weights: high = along vessel, low = across
    weights = direction_alignment_weight(pc, knn_idx, directions, alpha=alpha)

    # Gather neighbor log-deltas
    log_j = torch.gather(log_delta.unsqueeze(1).expand(-1, N, -1), 2, knn_idx)
    log_i = log_delta.unsqueeze(-1)

    # Weighted smoothness: only penalize along-vessel neighbors
    diff = log_i - log_j
    weighted_loss = torch.linalg.norm(weights * diff, dim=2)
    return weighted_loss.mean()

def topological_consistency_loss(atlas: torch.Tensor, deform_field: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Topological consistency loss. Points close together on the atlas
    should remain close together after deformation.
    """
    B, N, _ = atlas.shape

    # Pairwise squared distances
    diff = atlas[:,:,None,:] - atlas[:,None,:,:]    # [B, N, N, d]
    dist = (diff ** 2).sum(-1).sqrt()  # [B, N, N]

    # KNN
    knn_idx = dist.topk(k+1, largest=False).indices # [B, N, k]
    knn_idx = knn_idx[:,:,1:].unsqueeze(-1).expand(-1, -1, -1, atlas.size(-1))

    # Gather neighbor log-deltas
    atlas = atlas + deform_field  # [B, N, d]
    center = atlas.unsqueeze(2) # [B, N, 1, d]
    neighbors = torch.gather(atlas.unsqueeze(1).expand(-1, N, -1, -1), 2, knn_idx)  # [B, N, k, d]

    # Compute loss
    loss = torch.linalg.norm((center - neighbors).flatten(1), dim=1)
    return loss.mean()

@torch.no_grad()
def visualize_pc(model: torch.nn.Module, dataset: Any, vessel_graph: Any, pc_gt: torch.Tensor, device: torch.device, output_path: str, epoch: int):
    """
    Visualize:
        1. Atlas and deformed point cloud
        2. Vessel Graph
    """

    os.makedirs(output_path, exist_ok=True)
    model.eval()

    # 1. Comparison of Predicted and GT Point Clouds
    pred_pc = model(vessel_graph, sampling=DeltaSampling.DOUBLE_PC)[0]

    pc_gt = pc_gt[0].cpu().numpy()
    pred_pc = pred_pc[0].cpu().numpy()
    atlas = model.get_atlas(sampling=DeltaSampling.DOUBLE_PC).cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    axs[0].scatter(pc_gt[:, 0], -pc_gt[:, 1], s=5)
    axs[0].set_xlim([-1, 1])
    axs[0].set_ylim([-1, 1])
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].set_title('GT')

    axs[1].scatter(pred_pc[:, 0], -pred_pc[:, 1], s=5)
    axs[1].set_xlim([-1, 1])
    axs[1].set_ylim([-1, 1])
    axs[1].set_aspect('equal', adjustable='box')
    axs[1].set_title('Predicted')

    axs[2].scatter(atlas[:, 0], -atlas[:, 1], s=5)
    axs[2].set_xlim([-1, 1])
    axs[2].set_ylim([-1, 1])
    axs[2].set_aspect('equal', adjustable='box')
    axs[2].set_title('Atlas')

    plt.tight_layout()
    plt.savefig(f'{output_path}/compare_{epoch+1}.png', dpi=150)
    plt.close()

    # 2. Visualize the vessel graph
    vessel_graph = vessel_graph.to('cpu')
    vessel_graph.x[:,2] = vessel_graph.x[:,2] * torch.from_numpy(dataset.sig_node[0][None,]) + torch.from_numpy(dataset.mu_node[0][None,])

    G = utils.to_networkx(
        vessel_graph.to('cpu'),
        to_undirected=True,
        node_attrs=['x'],
        edge_attrs=['edge_attr'] if hasattr(vessel_graph, 'edge_attr') else None
    )

    if hasattr(vessel_graph, 'pos') and vessel_graph.pos is not None:
        pos = vessel_graph.pos.cpu().numpy()
        pos_dict = {i: pos[i] for i in range(pos.shape[0])}
    else:
        pos_dict = None

    node_colors = [G.nodes[i]['x'][2] for i in G.nodes]
    fig, ax = plt.subplots(figsize=(6, 6))

    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos_dict,
        ax=ax,
        node_size=10,
        node_color=node_colors,
        cmap='magma'
    )
    edges = nx.draw_networkx_edges(
        G,
        pos=pos_dict,
        ax=ax,
        width=1.0,
        edge_color='gray'
    )

    nodes.set_clim(vmin=0.0)
    cbar_nodes = plt.colorbar(nodes, ax=ax)
    cbar_nodes.set_label('Vessel Width (Node attr)')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'Vessel Graph - Epoch {epoch+1}')
    plt.tight_layout()
    plt.savefig(f'{output_path}/graph_{epoch+1}.png', dpi=150)
    plt.close()

def main():

    opt = parse_args()

    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    # Set GPU device
    device = torch.device('cpu') if opt.gpu is None else torch.device(f'cuda:{opt.gpu}')

    # Load dataset
    files = {fund_path: seg_path for fund_path, seg_path in zip(opt.fundus_train, opt.vessel_train)}
    dataset = FundusVesselDataset(files, save_path=opt.loader_cache, n_points_pc=opt.n_pc, n_nodes=opt.n_nodes)
    n_samples = len(dataset)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=opt.batch_size * (opt.n_templ + 1), shuffle=True, 
                            num_workers=opt.workers, persistent_workers=True, drop_last=True)

    # Create tensor of the degree of all nodes in the training dataset
    deg = torch.cat([utils.degree(vessel_graph.edge_index[1], num_nodes=vessel_graph.num_nodes) 
                     for _, vessel_graph, _ in dataloader])
    
    # Create output directory
    os.makedirs(opt.output_dir, exist_ok=True)

    # Save training dataset file paths and degrees
    with open(f'{opt.output_dir}/training_dataset.json', 'w') as json_file:
        json.dump(files, json_file, indent=4)
    torch.save({'deg': deg}, f'{opt.output_dir}/deg.pt')
    
    # Initialize VANTAGE model
    print(f'No. samples: {n_samples}')
    model = VANTAGE(node_channels=6, edge_channels=4, deg=deg, num_pts=opt.n_pc // 2, latent_dim=opt.latent_dim).to(device)
    print(model.train())

    # Randomly sample from the dataset to use as the atlas if necessary
    if opt.rand_atlas:
        init_pc = iter(dataloader).__next__()[0][0].unsqueeze(0).to(device)
        model.atlas_pc = torch.nn.Parameter(init_pc, requires_grad=False)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # Initialize APML point cloud matching loss
    apm_loss = AdaptiveProbabilisticMatchingLoss()

    # Proceed with training loop
    for e in tqdm(range(opt.epochs)):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Initialize progress bar
        pbar = tqdm(dataloader, desc=f'Epoch {e+1}/{opt.epochs}', leave=False)

        # Iterate through dataset samples
        for pc_gt, vessel_graph, log_radii_distrib in pbar:
            # Move batch onto compute device
            pc_gt = pc_gt.to(device)[::opt.n_templ+1]
            vessel_graph = vessel_graph.to(device)
            log_radii_distrib = log_radii_distrib.to(device)
            
            # Forward pass through model
            pred_pc, deform, log_delta, deform_delta, jac, jac_delta = model(
                vessel_graph, n_templates=opt.n_templ, compute_jacob=True, sampling=DeltaSampling.DOUBLE_PC
            )
            
            # Compute point cloud distance loss
            pc_loss = opt.lambda_pc * apm_loss(pred_pc, pc_gt)

            # Compute deformation magnitude regularization loss (radii component is undefined if atlas is a fixed point cloud)
            mag = torch.linalg.norm(deform, dim=(1, 2)).mean()
            mag_delta = 0.0 if opt.rand_atlas else torch.linalg.norm(deform_delta, dim=1).mean()

            # Compute jacobian norm regularization loss (radii component is undefined if atlas is a fixed point cloud)
            jac = torch.linalg.norm(jac, dim=(1, 2)).mean()
            jac_delta = 0.0 if opt.rand_atlas else torch.linalg.norm(jac_delta, dim=1).mean()
            
            mag_loss = opt.lambda_mag * (mag + mag_delta) + opt.lambda_jac * (jac + jac_delta)
            
            # Compute topological consistency loss
            atlas = model.get_atlas(DeltaSampling.BETA)[None,]
            topo_loss = opt.lambda_topo * topological_consistency_loss(atlas, deform)

            # Compute radii distribution matching loss
            if opt.rand_atlas:
                # Radii loss term is undefined if atlas is a fixed point cloud
                radii_loss = 0.0
            else:
                gt_radii_distrib = F.avg_pool1d(log_radii_distrib[::opt.n_templ+1], kernel_size=2, stride=2)
                pred_radii_distrib = torch.sort(log_delta)[0]
                distrib_loss = torch.linalg.norm(pred_radii_distrib - gt_radii_distrib, dim=1).mean()

                # Compute radii smoothness loss
                cl = model.atlas_pc.repeat(deform.size(0), 1, 1) + deform
                smooth_loss = logdelta_smoothness_loss(cl, log_delta)

                radii_loss = opt.lambda_radii * (distrib_loss + smooth_loss)

            # Sum overall loss
            loss = pc_loss + mag_loss + topo_loss + radii_loss

            # Backprop and update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track epoch loss 
            epoch_loss += loss.item()
            num_batches += 1

            # Create dictionary of batch losses
            losses_logging = {
                'loss': f'{loss.item():.2f}',
                'pc': f'{pc_loss.item():.2f}',
                'mag': f'{mag_loss.item():.2f}',
                'topo': f'{topo_loss.item():.2f}'
            }
            
            # Display radii loss term if defined
            if not opt.rand_atlas:
                losses_logging['radii'] = f'{radii_loss.item():.2f}'

            # Update tqdm bar to display current batch loss
            pbar.set_postfix(losses_logging)

        # Average loss after epoch
        avg_loss = epoch_loss / num_batches
        print(f'Epoch {e+1}: avg_loss = {avg_loss:.4f}')

        # Periodically save model weights and visualize atlas
        if (e + 1) % opt.log_interval == 0 or e == 0:
            with torch.no_grad():
                # Pick 1 graph from the last batch
                visualize_pc(
                    model,
                    dataset,
                    Batch.from_data_list([vessel_graph.get_example(i) for i in range(opt.n_templ + 1)]),        # uses last graph from epoch
                    pc_gt,
                    device,
                    opt.output_dir,
                    e
                )

                # Save/update model checkpoint
                torch.save(model.state_dict(), f'{opt.output_dir}/vantage.pth')

if __name__ == '__main__':
    main()