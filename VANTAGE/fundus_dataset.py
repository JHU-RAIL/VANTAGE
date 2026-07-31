import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
from scipy import ndimage
from scipy.stats import norm
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm
import os
import math
import cv2
import json
import warnings
import multiprocessing as mp
import fpsample
from queue import PriorityQueue

from PVBM.DiscSegmenter import DiscSegmenter
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

# module load cuda12.8/toolkit/12.8.1

class FundusVesselDataset(Dataset):
    def __init__(
        self,
        files: Dict[str, str],
        save_path: str = './data_prepr/fundus_vasc_pc_deform',
        com: Optional[Tuple[int, int]] = None,
        n_points_pc: int = 2048,
        n_nodes: int = 4096,
        load_norm_const: Optional[str] = None,
        attr_noise_std: Optional[float] = 0.025,
        pad_3d: bool = False,
        eval: bool = False,
        sample_ids: bool = False,
        workers: int = 8,
        recompute_dataset: bool = False
    ) -> None:

        default_mp_method = mp.get_start_method()
        mp.set_start_method('spawn', force=True)

        # Initialize PVBM fundus optic disc segmenter
        self.segmenter = DiscSegmenter()

        # Number of points to sample for the point cloud and vessel graph
        self.n_points_pc = n_points_pc
        self.n_nodes = n_nodes

        # Graph attribute random noise purturbation std scale factor
        self.attr_noise_std = attr_noise_std

        # Whether to pad 2D points with 0s to 3D coordinates and return sample IDs
        self.pad_3d = pad_3d
        self.sample_ids = sample_ids

        # Output file save path and number of multiprocessing/multi-threading workers
        self.save_path = save_path
        self.workers = workers

        # Whether to recompute and overwrite all cached dataset files
        self.recompute_dataset = recompute_dataset

        # Whether dataset is loaded for evaluation
        self.eval = eval

        # Create list of loaded samples
        unaligned = []
        dataset_coms = []

        # Create input arguments
        args = [(orig, seg) for orig, seg in files.items()]

        # Load all the fundus images and compute the center of mass of the optic disc
        with mp.Pool(self.workers) as pool:
            for result in tqdm(pool.imap(self.load_sample, args), total=len(args), desc='Loading Samples'):
                # Skip if failed to load sample
                if result is None:
                    continue

                # Unpack outputs and save results
                orig, seg, raw_com, path = result
                unaligned.append((orig, seg, path))
                dataset_coms.append(raw_com)

        mp.set_start_method(default_mp_method, force=True)

        # List of point clouds
        self.pc = []

        # List of graph attributes
        self.node_feats = []
        self.node_pos = []
        self.node_degree = []
        self.edge_idx = []
        self.edge_feats = []

        # List of sample IDs
        self.ids = []

        # List of vessel radii and log-normal distribution parameters
        # log_radii_data = []
        self.log_radii_distrib = []

        # List of file paths to the original data sample
        self.file_paths = []

        # Check that dataset is not empty
        if len(dataset_coms) == 0:
            return

        # Target optic disc center of mass for the dataset
        if com is None:
            com = np.mean(np.array(dataset_coms), axis=0)
            self.com = com.tolist()
        else:
            self.com = com

        print(f'\nUsing target center of mass (x, y) = ({self.com[1]:.1f}, {self.com[0]:.1f}).\n')

        # Perform COM alignment and convert segmentations into a point cloud and SDF
        i = 0
        for (_, seg, path), raw_com in tqdm(zip(unaligned, dataset_coms), total=len(unaligned), desc='Extracting Point Clouds and Vessel Graphs'):
            # Extract sample ID from file name
            sample_id = os.path.splitext(os.path.basename(path))[0]
            
            # Optic disc COM alignment
            aligned_seg = self.get_com_aligned(seg, sample_id, raw_com)
            
            # Extract point cloud
            pts_pc_xy = self.get_pt_cloud(aligned_seg, path, self.n_points_pc)

            # Skip if segmentation contains too few points
            if pts_pc_xy is None:
                continue

            # Extract vessel graph
            result = self.get_vessel_graph(aligned_seg, sample_id, self.n_nodes)

            # Skip if skeletonized vasculature contains too few points
            if result is None:
                continue

            # Unpack outputs
            node_feats, node_pos, degrees, edge_idx, edge_feats = result

            # Extract vessel radii log-normal distribution parameters
            result = self.get_log_radii_distrib(aligned_seg, sample_id)

            # Skip if skeletonized vasculature contains no vessels
            if result is None:
                continue

            # Unpack outputs
            log_radii, log_radii_mean, log_radii_std = result

            # Add point cloud to the dataset
            self.pc.append(pts_pc_xy)

            # Add graph attributes to the dataset
            self.node_feats.append(node_feats)
            self.node_pos.append(node_pos)
            self.node_degree.append(degrees)
            self.edge_idx.append(edge_idx)
            self.edge_feats.append(edge_feats)

            # Add vessel radii and log-normal distribution to the dataset
            # log_radii_data.append(log_radii)
            # self.log_radii_distrib.append(np.array([log_radii_mean, log_radii_std], dtype=np.float32))
            self.log_radii_distrib.append(
                np.log(self.interp_radii_distrib(np.exp(log_radii), self.n_points_pc, sample_id))
            )

            # Add sample ID to the dataset
            self.ids.append(i)
            i += 1

            # Add file path to the dataset
            self.file_paths.append(path)

        # Compute normalization statistics or load from disk
        if load_norm_const is None:
            # Compute mean and std of the vascular graph node and edge features
            self.mu_node, self.sig_node = self.compute_mu_sig(np.concatenate(self.node_feats, axis=0))
            self.mu_edge, self.sig_edge = self.compute_mu_sig(np.concatenate(self.edge_feats, axis=0))

            # No normalization needed for boolean node features
            self.mu_node[1:3] = 0.0
            self.sig_node[1:3] = 1.0

            # Save normalization constants
            norm_const_path = os.path.join(self.save_path, 'norm_const.json')
            with open(norm_const_path, 'w') as f:
                json.dump({'mu_node': self.mu_node.tolist(), 'sig_node': self.sig_node.tolist(),
                           'mu_edge': self.mu_edge.tolist(), 'sig_edge': self.sig_edge.tolist()}, f)
        else:
            try:
                with open(load_norm_const) as f:
                    data = json.load(f)
                    self.mu_node, self.sig_node = np.array(data['mu_node']), np.array(data['sig_node'])
                    self.mu_edge, self.sig_edge = np.array(data['mu_edge']), np.array(data['sig_edge'])
            except Exception as e:
                raise OSError(f'Unable to open or parse normalization constants JSON file "{load_norm_const}". Perhaps the JSON file is corrupted?')

        # # Compute mean and std of the log-radii data
        # self.mu_log_radii, self.sig_log_radii = self.compute_mu_sig(np.concatenate(log_radii_data, axis=0))

        # # Plot histogram of the log-vessel radii data across all samples and the log-normal PDF curve
        # vis_path = os.path.join(self.save_path, 'log_radii_vis.png')
        # if not os.path.exists(vis_path) or self.recompute_dataset:
        #     # Plot unnormalized log-radii data (assumes all segmentation masks are square and the same shape)
        #     self.plot_hist(np.concatenate(log_radii_data, axis=0).flatten() + math.log(0.5 * max(aligned_seg.shape)),
        #                    vis_path, fit_gaussian=True, bins=45)

    def load_sample(self, args: List[Any]) -> Optional[Tuple[Any, Any, Tuple[float, float], str]]:
        """
        Loads dataset image, segments the optic disc, and computes 
        the center of mass of the segmentation.
        """
        # Unpack input arguments
        orig_path, seg_path = args

        # Load dataset images and segmentations
        orig = np.array(Image.open(orig_path))   # Original color image
        seg = np.array(Image.open(seg_path).convert('L')).astype(np.float32)    # Vessel segmentations

        # Normalize segmentations
        if seg.max() - seg.min() > 0:
            seg = (seg - seg.min()) / (seg.max() - seg.min())

        # Extract sample ID and define segmentation/COM data storage paths
        sample_id = os.path.splitext(os.path.basename(orig_path))[0]
        disc_path = os.path.join(self.save_path, sample_id, 'seg.png')
        cm_path = os.path.join(self.save_path, sample_id, 'cm.json')

        # Create parent directory if necessary
        os.makedirs(os.path.dirname(disc_path), exist_ok=True)

        # Segment the optic disc or load the segmentations
        if os.path.exists(disc_path) and not self.recompute_dataset:
            optic_disc = Image.open(disc_path)
        else:
            optic_disc = self.segmenter.segment(image_path=orig_path)
            optic_disc.save(disc_path)

        # Normalize optic disc segmentations
        optic_disc = np.array(optic_disc).astype(np.float32)
        optic_disc /= 255

        # Skip scan if optic disc is not present
        if np.isclose(optic_disc.max(), 0.0):
            warnings.warn(f'Could not find optic disc for scan "{orig_path}". Skipping.')
            return None

        # Find center of mass (COM) or load the precomputed COMs
        if os.path.exists(cm_path) and not self.recompute_dataset:
            try:
                with open(cm_path) as f:
                    disc_com = json.load(f)
            except Exception as e:
                raise OSError(f'Unable to open or parse optic disc center of mass location JSON file "{cm_path}". Perhaps the JSON file is corrupted?')
        else:
            disc_com = ndimage.center_of_mass(optic_disc)
            with open(cm_path, 'w') as f:
                json.dump(list(disc_com), f)

        # Based on the optic disc center of mass, infer whether scan is a left or right eye
        type = 'OS' if disc_com[1] < orig.shape[1] // 2 else 'OD'
        
        # Coarse alignment of OD/OS fundus images determined based on optic disc COM
        if type == 'OS':
            # Flip the image if scan is of the left eye
            seg = np.flip(seg, axis=1)
            optic_disc = np.flip(optic_disc, axis=1)
            disc_com = ndimage.center_of_mass(optic_disc)

        return orig, seg, tuple(disc_com), seg_path

    def get_com_aligned(self, seg: np.ndarray, sample_id: str, raw_com: Tuple[float, float]) -> np.ndarray:
        """
        Performs center of mass alignment of a segmentation mask.
        """
        # Extract sample ID and define point cloud storage paths
        aligned_path = os.path.join(self.save_path, sample_id, 'aligned_seg.png')

        # Create parent directory if necessary
        os.makedirs(os.path.dirname(aligned_path), exist_ok=True)

        # Perform COM alignment of the segmentation mask or load pre-computed aligned mask
        if os.path.exists(aligned_path) and not self.recompute_dataset:
            aligned_seg = np.array(Image.open(aligned_path).convert('L')).astype(np.float32)
            aligned_seg /= 255
        else:
            # Create transformation matrix
            matrix = np.float32([[1.0, 0.0, self.com[1] - raw_com[1]],
                                 [0.0, 1.0, self.com[0] - raw_com[0]]])

            # Transform the segmentations
            aligned_seg = cv2.warpAffine(seg, matrix, (seg.shape[1], seg.shape[0]), flags=cv2.INTER_NEAREST)
            Image.fromarray(np.stack(3 * [aligned_seg], axis=-1).astype(np.uint8)).save(aligned_path)

        return aligned_seg

    def get_pt_cloud(self, seg: np.ndarray, path: str, n_points: int) -> Optional[np.ndarray]:
        """
        Extracts a point cloud from a segmentation mask using the
        farthest point sampling algorithm.
        """
        # Extract sample ID and define point cloud storage paths
        sample_id = os.path.splitext(os.path.basename(path))[0]
        pt_cloud_path = os.path.join(self.save_path, sample_id, 'seg_pt_cloud.npy')

        # Create parent directory if necessary
        os.makedirs(os.path.dirname(pt_cloud_path), exist_ok=True)

        # Extract point cloud from segmentation or load pre-computed point cloud
        if os.path.exists(pt_cloud_path) and not self.recompute_dataset:
            seg_pts = np.load(pt_cloud_path)
        else:
            # Extract coordinates of all segmented pixels
            y, x = np.nonzero(seg)
            seg_pts = np.array(list(zip(x, y)))

            # Check if there are enough segmented points to sample
            if seg_pts.shape[0] < n_points:
                warnings.warn(f'Number of segmented pixels {seg_pts.shape[0]} for scan "{path}" < target number of points {n_points}. Skipping.')
                return None

            # Farthest point sampling to extract point cloud from segmentation mask
            idx = fpsample.bucket_fps_kdline_sampling(seg_pts, n_points, h=5)
            seg_pts = seg_pts[idx]
            np.save(pt_cloud_path, seg_pts)

        # Normalize point cloud between [-1, 1]
        seg_pts = self.normalize_pts(seg_pts, (0, 0, *seg.shape[::-1]))
        return seg_pts

    def get_vessel_graph(self, seg: np.ndarray, path: str, n_nodes: int) -> Tuple[Any, Any, Any, Any, Any]:
        """
        Constructs an undirected vessel graph from a segmentation mask,
        where nodes are points along the skeletonized vasculature selected
        using farthest point sampling. Represents the vessel graph using
        as node features, node positions, the degree of the node, an
        adjacency list of edges, and edge features. Node features consist of
        vessel radius, whether node is occupied by vessel or background, 
        whether node is connected to the optic disc COM (used as a de facto 
        "centroid"), and the geodesic distance to the optic disc COM. 
        Edge features consist of geodesic distance between two nodes, computed 
        using Dijkstra's shortest path algorithm, the euclidean distance 
        between the two nodes, and mean/std of the discretized vessel radii.
        
        Node and edge features are purturbed with gaussian noise, if specified,
        using a standard deviation (std) that's a constant scale factor of the 
        unpurturbed graph attribute's std.

        *** Outputs: ***
        node_feats: n_nodes x n_node_feats
        node_pos: n_nodes x n_dims
        degrees: n_nodes-dimension vector
        edge_idx: 2 x n_edges
        edge_feats: n_edges x n_edge_feats
        """
        # Extract sample ID and define skeletonized vasculature and graph output paths
        sample_id = os.path.splitext(os.path.basename(path))[0]
        sk_path = os.path.join(self.save_path, sample_id, 'skeletonized.png')
        graph_path = os.path.join(self.save_path, sample_id, 'graph.npz')

        # Create directory if necessary
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)

        # Skeletonize the vessel segmentation mask or load pre-computed skeletonization
        if os.path.exists(sk_path) and not self.recompute_dataset:
            sk_seg = np.array(Image.open(sk_path).convert('L')).astype(np.float32)
            sk_seg /= 255
        else:
            sk_seg = skeletonize(seg)
            Image.fromarray(np.stack(3 * [255 * sk_seg], axis=-1).astype(np.uint8)).save(sk_path)

        # Check that skeletonized vessel map contains vasculature
        if np.all(sk_seg == 0):
            warnings.warn(f'Skeletonized vessel map contains no vasculature for scan "{path}". Perhaps the segmentation mask is empty? Skipping.')
            return None

        # Compute grid dimensions given total number of nodes and image aspect ratio
        aspect = sk_seg.shape[1] / sk_seg.shape[0]
        H = (n_nodes / aspect) ** 0.5
        W = (n_nodes * aspect) ** 0.5

        # Check that integer dimensions exist for the specified number of nodes 
        if not W.is_integer() or not W.is_integer():
            warnings.warn(f'Specified number of graph nodes {n_nodes} is invalid for aspect ratio {aspect:.1f}. Could not compute integer HxW dimensions, got {H:.2f} x {W:.2f} instead for scan "{path}"! Skipping.')
            return None

        # Extract integer grid dimensions and patch dimensions from the original segmentation
        H, W = int(H), int(W)
        self.H, self.W = H, W
        pH, pW = math.ceil(sk_seg.shape[0] / H), math.ceil(sk_seg.shape[1] / W)

        # Construct vessel graph or load pre-computed graph from disk
        if os.path.exists(graph_path) and not self.recompute_dataset:
            data = np.load(graph_path)
            node_feats = data['node_feats']
            node_pos = data['node_pos']
            degrees = data['node_degree']
            edge_idx = data['edge_idx']
            edge_feats = data['edge_feats']
        else:
            # Extract centerpoints of grid patches as nodes or closest skeletonized point
            sk_seg_pts = []
            for i in range(H):
                for j in range(W):
                    # Compute patch dimensions
                    patchH = min(pH, sk_seg.shape[0] - i * pH)
                    patchW = min(pW, sk_seg.shape[1] - j * pW)

                    # Compute coordinates of patch center on the skeletonized vessel segmentation
                    center = [i * pH + patchH // 2, j * pW + patchW // 2]

                    # Extract corresponding patch on the skeletonized segmentation mask
                    patch = sk_seg[i*pH:i*pH+patchH, j*pW:j*pW+patchW]

                    # Find closest skeletonized point if patch contains vessel structures
                    if patch.any():
                        # Extract coordinates of skeletonized vasculature within the patch
                        coords = np.column_stack(np.nonzero(patch))

                        # Offset the patch coordinates
                        offset = np.zeros((1, 2), dtype=coords.dtype)
                        offset[:,0] = i * pH
                        offset[:,1] = j * pW
                        coords += offset
                        
                        # Find the skeletonized vessel closest to the centerpoint of the patch 
                        closest = coords[np.argmin(np.linalg.norm(coords - np.array(center), axis=1))].tolist()
                    else:
                        # No skeletonized vasculature within the patch
                        closest = center.copy()

                    # Add patch centerpoint and closest skeletonized point to list of nodes to process
                    sk_seg_pts.append((tuple(center), tuple(closest)))

            # Create graph of skeletonized vasculature
            G_sk = self._skeleton_to_graph(sk_seg)

            # Find the closest point on the skeletonized vasculature to the optic disc COM
            coords = np.column_stack(np.nonzero(sk_seg))
            centroid = tuple(coords[np.argmin(np.linalg.norm(coords - np.array(self.com), axis=1))].tolist())

            # Create a vessel radius map
            distance_map = distance_transform_edt(seg)
            radius_map = sk_seg * distance_map

            # Compute geodesic distance from the vascular centroid (optic disc)
            geod_centroid = nx.single_source_dijkstra_path_length(G_sk, centroid, weight='weight')

            # Create a dictionary of all points and node metadata, i.e. a unique node ID, original centerpoint, 
            # vessel radius, whether node occupied vessel or background, whether node is connected to the 
            # optic disc COM (used as a de facto "centroid"), and geodesic distance from optic disc COM
            int_tuple = lambda x: tuple(map(int, x))
            all_pts_dict = {pt_cl: (i, pt, radius_map[pt_cl[0], pt_cl[1]], 1.0 if sk_seg[pt_cl[0], pt_cl[1]] > 0 else 0.0,
                                    1.0 if int_tuple(pt_cl) in geod_centroid else 0.0, 
                                    geod_centroid[int_tuple(pt_cl)] if int_tuple(pt_cl) in geod_centroid else 0.0)
                                    for i, (pt, pt_cl) in enumerate(sk_seg_pts)}
            
            # Define undirected vessel graph attributes
            node_feats = [] # Contains [vessel radius, is vessel, centroid-connected, centroid geodesic] for each node, n_nodes × n_node_feats matrix
            node_pos = []   # Contains [x, y] for each node, n_nodes × n_dims matrix
            degrees = []    # Contains [degree] for each node, n_nodes-dimension vector
            edge_idx = []   # Contains [n1, n2] for each edge, 2 × n_edges matrix
            edge_feats = [] # Contains [geodesic, euclidean, radii mu, radii std] for each edge, n_edges × n_edge_feats matrix

            # Find all adjacent points for each point in the point cloud and populate attributes
            for i, (pt, pt_cl) in enumerate(sk_seg_pts):
                # Extract point and its associated metadata
                meta = all_pts_dict[pt_cl]
                assert i == meta[0], f'Node indices {i} and {meta[0]} are misaligned. Iteration through a dictionary was expected to be in a deterministic order.'

                # Add node features and position to the graph
                node_feats.append([*meta[2:]])
                node_pos.append(list(pt))
                
                # Apply Dijkstra's algorithm to search for all adjacent points in all_pts_dict,
                # their corresponding geodesic distances, and list of discretized edge radii
                adj_pts, adj_meta, adj_dist, adj_radii = self.find_adjacent_pts(sk_seg, radius_map, pt_cl, all_pts_dict)

                # Record degree of the node as the number of adjacent points
                degrees.append(len(adj_pts))

                # Add edges and edge features
                for adj_pt, (j, _, _, _, _, _), geodist, radii in zip(adj_pts, adj_meta, adj_dist, adj_radii):
                    # Add undirected edge to the graph
                    edge_idx.append([i, j])
                    edge_idx.append([j, i])

                    # Compute euclidean distance between adjacent points
                    eucdist = float(sum((a - b) ** 2 for a, b in zip(pt_cl, adj_pt))) ** 0.5

                    # Compute mean and standard deviation of discretized edge radii
                    radii = np.array(radii, dtype=np.float32)
                    mu_radii, std_radii = radii.mean(), radii.std()

                    # Add geodesic distance, euclidean distance, and mean/std edge radii as undirected edge features
                    edge_feats.append([geodist, eucdist, mu_radii, std_radii])
                    edge_feats.append([geodist, eucdist, mu_radii, std_radii])

            # Convert vessel graph attributes into numpy array
            node_feats = np.array(node_feats).astype(np.float32)
            node_pos = np.array(node_pos).astype(np.float32)
            degrees = np.array(degrees)
            edge_idx = np.array(edge_idx).T
            edge_feats = np.array(edge_feats).astype(np.float32)

            # Save vessel graph to disk
            np.savez(graph_path, node_feats=node_feats, node_pos=node_pos, node_degree=degrees,
                     edge_idx=edge_idx, edge_feats=edge_feats)

        # Normalize node positions between [-1, 1]
        node_pos = self.normalize_pts(node_pos, (0, 0, *seg.shape))
        return node_feats, node_pos[:,::-1].copy(), degrees, edge_idx, edge_feats

    def get_log_radii_distrib(self, seg: np.ndarray, sample_id: str) -> Tuple[float, float]:
        """
        Computes vessel radii and parameterizes it as a log-normal distribution N(μ,σ).
        """
        # Extract sample ID and define skeletonized vasculature and radii distribution + visualization output paths
        sk_path = os.path.join(self.save_path, sample_id, 'skeletonized.png')
        radii_path = os.path.join(self.save_path, sample_id, 'log_radii.npy')
        radii_distrib_path = os.path.join(self.save_path, sample_id, 'log_radii_distrib.json')
        vis_path = os.path.join(self.save_path, sample_id, 'radii_vis.png')
        hist_path = os.path.join(self.save_path, sample_id, 'log_radii_hist.png')

        # Create directory if necessary
        os.makedirs(os.path.dirname(radii_path), exist_ok=True)

        # Estimate distribution of vessel radii or load pre-computed parameters from disk
        if os.path.exists(radii_path) and os.path.exists(radii_distrib_path) and not self.recompute_dataset:
            radii = np.load(radii_path)
            try:
                with open(radii_distrib_path) as f:
                    data = json.load(f)
                    mean, std = data['mean'], data['std']
            except Exception as e:
                raise OSError(f'Unable to open or parse vessel radii log-normal parameters JSON file "{radii_distrib_path}". Perhaps the JSON file is corrupted?')
        else:
            # Skeletonize the vessel segmentation mask or load pre-computed skeletonization
            seg = seg > 0
            if os.path.exists(sk_path): # Will be recomputed if necessary when creating graph
                sk_seg = np.array(Image.open(sk_path).convert('L')).astype(np.float32)
                sk_seg /= 255
            else:
                sk_seg = skeletonize(seg)
                Image.fromarray(np.stack(3 * [255 * sk_seg], axis=-1).astype(np.uint8)).save(sk_path)

            # Check that skeletonized vessel map contains vasculature
            if np.all(sk_seg == 0):
                warnings.warn(f'Skeletonized vessel map contains no vasculature for scan "{path}". Is the segmentation mask empty? Skipping.')
                return None

            # Create a vessel radius map
            distance_map = distance_transform_edt(seg)
            radius_map = sk_seg * distance_map

            # Compute mean and std of log-vessel radii
            assert np.all(radius_map[radius_map > 0] > 0), 'Something went wrong. All vessel radii should be > 0!'
            radii = np.log(radius_map[radius_map > 0])
            mean, std = radii.mean(), radii.std()

            # Plot vessel radius map
            plt.figure(figsize=(4, 4), dpi=300)
            im = plt.imshow(distance_map, cmap='hot')
            cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
            cbar.set_label('Distance to Vessel Boundary (pixels)', fontsize=9)
            cbar.ax.tick_params(labelsize=8)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot histogram of the log-vessel radii data and the log-normal PDF curve
            self.plot_hist(radii, hist_path, fit_gaussian=True)

            # Save vessel radii log-normal distribution parameters to disk
            with open(radii_distrib_path, 'w') as f:
                json.dump({'mean': float(mean), 'std': float(std)}, f)

            # Save vessel radii data
            np.save(radii_path, radii)

        # Computes normalization constant to convert radii into the normalized [-1, 1] 
        # coordinate space and shift the mean (NOTE: Assumes square 2D coordinate space)
        norm_scale = math.log(0.5 * max(seg.shape))
        return radii - norm_scale, mean - norm_scale, std

    def interp_radii_distrib(self, samples: np.ndarray, n: int, sample_id: str) -> np.ndarray:
        """
        Interpolates an set of m radii samples from a distribution into
        a set of n-ordered samples.
        """
        # Extract sample ID and define interpolated radii distribution path
        radii_interp_path = os.path.join(self.save_path, sample_id, 'radii_interp.npy')

        # Create directory if necessary
        os.makedirs(os.path.dirname(radii_interp_path), exist_ok=True)
        
        # Interpolate samples or load pre-computed interpolations from disk
        if os.path.exists(radii_interp_path) and not self.recompute_dataset:
            n_ordered_interp = np.load(radii_interp_path)
        else:
            # Sort the samples from the distribution
            sorted = np.sort(samples)

            # Find lower and upper bound index for interpolation
            idx = np.linspace(0, 1, n) * (len(sorted) - 1)
            idx_low = np.floor(idx).astype(int)
            idx_high = np.ceil(idx).astype(int)

            # Compute linear interpolation of samples
            alpha = idx - idx_low
            n_ordered_interp = (1. - alpha) * sorted[idx_low] + alpha * sorted[idx_high]
            n_ordered_interp = n_ordered_interp.astype(np.float32)

            # Save interpolated samples
            np.save(radii_interp_path, n_ordered_interp)
        return n_ordered_interp

    def find_adjacent_pts(
        self,
        skeletonized: np.ndarray,
        radius_map: np.ndarray,
        query: Tuple[int, int],
        all_pts_dict: Dict[Tuple[int, int], Tuple[Any, ...]]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[Any, ...]], List[float]]:
        """
        Apply Dijkstra's algorithm to find a set of adjacent points
        from all_pts_dict in the skeletonized vasculature and their 
        corresponding shortest-path geodesic distances.
        """
        adj_pts = []    # Final list of adjacent points in the vasculature
        adj_meta = []   # Metadata associated with each adjacent point
        adj_dist = []   # Closest distance to each adjacent point
        adj_radii = []  # Discretized set of radii between adjacent points
        neighbors = PriorityQueue() # Neighboring points to explore, from closest to furthest euclidean distance
        explored = np.zeros(skeletonized.shape, dtype=bool) # Points already explored -- no need to explore them again

        # Add the query as the first node the explore
        neighbors.put((0.0, query, [radius_map[query[0], query[1]]]))
        
        # Mark query point as explored
        explored[query[0], query[1]] = True

        # Keep searching if there are still more neighbors to explore
        while not neighbors.empty():
            # Pop the first element of the queue
            dist, node, radii = neighbors.get()

            # Explore 8-connected neighbors
            for xdel, ydel in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
                new_pt = (node[0] + ydel, node[1] + xdel)

                # Skip neighboring points if x-coordinate is out of bounds
                if new_pt[1] < 0 or new_pt[1] >= skeletonized.shape[1]:
                    continue

                # Skip neighboring points if y-coordinate is out of bounds
                if new_pt[0] < 0 or new_pt[0] >= skeletonized.shape[0]:
                    continue

                # Skip neighboring points that are outside the skeletonized network
                if skeletonized[new_pt[0], new_pt[1]] == 0:
                    continue

                # Process the new point only if it hasn't been explored before
                if not explored[new_pt[0], new_pt[1]]:
                    # Compute distance to node
                    new_dist = dist + float(xdel ** 2 + ydel ** 2) ** 0.5
                    if new_pt in all_pts_dict:
                        # Encountered an adjacent point in the skeletonized vasculature
                        adj_pts.append(new_pt)
                        adj_meta.append(all_pts_dict[new_pt])
                        adj_dist.append(new_dist)
                        adj_radii.append(radii + [radius_map[new_pt[0], new_pt[1]]])
                    else:
                        # Encountered an unexplored neighbor -- explore this point
                        neighbors.put((new_dist, new_pt, radii + [radius_map[new_pt[0], new_pt[1]]]))
                        explored[new_pt[0], new_pt[1]] = True
        
        assert len(adj_pts) == len(adj_meta) == len(adj_dist) == len(adj_radii), 'Oops, something went wrong. ' \
            'Lengths of lists, which reflects the number of adjacent points, should match exactly.'
        return adj_pts, adj_meta, adj_dist, adj_radii

    def _skeleton_to_graph(self, skeletonized: np.ndarray) -> nx.Graph:
        """
        Constructs a graph from a skeletonized segmentation map.
        """
        # Initialize graph
        G = nx.Graph()
        
        # Extract skeletonized map coordinates
        coords = np.column_stack(np.nonzero(skeletonized))
        coord_set = set(map(tuple, coords))
        
        # Explore 8-connected neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        # Create graph
        for r, c in coord_set:
            G.add_node((r, c))
            for dr, dc in neighbors:
                nbr = (r + dr, c + dc)
                if nbr in coord_set:
                    w = (dr ** 2 + dc ** 2) ** 0.5
                    G.add_edge((r, c), nbr, weight=w)
        
        return G

    def normalize_pts(self, pts: np.ndarray, roi_bbox: Tuple[float, float, float, float]) -> np.ndarray:
        """
        Normalize points (x, y) to [-1, 1] given a region of interest 
        surrounding the point cloud, represented as a bounding box of 
        dimension (x, y, width, height).
        """
        # Check that width and height are positive numbers
        if roi_bbox[2] <= 0 or roi_bbox[3] <= 0:
            raise ValueError('The ROI bounding box must have a positive width and height, ' \
                             f'but got w = {roi_bbox[2]} and h = {roi_bbox[3]} instead.')

        # Compute min/max x and y coordinates of the ROI bounding box
        min_x, max_x = float(roi_bbox[0]), float(roi_bbox[0] + roi_bbox[2])
        min_y, max_y = float(roi_bbox[1]), float(roi_bbox[1] + roi_bbox[3])

        # Normalize the coordinates to [0, 1], then to [-1, 1]
        pts = pts.astype(np.float32)
        pts[:,0] = (pts[:,0] - min_x) / (max_x - min_x)
        pts[:,1] = (pts[:,1] - min_y) / (max_y - min_y)
        pts = (pts - 0.5) / 0.5

        return pts

    def purturb_sample_randn(self, sample: np.ndarray, std_scale: float, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Purturbs a sample of data with n > 1 with random gaussian noise,
        where the standard deviation of the added noise is the standard
        deviation of the raw, unpurturbed sample scaled by a constant.
        Expects an N x * dimension sample, where the first dimension
        represents the N samples of data followed by an arbitrary number
        of dimensions (represented above as '*').
        """
        # Create a noise mask to avoid purturbing features that shouldn't be purturbed
        mask = np.ones(sample.shape, dtype=sample.dtype) if mask is None else mask

        # Sample and apply random noise with std as a constant scale factor of unpurturbed sample std
        sigma = std_scale * sample.std(axis=0, keepdims=True)
        noise = np.random.randn(*sample.shape).astype(sample.dtype) * sigma
        return sample + noise * mask

    def plot_hist(
        self,
        data: np.ndarray,
        save_path: str,
        fit_gaussian: bool = False,
        bins: Optional[int] = None
    ) -> None:
        """
        Plot histogram of data, gaussian PDF curve, and save figure to disk.
        """
        # Compute statistics
        mean, std = data.mean(), data.std()

        # Estimate optimal number of bins for visualizing the data distribution if necessary
        if bins is None:
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            bw = 2.0 * iqr / (len(data) ** (1. / 3))
            bins = int((data.max() - data.min()) / bw)

        # Create figure and plot histogram
        plt.figure(figsize=(5,3), dpi=150)
        sns.histplot(data, bins=bins, stat='density', color='royalblue', alpha=0.35)

        # Fit a gaussian curve to the data and plot if necessary
        if fit_gaussian:
            x = np.linspace(data.min(), data.max(), 300)
            pdf = norm.pdf(x, mean, std)
            plt.plot(x, pdf, 'r--', linewidth=1.5, label='Gaussian Fit')
        
        plt.xlabel('log(radius)', fontsize=9)
        plt.ylabel('Density', fontsize=9)
        plt.title('Vessel radius distribution (log-space)', fontsize=10)
        plt.text(0.03, 0.94, f'$\\mu={mean:.2f}$\n$\\sigma={std:.2f}$',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def compute_min_max(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find minimum and maximum of the dataset.
        """
        # Compute min and max across the dataset
        return data.min(axis=0), data.max(axis=0)
    
    def compute_mu_sig(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find mean and standard deviation of the dataset.
        """
        # Compute mean and standard deviation across the dataset
        return data.mean(axis=0), data.std(axis=0)

    def __len__(self) -> int:
        """
        Returns number of dataset samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Data, Tuple]:
        """
        Returns a dataset sample.
        """
        if idx >= len(self): raise IndexError

        # Retrieve point cloud
        pc = torch.from_numpy(self.pc[idx])

        # Retrieve graph node and edge features
        node_feats = self.node_feats[idx]
        degree = self.node_degree[idx]
        edge_idx = self.edge_idx[idx]
        edge_feats = self.edge_feats[idx]

        # Purturb graph attributes with random noise if necessary
        if self.attr_noise_std is not None:
            # Apply noise to node positions and clip to [-1, 1]
            # node_pos = self.purturb_sample_randn(node_pos, attr_noise_std)
            # node_pos = np.clip(node_pos, -1.0, 1.0)

            # Create noise mask for node features to avoid purturbing boolean features
            mask = np.ones((1, node_feats.shape[1]), dtype=node_feats.dtype)
            mask[:,1:3] = 0.0

            # Apply noise to node features and clip to a non-negative value
            node_feats = self.purturb_sample_randn(node_feats, self.attr_noise_std, mask=mask)
            node_feats = np.clip(node_feats, 0.0, None)

            # Apply noise to edge features and clip to a non-negative value
            edge_feats = self.purturb_sample_randn(edge_feats, self.attr_noise_std)
            edge_feats = np.clip(edge_feats, 0.0, None)
            
            # Randomly dropout patches of the vessel graph
            if not self.eval and np.random.rand() < 0.5:
                node_feats, edge_idx, edge_feats, degree = self.vessel_block_dropout(
                    node_feats, edge_idx, edge_feats, degree, H=self.H, W=self.W, block_size=min(self.H, self.W) // 8
                )

        # Retrieve dataset sample for graph attributes
        node_feats = torch.from_numpy((node_feats - self.mu_node[None,]) / self.sig_node[None,])
        node_pos = torch.from_numpy(self.node_pos[idx]) # Already normalized between [-1, 1]
        degree = torch.from_numpy(degree)
        edge_idx = torch.from_numpy(edge_idx)
        edge_feats = torch.from_numpy((edge_feats - self.mu_edge[None,]) / self.mu_edge[None,])

        # Retrieve sample IDs
        ids = torch.tensor(self.ids[idx])

        # Concatenate the node coordinates and features
        node_feats = torch.cat([node_pos, node_feats], dim=1)

        # Retrieve dataset sample for vessel radii log-normal parameterization
        log_radii_distrib = torch.from_numpy(self.log_radii_distrib[idx])
        
        # Retrieve dataset sample file name
        # sample_name = self.file_paths[idx]

        # Pad 2D to 3D coordinates if necessary
        if self.pad_3d:
            pc = torch.cat([pc, torch.zeros((pc.size(0), 1))], dim=1)
            node_pos = torch.cat([node_pos, torch.zeros((node_pos.size(0), 1))], dim=1)

        # Construct graph object
        vessel_graph = Data(x=node_feats, pos=node_pos, degree=degree, edge_index=edge_idx,
                            edge_attr=edge_feats, num_nodes=node_feats.size(0))
        
        # Return dataset sample
        if self.sample_ids:
            return ids, pc, vessel_graph, log_radii_distrib
        else:
            return pc, vessel_graph, log_radii_distrib

    def vessel_block_dropout(
        self,
        node_feats: np.ndarray,
        edge_idx: np.ndarray,
        edge_feats: np.ndarray,
        degree: np.ndarray,
        H: int = 64,
        W: int = 64,
        block_size: int = 8,
        num_blocks: int = 6,
        vessel_channel: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Randomly drops out all features and connections for a square
        block of vessel nodes.
        """
        # Make a copy of input feature numpy arrays
        N = node_feats.shape[0]
        node_feats = node_feats.copy()
        edge_feats = edge_feats.copy()
        degree = degree.copy()

        # Create vessel and node dropout mask
        vessel_mask = node_feats[:,vessel_channel] > 0.5
        drop_mask = np.zeros(N, dtype=bool)

        # Randomly dropout blocks of vessel nodes
        r = block_size // 2
        for _ in range(num_blocks):
            # Select centerpoint
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)

            # Define dropout block bounding box
            x0 = max(cx - r, 0)
            x1 = min(cx + r + 1, W)
            y0 = max(cy - r, 0)
            y1 = min(cy + r + 1, H)

            # Update the node dropout mask
            ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
            idx = (ys * W + xs).flatten()
            drop_mask[idx] = True

        # Update node dropout mask to only affect vessel nodes
        drop_mask &= vessel_mask

        # Drop node information
        node_feats[drop_mask] = 0.0
        degree[drop_mask] = 0.0
        
        # Remove edges touching dropped nodes
        src, dst = edge_idx
        keep_edge_mask = (~drop_mask[src]) & (~drop_mask[dst])
        edge_idx = edge_idx[:,keep_edge_mask]
        edge_feats = edge_feats[keep_edge_mask]

        return node_feats, edge_idx, edge_feats, degree
