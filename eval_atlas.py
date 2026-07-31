import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import argparse
import os
import csv
from typing import Optional, Tuple, List, Any
from PIL import Image
from tqdm import tqdm
import pickle
import warnings
import random

from VANTAGE.fundus_dataset import FundusVesselDataset
from VANTAGE.vantage import VANTAGE, DeltaSampling
from utils.vessel_quantifier import VesselQuantifier
from utils.atlas import VANTAGE_Atlas
from PVBM.DiscSegmenter import DiscSegmenter
from quantify_vasculature import quantify_vessels

def parse_args():

    # Sample command line call:
    # >>> python3 eval_atlas.py --fundus_train ../datasets/FIVES/train/Original/*_N.png --vessel_train ../datasets/FIVES/train/Ground\ truth/*_N.png --fundus_test ../datasets/FIVES/test/Original/*_N.png --vessel_test ../datasets/FIVES/test/Ground\ truth/*_N.png --disease_fundus_test ../datasets/FIVES/test/Original/*_D.png --disease_vessel_test ../datasets/FIVES/test/Ground\ truth/*_D.png --disease_label DR --model_ckpt ./results/train/vantage.pth --deg_path ./results/train/deg.pt --atlas_seg ./results/example/atlas_fives/atlas_mask.png --disc_seg ./results/example/atlas_fives/optic_disc_mask.png --pc_labels ./results/example/atlas_fives/labeled/atlas_point_cloud_labeled.npz --output_path ./results/eval_atlas/ --gpu 0

    parser = argparse.ArgumentParser()

    parser.add_argument('--fundus_train', nargs='+', type=str, required=True, help='Path to retinal fundus training files.')
    parser.add_argument('--vessel_train', nargs='+', type=str, required=True, help='Path to the corresponding vessel segmentation training files.')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the trained VANTAGE model checkpoint.')
    parser.add_argument('--deg_path', type=str, required=True, help='Path to the vessel graph degree statistics for the trained VANTAGE model.')
    parser.add_argument('--loader_cache', type=str, default='./data_prepr/fundus_vasc_pc_deform/', help='Dataloader file cache.')
    
    parser.add_argument('--atlas_seg', type=str, default=None, help='Path to the atlas segmentation mask (--disc_seg is also required for evaluation of the atlas segmentation mask).')
    parser.add_argument('--disc_seg', type=str, default=None, help='Path to optic disc atlas segmentation mask (--atlas_seg is also required for evaluation of the atlas segmentation mask).')
    parser.add_argument('--pc_labels', type=str, default=None, help='Path to the atlas point cloud segmentation labels as a *.npz file.')
    parser.add_argument('--fundus_test', nargs='+', type=str, default=None, help='Path to retinal fundus testing files.')
    parser.add_argument('--vessel_test', nargs='+', type=str, default=None, help='Path to the corresponding vessel segmentation testing files.')
    parser.add_argument('--metrics', nargs='+', type=str, default=['bifurcations', 'endpoints', 'segments', 'tortuosity', 'diameter', 'length', 'fractal_cap', 'fractal_entr', 'fractal_corr'], help='Vessel quantification metrics to evaluate. Options include "bifurcations", "endpoints", "segments", "tortuosity", "diameter", "length", "fractal_cap", "fractal_entr", and "fractal_corr".')
    parser.add_argument('--disease_fundus_test', nargs='+', type=str, default=None, help='Path to diseased retinal fundus testing files (all other --disease_... flags are also required for evaluation against diseased subjects).')
    parser.add_argument('--disease_vessel_test', nargs='+', type=str, default=None, help='Path to the corresponding diseased vessel segmentation testing files (all other --disease_... flags are also required for evaluation against diseased subjects).')
    parser.add_argument('--disease_label', type=str, default=None, help='Optional label for the disease being evaluated (for plotting purposes only).')
    
    parser.add_argument('--n_nodes', type=int, default=4096, help='Number of input vessel graph nodes for the dataloader.')
    parser.add_argument('--n_templ', type=int, default=2, help='Number of VANTAGE template vessel graphs.')
    parser.add_argument('--test_iters', type=int, default=50, help='Number of test time latent space optimization iterations.')
    parser.add_argument('--pca_modes', type=int, default=20, help='Number of Principal Component Analysis modes to visualze for the deformation fields.')
    parser.add_argument('--output_path', type=str, default='./results/eval_atlas/', help='Output path for constructed atlas evaluation results.')
    parser.add_argument('--seed', type=int, default=42, help='Set random seed for reproducibility.')
    parser.add_argument('--gpu', type=int, default=None, help='Inference on a GPU device.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader.')

    opt = parser.parse_args()
    return opt

def plot_pca_var_curve(
    explained_var: np.ndarray,
    max_modes: int = 100,
    figsize: Tuple[float, float] = (7, 4),
    dpi: int = 300
) -> Tuple[Any, Any]:
    """
    Plot Principal Component Analysis (PCA) explained variance curve against
    the number of principal modes.
    """
    # Compute cumulative variance explained
    cumulative = np.cumsum(explained_var)
    modes = np.arange(1, len(explained_var) + 1)
    N = min(max_modes, len(explained_var))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot the PCA cumulative variance curve
    ax.fill_between(modes[:N], cumulative[:N] * 100, alpha=0.08, color='#2d6a9f')
    ax.plot(modes[:N], cumulative[:N] * 100, color='#2d6a9f', linewidth=2, zorder=3)

    # Draw cumulative variance threshold lines
    for t, label in [(0.50, '50%'), (0.90, '90%'), (0.95, '95%')]:
        m = np.searchsorted(cumulative, t) + 1
        ax.axhline(t * 100, color='#999999', linewidth=1.5, linestyle='--', zorder=1)
        ax.text(N + 1, t * 100, f'{label} ({m})', fontsize=9, color='#999999', va='center')
    
    # Configure axis labels
    ax.set_xlabel('Principal Mode', fontsize=11)
    ax.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax.set_title('Atlas Deformation Field PCA Variance Plot', fontsize=14, pad=10)

    # Configure axis range
    ax.set_xlim(1, N)
    ax.set_ylim(0, 102)
    ax.tick_params(labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('black')

    plt.tight_layout()
    return fig, ax

def visualize_point_clouds(
    pc_gt: np.ndarray,
    pc_atlas: np.ndarray,
    pc_deform: np.ndarray,
    gt_dist: np.ndarray,
    deform_dist: np.ndarray,
    figsize: Tuple[float, float] = (12, 6),
    point_size: float = 2,
    cmap: str = 'turbo',
    dpi: int = 300
) -> Tuple[Any, Any]:
    """
    Visualize the ground-truth, atlas, and deformed point cloud
    given per-point closest distances.  
    """
    # Set background color and normalize the colorbar
    BG = '#0d0d0d'
    BG = 'white'
    norm = mcolors.Normalize(vmin=0, vmax=np.concatenate([gt_dist, deform_dist]).max())
    cmap = plt.get_cmap(cmap)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor(BG)

    # Wrap data into a dictionary
    data = [{'coords': pc_gt, 'distances': gt_dist, 'title': f'Ground Truth (Avg Dist. = {gt_dist.mean():.1f})'},
            {'coords': pc_deform, 'distances': deform_dist, 'title': f'Deformed Atlas (Avg Dist. = {deform_dist.mean():.1f})'}]

    # Create each subplot of the figure
    for ax, cfg in zip(axes, data):
        # Set background color of the subplot
        ax.set_facecolor(BG)

        # Plot point cloud colored based on distance
        sc = ax.scatter(cfg['coords'][:,0], cfg['coords'][:,1], c=cfg['distances'], s=point_size,
                        cmap=cmap, norm=norm, linewidths=2, zorder=2, rasterized=True)

        # Create colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.08)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label('Distance (px)', fontsize=11, color='black', labelpad=8)
        cbar.ax.yaxis.set_tick_params(color='black', labelsize=10, labelcolor='black')
        cbar.outline.set_edgecolor('#444444')

        # Configure axes
        ax.set_aspect('equal')
        ax.invert_yaxis()
        pad = 30
        all_coords = np.vstack([pc_atlas, cfg['coords']])
        ax.set_xlim(all_coords[:,0].min() - pad, all_coords[:,0].max() + pad)
        ax.set_ylim(all_coords[:,1].max() + pad, all_coords[:,1].min() - pad)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Set subplot title
        ax.set_title(cfg['title'], fontsize=14, color='black', pad=10, fontweight='normal')

    # Set global figure title
    fig.suptitle('Atlas Registration Point Cloud Distance', fontsize=16, color='black', fontweight='normal')
    plt.tight_layout(w_pad=2.5)
    return fig, axes

def chamfer_distance(pc_a: torch.Tensor, pc_b: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """
    Computes Chamfer distance between point cloud A and B.
    Finds closest point in point cloud B for every point in
    point cloud A. Pointwise scores are provided in simple
    Euclidean distance.
    """
    # Extract point cloud dimensions
    B, N, d = pc_a.shape
    M = pc_b.size(1)
    
    # Compute minimum distances for every point in each set of point clouds
    scores = []
    for b in range(B):
        x = pc_a[b]
        p = pc_b[b]

        # Process in chunks to reduce memory footprint
        min_dists = []
        for i in range(0, N, chunk):
            # Chunk the point cloud
            x_chunk = x[i:i+chunk]

            # For every point in point cloud A, find closest point in point cloud B 
            dists = torch.cdist(x_chunk, p)
            min_d = torch.min(dists, dim=1)[0]
            min_dists.append(min_d)

        scores.append(torch.cat(min_dists))
    
    # Compute Chamfer distance
    scores = torch.stack(scores)
    cd = torch.mean(scores ** 2, dim=1)
    return cd, scores

def compute_auc_ci(
    y_gt: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Tuple[float, float, float]:
    """
    Computes AUC and estimates confidence intervals using
    stratified percentile bootstrapping.
    """
    # Identify samples from each class
    pos_idx = np.where(y_gt == 1)[0]
    neg_idx = np.where(y_gt == 0)[0]
    
    # Perform bootstrapping to estimate confidence intervals
    boot_aucs = []
    for _ in range(n_bootstrap):
        # Randomly resample each class independently to preserve class proportions
        resampled_pos = np.random.choice(pos_idx, len(pos_idx), replace=True)
        resampled_neg = np.random.choice(neg_idx, len(neg_idx), replace=True)
        idx = np.concatenate([resampled_pos, resampled_neg])
        
        # Resample classes and extract scores
        y_resampled = y_gt[idx]
        scores_resampled = scores[idx]
        
        # Skip if resampled does not include both classes
        if len(np.unique(y_resampled)) < 2:
            continue
        
        # Compute resampled AUC
        boot_aucs.append(roc_auc_score(y_resampled, scores_resampled))
    
    # Compute observed and bootstrapped AUCs
    auc_obs = roc_auc_score(y_gt, scores)
    boot_aucs = np.array(boot_aucs)
    
    # Use percentile bootstrap to compute CI
    ci_low = np.percentile(boot_aucs, 100 * alpha / 2.0)
    ci_high = np.percentile(boot_aucs, 100 * (1. - alpha / 2.0))
    return auc_obs, ci_low, ci_high

def compute_distance_matrix(query_dl: DataLoader, templ_dl: DataLoader):
    """
    Compute pairwise point cloud chamfer distances.
    """
    # Collect all query point clouds
    query_pcs = torch.cat([pc for pc, _, _ in query_dl], dim=0)

    # Collect all template point clouds and vessel graphs
    templ_pcs, templ_graphs = [], []
    for pc, graph, _ in templ_dl:
        templ_pcs.append(pc)
        templ_graphs.append(graph)
    templ_pcs = torch.cat(templ_pcs, dim=0)
    
    # Create point cloud distance matrix
    N = query_pcs.size(0)
    M = templ_pcs.size(0)
    dist_matrix = torch.zeros(N, M)
    
    # Compute pairwise chamfer distances
    for i in range(N):
        for j in range(M):
            d_q = chamfer_distance(query_pcs[i][None,], templ_pcs[j][None,])[0]
            d_t = chamfer_distance(templ_pcs[j][None,], query_pcs[i][None,])[0]
            dist_matrix[i,j] = d_q + d_t
    
    return dist_matrix, templ_graphs

def create_violinplot(
    data: List[List[float]],
    labels: List[str],
    title: Optional[str] = None,
    y_axis_label: Optional[str] = None,
    dpi: int = 300
) -> Tuple[Any, Any]:
    """
    Creates a violin plot of the data.
    """
    # Create dataframe
    df = pd.concat([pd.DataFrame({'value': np.array(d), 'group': l}) for d, l in zip(data, labels)])

    # Set colors
    BG = '#fafaf8'
    DARK = '#1a1a2e'
    COLOR = '#2d6a9f'

    # Create figure and set background color
    fig, ax = plt.subplots(figsize=(max(6, len(data) * 1.4), 4), dpi=dpi)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Create violin plot
    sns.violinplot(data=df, x='group', y='value', order=labels, color=COLOR, inner='box', density_norm='width',
                   linewidth=3.5, saturation=0.85, ax=ax, cut=0)

    # Set title and axis labels
    ax.set_title(title, fontsize=14, color=DARK, pad=12)
    ax.set_xlabel('', fontsize=11)
    ax.set_ylabel(y_axis_label, fontsize=11, color=DARK, labelpad=8)

    # Configure axes
    ax.tick_params(axis='both', labelsize=11, colors=DARK)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#cccccc')

    plt.tight_layout()
    return fig, ax

def main():
    # Parse command-line arguments
    opt = parse_args()

    # Display warning if only one of --atlas_seg and --disc_seg is specified, when both are required
    atlas_mask = opt.atlas_seg is not None and opt.disc_seg is not None
    if (opt.atlas_seg is not None or opt.disc_seg is not None) and not atlas_mask:
        warnings.warn('Only one of --atlas_seg and --disc_seg was specified, but both are required ' \
                      'for evaluation of the atlas segmentation mask. The specified argument will have ' \
                      'no effect without the other flag also specified.')

    # Ensure specified quantificaton metrics are valid and case-insensitive
    if atlas_mask:
        metrics, valid = [], ['bifurcations', 'endpoints', 'segments', 'tortuosity', 'diameter', 'length',
                              'fractal_cap', 'fractal_entr', 'fractal_corr']
        for m in opt.metrics:
            if m.lower() in valid:
                metrics.append(m.lower())
            else:
                raise ValueError('Expected quantification metric to be "bifurcations", "endpoints", "tortuosity", ' \
                                 '"diameter", "length", "fractal_cap", "fractal_entr", or "fractal_corr", ' \
                                 f'but got "{m}" instead!')

    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    # Load model state dictionary and extract point cloud size
    device = torch.device('cpu') if opt.gpu is None else torch.device(f'cuda:{opt.gpu}')
    state_dict = torch.load(opt.model_ckpt, map_location=device)
    n_pc = state_dict['atlas_pc'].size(1)
    learned_radii = 'atlas_delta' in state_dict

    # Load weights for the trained VANTAGE model
    deg = torch.load(opt.deg_path)['deg']
    model = VANTAGE(node_channels=6, edge_channels=4, deg=deg, num_pts=n_pc,
                    pc_norm_offset=learned_radii).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'\nSuccessfully loaded VANTAGE model, which has an atlas with {n_pc} learned points.\n')

    # Create list of data files
    datasets, ds_labels = {}, ['Train', 'Test', 'Disease Test']
    ds_files = [(opt.fundus_train, opt.vessel_train), (opt.fundus_test, opt.vessel_test),
                (opt.disease_fundus_test, opt.disease_vessel_test)]
    
    # Initialize datasets for constructing input vessel graphs and ground-truth vessel points clouds
    com = None
    for (fundus, vessel_seg), label in zip(ds_files, ds_labels):
        if fundus is not None and vessel_seg is not None:
            files = {fundus: vessel_seg for fundus, vessel_seg in zip(fundus, vessel_seg)}
            datasets[label] = FundusVesselDataset(files, save_path=opt.loader_cache, n_points_pc=2*n_pc, n_nodes=opt.n_nodes,
                                                  com=com, eval=True)
            com = datasets[label].com
            print()

    # Load the first image to extract dimensions (NOTE: We assume it's a square where width = height)
    img_size, _ = Image.open(opt.vessel_train[0]).size

    # Construct dataloader for all provided datasets
    dataloaders = {}
    for label, dataset in datasets.items():
        dataloaders[label] = DataLoader(dataset, batch_size=1, num_workers=opt.workers, persistent_workers=True)
    
    ###
    # Phase 1: Evaluation of the Atlas Point Cloud
    ###

    # Initialize helper class for constructing atlas
    atlas = VANTAGE_Atlas(model, image_size=img_size)

    # Load point cloud segmentation labels if necessary
    pc_labels = None
    label_names = None
    if opt.pc_labels is not None:
        with np.load(opt.pc_labels) as data:
            # Load labels and names
            pc_labels = data['labels'].astype(int)
            label_names = data['names']
    
    # Create output directory if necessary
    base_dir_pc = f'{opt.output_path}/point_cloud/'
    os.makedirs(base_dir_pc, exist_ok=True)
    print('\n##### Evaluating Atlas Point Cloud #####')

    # Create dictionary for storing all the results
    results = {}

    # Extract the vessel radii (if available), convert it to pixel space, and compute diameter
    centerlines, radii = atlas.get_raw_data()
    if radii is not None:
        radii = np.exp(radii) * img_size / 2.0
        diameters = 2.0 * radii
        results['median_diameter'] = np.median(diameters)

        # Plot the distribution of the VANTAGE atlas diameter
        VesselQuantifier.plot_distribution(diameters, title='Atlas Per-Point Vessel Diameter Distribution',
                                           x_label='Diameter', y_label='Frequency', bins=30)
        plt.savefig(f'{base_dir_pc}/diameter_distrib.png')
        plt.close()

        # Save the unnormalized vessel diameters
        np.save(f'{base_dir_pc}/diameter.npy', diameters)

    # Create csv file with point cloud evaluation results
    with open(f'{base_dir_pc}/point_cloud_eval.csv', 'w', newline='') as f:
        # Initialize csv writer and write header
        writer = csv.writer(f)
        headers = ['median_diameter', 'train_global_deform_vec_mu', 'train_global_deform_vec_sigma', 'test_global_deform_vec_mu',
                'test_global_deform_vec_sigma', 'disease_test_global_deform_vec_mu', 'disease_test_global_deform_vec_sigma',
                'train_global_deform_mag_mu', 'train_global_deform_mag_sigma', 'test_global_deform_mag_mu', 'test_global_deform_mag_sigma',
                'disease_test_global_deform_mag_mu', 'disease_test_global_deform_mag_sigma', 'train_chamfer_dist_mu', 'train_chamfer_dist_std',
                'test_chamfer_dist_mu', 'test_chamfer_dist_std', 'disease_test_chamfer_dist_mu', 'disease_test_chamfer_dist_std', 'test_cd_auc',
                'test_cd_auc_CI_lower', 'test_cd_auc_CI_upper', 'test_mannwhitneyu_p_val']
        writer.writerow(headers)

        # Atlas distances to the closest point in the ground-truth point cloud for training dataset
        atlas_dist_gt_train = None
            
        # Evaluate on training and testing (if provided) dataset
        cd_results, norm_results = {}, {}
        for ds_type, loader in dataloaders.items():
            # Define base output directory for the dataset evaluation and VANTAGE inferencing results if necessary
            fname = ds_type.lower().replace(' ', '_')
            base_ds_dir_pc = f'{base_dir_pc}/{fname}/'
            base_dir_inference = f'{base_ds_dir_pc}/inference/'

            # Create visualization and result point cloud result directories
            base_dir_inference_vis = f'{base_dir_inference}/visualizations/'
            base_dir_inference_raw = f'{base_dir_inference}/raw_results/'
            os.makedirs(base_dir_inference_vis, exist_ok=True)
            os.makedirs(base_dir_inference_raw, exist_ok=True)

            # Pre-compute pairwise distances with the template vessel dataloaders
            dist_matrix, graphs = compute_distance_matrix(loader, dataloaders['Train'])
            dist_matrix = dist_matrix.to(device)
            
            # List for storing predicted deformations and symmetric chamfer distances across the training dataset
            deformations = []
            chamfer_dists = []

            # List of distances to the closest point in the ground-truth point cloud
            atlas_dist_gt = []

            #### Inference VANTAGE model on the datasets ####
            for i, (pc_gt, vessel_graph, _) in enumerate(tqdm(loader, desc=f'Inferencing VANTAGE ({ds_type}ing)')):
                # Retrieve nearest neighbor templates
                if 'Train' in ds_type:
                    sample_idx = dist_matrix[i].topk(opt.n_templ + 1, largest=False).indices.to(device)
                    sample_idx = sample_idx[1:]
                else:
                    sample_idx = dist_matrix[i].topk(opt.n_templ, largest=False).indices.to(device)

                # Create template graph data list
                templ_graph = []
                for idx in sample_idx:
                    templ_graph = templ_graph + graphs[idx].to(device).to_data_list()

                # Move all data to device
                pc_gt = pc_gt.to(device)
                vessel_graph = vessel_graph.to(device)
                
                # Concatenate vessel query graph with templates
                graph = Batch.from_data_list(vessel_graph.to_data_list() + templ_graph).to(device)

                # Inference VANTAGE to deform atlas to query vessel
                pc_pred, deform = atlas.align_sample(graph, pc_gt, n_iters=0 if 'train' in ds_type.lower() else opt.test_iters)

                # Save predicted deformation
                deformations.append(deform.cpu().numpy()[0])

                # Compute symmetric Chamfer distance between the predicted and ground-truth point cloud
                cd_gt_pred, gt_dist = chamfer_distance(pc_gt * img_size / 2.0, pc_pred * img_size / 2.0)
                cd_pred_gt, deform_dist = chamfer_distance(pc_pred * img_size / 2.0, pc_gt * img_size / 2.0)
                cd_symmetric = cd_gt_pred[0] + cd_pred_gt[0]
                chamfer_dists.append(cd_symmetric.item())
                atlas_dist_gt.append(deform_dist.cpu().numpy()[0])

                # Convert predicted and ground-truth point cloud tensors into numpy arrays
                pc_gt = pc_gt[0].cpu().numpy()
                pc_pred = pc_pred[0].cpu().numpy()

                # Save point clouds
                np.save(f'{base_dir_inference_raw}/query_vessel_{i+1}.npy', pc_gt)
                np.save(f'{base_dir_inference_raw}/deformed_atlas_{i+1}.npy', pc_pred)

                # Unnormalize point clouds and distances to closest points
                pc_gt_unnorm = (pc_gt * 0.5 + 0.5) * img_size
                pc_pred_unnorm = (pc_pred * 0.5 + 0.5) * img_size
                gt_dist_norm = gt_dist[0].cpu().numpy()
                deform_dist_norm = deform_dist[0].cpu().numpy()

                # Visualize VANTAGE inference output
                visualize_point_clouds(pc_gt_unnorm, atlas.get_point_cloud(), pc_pred_unnorm, 
                                    gt_dist_norm, deform_dist_norm)
                plt.savefig(f'{base_dir_inference_vis}/alignment_{i+1}.png')
                plt.close()

                # Transfer, save, and visualize deformed atlas point cloud segmentation labels if necessary
                if pc_labels is not None and label_names is not None:
                    contours = VANTAGE_Atlas.visualize_aligned_segs(pc_gt, pc_pred, pc_labels, label_names)[3]
                    with open(f'{base_dir_inference_raw}/contours_{i+1}.pkl', 'wb') as file:
                        pickle.dump(contours, file)
                    plt.savefig(f'{base_dir_inference_vis}/labeled_{i+1}.png')
                    plt.close()
            
            # Stack all deformations into a single array for analysis in pixel space
            deformations = np.stack(deformations)

            # Stack and save all atlas per-point distances
            atlas_dist_gt = np.stack(atlas_dist_gt)
            np.save(f'{base_ds_dir_pc}/atlas_point_dist.npy', atlas_dist_gt)

            # Save training dataset atlas per-point distances
            if ds_type == 'Train':
                atlas_dist_gt_train = atlas_dist_gt.mean(0)

            # Visualize the atlas average per-point distances to the closest ground-truth point
            VANTAGE_Atlas.visualize_labeled_atlas_pc(atlas.get_point_cloud(unnormalize=False), atlas_dist_gt.mean(0),
                                                    title='Deformed Atlas Average Distance',
                                                    cbar_label='Average Distance to Ground-truth')
            plt.savefig(f'{base_ds_dir_pc}/atlas_avg_dist.png')
            plt.close()

            # Visualize the healthy-subtracted atlas average per-point distances to the closest ground-truth point
            if 'Train' not in ds_type and atlas_dist_gt_train is not None:
                sub_dists = np.clip(atlas_dist_gt.mean(0) - atlas_dist_gt_train, a_min=0, a_max=None)
                VANTAGE_Atlas.visualize_labeled_atlas_pc(atlas.get_point_cloud(unnormalize=False), sub_dists,
                                                        title='Deformed Atlas Average Distance',
                                                        cbar_label='Average Distance to Ground-truth')
                plt.savefig(f'{base_ds_dir_pc}/atlas_healthy_sub_avg_dist.png')
                plt.close()

            # Compute mean and std Chamfer distance across the dataset
            cd_results[ds_type] = chamfer_dists
            chamfer_dists = np.array(chamfer_dists)
            results[f'{fname}_chamfer_dist_mu'] = chamfer_dists.mean()
            results[f'{fname}_chamfer_dist_std'] = chamfer_dists.std()

            # Save the per-sample Chamfer distances
            np.save(f'{base_ds_dir_pc}/chamfer_dist.npy', chamfer_dists)

            #### Compute mean global deformation ####
            deform_px = deformations * img_size / 2.0

            # Compute mean and std of the global deformation vector
            pc_mean_deform_vec = np.linalg.norm(np.mean(deform_px, axis=0), axis=1)
            results[f'{fname}_global_deform_vec_mu'] = np.mean(pc_mean_deform_vec)
            results[f'{fname}_global_deform_vec_sigma'] = np.std(pc_mean_deform_vec)

            # Compute mean and std of the global deformation magnitude
            pc_mean_deform_mag = np.mean(np.linalg.norm(deform_px, axis=2), axis=1)
            results[f'{fname}_global_deform_mag_mu'] = np.mean(pc_mean_deform_mag)
            results[f'{fname}_global_deform_mag_sigma'] = np.std(pc_mean_deform_mag)

            # Visualize the atlas deformation field averaged across the population
            atlas.visualize_deformation(deformations.mean(0), title=f'Mean Atlas Deformation Field (n = {len(datasets[ds_type])})')
            plt.savefig(f'{base_ds_dir_pc}/mean_atlas_deform.png')
            plt.close()

            # Save the per-sample deformation fields
            np.save(f'{base_ds_dir_pc}/deformations.npy', deformations)

            #### Perform Principal Component Analysis (PCA) on the deformation field ####
            pca = PCA()
            coefficients = pca.fit_transform(deformations.reshape(deformations.shape[0], -1))

            # Create output directory for PCA results
            base_dir_pca = f'{base_ds_dir_pc}/pca/'
            os.makedirs(base_dir_pca, exist_ok=True)

            # Plot cumulative (explained) variance curve
            plot_pca_var_curve(pca.explained_variance_ratio_)
            plt.savefig(f'{base_dir_pca}/pca_variance.png')
            plt.close()

            # Compute max of deformation field for the top-20
            n_modes = min(opt.pca_modes, len(pca.components_))
            deform_norm = np.linalg.norm(pca.components_[:n_modes].reshape(n_modes, -1, 2), axis=2)
            max_deform = np.percentile(deform_norm, 99.5)

            # Visualize the deformation fields of the top-20 modes
            for i in tqdm(range(n_modes), desc='Saving PCA Modes'):
                mode = pca.components_[i].reshape(n_pc, 2)
                atlas.visualize_deformation(mode, title=f'Principal Mode {i+1} Deformation Field (PVE = {100 * pca.explained_variance_ratio_[i]:.1f}%)',
                                            clim=(0, max_deform))
                plt.savefig(f'{base_dir_pca}/pca_mode_{i+1}.png')
                plt.close()

            # Save PCA variance ratio and modes
            np.save(f'{base_dir_pca}/pca_variance.npy', pca.explained_variance_ratio_)
            np.save(f'{base_dir_pca}/pca_modes.npy', pca.components_.reshape(-1, n_pc, 2))
        
        # Create violin plot of the per-sample Chamfer distances of all datasets
        create_violinplot([np.sqrt(np.array(cd_results[l])) for l in ds_labels], [f'{l}ing' for l in ds_labels], 
                        title='Dataset Per-Sample Chamfer Distances', y_axis_label='√ Symmetric Chamfer Distance')
        plt.savefig(f'{base_dir_pc}/dataset_all_cd.png')
        plt.close()

        # Evaluate and visualize test data Chamfer distance disease classification performance
        if ds_labels[1] in datasets and ds_labels[2] in datasets:
            # Create list of test dataset Chamfer distances and their corresponding labels
            cd, y, test_ds_labels = [], [], ['Test', 'Disease Test']
            for l in test_ds_labels:
                cd = cd + cd_results[l]
                one_hot = 0.0 if l == 'Test' else 1.0
                y = y + [one_hot for _ in range(len(cd_results[l]))]
            
            # Compute test dataset Chamfer distance AUC and bootstrapped confidence intervals
            auc, lower_CI, upper_CI = compute_auc_ci(np.array(y), np.array(cd))
            results['test_cd_auc'] = auc
            results['test_cd_auc_CI_lower'] = lower_CI
            results['test_cd_auc_CI_upper'] = upper_CI

            # Conduct Mann–Whitney U Test for significance above chance (i.e. AUC > 0.5)
            results['test_mannwhitneyu_p_val'] = mannwhitneyu(cd_results['Test'], cd_results['Disease Test'],
                                                            alternative='two-sided')[1]

            # Create violin plot of the per-sample Chamfer distances of the test datasets
            test_ds_labels = ['Test', 'Disease Test']
            create_violinplot([np.sqrt(np.array(cd_results[l])) for l in test_ds_labels],
                                ['Healthy Subjects', 'Diseased Subjects' if opt.disease_label is None else f'{opt.disease_label} Subjects'], 
                                title='Test Dataset Per-Sample Chamfer Distances', y_axis_label='√ Symmetric Chamfer Distance')
            plt.savefig(f'{base_dir_pc}/dataset_test_cd.png')
            plt.close()

        # Append point cloud evaluation results to the csv file
        writer.writerow([results.get(h, '') for h in headers])

    print('Finished.')

    ###
    # Phase 2: Evaluation of the Atlas Segmentation Mask (if provided)
    ###

    # Load atlas segmentation mask (if provided) and compute quantification metrics
    if atlas_mask:
        # Create output directory if necessary
        base_dir_seg = f'{opt.output_path}/segmentation/'
        os.makedirs(base_dir_seg, exist_ok=True)
        
        # Quantify vessel metrics on the atlas segmentation mask
        print('\n##### Evaluating Atlas Segmentation Mask #####')
        print(f'Quantifying vessel mask across {len(opt.metrics)} metrics.')
        result = quantify_vessels((opt.atlas_seg, opt.disc_seg, opt.metrics, DiscSegmenter(), base_dir_seg))

        # Search through all quantification metric results
        aggr_results = []
        for metric in opt.metrics:
            # Save aggregated atlas quantification result
            aggr_results.append(result[metric][1])

        # Write atlas vessel quantification to output csv file
        with open(f'{base_dir_seg}/vessel_quantification.csv', 'w', newline='') as f:
            # Initialize csv writer and write header
            writer = csv.writer(f)
            writer.writerow(['case_id', 'vessel_path', 'disc_path'] + opt.metrics)
            
            # Append aggregated atlas quantification result to the csv results file
            writer.writerow([result['id'], result['vessel_path'], result['disc_path']] + aggr_results)
        
        print('Finished.')
    
    print(f'\nDone. Finished evaluating atlas.\n')

if __name__ == '__main__':
    main()