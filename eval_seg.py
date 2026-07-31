import numpy as np
from scipy.spatial import KDTree
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.colors import to_rgba
import os
import argparse
from typing import Optional, Tuple, List, Dict, Any
from tqdm import tqdm
import csv
import pickle
from utils.label_point_cloud import get_colormap

def parse_args():

    # Sample command line call:
    # >>> python3 eval_seg.py --labels ./results/example/labeled_healthy_test_fives/*.npz --pred_contours ./results/eval_atlas/point_cloud/test/inference/raw_results/contours_*.pkl --n_labels 4

    parser = argparse.ArgumentParser()

    parser.add_argument('--labels', nargs='+', type=str, required=True, help='Path to the point cloud and ground-truth segmentation labels as a *.npz file.')
    parser.add_argument('--pred_contours', nargs='+', type=str, required=True, help='Path to the corresponding predicted segmentation contours as a *.pkl file.')
    parser.add_argument('--img_size', type=int, default=2048, help='Original image size from which the point cloud was extracted.')
    parser.add_argument('--n_labels', type=int, default=None, help='First-n labels for point cloud segmentation evaluation.')
    parser.add_argument('--output_path', type=str, default='./results/eval_seg/', help='Output path for the point cloud segmentation evaluation results.')

    opt = parser.parse_args()
    return opt

def boundary_points(
    points: np.ndarray,
    labels: np.ndarray,
    k: int = 5
) -> None:
    # points with at least one neighbor of different class
    tree = KDTree(points)
    _, idx = tree.query(points, k=k)
    neighbor_labels = labels[idx]  # [N, k]
    is_boundary = (neighbor_labels != labels[:, None]).any(axis=1)
    return points[is_boundary]

def visualize_segs(
    query_pc: np.ndarray,
    gt_labels: np.ndarray,
    contours: Dict[int, List[List[float]]],
    label_names: Optional[List[str]] = None,
    lasso_lw: float = 2.0,
    title: Optional[str] = 'Retinal Vasculature — Atlas Label Transfer Contours vs. Ground-truth Labels',
    point_size: float = 6.0,
    figsize: Tuple[float, float] = (11, 9),
    dpi: int = 200
) -> Tuple[Any, Any]:
    """
    Transfer labels from the deformed atlas to the query point cloud using a
    k-nearest neighbor approach and visualize with coarse-grained contours
    overlaid for each label constructed using kernel density estimation (KDE).
    """
    # Define color schemes
    BACKGROUND = '#0a0a0f'
    PANEL_BG = '#0f0f18'
    GRID_COLOR = '#1e1e2e'
    TEXT_COLOR = '#e8e8f0'
    SUBTLE = '#6a6a8a'

    # Define colormap
    n_labels = gt_labels.max() if label_names is None else len(label_names)
    cmap = get_colormap(n_labels)
    
    # Create label color mapping
    color_mapping = {0: ('Unlabeled', 'gray')}
    if label_names is None:
        color_mapping.update({i: (f'Label {i}', cmap(i-1)) for i in range(1, n_labels+1)})
    else:
        color_mapping.update({i: (label_names[i-1], cmap(i-1)) for i in range(1, n_labels+1)})

    # Create figure
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BACKGROUND)
    fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=16, fontweight='600',
             color=TEXT_COLOR, fontfamily='monospace')
    fig.text(0.5, 0.945, 'Nearest-neighbor label transfer  ·  KDE contour approximation', fontweight='600',
             ha='center', va='top', fontsize=13, color=SUBTLE, fontfamily='monospace')

    # Configure figure axes
    gs = fig.add_gridspec(1, 1, left=0.04, right=0.78, top=0.91, bottom=0.05)
    ax = fig.add_subplot(gs[0])
    ax.set_facecolor(PANEL_BG)
    ax.set_aspect('equal')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a3e')
        spine.set_linewidth(0.8)
    ax.tick_params(colors=SUBTLE, labelsize=10, length=3, width=0.6)
    ax.grid(True, color=GRID_COLOR, linewidth=0.4, linestyle='--', alpha=0.7)
    ax.invert_yaxis()

    # Plot unlabeled points
    unlabeled_mask = gt_labels == 0
    if unlabeled_mask.any():
        unlabeled_color = to_rgba(color_mapping[0][1])
        ax.scatter(query_pc[unlabeled_mask,0], query_pc[unlabeled_mask,1], s=point_size,
                   color=unlabeled_color, linewidths=1, zorder=2, rasterized=True, alpha=0.90)

    # Plot query vessel point cloud with their transfered labels and coarse KDE lasso contours
    for label_id, label_contours in contours.items():
        # Create label mask and process if the mask contains at least one point
        mask = gt_labels == label_id
        if not mask.any():
            continue

        # Plot vessel points for each segmentation label
        label_color = color_mapping[label_id][1] if label_id in color_mapping else color_mapping[0][1]
        ax.scatter(query_pc[mask,0], query_pc[mask,1], s=point_size, color=to_rgba(label_color),
                   linewidths=1, zorder=2, rasterized=True, alpha=0.90)
        
        # Skip if the contour label ID is not a visualized label
        if label_id not in color_mapping:
            continue

        # Get label color
        color = color_mapping[label_id][1]
        rgba = to_rgba(color)

        # Draw the contours
        for contour in label_contours:
            # Fill in contours
            contour = np.array(contour)
            path = Path(contour)
            patch = PathPatch(path, facecolor=rgba, edgecolor='none', zorder=3, alpha=0.4)
            ax.add_patch(patch)
    
            # Draw dashed outline of the contour
            ax.plot(contour[:,0], contour[:,1], color=color, linewidth=lasso_lw, linestyle='dashed',
                    alpha=0.85, zorder=4, solid_capstyle='round', dash_capstyle='round')

    # Configure legend
    leg = fig.add_axes([0.80, 0.05, 0.18, 0.86])
    leg.set_facecolor(PANEL_BG)
    leg.set_xlim(0, 1)
    leg.set_ylim(0, 1)
    for spine in leg.spines.values():
        spine.set_edgecolor('#2a2a3e')
        spine.set_linewidth(0.8)
    leg.set_xticks([])
    leg.set_yticks([])

    # Add legend title text
    leg.text(0.5, 0.982, 'LABELS', ha='center', va='top', fontsize=13, color=SUBTLE, 
                fontfamily='monospace', fontweight='600')
    leg.axhline(0.945, color='#2a2a3e', linewidth=1.0)

    # Define configuration parameters for the legend
    named = [(lid, cfg) for lid, cfg in color_mapping.items()]
    n = len(named) + 1
    y_top = 0.905
    y_bot = 0.13
    y_step = (y_top - y_bot) / n

    # Create function for adding entries to the legend
    def add_entry(i: int, color: Any, name: str):
        # Add entries at particular offsets from the top of the legend
        y = y_top - i * y_step
        leg.add_patch(plt.Rectangle((0.07, y - 0.022), 0.15, 0.040,
                        facecolor=color, edgecolor='none', transform=leg.transAxes))
        words = name.split()
        lines, line = [], ''

        # Add text for each entry
        for w in words:
            if len(line) + len(w) + 1 > 15:
                lines.append(line.strip())
                line = w + ' '
            else:
                line += w + ' '
        if line.strip():
            lines.append(line.strip())
        leg.text(0.27, y, '\n'.join(lines), ha='left', va='center',
                    fontsize=10, color=TEXT_COLOR, fontfamily='monospace',
                    linespacing=1.35, transform=leg.transAxes)

    # Add entries
    for i, (lid, cfg) in enumerate(named):
        if i >= 10: break
        add_entry(i, cfg[1], cfg[0])

    return fig, ax

def main():
    # Parse command-line arguments
    opt = parse_args()

    # Number of point cloud label and predicted segmentation contour files must match
    if len(opt.labels) != len(opt.pred_contours):
        raise ValueError(f'The number of provided vessel point cloud and vessel label ' \
                         f'files {len(opt.labels)} and predicted segmentation countour ' \
                         f'files {len(opt.pred_contours)}.')
    
    # Number of labels must be a positive integer
    if opt.n_labels is not None and opt.n_labels <= 0:
        raise ValueError(f'First-n labels must be a positive integer > 0, but got {opt.n_labels} instead.')
    
    # Stores reference label names to determine if label classes are identical for all samples
    names_ref = None
    same_labels = True
    
    # Evaluate predicted segmentation contours with the ground-truth vessel labels
    dataset_iou, dataset_iou_per_class = [], []
    dataset_hd95, dataset_hd95_per_class = [], []
    for label_path, contour_path in tqdm(zip(opt.labels, opt.pred_contours)):
        # Load the point cloud and ground-truth label file
        label_file = np.load(label_path)
        names = label_file['names'] if opt.n_labels is None else label_file['names'][:opt.n_labels]
        labels = label_file['labels']

        # Load predicted contours
        with open(contour_path, 'rb') as file:
            contours = pickle.load(file)

        # Determine if label names match all other samples in the dataset
        if names_ref is None:
            names_ref = names
        elif not (names_ref == names).all():
            same_labels = False
        
        # Extract the point cloud
        pc = label_file['point_cloud']
        
        # Iterate through each class and compute per-class IoU
        iou_per_class, hd95_per_class = [], []
        for i in range(1, len(names)+1):
            # Load predicted contours to extract predicted labels for each point
            pred_label = np.zeros_like(labels, dtype=bool)
            if i in contours:
                for ct in contours[i]:
                    ct_mask = Path(ct).contains_points(pc)
                    pred_label = pred_label | ct_mask
            
            # Compute ground-truth label mask
            gt_label = labels == i

            # Compute the class-level IoU
            intersection = (pred_label & gt_label).sum()
            union = (pred_label | gt_label).sum()
            if pred_label.sum() == 0 or gt_label.sum() == 0:
                iou = np.nan
            else:
                iou = intersection / union
            iou_per_class.append(iou)

            # Compute class-level hausdorff distance
            pred_boundary = boundary_points((pc * 0.5 + 0.5) * opt.img_size, pred_label)
            gt_boundary = boundary_points((pc * 0.5 + 0.5) * opt.img_size, gt_label)

            # Compute forward distances between predicted to ground-truth labeled points
            tree_gt = KDTree(gt_boundary)
            dists_forward = tree_gt.query(pred_boundary)[0]

            # Compute reverse distances between ground-truth to predicted labeled points
            tree_pred = KDTree(pred_boundary)
            dists_rev = tree_pred.query(gt_boundary)[0]

            # Compute HD95
            if len(dists_forward) == 0 or len(dists_rev) == 0:
                hd95 = np.nan
            else:
                hd95 = max(np.percentile(dists_forward, 95), np.percentile(dists_rev, 95))
            hd95_per_class.append(hd95)
        
        # Compute overall IoU
        miou = np.nanmean(np.array(iou_per_class))

        # Compute overall HD95
        mhd95 = np.nanmean(np.array(hd95_per_class))

        # Create output directory if necessary
        per_sample_base_dir = f'{opt.output_path}/samples/'
        os.makedirs(per_sample_base_dir, exist_ok=True)

        # Create csv file with sample-level point cloud segmentation evaluation results
        fname = os.path.splitext(os.path.basename(label_path))[0].replace('_labeled', '')
        with open(f'{per_sample_base_dir}/{fname}.csv', 'w', newline='') as f:
            # Initialize csv writer and write header
            writer = csv.writer(f)
            writer.writerow(names.tolist() + ['Mean IoU'])
            writer.writerow(iou_per_class + [miou])
            writer.writerow([])
            writer.writerow(names.tolist() + ['Mean HD95'])
            writer.writerow(hd95_per_class + [mhd95])

        # Visualize ground-truth labels and predicted contours
        visualize_segs(pc, labels, contours, names.tolist())
        plt.savefig(f'{per_sample_base_dir}/{fname}.png')
        plt.close()

        # Save per-class IoU results
        dataset_iou.append(miou)
        dataset_iou_per_class.append(iou_per_class)

        # Save per-class HD95 results
        dataset_hd95.append(mhd95)
        dataset_hd95_per_class.append(hd95_per_class)

    # Create csv file with dataset-level point cloud segmentation evaluation results
    with open(f'{opt.output_path}/dataset_eval.csv', 'w', newline='') as f:
        # Initialize csv writer and write header
        writer = csv.writer(f)

        # Compute per-class evaluation results across the dataset if identical labels for all samples
        if same_labels:
            # Compute per-class mean and std IoU across the dataset
            per_class_iou = np.array(dataset_iou_per_class)
            per_class_iou_mu = np.nanmean(per_class_iou, axis=0)
            per_class_iou_std = np.nanstd(per_class_iou, axis=0)

            # Compute per-class mean and std HD95 across the dataset
            per_class_hd95 = np.array(dataset_hd95_per_class)
            per_class_hd95_mu = np.nanmean(per_class_hd95, axis=0)
            per_class_hd95_std = np.nanstd(per_class_hd95, axis=0)

            # Collate results
            results = (names_ref, per_class_iou_mu, per_class_iou_std, 
                       per_class_hd95_mu, per_class_hd95_std)

            # Construct headers and results array
            headers, results_iou, results_hd95 = [], [], []
            for name, iou_mu, iou_sigma, hd95_mu, hd95_sigma in zip(*results):
                headers = headers + [f'{name} (mu)', f'{name} (sigma)']
                results_iou = results_iou + [iou_mu, iou_sigma]
                results_hd95 = results_hd95 + [hd95_mu, hd95_sigma]
            
            # Compute mean IoU across the dataset
            miou = np.nanmean(per_class_iou, axis=1)
            iou_mu, iou_sigma = miou.mean(), miou.std()

            # Compute mean HD95 across the dataset
            mhd95 = np.nanmean(per_class_hd95, axis=1)
            hd95_mu, hd95_sigma = mhd95.mean(), mhd95.std()

            # Write results to csv
            writer.writerow(headers + ['Mean IoU (mu)', 'Mean IoU (sigma)'])
            writer.writerow(results_iou + [iou_mu, iou_sigma])
            writer.writerow([])
            writer.writerow(headers + ['Mean HD95 (mu)', 'Mean HD95 (sigma)'])
            writer.writerow(results_hd95 + [hd95_mu, hd95_sigma])
        else:
            # Compute mean IoUs
            per_sample_iou = [np.nanmean(np.array(sample)) for sample in dataset_iou_per_class]
            per_sample_iou = np.array(per_sample_iou)

            # Compute mean HD95s
            per_sample_hd95 = [np.nanmean(np.array(sample)) for sample in dataset_iou_per_class]
            per_sample_hd95 = np.array(per_sample_hd95)

            # # Write results to csv
            writer.writerow(['Mean IoU', 'Std IoU', 'Mean HD95', 'Std HD95'])
            writer.writerow([per_sample_iou.mean(), per_sample_iou.std(), 
                             per_sample_hd95.mean(), per_sample_hd95.std()])

if __name__ == '__main__':
    main()