import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba
import argparse
import os
from typing import Optional, Tuple, List, Any
import warnings

def parse_args():

    # Sample command line call:
    #   To label from scratch:  >>> python3 ./utils/label_point_cloud.py --pc ./results/example/atlas_fives/atlas_point_cloud.npy --names ./results/example/atlas_fives/labeled/label_names.txt
    #   To edit labels:         >>> python3 ./utils/label_point_cloud.py --pc ./results/example/atlas_fives/atlas_point_cloud.npy --labels ./results/example/atlas_fives/labeled/atlas_point_cloud_labeled.npz

    parser = argparse.ArgumentParser()

    parser.add_argument('--pc', type=str, required=True, help='Path to the point cloud as a *.npy file.')
    parser.add_argument('--labels', type=str, default=None, help='Path to the point cloud segmentation labels to pre-load as a *.npz file (must specify one of --labels, --names, or --n_labels).')
    parser.add_argument('--names', type=str, default=None, help='Path to the point cloud segmentation label names as a *.txt file (must specify one of --labels, --names, or --n_labels).')
    parser.add_argument('--n_labels', type=int, default=None, help='Number of foreground segmentation labels  (must specify one of --labels, --names, or --n_labels).')
    parser.add_argument('--output_path', type=str, default='./results/labeled/', help='Output path for the labeled/segmented atlas.')

    opt = parser.parse_args()
    return opt

def get_colormap(n_labels: int) -> Any:
    """
    Create colormap for point cloud segmentations.
    """
    if n_labels <= 8:
        return plt.get_cmap('Dark2')
    elif n_labels <= 20:
        return plt.get_cmap('tab20b')
    else:
        import distinctipy  # Only import distinctipy as a last resort if many labels are required
        return distinctipy.get_colormap(distinctipy.get_colors(n_labels))

def build_pc_selector(
    point_cloud: np.ndarray,
    labels: np.ndarray,
    n_labels: int,
    names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8.5, 7),
    point_size: float = 7.0
) -> np.ndarray:
    """
    Build a point cloud lasso selector and segmentation GUI.
    NOTE: Requires Matplotlib interactive GUI enabled, which
    may be disabled on linux servers and compute nodes.
    """
    # Number of label names should be more than the number of point cloud segmentation labels
    if names is not None and len(names) < n_labels:
        raise ValueError(f'Expected {n_labels} label names, but only {len(names)} were provided!')

    # Define colormap
    cmap = get_colormap(n_labels)
    
    # Create label color mapping
    color_mapping = {0: ('Unlabeled', 'lightgray')}
    if names is None:
        color_mapping.update({i: (f'Label {i}', cmap(i-1)) for i in range(1, n_labels+1)})
    else:
        color_mapping.update({i: (names[i-1], cmap(i-1)) for i in range(1, n_labels+1)})

    # Create figure and scatterplot
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(point_cloud[:,0], point_cloud[:,1], s=point_size, linewidths=1,
                         c=[to_rgba(color_mapping[l][1]) for l in labels])

    # Create legend outside the plot to avoid obscuring the point cloud
    legend_elements = []
    for i, (name, c) in color_mapping.items():
        # Add legend elements, truncating after 27 elements
        if i > 27: break
        legend_elements.append(Patch(facecolor=c, label=name))
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), borderpad=1.0)

    # Label indicator text
    current_label = [1]
    ax.set_title(f'Active label: {color_mapping[current_label[0]][0]}  |  Label number + "Enter" to switch, Z to clear',
                 fontsize=12.5, pad=12)

    # Define point selection handler
    def on_select(verts) -> None:
        """
        Set labels and color for selected points.
        """
        path = Path(verts)
        mask = path.contains_points(point_cloud)
        labels[mask] = current_label[0]
        scatter.set_facecolor([to_rgba(color_mapping[l][1]) for l in labels])
        fig.canvas.draw_idle()

    # Buffer for storing key inputs 
    key_buffer = []

    # Define key press handler
    def on_key(event) -> None:
        """
        Change selected point cloud label or reset all point cloud labels.
        """
        # Add number to key input buffer
        if event.key in '0123456789':
            key_buffer.append(event.key)
        
        # Change selected label
        elif event.key == 'enter':
            if key_buffer:
                num = int(''.join(key_buffer))
                key_buffer.clear()
                if num in color_mapping:
                    current_label[0] = num
                    ax.set_title(f'Active label: {color_mapping[current_label[0]][0]}',
                                 fontsize=12.5, pad=12)
                    fig.canvas.draw_idle()
        
        # Reset all points with current label back to 0
        elif event.key == 'z':
            labels[labels == current_label[0]] = 0
            scatter.set_facecolor([to_rgba(color_mapping[l][1]) for l in labels])
            fig.canvas.draw_idle()

    # Create grid marks
    ax.grid(True, linewidth=0.5, alpha=0.5, color='gray', linestyle='--')
    ax.set_aspect('equal')
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Define lasso point cloud selector
    fig._lasso = LassoSelector(ax, on_select)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout()
    return labels, fig, ax

def visualize_pc_segmentation(
    point_cloud: np.ndarray,
    labels: np.ndarray,
    names: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (10, 7),
    title: Optional[str] = 'Point Cloud Segmentation Labels',
    point_size: float = 7.0,
    dpi: int = 300
) -> Tuple[Any, Any]:
    """
    Create a scatterplot visualizing point cloud segmentation labels.
    """
    # Number of label names should be more than the number of point cloud segmentation labels
    if names is not None and len(names) < labels.max():
        raise ValueError(f'Expected {labels.max()} label names, but only {len(names)} were provided!')
    
    # Define colormap
    n_labels = labels.max() if names is None else len(names)
    cmap = get_colormap(n_labels)
    
    # Create label color mapping
    color_mapping = {0: ('Unlabeled', 'lightgray')}
    if names is None:
        color_mapping.update({i: (f'Label {i}', cmap(i-1)) for i in range(1, n_labels+1)})
    else:
        color_mapping.update({i: (names[i-1], cmap(i-1)) for i in range(1, n_labels+1)})

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot each class separately
    for label_idx, (name, color) in color_mapping.items():
        # Create mask of points of the particular label
        mask = labels == label_idx
        if not mask.any(): continue

        # Plot points of the label as a scatterplot
        ax.scatter(point_cloud[mask,0], point_cloud[mask,1], c=[color], s=point_size, label=name,
                   linewidths=1, alpha=0.9, rasterized=True)

    # Create legend outside the plot to avoid obscuring the point cloud
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), edgecolor='#cccccc',
              fontsize=10, markerscale=2.5, borderpad=1.0)

    # Configure figure axes
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create grid marks
    ax.grid(True, linewidth=0.4, alpha=0.4, color='gray', linestyle='--')
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    plt.tight_layout(pad=3)
    return fig, ax, [color_mapping[i][0] for i in range(1, labels.max()+1)]

def main():
    # Parse command-line arguments
    opt = parse_args()

    # Load the point cloud and create segmentation labels
    pc = np.load(opt.pc)
    fname = os.path.splitext(os.path.basename(opt.pc))[0]

    # Load label names if necessary
    if opt.names is not None:
        label_names = []
        with open(opt.names, 'r') as f:
            label_names = f.read().splitlines()
    else:
        label_names = None

    # Set number of labels if necessary
    if opt.names is not None and opt.n_labels is None:
        opt.n_labels = len(label_names)

    # Pre-load segmentation labels if necessary
    labels = np.zeros(pc.shape[0], dtype=int)
    if opt.labels is not None:
        with np.load(opt.labels) as data:
            # Check that the loaded point cloud segmentation labels are associated with the correct point cloud
            if not np.isclose(data['point_cloud'], pc).all():
                raise OSError('Loaded point cloud segmentation labels as associated with a different ' \
                              'point cloud than that specified by --pc!')
            
            # Load labels
            labels = data['labels'].astype(int)

            # Use pre-loaded segmentation labels if number of labels is not specified
            if opt.n_labels is None:
                opt.n_labels = labels.max()

            # Use pre-loaded segmentation label names if not provided
            if opt.names is None:
                label_names = data['names']

            # Check that the number of labels matches the specified number of labels
            if labels.max() > opt.n_labels:
                warnings.warn(f'Loaded point cloud segmentation labels contains {labels.max()} labels, ' \
                              f'but only {opt.n_labels} labels were specified by --n_labels. Overriding the ' \
                              'value of --n_labels.')
                opt.n_labels = labels.max()

    # Number of labels must be specified
    if opt.n_labels is None:
        raise TypeError('Must specify one of the following arguments: --labels, --names, or --n_labels!')

    # Build point cloud selector to label an point cloud
    print('Loading point cloud selector.')
    pc_labels = build_pc_selector(pc, labels, opt.n_labels)[0]
    plt.show()

    # Save labeling outputs if necessary
    if (pc_labels == 0).all():
        print('Done. No point cloud segmentations to save.')
    else:
        # Create output directory if necessary
        os.makedirs(opt.output_path, exist_ok=True)
        
        # Visualize the point cloud segmentation labels
        label_names = visualize_pc_segmentation(pc, pc_labels, label_names)[2]
        plt.savefig(f'{opt.output_path}/{fname}_labeled.png')
        plt.close()

        # Save point cloud raw data and labels
        pc_labeled_path = f'{opt.output_path}/{fname}_labeled.npz'
        np.savez(pc_labeled_path, point_cloud=pc, labels=pc_labels, names=label_names)
        print('Done. Saved point cloud segmentations.')

if __name__ == '__main__':
    main()