import torch
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from matplotlib.colors import to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import label
from scipy.spatial import KDTree, cKDTree
from scipy import stats
import cv2
from typing import Tuple, Optional, Dict, List, Any
from tqdm import tqdm

from VANTAGE.vantage import VANTAGE, DeltaSampling
from util.label_point_cloud import get_colormap
from util.apml import AdaptiveProbabilisticMatchingLoss

class VANTAGE_Atlas():
    """
    Helper class for a VANTAGE-based 2D point cloud atlas containing
    learned centerline and radial offsets.
    """
    def __init__(self, model: VANTAGE, image_size: int = 2048) -> None:
        # Centerline points as an unordered point cloud and their corresponding log-radii
        self.centerline_pts = model.atlas_pc.detach().cpu().numpy()[0]
        if model.pc_norm_offset:
            self.log_radii = model.atlas_delta.detach().cpu().numpy()[0]

        # Model and output image size
        self.model = model
        self.image_size = image_size

    def align_sample(
        self,
        query_graph: Batch,
        query_pc: torch.Tensor,
        n_templates: int = 2,
        n_iters: int = 0,
        lr: float = 0.001,
        weight_decay: float = 0.001,
        sampling: DeltaSampling = DeltaSampling.DOUBLE_PC
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align the learned VANTAGE-based atlas to a query vessel point cloud through
        deformable registration given its corresponding vessel graph.
        """
        # Predict the VANTAGE encoder to get vessel graph embeddings
        self.model.eval()
        z = self.model.encoder(query_graph, n_templates)
        
        # Optimize for a better latent embedding
        if n_iters > 0:
            # Initialize point cloud matching APML
            apm_loss = AdaptiveProbabilisticMatchingLoss()
            
            # Save encoder-predicted embeddings and wrap as a parameter
            z_init = z.detach().clone()
            z = torch.nn.Parameter(z)

            # Initialize latent space optimizer
            optim = torch.optim.AdamW([z], lr=lr, weight_decay=weight_decay)
            
            # Optimize for a better latent embedding on testing datasets
            pbar = tqdm(range(n_iters), leave=False)
            for _ in pbar:
                # Predict the VANTAGE decoder
                if self.model.pc_norm_offset:
                    pc_pred, deform, _, deform_delta = self.model.decode(z)
                    mag_loss = torch.linalg.norm(deform_delta, dim=1).mean()
                else:
                    pc_pred, deform = self.model.decode(z)
                    mag_loss = 0
                
                # Compute magnitude regularization
                mag_loss = mag_loss + torch.linalg.norm(deform, dim=(1, 2)).mean()

                # Compute embedding regularization
                reg_loss = torch.linalg.norm(z_init - z, dim=1).mean()

                # Compute loss and update progress bar
                loss = apm_loss(pc_pred, query_pc) + mag_loss + reg_loss
                pbar.set_postfix({'loss': f'{loss.item():.2f}'})

                # Backpropagate and update parameters
                optim.zero_grad()
                loss.backward()
                optim.step()

        # Switch off gradients and inference the VANTAGE decoder
        with torch.inference_mode():
            if self.model.pc_norm_offset:
                pc_pred, deform, _, _ = self.model.decode(z)
            else:
                pc_pred, deform = self.model.decode(z)

        return pc_pred, deform

    def get_raw_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Outputs raw centerline point cloud and log-radius atlas representation, if
        available, in normalized [-1, 1] coordinate space without post-processing.
        """
        return self.centerline_pts, self.log_radii if self.model.pc_norm_offset else None

    def get_point_cloud(
        self,
        sampling: DeltaSampling = DeltaSampling.DOUBLE_PC,
        unnormalize: bool = True
    ) -> np.ndarray:
        """
        Constructs a full atlas point cloud from unordered centerline points and
        their corresponding radii by using principal component analysis (PCA) to
        estimate the normal vectors along the vessel trajectory.
        """
        # Get atlas point cloud as (x, y) coordinates in normalized [-1, 1] coordinate space
        atlas_pc = self.model.get_atlas(sampling).detach().cpu().numpy()

        # Unnormalize point cloud into (x, y) coordinates in image space [0, image_size] if necessary
        if unnormalize:
            atlas_pc = atlas_pc * 0.5 + 0.5
            atlas_pc *= self.image_size
        
        return atlas_pc

    def get_segmask(self) -> np.ndarray:
        """
        Constructs a full atlas segmentation mask from unordered centerline points
        and their corresponding radii by using a k-nearest neighbor approach.
        """
        # Implementation incompatible with a VANTAGE atlas learned without radial offsets
        if not self.model.pc_norm_offset:
            raise ValueError('Implementation of the segmentation mask reconstruction algorithm is currently ' \
                             'only compatible with a VANTAGE atlas with both centerlines and radial offsets!')

        # Construct segmentation mask
        return self.construct_segmask(self.centerline_pts, self.log_radii)

    def construct_segmask(self, centerline_pts: np.ndarray, log_radii: np.ndarray) -> np.ndarray:
        """
        Constructs a segmentation mask from given a set of unordered centerline points
        and their corresponding radii by using a k-nearest neighbor approach.

        Expects centerline points and log-radii in normalized [-1, 1] coordinate space.
        """
        # Construct segmentation mask
        mask = self._atlas_to_segmask(centerline_pts, np.exp(log_radii), self.image_size)

        # Apply light post-processing to improve atlas quality and topological continuity
        return self._close_fragments(mask)

    @staticmethod
    def visualize_atlas(
        atlas_points: np.ndarray,
        atlas_mask: Optional[np.ndarray] = None,
        mean_disc_mask: Optional[np.ndarray] = None,
        point_size: float = 3,
        figsize: Tuple[float, float] = (12, 4.5),
        dpi: int = 300
    ) -> Tuple[Any, Any]:
        """
        Visualize the atlas point cloud, optionally with a vessel segmentation
        mask and the optic disc overlaid. Expects the atlas point cloud in
        normalized [-1, 1] coordinate space.
        """
        # Extract image dimensions and set background color
        if atlas_mask is not None:
            H, W = atlas_mask.shape
        elif mean_disc_mask is not None:
            H, W = mean_disc_mask.shape
        else:
            H, W = None, None
        bg_color = '#0d1117'

        # Create figure and subplots
        n_subplots = 1 + (atlas_mask is not None) + (mean_disc_mask is not None)
        fig, axes = plt.subplots(1, n_subplots, figsize=figsize, facecolor=bg_color, dpi=dpi)
        
        ###
        # Panel 1: Display the mean optic disc (if provided) 
        # and contours based on population consensus
        ###
        if mean_disc_mask is not None:
            # Create optic disc overlay based on the amount of overlap across the population
            disc_rgba = np.zeros((H, W, 4), dtype=np.float32)
            disc_rgba[:,:,0] = mean_disc_mask
            disc_rgba[:,:,1] = mean_disc_mask * 0.6
            disc_rgba[:,:,2] = mean_disc_mask * 0.1
            disc_rgba[:,:,3] = mean_disc_mask

            # Set panel background color
            axes[0].set_facecolor(bg_color)

            # Display the optic disc with 25%, 50%, and 75% population consensus contours
            axes[0].set_title('Mean Optic Disc', color='white', fontsize=14, pad=10)
            axes[0].imshow(np.zeros((H, W, 3)), extent=[-1, 1, -1, 1])
            axes[0].imshow(disc_rgba, extent=[-1, 1, -1, 1], origin='lower')
            axes[0].contour(np.linspace(-1, 1, W), np.linspace(-1, 1, H), mean_disc_mask,
                            levels=[0.25, 0.5, 0.75], colors=['#FF6B6B', '#FFD93D', '#FFFFFF'],
                            linewidths=[1.4, 1.2, 1.0], alpha=0.9)

            # Add legend
            legend_disc = [plt.Line2D([0], [0], color='#FF6B6B', linewidth=1.0, label='25% consensus'),
                           plt.Line2D([0], [0], color='#FFD93D', linewidth=1.2, label='50% consensus'),
                           plt.Line2D([0], [0], color='#FFFFFF', linewidth=1.4, label='75% consensus')]
            axes[0].legend(handles=legend_disc, loc='lower left', fontsize=11, framealpha=0.3, 
                           facecolor='#1a1a2e', labelcolor='white', edgecolor='none')
            axes[0].set_xlim(-1, 1)
            axes[0].set_ylim(-1, 1)
            axes[0].axis('off')
            axes[0].invert_yaxis()

        ###
        # Panel 2: Display atlas point cloud with the 50% consensus
        # mean optic disc overlaid (if provided)
        ###
        panel_idx = 0 if mean_disc_mask is None else 1

        # Set panel background color
        axes[panel_idx].set_facecolor(bg_color)

        # Display scatterplot of the atlas point cloud
        if H is not None and W is not None:
            axes[panel_idx].imshow(np.zeros((H, W, 3)), extent=[-1, 1, -1, 1])
        axes[panel_idx].scatter(atlas_points[:,0], atlas_points[:,1], c='red', s=point_size,
                                linewidths=1, zorder=2)

        # Overlay the optic disc over atlas point cloud if provided
        axes[panel_idx].set_title('VANTAGE Point Cloud', color='white', fontsize=14, pad=10)
        if mean_disc_mask is not None:
            # Create the optic disc overlay with the 50% population consensus contour
            axes[panel_idx].imshow(disc_rgba, extent=[-1, 1, -1, 1], origin='lower', alpha=0.7, zorder=2)
            axes[panel_idx].contour(np.linspace(-1, 1, W), np.linspace(-1, 1, H), mean_disc_mask,
                                    levels=[0.5], colors=['#FFD93D'], linewidths=[1.5], zorder=3)
            
            # Display legend
            legend_overlay = [plt.scatter([], [], c='red', s=20, alpha=0.8, label='Vascular atlas'),
                              plt.Line2D([0], [0], color='#FFD93D', linewidth=1.5, label='50% consensus boundary')]
            axes[panel_idx].legend(handles=legend_overlay, loc='lower left', fontsize=11, framealpha=0.3, 
                                   facecolor='#1a1a2e', labelcolor='white', edgecolor='none')
        
        # Configure subplot axes
        axes[panel_idx].set_xlim(-1, 1)
        axes[panel_idx].set_ylim(-1, 1)
        axes[panel_idx].set_aspect('equal')
        axes[panel_idx].axis('off')
        axes[panel_idx].invert_yaxis()

        ###
        # Panel 3: Display atlas segmentation mask (if provided) with the 
        # 50% consensus mean optic disc overlaid (if provided)
        ###
        if atlas_mask is not None:
            panel_idx = 1 if mean_disc_mask is None else 2

            # Set panel background color
            axes[panel_idx].set_facecolor(bg_color)

            # Set RGBA color of the vessel mask
            vessel_rgba = np.zeros((H, W, 4), dtype=np.float32)
            vessel_rgba[atlas_mask > 0] = [0.12, 0.68, 0.66, 1.0]

            # Display segmentation mask
            axes[panel_idx].imshow(np.zeros((H, W, 3)), extent=[-1, 1, -1, 1])
            axes[panel_idx].imshow(vessel_rgba, extent=[-1, 1, -1, 1], origin='lower', zorder=1)
            
            # Overlay the optic disc over atlas segmentation mask if provided
            axes[panel_idx].set_title('VANTAGE Segmentation Mask', color='white', fontsize=14, pad=10)
            if mean_disc_mask is not None:
                # Create the optic disc overlay with the 50% population consensus contour
                axes[panel_idx].imshow(disc_rgba, extent=[-1, 1, -1, 1], origin='lower', alpha=0.7, zorder=2)
                axes[panel_idx].contour(np.linspace(-1, 1, W), np.linspace(-1, 1, H), mean_disc_mask,
                                        levels=[0.5], colors=['#FFD93D'], linewidths=[1.5], zorder=3)
                
                # Display legend
                legend_overlay = [Patch(facecolor='#1FAEAA', alpha=0.8, label='Vascular atlas'),
                                plt.Line2D([0], [0], color='#FFD93D', linewidth=1.5, label='50% consensus boundary')]
                axes[panel_idx].legend(handles=legend_overlay, loc='lower left', fontsize=11, framealpha=0.3,
                                    facecolor='#1a1a2e', labelcolor='white', edgecolor='none')
            
            axes[panel_idx].set_xlim(-1, 1)
            axes[panel_idx].set_ylim(-1, 1)
            axes[panel_idx].axis('off')
            axes[panel_idx].invert_yaxis()

        plt.subplots_adjust(wspace=0.05, left=0.02, right=0.98, top=0.92, bottom=0.02)
        return fig, axes

    @staticmethod
    def visualize_labeled_atlas_pc(
        atlas_pc: np.ndarray,
        distance: np.ndarray,
        figsize: Tuple[float, float] = (8, 7),
        title: Optional[str] = 'Deformed Atlas Average Distance',
        cbar_label: Optional[str] = 'Average Distance to Ground-truth (px)',
        dpi: int = 300
    ) -> Tuple[Any, Any]:
        """
        Visualize the atlas with per-point labels.
        """
        # Create colormap
        colors_list = [(0.05, 0.10, 0.30), (0.10, 0.45, 0.70), (0.20, 0.80, 0.80),
                       (0.95, 0.85, 0.30)]
        cmap = mcolors.LinearSegmentedColormap.from_list('vascular', colors_list, N=512)

        # Set color normalization scheme
        r_min, r_max = np.percentile(distance, 2), np.percentile(distance, 98)
        norm = mcolors.Normalize(vmin=r_min, vmax=r_max)

        # Create figure
        fig = plt.figure(figsize=figsize, facecolor='#0a0a0a', dpi=dpi)
        ax = fig.add_axes([0.08, 0.08, 0.82, 0.82], facecolor='#0a0a0a')

        # Plot color-coded scatterplot of the atlas
        sc = ax.scatter(atlas_pc[:,0], -atlas_pc[:,1], c=distance, cmap=cmap, norm=norm,
                        s=20, alpha=0.88, linewidths=0, rasterized=True)

        # Configure axes
        ax.set_xlim([-1.02, 1.02])
        ax.set_ylim([-1.02, 1.02])
        ax.set_aspect('equal', adjustable='box')

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        ax.set_title(title, color='white', fontsize=20, pad=10)

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.12)
        cax.set_facecolor('#0a0a0a')

        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label(cbar_label, color='white', fontsize=14, labelpad=8)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=12, labelcolor='white')
        cbar.outline.set_edgecolor('#333333')

        # Set colorbar tickmark labels
        tick_vals = np.linspace(r_min, r_max, 5)
        cbar.set_ticks(tick_vals)
        cbar.set_ticklabels([f'{v:.1f}' for v in tick_vals])
        return fig, ax

    @staticmethod
    def visualize_aligned_segs(
        query_pc: np.ndarray,
        deformed_atlas: np.ndarray,
        atlas_labels: np.ndarray,
        label_names: Optional[List[str]] = None,
        k: int = 9,
        max_distance: Optional[float] = None,
        kde_level: float = 0.25,
        lasso_lw: float = 2.0,
        title: Optional[str] = 'Retinal Vasculature — Coarse Atlas Label Transfer',
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
        n_labels = atlas_labels.max() if label_names is None else len(label_names)
        cmap = get_colormap(n_labels)
        
        # Create label color mapping
        color_mapping = {0: ('Unlabeled', 'gray')}
        if label_names is None:
            color_mapping.update({i: (f'Label {i}', cmap(i-1)) for i in range(1, n_labels+1)})
        else:
            color_mapping.update({i: (label_names[i-1], cmap(i-1)) for i in range(1, n_labels+1)})

        # Transfer labels from the deformed atlas to the query vessels
        transferred = VANTAGE_Atlas._transfer_labels(
            query_pc, deformed_atlas, atlas_labels, k=k, max_distance=max_distance
        )

        # Create coarse lassos with kernel density estimation around each atlas label
        lassos = VANTAGE_Atlas._compute_kde_lasso(deformed_atlas, atlas_labels)

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

        # Plot query vessel point cloud with their transfered labels
        for label_id, cfg in color_mapping.items():
            # Create label mask and process if the mask contains at least one point
            mask = transferred == label_id
            if not mask.any():
                continue

            # Plot vessel points for each segmentation label
            ax.scatter(query_pc[mask,0], query_pc[mask,1], s=point_size, color=to_rgba(cfg[1]),
                       linewidths=1, zorder=2, rasterized=True, alpha=0.90)

        # Plot coarse KDE lasso contours
        contours = {}
        for label_id, (xx, yy, density) in lassos.items():
            # Get label color
            color = color_mapping[label_id][1]
            rgba = to_rgba(color)
    
            # Create temporary subplot to gather contours
            fig_tmp, ax_tmp = plt.subplots()
            cs = ax_tmp.contour(xx, yy, density, levels=[kde_level])

            # Collate contours into a single list
            kde_contours = []
            for seg_group in cs.allsegs:
                for seg in seg_group:
                    kde_contours.append(seg)
            plt.close(fig_tmp)

            # Draw the contours
            contours[label_id] = []
            for contour in kde_contours:
                # Contours must have 3+ points to create a 2D shape
                if contour is None or len(contour) < 3:
                    continue

                # Save contour to global dictionary of all contours
                contours[label_id].append(contour.tolist())
        
                # Fill in contours
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
    
        return fig, ax, transferred, contours

    @staticmethod
    def _transfer_labels(
        query_pc: np.ndarray,
        deformed_atlas: np.ndarray,
        atlas_labels: np.ndarray,
        k: int = 9,
        max_distance: Optional[float] = None
    ) -> np.ndarray:
        """
        Transfer labels from the deformed atlas point cloud to
        a query point cloud using a k-nearest neighbor majority
        vote approach.
        """
        # Create a KDTree for point clouds on the deformed atlas
        tree = cKDTree(deformed_atlas)

        # Find k-nearest neighbors for every point on the query point cloud
        distances, indices = tree.query(query_pc, k=max(k, 1))

        # Transfer labels directly for nearest neighbor (k = 1)
        if k == 1:
            transferred = atlas_labels[indices].copy()
            if max_distance is not None:
                transferred[distances > max_distance] = 0
            return transferred

        # Extract the labels for the multiple points for k > 1
        neighbor_labels = atlas_labels[indices]
        transferred = np.full(len(query_pc), -1, dtype=int)

        # Majority vote across the points to assign label
        for i, (labs, dists) in enumerate(zip(neighbor_labels, distances)):
            valid = labs[dists <= max_distance] if max_distance is not None else labs
            valid = valid[valid != -1]
            if len(valid) > 0:
                transferred[i] = stats.mode(valid, keepdims=False).mode
        
        return transferred

    @staticmethod
    def _compute_kde_lasso(
        point_cloud: np.ndarray,
        labels: np.ndarray,
        grid_size: int = 400,
        bw_method: float = 0.35
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create lassos around point clouds for each label using
        kernel density estimation (KDE).
        """
        # Create a fixed grid for KDE
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        xx, yy = np.meshgrid(x, y)
        grid = np.vstack([xx.ravel(), yy.ravel()])

        # Create KDE lassos for each atlas label
        lassos = {}
        for label_id in np.unique(labels):
            # Skip unlabeled points
            if label_id <= 0:
                continue

            # Filter deformed atlas points for the particular label
            pts = point_cloud[labels == label_id]
            if len(pts) < 5:
                continue
            try:
                # Initialize KDE estimator 
                kde = stats.gaussian_kde(pts.T, bw_method=bw_method)

                # Estimate probability density on a fixed grid
                density = kde(grid).reshape(grid_size, grid_size)
                density /= density.max()
                lassos[label_id] = (xx, yy, density)
            except Exception as e:
                continue
        
        return lassos

    def visualize_deformation(
        self,
        deform_field: np.ndarray,
        title: Optional[str] = None,
        cmap: str = 'inferno',
        sampling: DeltaSampling = DeltaSampling.DOUBLE_PC,
        figsize: Tuple[float, float] = (7, 7),
        quiver_width: float = 0.0015,
        clim: Optional[Tuple[float, float]] = None,
        dpi: int = 300
    ) -> Tuple[Any, Any]:
        """
        Visualize deformation field of the atlas as a quiver plot.
        Expects the deformation field in normalized [-1, 1] coordinate
        space.
        """
        # Sample atlas point cloud
        point_cloud = self.get_point_cloud(sampling)

        # Construct deformation field and per-point deformation magnitude
        if self.model.pc_norm_offset and sampling == DeltaSampling.DOUBLE_PC:
            deform_field = np.concatenate(2 * [deform_field], axis=0)
        deform_field *= self.image_size / 2.0
        deform_mag = np.linalg.norm(deform_field, axis=1)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor('#0d0d0d')
        ax.set_facecolor('#0d0d0d')

        # Create scatter plot of the atlas point cloud
        ax.set_title(title, fontsize=20, color='white', pad=14)
        ax.scatter(point_cloud[:,0], point_cloud[:,1], s=1.0, c='#2a2a2a', linewidths=1, zorder=1, rasterized=True)

        # Normalize data range and create colormap
        if clim is None:
            norm = mcolors.Normalize(vmin=0, vmax=deform_mag.max())
        else:
            norm = mcolors.Normalize(vmin=clim[0] * self.image_size / 2.0, vmax=clim[1] * self.image_size / 2.0)
        cmap = plt.get_cmap(cmap)
        colors = cmap(norm(deform_mag))

        # Create quiver plot of the deformation field
        ax.quiver(point_cloud[:,0], point_cloud[:,1], deform_field[:,0], deform_field[:,1], color=colors, 
                  angles='xy', scale_units='xy', scale=1, width=quiver_width, headwidth=6, headlength=6,
                  headaxislength=5, zorder=3, rasterized=True)

        # Configure colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.1)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label('Deformation Magnitude (px)', fontsize=14, color='white', labelpad=10)
        cbar.ax.yaxis.set_tick_params(color='white', labelsize=12, labelcolor='white')
        cbar.outline.set_edgecolor('#444444')

        # Configure figure axes
        ax.set_aspect('equal')
        ax.set_xlim(0, self.image_size)
        ax.set_ylim(self.image_size, 0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        plt.tight_layout()
        return fig, ax

    def _atlas_to_segmask(
        self,
        centerline: np.ndarray,
        radii: np.ndarray,
        image_size: int = 2048,
        k_neighbors: int = 3,
        max_edge_length: float = 0.055,
        angle_threshold_deg: float = 50.0,
        max_bridge_dist: float = 0.06,
        min_fragment_px: int = 10
    ) -> np.ndarray:
        """
        Estimates the segmentation mask from centerlines and radial offsets
        using a k-nearest neighbors approach. Prospective nearest neighbor
        candidate points are filtered based on consistency with tangent vectors
        estimated with local neighborhood principal component analysis (PCA).
        Neighboring fragments are subsequently bridged to improve connectivity.

        Expects the centerline points, radii, and maximum edge length to be in
        normalized [-1, 1] and (x, y) coordinate space.
        """
        # Convert to pixel space
        cl_px = centerline * 0.5 + 0.5
        cl_px *= image_size
        radii_px = radii * image_size / 2.0

        # KNN on centerline
        tree = KDTree(centerline)
        distances, indices = tree.query(centerline, k=k_neighbors+1)

        # Compute tangent vectors using PCA
        tangents = self._estimate_tangents_pca(centerline, indices, distances, max_edge_length)
        cos_thresh = np.cos(np.radians(angle_threshold_deg))

        # Array for the atlas segmentation mask
        mask = np.zeros((image_size, image_size), dtype=np.uint8)

        # Fill in mask for the corresponding radius at every centerline point and connect with neighbors
        for i in range(len(centerline)):
            # Draw a circle at every centerline point
            pt = (int(round(cl_px[i,0])), int(round(cl_px[i,1])))
            r = max(1, int(round(radii_px[i])))
            cv2.circle(mask, pt, r, 1, -1)

            # Connect to neighbors with thick capsules
            for k in range(1, indices.shape[1]):
                # Get index of and distance to neigbor
                j, dist = indices[i,k], distances[i,k]

                # Skip nearest points that are either too far or have already been processed
                if dist > max_edge_length: break
                if j <= i: continue

                # Extract the unit direction vector of the nearest neighbor
                edge_vec = centerline[j] - centerline[i]
                edge_len = np.linalg.norm(edge_vec)

                if edge_len < 1e-6:
                    continue
                
                edge_dir = edge_vec / edge_len

                # Do not connect the two points if the edge deviates too far from their tanget vectors
                cos_i = abs(np.dot(edge_dir, tangents[i]))
                cos_j = abs(np.dot(edge_dir, tangents[j]))
                
                if cos_i < cos_thresh and cos_j < cos_thresh:
                    continue
                
                # Edge is consistent with their tangent vectors -- connect the two centerline points
                pt_i = (int(round(cl_px[i,0])), int(round(cl_px[i,1])))
                pt_j = (int(round(cl_px[j,0])), int(round(cl_px[j,1])))
                r_avg = max(1, int(round((radii_px[i] + radii_px[j]) / 2.0)))
                cv2.line(mask, pt_i, pt_j, 1, thickness=2*r_avg, lineType=cv2.LINE_8)

        # Post-process the segmentation mask by fusing disconnected fragments
        mask = self._defragment_segmask(mask, cl_px, radii_px, max_bridge_dist, min_fragment_px)
        return mask

    def _defragment_segmask(
        self,
        mask: np.ndarray,
        centerline_px: np.ndarray,
        radii_px: np.ndarray,
        max_bridge_dist: float = 0.06,
        min_fragment_px: int = 10
    ) -> np.ndarray:
        """
        De-fragments a segmentation mask by bridging disconnected, neighboring
        connected components.

        Input centerline points and radii are assumed to be in image space 
        and (x, y) coordinate space. Maximum bridge gap is expected to be in
        normalized [-1, 1] coordinate space for metric consistency with
        previous functions.
        """
        # Identify connected components
        mask = mask.astype(np.uint8)
        n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(mask)
        filtered = np.zeros_like(mask)
        component_of_point = np.zeros(len(centerline_px), dtype=int)
        
        # Only connected components larger than the minimum size are eligible to be bridged
        for i in range(1, n_labels):
            if stats[i,cv2.CC_STAT_AREA] >= min_fragment_px:
                filtered[labeled == i] = 1
        
        # Tag each centerline point with its connected component ID
        for i, (row, col) in enumerate(centerline_px.astype(int)):
            row = np.clip(row, 0, mask.shape[0]-1)
            col = np.clip(col, 0, mask.shape[1]-1)
            component_of_point[i] = labeled[row,col]
        
        # Convert maximum bridge distance to pixel space
        image_size = mask.shape[0]
        max_bridge_px = max_bridge_dist * image_size / 2.0
        
        # Find pairs of points from different connected components within the maximum bridge distance
        tree = KDTree(centerline_px)
        pairs = tree.query_pairs(r=max_bridge_px)
        
        bridge_mask = filtered.copy()
        for i, j in pairs:
            # Extract components of each point in the pair
            ci = component_of_point[i]
            cj = component_of_point[j]
            
            # Skip points that have label 0 (noise) or are same-component pairs
            if ci == 0 or cj == 0 or ci == cj:
                continue
            
            # Identify the points within the maximum bridge distance from different components and average radii
            pt_i = (int(round(centerline_px[i,0])), int(round(centerline_px[i,1])))
            pt_j = (int(round(centerline_px[j,0])), int(round(centerline_px[j,1])))
            r_avg = max(1, int(round((radii_px[i] + radii_px[j]) / 2.0)))
            
            # Connect the points
            cv2.line(bridge_mask, pt_i, pt_j, 1, thickness=2*r_avg, lineType=cv2.LINE_8)
            cv2.circle(bridge_mask, pt_i, r_avg, 1, -1)
            cv2.circle(bridge_mask, pt_j, r_avg, 1, -1)
        
        return bridge_mask.astype(np.uint8)

    def _close_fragments(
        self,
        mask: np.ndarray, 
        bridge_radius: int = 3,
        min_fragment_size: int = 30
    ) -> np.ndarray:
        """
        Post-process a vessel mask to join nearby fragments and remove noise
        through dilation, morphological closing, and erosion of the mask.
        """
        # Step 1: Remove tiny isolated specks first (they cause false bridges)
        labeled, n = label(mask)
        cleaned = np.zeros_like(mask)
        for i in range(1, n + 1):
            if (labeled == i).sum() >= min_fragment_size:
                cleaned |= (labeled == i).astype(np.uint8)

        # Step 2: Dilate to find which fragments are close to each other
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*bridge_radius+1, 2*bridge_radius+1))
        dilated = cv2.dilate(cleaned, kernel)

        # Step 3: Where dilated regions overlap = there's a bridgeable gap
        # Find the skeleton of the dilated mask to get thin bridges and take the union with original
        bridged = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Step 4: Erode back to original vessel width (dilation expands vessels, erosion restores shape)
        eroded = cv2.erode(bridged, kernel)

        # Step 5: Union -- keep original details + add bridges
        result = cv2.bitwise_or(cleaned, eroded)
        return result.astype(np.uint8)

    def _estimate_tangents_pca(
        self,
        centerline: np.ndarray,
        indices: np.ndarray,
        distances: np.ndarray,
        max_radius: float
    ) -> np.ndarray:
        """
        Fits a local line through each point's neighborhood using 
        Principal Component Analysis (PCA) to estimate tangent vectors.
        """
        # Array for holding the predicted tangent vectors
        tangents = np.zeros_like(centerline)
        
        for i in range(len(centerline)):
            # Collect all neighbors within a maximum radius around the centerline point
            neighbors = [centerline[indices[i,k]] for k in range(1, indices.shape[1]) if distances[i,k] <= max_radius]
            
            if len(neighbors) >= 2:
                # Two or more neighbors -- perform PCA on local neighborhood using SVD
                pts = np.array([centerline[i]] + neighbors)
                pts_centered = pts - pts.mean(axis=0)
                Vt = np.linalg.svd(pts_centered, full_matrices=False)[2]
                tangents[i] = Vt[0]

            elif len(neighbors) == 1:
                # Only one neighbor -- use that direction as the tangent
                vec = neighbors[0] - centerline[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-6:
                    tangents[i] = vec / norm
        
        return tangents