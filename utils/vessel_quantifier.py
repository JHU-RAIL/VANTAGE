import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats
from scipy.ndimage import label
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from typing import Union, List, Tuple, Optional, Any

from PVBM.DiscSegmenter import DiscSegmenter
from PVBM.GeometryAnalysis import GeometricalVBMs
from PVBM.FractalAnalysis import MultifractalVBMs

class VesselQuantifier():
    """
    Computes vessel quantification metrics.
    Adapted from the PVBM library source code: https://github.com/aim-lab/PVBM/blob/main/PVBM/GeometryAnalysis.py.
    """
    def __init__(
        self,
        vessel_seg: np.ndarray,
        disc_seg: np.ndarray,
        x_center: int,
        y_center: int,
        radius: float
    ) -> None:
        # Check that vessel mask contains signal
        if vessel_seg.max() == vessel_seg.min():
            raise ValueError('Vessel segmentation mask must be non-empty!')

        # Compute skeletonized vasculature and vessel diameter map
        self.skeletonized = skeletonize(vessel_seg)
        self.diameter_map = 2.0 * distance_transform_edt(vessel_seg) * (vessel_seg > 0)
        self.disc_seg = disc_seg
        self.vessel_seg = vessel_seg
        self.xc = x_center
        self.yc = y_center
        self.radius = radius

        # Extract vessel subgraphs, identify vessel starting points, and filter irregularities
        B, D = self._extract_subgraphs(graphs=self.skeletonized.copy(), x_c=x_center, y_c=y_center)
        self.starting_points = self._extract_starting_points(B, D, x_center, y_center, radius, self.skeletonized.shape)
        self.prepr_skel = self._filter_vessel_graph(self.skeletonized, self.starting_points)

        # Initialize multifractal and geometric analysis helper classes
        self.fractal_analysis = MultifractalVBMs(n_rotations=25, optimize=True, min_proba=0.0001, maxproba=0.9999)
        self.geometric_analysis = GeometricalVBMs()
        self.disc_segmenter = None

    def keypoints(self) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Compute vessel branching and end points.
        """
        # Extract vessel topological attributes if necessary
        if not hasattr(self, 'topological_attr'):
            self._extract_topological_attr()

        # Extract (i, j) coordinates of vessel branching and end points
        branch_pts = [[int(i), int(j)] for j, i in zip(*np.nonzero(self.branch_pts))]
        end_pts = [[int(i), int(j)] for j, i in zip(*np.nonzero(self.end_pts))]
        return branch_pts, end_pts

    def segments(self) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Compute number of vessel segments.
        """
        # Extract vessel topological attributes if necessary
        if not hasattr(self, 'topological_attr'):
            self._extract_topological_attr()

        segments = []
        n_segments = 0
        
        # Get vessel segments
        for val in self.topological_attr.values():
            segments.append(val['path'])
            n_segments += 1
        
        return n_segments, segments

    def tortuosity(self) -> Tuple[List[float], List[Tuple[int, int]], float]:
        """
        Compute per-segment vessel tortuosity and median tortuosity.
        """
        # Extract vessel topological attributes if necessary
        if not hasattr(self, 'topological_attr'):
            self._extract_topological_attr()

        seg_tortuosities = []
        segments = []
        
        # Get vessel segment tortuosity values
        for val in self.topological_attr.values():
            seg_tortuosities.append(val['tortuosity'])
            segments.append(val['path'])
        
        return seg_tortuosities, segments, np.median(seg_tortuosities)
    
    def diameter(self) -> Tuple[List[float], List[Tuple[int, int]], float]:
        """
        Compute per-segment vessel diameter and median diameter.
        """
        # Extract vessel topological attributes if necessary
        if not hasattr(self, 'topological_attr'):
            self._extract_topological_attr()

        seg_diameters = []
        segments = []
        
        # Get vessel segment tortuosity values
        for val in self.topological_attr.values():
            seg_diameters.append(val['diameter'])
            segments.append(val['path'])
        
        return seg_diameters, segments, np.median(seg_diameters)

    def length(self) -> Tuple[List[float], List[Tuple[int, int]], float]:
        """
        Compute per-segment vessel lengths and median length.
        """
        # Extract vessel topological attributes if necessary
        if not hasattr(self, 'topological_attr'):
            self._extract_topological_attr()

        seg_lengths = []
        segments = []
        
        # Get vessel segment tortuosity values
        for val in self.topological_attr.values():
            seg_lengths.append(val['arc'])
            segments.append(val['path'])
        
        return seg_lengths, segments, np.median(seg_lengths)

    def fractals(self, disc_segmenter: Optional[DiscSegmenter] = None):
        """
        Performs multifractal analysis of the retinal vasculature
        across the full image.
        """
        # Binarize the vessel segmentation mask
        seg_binarized = (self.vessel_seg > 0).astype(np.float32)

        # Perform fractal analysis
        D0, D1, D2, SL = self.fractal_analysis.compute_multifractals(seg_binarized)
        return D0, D1, D2, SL
    
    def visualize_keypoints(
        self,
        branch_pts: Optional[np.ndarray] = None,
        end_pts: Optional[np.ndarray] = None,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (7, 6),
        legend_loc: Optional[str] = None,
        dpi: int = 300
    ) -> Tuple[Any, Any]:
        """
        Create figure to visualize vessel branching and end points.
        """
        # Normalize vessel segmentation mask
        vessel_seg = (self.vessel_seg - self.vessel_seg.min()) / (self.vessel_seg.max() - self.vessel_seg.min())
        plotted = False

        # Create figure and add vessel segmentation mask
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(vessel_seg, cmap='gray', clim=(0, 2))
        ax.set_title(title, fontsize=15, pad=10)
        
        # Add branching points if necessary
        if branch_pts is not None and len(branch_pts) > 0:
            ax.scatter(branch_pts[:,0], branch_pts[:,1], s=22, c='yellow', alpha=0.9, marker='o',
                       label=f'Branching Points (n = {branch_pts.shape[0]})')
            plotted = True
        
        # Add end points if necessary
        if end_pts is not None and len(end_pts) > 0:
            ax.scatter(end_pts[:,0], end_pts[:,1], s=25, c='red', alpha=0.9, marker='x',
                       label=f'End Points (n = {end_pts.shape[0]})')
            plotted = True

        plt.tight_layout()
        if plotted:
            plt.legend(loc='lower right')
        return fig, ax
        
    def visualize_topology(
        self,
        paths: List[Tuple[int, int]],
        values: Union[List[float], np.ndarray],
        title: Optional[str] = None,
        cbar_label: Optional[str] = None,
        c_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
        figsize: Tuple[int, int] = (8, 7),
        cmap: str = 'hot',
        decimals: int = 2,
        dpi: int = 300
    ) -> Tuple[Any, Any]:
        """
        Create figure to visualize per-segment vessel quantification metrics.
        """
        # Check that vessel mask contains signal
        if self.vessel_seg.max() == self.vessel_seg.min():
            raise ValueError('Vessel segmentation mask must be non-empty!')

        # Normalize vessel segmentation mask
        vessel_seg = (self.vessel_seg - self.vessel_seg.min()) / (self.vessel_seg.max() - self.vessel_seg.min())

        # Create figure and add vessel segmentation mask
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(vessel_seg, cmap='gray', clim=(0, 2))

        # Determine range of the colormap
        min = np.min(values) if c_range is None or c_range[0] is None else c_range[0]
        max = np.max(values) if c_range is None or c_range[1] is None else c_range[1]

        # Extract colormap and normalize based on specified range
        cmap = plt.get_cmap(cmap)
        norm = mpl.colors.Normalize(vmin=min, vmax=max)

        # Add vessel segments overlaid on top of the segmentation mask
        for path, val in zip(paths, values):
            # Skip segments with invalid values
            if val is None or np.isnan(val) or np.isinf(val):
                continue

            # Plot the vessel segment
            ys, xs = zip(*path)
            ax.plot(xs, ys, color=cmap(norm(val)), linewidth=2)

            # Label segment with value at vessel midpoint
            mid_idx = len(path) // 2
            ax.text(xs[mid_idx], ys[mid_idx], f'{val:.{decimals}f}',
                    color='white', fontsize=8,
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

        # Add colorbar
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=14)

        # Set title
        ax.set_title(title, fontsize=16, pad=12)
        return fig, ax

    @staticmethod
    def plot_distribution(
        data: np.ndarray,
        title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        bins: Optional[int] = None,
        percentiles: Tuple[float, float] = (0, 100),
        x_range: Optional[Tuple[float, float]] = None,
        include_mu_sig: Optional[bool] = None,
        figsize: Tuple[int, int] = (5, 3),
        color: str = 'royalblue',
        dpi: int = 300
    ) -> None:
        """
        Plot histogram of a dataset. If multiple distributions are provided,
        the average histogram across the dataset is plotted.
        """
        # Perform test for normality to determine whether to include μ and σ if necessary
        if include_mu_sig is None and len(data) > 3:
            # Use Shapiro-Wilk for 3 < n <= 5000 and D'Agostino-Pearson for n > 5000
            _, p_value = stats.shapiro(data) if len(data) <= 5000 else stats.normaltest(data)
            include_mu_sig = p_value > 0.05
        
        # Estimate optimal number of bins for visualizing the data distribution if necessary
        if bins is None:
            iqr = np.percentile(data, 75) - np.percentile(data, 25)
            bw = 2.0 * iqr / (len(data) ** (1. / 3))
            bins = int((data.max() - data.min()) / bw)

        # Create figure and plot histogram
        plt.figure(figsize=figsize, dpi=dpi)
        plt.title(title, fontsize=10)
        
        # Clip the data range based on the specified percentiles
        p_lower, p_upper = np.percentile(data, percentiles[0]), np.percentile(data, percentiles[1])
        clipped = np.clip(data, p_lower, p_upper)

        # Construct histogram
        sns.histplot(clipped, bins=bins, color='coral')
        if include_mu_sig:
            mean, std = data.mean(), data.std()
            plt.text(0.03, 0.94, f'$\\mu={data.mean():.2f}$\n$\\sigma={data.std():.2f}$',
                    transform=plt.gca().transAxes, verticalalignment='top',
                    fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.xlabel(x_label, fontsize=9)
        plt.ylabel(y_label, fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        if x_range is not None:
            plt.xlim(*x_range)
        plt.tight_layout()
    
    def _extract_topological_attr(self):
        """
        Extract topological attributes from vessel graph (i.e. vessel segment tortuosity,
        diameter, length, start points, branching points, and end points).
        """
        # Initialize arrays for storing start, branch, and end points
        self.start_pts = np.zeros((self.vessel_seg.shape[0], self.vessel_seg.shape[1]))
        self.branch_pts = np.zeros((self.vessel_seg.shape[0], self.vessel_seg.shape[1]))
        self.end_pts = np.zeros((self.vessel_seg.shape[0], self.vessel_seg.shape[1]))

        # Initialize dictionary for storing the results
        self.topological_attr = {}

        # Extract topological attributes
        B = np.zeros((self.vessel_seg.shape[0], self.vessel_seg.shape[1]))
        for idx_start in self.starting_points:
            i, j = idx_start
            self.start_pts[i,j] = 1
            self._iterative_topology(self.prepr_skel.copy(), B, idx_start[0], idx_start[1], 1, np.inf, self.xc, self.yc,
                                    self.end_pts, self.branch_pts, i, j, self.topological_attr, 0, None, self.diameter_map)

    def _extract_starting_points(
        self,
        subgraphs: np.ndarray,
        dist_graph: np.ndarray,
        xc: int,
        yc: int,
        radius: float,
        dimensions: Tuple[int, int]
    ) -> np.ndarray:
        """
        Extracts starting points from vessel skeleton graph.
        """
        starting_points = np.zeros(dimensions, dtype=np.float32)
        for i in set(list(subgraphs.reshape(-1))) - {0}:
            mask = subgraphs == i
            if mask.sum() >= 50:
                min_index = (dist_graph * mask + (1 - mask) * 1e10).argmin()
                min_coordinates = np.unravel_index(min_index, dist_graph.shape)
                if ((min_coordinates[0] - yc) ** 2 + (min_coordinates[1] - xc) ** 2) ** 0.5 < radius + (dimensions[0] * 0.02) ** 2:
                    starting_points[min_coordinates[0], min_coordinates[1]] = 1
        return np.argwhere(starting_points == 1)

    def _filter_vessel_graph(self, skeletonized: np.ndarray, starting_points: np.ndarray) -> np.ndarray:
        """
        Applies tree regularization to remove irregularities in graph topology.
        """
        result = np.zeros(skeletonized.shape, dtype=np.float32)
        
        for idx_start in starting_points:
            A = skeletonized.copy()

            # Build tree iteratively using explicit stack with entries (i, j, parent_node)
            root = {'plot': (idx_start[0], idx_start[1]), 'children': [], 'size': 0}
            stack = [(idx_start[0], idx_start[1], root)]
            A[idx_start[0], idx_start[1]] = 0
            all_nodes = [root]
            
            while stack:
                ci, cj, parent = stack.pop()
                neighbors = [
                    (ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1),
                    (ci-1, cj-1), (ci-1, cj+1), (ci+1, cj-1), (ci+1, cj+1)
                ]
                for ni, nj in neighbors:
                    if (0 <= ni < A.shape[0] and 
                        0 <= nj < A.shape[1] and 
                        A[ni, nj] == 1):
                        child = {
                            'plot': (ni, nj), 
                            'children': [], 
                            'size': 0
                        }
                        parent['children'].append(child)
                        all_nodes.append(child)
                        A[ni, nj] = 0
                        stack.append((ni, nj, child))
            
            # Compute subtree sizes bottom-up by process nodes in reverse order (children before parents)
            for node in reversed(all_nodes):
                node['size'] = 1 + sum(
                    c['size'] for c in node['children']
                )
            
            # Render nodes with subtree size >= 10
            for node in all_nodes:
                if node['size'] >= 10:
                    result[node['plot']] = 1
        
        return result

    def _iterative_topology(
        self, A, B, i, j, n, max_radius, x_c, y_c, endpoints, interpoints,
        i_or, j_or, dico, bacount, bapos, diameter_map, dist=0
    ) -> None:
        """
        Iteratively compute and analyze the topology of a segmented image using a stack.
        """
        # Initialize the stack with the initial node's state
        stack = []
        initial_frame = {
            'i': i,
            'j': j,
            'n': n,
            'i_or': i_or,
            'j_or': j_or,
            'bacount': bacount,
            'bapos': bapos,
            'dist': dist,
            'diameter': [diameter_map[i,j]],
            'path': [(i, j)],
            'state': 'process_node'  # Possible states: 'process_node', 'process_children'
        }
        stack.append(initial_frame)
        A[i,j] = 0  # Mark the starting node as visited

        while stack:
            # Peek at the last frame on the stack
            frame = stack[-1]

            current_i = frame['i']
            current_j = frame['j']
            current_n = frame['n']
            current_i_or = frame['i_or']
            current_j_or = frame['j_or']
            current_bacount = frame['bacount']
            current_bapos = frame['bapos']
            current_dist = frame['dist']
            current_diam = frame['diameter']
            current_path = frame['path']
            state = frame['state']

            if state == 'process_node':
                # Calculate the Euclidean distance from the center
                distance_from_center = ((y_c - current_i) ** 2 + (x_c - current_j) ** 2) ** 0.5

                # Base Case 1: If beyond the allowed radius
                if distance_from_center > max_radius:
                    endpoints[current_i, current_j] = 1
                    true_distance = ((current_i_or - current_i) ** 2 + (current_j_or - current_j) ** 2) ** 0.5
                    dico[(current_i_or, current_j_or, current_i, current_j)] = {
                        'arc': current_dist,
                        'chord': true_distance,
                        'tortuosity': current_dist / true_distance if current_dist != 0 else float('inf'),
                        'diameter': sum(current_diam) / len(current_diam) if len(current_diam) != 0 else float('inf'),
                        'path': current_path,
                        'bapos': current_bapos
                    }
                    stack.pop()  # Remove frame from stack
                    continue  # Proceed to next frame

                # Define all 8 neighbors and their corresponding distances
                up = (current_i - 1, current_j)
                down = (current_i + 1, current_j)
                left = (current_i, current_j - 1)
                right = (current_i, current_j + 1)
                up_left = (current_i - 1, current_j - 1)
                up_right = (current_i - 1, current_j + 1)
                down_left = (current_i + 1, current_j - 1)
                down_right = (current_i + 1, current_j + 1)
                points = [up, down, left, right, up_left, up_right, down_left, down_right]
                distances = [1, 1, 1, 1, 2 ** 0.5, 2 ** 0.5, 2 ** 0.5, 2 ** 0.5]

                # Compute the number of children
                children = 0
                valid_children = []
                child_distances = []
                for point, distance in zip(points, distances):
                    pi, pj = point
                    if 0 <= pi < A.shape[0] and 0 <= pj < A.shape[1]:
                        if A[pi, pj] == 1:
                            children += 1
                            valid_children.append(point)
                            child_distances.append(distance)

                # Store valid children and distances in the frame
                frame['valid_children'] = valid_children
                frame['child_distances'] = child_distances
                frame['child_index'] = 0  # Index of next child to process

                # Base Case 2: No children and sufficient depth
                if children == 0 and current_n >= 10:
                    endpoints[current_i, current_j] = 1
                    true_distance = ((current_i_or - current_i) ** 2 + (current_j_or - current_j) ** 2) ** 0.5
                    dico[(current_i_or, current_j_or, current_i, current_j)] = {
                        'arc': current_dist,
                        'chord': true_distance,
                        'tortuosity': current_dist / true_distance if current_dist != 0 else float('inf'),
                        'diameter': sum(current_diam) / len(current_diam) if len(current_diam) != 0 else float('inf'),
                        'path': current_path,
                        'bapos': current_bapos
                    }
                    stack.pop()  # Remove frame from stack
                    continue

                # Base Case 3: More than one child and sufficient depth
                if children > 1 and current_n >= 10:
                    interpoints[current_i, current_j] = 1
                    true_distance = ((current_i_or - current_i) ** 2 + (current_j_or - current_j) ** 2) ** 0.5
                    dico[(current_i_or, current_j_or, current_i, current_j)] = {
                        'arc': current_dist,
                        'chord': true_distance,
                        'tortuosity': current_dist / true_distance if current_dist != 0 else float('inf'),
                        'diameter': sum(current_diam) / len(current_diam) if len(current_diam) != 0 else float('inf'),
                        'path': current_path,
                        'bapos': current_bapos
                    }
                    # Reset variables for this frame (affects only its children)
                    frame['i_or'] = current_i
                    frame['j_or'] = current_j
                    frame['dist'] = 0
                    frame['diameter'] = [diameter_map[current_i,current_j]]
                    frame['path'] = [(current_i, current_j)]
                    frame['n'] = 0
                    frame['bacount'] = 0
                    frame['bapos'] = None

                # Set state to 'process_children' to begin processing children
                frame['state'] = 'process_children'

            elif state == 'process_children':
                # Get the list of valid children and current child index
                valid_children = frame['valid_children']
                child_distances = frame['child_distances']
                child_index = frame['child_index']

                if child_index >= len(valid_children):
                    # All children have been processed, pop the frame
                    stack.pop()
                    continue

                # Get the next child to process
                point = valid_children[child_index]
                distance = child_distances[child_index]
                pi, pj = point

                # Increment child index in the parent frame
                frame['child_index'] += 1

                # Mark child as visited
                A[pi,pj] = 0

                # Update backup position if bacount reaches 30
                child_bacount = frame['bacount'] + 1
                child_bapos = frame['bapos']
                if child_bacount == 30:
                    child_bapos = (pi, pj)

                # Update cumulative distance
                child_dist = frame['dist'] + distance

                # Update segment path
                child_path = frame['path'] + [(pi, pj)]

                # Update list of vessel diameters
                child_diam = frame['diameter'] + [diameter_map[pi,pj]]

                # Create a new frame for the child node
                child_frame = {
                    'i': pi,
                    'j': pj,
                    'n': frame['n'] + 1,
                    'i_or': frame['i_or'],
                    'j_or': frame['j_or'],
                    'bacount': child_bacount,
                    'bapos': child_bapos,
                    'dist': child_dist,
                    'diameter': child_diam,
                    'path': child_path,
                    'state': 'process_node'
                }

                # Push the child frame onto the stack
                stack.append(child_frame)
        return

    def _extract_subgraphs(self, graphs, x_c, y_c):
        """
        Extract B, the a graph where each of the disconnected subgraph is labeled differently and D which contains the euclidian distance graph between each
        point in the graph and the optic disc.

        :param graphs: Original blood vessel segmentation graph
        :type graphs: array
        :param x_c: x axis of the optic disc center
        :type x_c: int
        :param y_c: y axis of the optic disc center
        :type y_c: int

        :return: B,D
        :rtype: tuple
        """
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        B, n = label(graphs, structure=structure)
        B = B.astype(np.float32)
        
        D = np.zeros_like(graphs, dtype=np.float32)
        ys, xs = np.where(graphs == 1)
        D[ys, xs] = np.sqrt((y_c - ys) ** 2 + (x_c - xs) ** 2)
        
        return B, D

    def _recursive_subgraph(self,A, B, D, i, j, n, x_c, y_c):
        """
        Recursively extract the value within B and D.

        :param A: Original blood vessel segmentation graph
        :type A: array
        :param B: A graph where each of the disconnected subgraph is labeled differentl, which is initialized by a zeros matrix and recursively built
        :type B: array
        :param D: Euclidian Distance graph between each point in A and the optic disc center (x_c,y_c), which is initialized by a zeros matrix and recursively built
        :type D: array
        :param i: Current x axis location within the graph
        :type i: int
        :param j: Current y axis location within the graph
        :type j: int
        :param n: Current number of point distance since the optic disc
        :type n: int
        :param x_c: x axis of the optic disc center
        :type x_c: int
        :param y_c: y axis of the optic disc center
        :type y_c: int

        :return: B,D
        :rtype: tuple
        """
        up = (i - 1, j)
        down = (i + 1, j)
        left = (i, j - 1)
        right = (i, j + 1)

        up_left = (i - 1, j - 1)
        up_right = (i - 1, j + 1)
        down_left = (i + 1, j - 1)
        down_right = (i + 1, j + 1)
        points = [up, down, left, right, up_left, up_right, down_left, down_right]
        for point in points:
            if point[0] >= 0 and point[0] < B.shape[0] and point[1] < B.shape[1] and point[1] >= 0:
                if A[point] == 1:
                    B[point] = n
                    A[point] = 0
                    D[point] = ((y_c - point[0]) ** 2 + (x_c - point[1]) ** 2) ** 0.5
                    self._recursive_subgraph(A, B, D, point[0], point[1], n, x_c, y_c)