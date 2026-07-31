import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import multiprocessing as mp
import warnings
import pickle
import csv
from typing import Dict, List, Any
from PIL import Image
from tqdm import tqdm

from PVBM.DiscSegmenter import DiscSegmenter
from PVBM.GeometryAnalysis import GeometricalVBMs
from utils.vessel_quantifier import VesselQuantifier

def parse_args():

    # Sample command line call:
    # >>> python3 quantify_vasculature.py --vessel_seg ./results/disc_seg/vessel/*_N.png --disc_seg ./results/disc_seg/disc/*_N.png

    parser = argparse.ArgumentParser()

    parser.add_argument('--vessel_seg', nargs='+', type=str, required=True, help='Path to vessel segmentation files.')
    parser.add_argument('--disc_seg', nargs='+', type=str, required=True, help='Path to optic disc segmentation files.')

    parser.add_argument('--metrics', nargs='+', type=str, default=['bifurcations', 'endpoints', 'segments', 'tortuosity', 'diameter', 'length', 'fractal_cap', 'fractal_entr', 'fractal_corr'], help='Vessel quantification metrics to evaluate. Options include "bifurcations", "endpoints", "segments", "tortuosity", "diameter", "length", "fractal_cap", "fractal_entr", and "fractal_corr".')
    parser.add_argument('--output_path', type=str, default='./results/quantification/', help='Output path for vessel quantification results.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for multiprocessing.')

    opt = parser.parse_args()
    return opt

def quantify_vessels(args: List[Any]) -> Dict[str, Any]:
    """
    Computes vessel quantification metrics.
    """
    # Extract arguments
    vessel_path, disc_path, metrics, disc_segmenter, output_path = args
    case_id = os.path.splitext(os.path.basename(vessel_path))[0]

    # Load the vessel and optic disc segmentation masks
    vessel_seg = np.array(Image.open(vessel_path).convert('L')).astype(np.float32)
    disc_seg = Image.open(disc_path)

    # Extract the center and radius of the optic disc
    center, radius, _, _ = disc_segmenter.post_processing(segmentation=disc_seg, max_roi_size=600) 
    disc_seg = np.array(disc_seg).astype(np.float32)

    # Vessel segmentation mask should contain segmented regions
    if (np.array(vessel_seg) == 0).all():
        warnings.warn(f'Vessel segmentation mask \"{vessel_path}\" is empty.')
        return None

    # Optic disc segmentation mask should contain segmented regions
    if (np.array(disc_seg) == 0).all():
        warnings.warn(f'Optic disc segmentation mask \"{vessel_path}\" is empty.')
        return None

    # Create output directory if necessary
    base_dir = f'{output_path}/{case_id}/'
    os.makedirs(base_dir, exist_ok=True)

    # Initialize vessel quantifier
    vessel = VesselQuantifier(vessel_seg, disc_seg, center[0], center[1], radius)

    # Variables to cache results of fractal analysis if necessary
    D0, D1, D2 = None, None, None

    # Perform quantification
    results = {'id': case_id, 'vessel_path': vessel_path, 'disc_path': disc_path}
    if 'bifurcations' in metrics and 'endpoints' in metrics:
        # Compute vessel branching and end points
        branch_pts, end_pts = vessel.keypoints()
        results['bifurcations'] = (None, len(branch_pts))   # In the form (Individual measurements, aggregated)
        results['endpoints'] = (None, len(end_pts)) # In the form (Individual measurements, aggregated)

        # Visualize vessel branching and end points
        vessel.visualize_keypoints(branch_pts=np.array(branch_pts), end_pts=np.array(end_pts), 
                                   title='Vessel Branching and End Points', legend_loc='lower right')
        plt.savefig(f'{base_dir}/keypoint_visualization.png')
        plt.close()

    elif 'bifurcations' in metrics:
        # Compute vessel branching points
        branch_pts = vessel.keypoints()[0]
        results['bifurcations'] = (None, len(branch_pts))   # In the form (Individual measurements, aggregated)

        # Visualize vessel branching points
        vessel.visualize_keypoints(branch_pts=np.array(branch_pts), title='Vessel Branching Points')
        plt.savefig(f'{base_dir}/branch_visualization.png')
        plt.close()

    elif 'endpoints' in metrics:
        # Compute vessel end points
        end_pts = vessel.keypoints()[1]
        results['endpoints'] = (None, len(end_pts)) # In the form (Individual measurements, aggregated)

        # Visualize vessel end points
        vessel.visualize_keypoints(end_pts=np.array(end_pts), title='Vessel End Points')
        plt.savefig(f'{base_dir}/end_visualization.png')
        plt.close()

    if 'segments' in metrics:
        # Compute number of vessel segments
        n_segments, segments = vessel.segments()
        results['segments'] = (None, n_segments)    # In the form (Individual measurements, aggregated)

    if 'tortuosity' in metrics:
        # Compute vessel tortuosities
        seg_tort, segments, med_tort = vessel.tortuosity()
        results['tortuosity'] = (seg_tort, med_tort)    # In the form (Individual measurements, aggregated)

        # Visualize vessel segment tortuosities
        vessel.visualize_topology(segments, seg_tort, title='Vessel Segment Tortuosity', cbar_label='Tortuosity', 
                                  c_range=(1, 2), decimals=2)
        plt.savefig(f'{base_dir}/tortuosity_visualization.png')
        plt.close()

        # Plot histogram of vessel segment tortuosities from a single sample
        VesselQuantifier.plot_distribution(np.array(seg_tort), title='Vessel Tortuosity Distribution', 
                                           x_label='Tortuosity', y_label='Frequency', bins=30, 
                                           percentiles=(0, 99), include_mu_sig=True)
        plt.savefig(f'{base_dir}/tortuosity_distrib.png')
        plt.close()

        # Save vessel segment tortuosities
        np.save(f'{base_dir}/tortuosity.npy', seg_tort)

    if 'diameter' in metrics:
        # Compute vessel diameters
        seg_diam, segments, med_diam = vessel.diameter()
        results['diameter'] = (seg_diam, med_diam)  # In the form (Individual measurements, aggregated)

        # Visualize vessel segment diameters
        vessel.visualize_topology(segments, seg_diam, title='Vessel Segment Diameters', cbar_label='Diameter', 
                                  decimals=1)
        plt.savefig(f'{base_dir}/diameter_visualization.png')
        plt.close()

        # Plot histogram of vessel segment diameters from a single sample
        VesselQuantifier.plot_distribution(np.array(seg_diam), title='Vessel Diameter Distribution', 
                                           x_label='Diameter', y_label='Frequency', bins=30)
        plt.savefig(f'{base_dir}/diameter_distrib.png')
        plt.close()

        # Save vessel segment diameters
        np.save(f'{base_dir}/diameter.npy', seg_diam)

    if 'length' in metrics:
        # Compute vessel lengths
        seg_len, segments, med_len = vessel.length()
        results['length'] = (seg_len, med_len)  # In the form (Individual measurements, aggregated)

        # Visualize vessel segment lengths
        vessel.visualize_topology(segments, seg_len, title=f'Vessel Segment Lengths', cbar_label='Length',
                                  decimals=0)
        plt.savefig(f'{base_dir}/length_visualization.png')
        plt.close()

        # Plot histogram of vessel segment lengths from a single sample
        VesselQuantifier.plot_distribution(np.array(seg_len), title='Vessel Length Distribution', 
                                           x_label='Length', y_label='Frequency', bins=30)
        plt.savefig(f'{base_dir}/length_distrib.png')
        plt.close()

        # Save vessel segment lengths
        np.save(f'{base_dir}/length.npy', seg_len)
    
    if 'fractal_cap' in metrics:
        # Compute fractal capacity dimension
        D0, D1, D2, _ = vessel.fractals(disc_segmenter)
        results['fractal_cap'] = (None, D0)

    if 'fractal_entr' in metrics:
        # Compute fractal entropy dimension if necessary
        if D1 is None:
            D0, D1, D2, _ = vessel.fractals(disc_segmenter)
        results['fractal_entr'] = (None, D1)

    if 'fractal_corr' in metrics:
        # Compute fractal correlation dimension if necessary
        if D2 is None:
            D0, D1, D2, _ = vessel.fractals(disc_segmenter)
        results['fractal_corr'] = (None, D2)

    if 'tortuosity' in metrics or 'diameter' in metrics or 'length' in metrics:
        # Save vessel segment paths
        with open(f'{base_dir}/path.pkl', 'wb') as f:
            pickle.dump(segments, f)
    
    return results

def plot_global_results(metrics_global: Dict[str, Dict[str, List[float]]], output_dir: str) -> None:
    """
    Creates histogram of per-segment and aggregated vessel
    quantification metrics across the dataset.
    """
    for metric, data in metrics_global.items():
        # Capitalize metric name for the purposes of plotting
        metric = metric.capitalize()

        # Number of dataset samples
        n = len(data['aggr'])

        # Create output directory if necessary
        os.makedirs(output_dir, exist_ok=True)

        # Plot histogram of per-segment vessel quantification across the dataset
        if len(data['indiv']) > 0:
            is_tortuosity = metric.lower() == 'tortuosity'
            VesselQuantifier.plot_distribution(np.array(data['indiv']), title=f'Vessel {metric} Distribution (n = {n})', x_label=metric,
                                               y_label='Frequency', bins=30, percentiles=(0, 99) if is_tortuosity else (0, 99.9), 
                                               include_mu_sig=is_tortuosity)
            plt.savefig(f'{output_dir}/{metric.lower()}_distrib.png')
            plt.close()
        
        # Plot histogram of aggregated per-sample vessel quantification across the dataset
        if len(data['aggr']) > 0:
            aggr_hist_title = f'No. of Vessel {metric} Distribution (n = {n})' if len(data['indiv']) == 0 else f'Median Vessel {metric} Distribution (n = {n})'
            fname = f'num_{metric.lower()}_distrib' if len(data['indiv']) == 0 else f'median_{metric.lower()}_distrib'
            VesselQuantifier.plot_distribution(np.array(data['aggr']), title=aggr_hist_title, x_label=metric, 
                                               y_label='Frequency', bins=30, include_mu_sig=True)
            plt.savefig(f'{output_dir}/{fname}.png')
            plt.close()

def main():
    # Parse command-line arguments
    opt = parse_args()

    # Ensure specified quantificaton metrics are valid and case-insensitive
    metrics, valid = [], ['bifurcations', 'endpoints', 'segments', 'tortuosity', 'diameter', 'length',
                          'fractal_cap', 'fractal_entr', 'fractal_corr']
    for m in opt.metrics:
        if m.lower() in valid:
            metrics.append(m.lower())
        else:
            raise ValueError('Expected quantification metric to be "bifurcations", "endpoints", "tortuosity", ' \
                             '"diameter", "length", "fractal_cap", "fractal_entr", or "fractal_corr", ' \
                             f'but got "{m}" instead!')
    
    # Remove duplicate values from list of quantification metrics
    opt.metrics = list(set(metrics))

    # Set multiprocessing mode
    mp.set_start_method('spawn', force=True)

    # Create input arguments
    disc_segmenter = DiscSegmenter()
    args = [(vessel, disc, opt.metrics, disc_segmenter, opt.output_path)
            for vessel, disc in zip(opt.vessel_seg, opt.disc_seg)]
    
    # Create output directory if necessary
    os.makedirs(opt.output_path, exist_ok=True)
    success = 0
    
    # Create dictionary storing quantification metrics and statistics across the dataset
    metrics_global = {metric: {'indiv': [], 'aggr': []} for metric in opt.metrics}

    # Run vessel quantification on the input files
    with open(f'{opt.output_path}/vessel_quantification.csv', 'w', newline='') as f:
        # Initialize csv writer and write header
        writer = csv.writer(f)
        writer.writerow(['case_id', 'vessel_path', 'disc_path'] + opt.metrics)

        # Compute vessel quantification metrics with multiprocessing
        print(f'\nQuantifying vessel mask across {len(opt.metrics)} metrics.')
        with mp.Pool(opt.workers) as pool:
            for result in tqdm(pool.imap(quantify_vessels, args), total=len(args), desc='Quantifying Vasculature'):
                # Skip if samples could not be quantified
                if result is None:
                    continue

                # Search through all quantification metric results
                aggr_results = []
                for metric in opt.metrics:
                    # Only tortuosity, diameter, and length have individual per-segment quantification results to save
                    if result[metric][0] is not None:
                        metrics_global[metric]['indiv'] = metrics_global[metric]['indiv'] + result[metric][0]
                    
                    # Save aggregated per-sample quantification result
                    metrics_global[metric]['aggr'].append(result[metric][1])
                    aggr_results.append(result[metric][1])

                # Append aggregated per-sample quantification result to the csv results file
                writer.writerow([result['id'], result['vessel_path'], result['disc_path']] + aggr_results)

                # Save global quantification metrics data
                with open(f'{opt.output_path}/vessel_quantification_raw.pkl', 'wb') as f:
                    pickle.dump(metrics_global, f)

                # Update plot of global results
                plot_global_results(metrics_global, opt.output_path)
                success += 1

            print(f'\nDone. Successfully computed vessel quantification metrics for [{success}/{len(args)}] samples.\n')

if __name__ == '__main__':
    main()