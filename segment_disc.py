import numpy as np
import argparse
import os
import multiprocessing as mp
import warnings
from typing import List, Any
from PIL import Image
from tqdm import tqdm
from PVBM.DiscSegmenter import DiscSegmenter

def parse_args():

    # Sample command line call:
    # >>> python3 segment_disc.py --fundus_path ../datasets/FIVES/train/Original/*.png --vessel_seg ../datasets/FIVES/train/Ground\ truth/*.png

    parser = argparse.ArgumentParser()

    parser.add_argument('--fundus_path', nargs='+', type=str, required=True, help='Path to retinal fundus files.')
    parser.add_argument('--vessel_seg', nargs='+', type=str, required=True, help='Path to the corresponding vessel segmentation files.')
    
    parser.add_argument('--output_path', type=str, default='./results/disc_seg/', help='Output path for optic disc segmentation.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for multiprocessing.')

    opt = parser.parse_args()
    return opt

def segment_disc(args: List[Any]) -> int:
    """
    Segments the optic disc of a retinal fundus image using the PVBM library.
    """
    # Extract arguments
    fundus_path, vessel_path, disc_segmenter, output_path = args
    case_id = os.path.splitext(os.path.basename(fundus_path))[0]

    # Load the fundus image and vessel segmentation mask 
    fundus = Image.open(fundus_path)
    vessel = Image.open(vessel_path)

    # Segment the optic disc
    optic_disc = disc_segmenter.segment(image_path=fundus_path)

    if (np.array(optic_disc) == 0).all():
        warnings.warn(f'Failed to detect optic disc for image \"{fundus_path}\".')
        return 1

    # Create output directories if necessary
    fundus_path = f'{output_path}/fundus/{case_id}.png'
    vessel_path = f'{output_path}/vessel/{case_id}.png'
    disc_path = f'{output_path}/disc/{case_id}.png'

    os.makedirs(os.path.dirname(fundus_path), exist_ok=True)
    os.makedirs(os.path.dirname(vessel_path), exist_ok=True)
    os.makedirs(os.path.dirname(disc_path), exist_ok=True)

    # Save output files
    fundus.save(fundus_path)
    vessel.save(vessel_path)
    optic_disc.save(disc_path)
    return 0

def main():
    # Parse command-line arguments
    opt = parse_args()

    # Set multiprocessing mode
    mp.set_start_method('spawn', force=True)

    # Create input arguments
    disc_segmenter = DiscSegmenter()
    args = [(fundus, vessel, disc_segmenter, opt.output_path) for fundus, vessel in zip(opt.fundus_path, opt.vessel_seg)]
    samples_success = 0

    # Segment the optic disc of the retinal fundus images
    with mp.Pool(opt.workers) as pool:
        for i in tqdm(pool.imap_unordered(segment_disc, args), total=len(args), desc='Segmenting Optic Disc'):
            samples_success += (1 - i)

        print(f'\nDone. Successfully segmented the optic disc for [{samples_success}/{len(opt.fundus_path)}] samples.\n')

if __name__ == '__main__':
    main()