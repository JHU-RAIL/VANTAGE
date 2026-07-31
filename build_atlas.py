import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import multiprocessing as mp
from typing import Tuple, List, Any
import cv2
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
import random

from VANTAGE.vantage import VANTAGE, DeltaSampling
from utils.atlas import VANTAGE_Atlas

def parse_args():

    # Sample command line call:
    # >>> python3 build_atlas.py --model_ckpt ./results/train/vantage.pth --deg_path ./results/train/deg.pt --disc_seg ./results/disc_seg/disc/*_N.png

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the trained VANTAGE model checkpoint.')
    parser.add_argument('--deg_path', type=str, required=True, help='Path to the vessel graph degree statistics for the trained VANTAGE model as a *.pt file.')
    parser.add_argument('--disc_seg', nargs='+', type=str, required=True, help='Path to optic disc segmentation files of the training dataset.')
    
    parser.add_argument('--ref_eye', type=str, default='OD', help='Optic disc segmentation mask of the opposite eye will be flipped to match the reference eye.')
    parser.add_argument('--output_path', type=str, default='./results/atlas/', help='Output path for constructed atlas.')
    parser.add_argument('--seed', type=int, default=42, help='Set random seed for reproducibility.')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for multiprocessing.')

    opt = parser.parse_args()
    return opt

def prepr_disc(args: List[Any]) -> Tuple[np.ndarray, List[float]]:
    """
    Loads optic disc segmentation, computes the center of mass,
    and performs OD/OS normalization.
    """
    # Extract arguments
    disc_path, ref_eye = args

    # Load optic disc segmentation mask
    disc_seg = np.array(Image.open(disc_path).convert('L')).astype(np.float32)

    # Normalize segmentations
    if disc_seg.max() - disc_seg.min() > 0:
        disc_seg = (disc_seg - disc_seg.min()) / (disc_seg.max() - disc_seg.min())

    # Compute the center of mass and flip to match the reference eye if necessary
    disc_com = ndimage.center_of_mass(disc_seg)
    type = 'OS' if disc_com[1] < disc_seg.shape[1] // 2 else 'OD'

    # Coarse alignment of OD/OS fundus images determined based on optic disc COM
    if type != ref_eye:
        # Flip the image if OD/OS does not match the reference eye
        disc_seg = np.flip(disc_seg, axis=1)
        disc_com = ndimage.center_of_mass(disc_seg)
    
    return disc_seg, list(disc_com)

def disc_com_alignment(args: List[Any]) -> np.ndarray:
    """
    Performs center-of-mass alignment of the optic disc
    segmentation mask to a target point.
    """
    # Extract arguments
    disc_seg, disc_com, target = args

    # Create COM alignment transformation matrix
    matrix = np.float32([[1.0, 0.0, target[1] - disc_com[1]],
                         [0.0, 1.0, target[0] - disc_com[0]]])

    # Transform the optic disc segmentation mask
    aligned_disc = cv2.warpAffine(disc_seg, matrix, (disc_seg.shape[1], disc_seg.shape[0]), flags=cv2.INTER_NEAREST)
    return aligned_disc

def main():
    # Parse command-line arguments
    opt = parse_args()

    # Set random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)

    # Load model state dictionary and extract point cloud size
    device = torch.device('cpu')
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

    # Set multiprocessing mode
    mp.set_start_method('spawn', force=True)

    # Ensure specified reference eye is valid and case insensitive
    opt.ref_eye = opt.ref_eye.upper()
    if opt.ref_eye not in ['OD', 'OS']:
        raise ValueError(f'Expected reference to be "OD" (right eye) or "OS" (left eye), but got "{opt.ref_eye}" instead!')

    # Create input arguments
    args = [(disc, opt.ref_eye) for disc in opt.disc_seg]

    # List storing optic disc and center of mass (COM) values across the entire dataset
    optic_discs = []
    coms = []

    # Preprocess optic disc segmentation by performing OD/OS normalization with multiprocessing
    with mp.Pool(opt.workers) as pool:
        for result in tqdm(pool.imap_unordered(prepr_disc, args), total=len(args), desc='Preprocessing Optic Disc'):
            disc_seg, disc_com = result
            optic_discs.append(disc_seg)
            coms.append(disc_com)
    
    # Compute the average COM
    target_com = np.array(coms).mean(0).tolist()
    print(f'Target Center of Mass (x, y) = ({target_com[1]:.1f}, {target_com[0]:.1f})')

    # Create input arguments
    args = [(disc, com, target_com) for disc, com in zip(optic_discs, coms)]
    optic_discs = []

    # Perform COM alignment of all optic discs
    with mp.Pool(opt.workers) as pool:
        for result in tqdm(pool.imap_unordered(disc_com_alignment, args), total=len(args), desc='Performing COM Alignment'):
            optic_discs.append(result)

    # Create output directory if necessary
    os.makedirs(opt.output_path, exist_ok=True)

    # Construct population consensus optic disc segmentation mask
    mean_disc_path = f'{opt.output_path}/mean_optic_disc.npy'
    mean_disc = np.array(optic_discs).mean(0)
    np.save(mean_disc_path, mean_disc)

    # Construct mean optic disc segmentation mask (50% consensus)
    mean_disc_seg_path = f'{opt.output_path}/optic_disc_mask.png'
    mean_disc_seg = (mean_disc >= 0.5).astype(np.float32)
    Image.fromarray(np.stack(3 * [255 * mean_disc_seg], axis=-1).astype(np.uint8)).save(mean_disc_seg_path)

    # Initialize VANTAGE-based atlas helper class
    atlas = VANTAGE_Atlas(model, image_size=mean_disc.shape[0])

    # Extract atlas raw data and point cloud representations
    atlas_centerlines, atlas_log_radii = atlas.get_raw_data()
    atlas_pc = atlas.get_point_cloud(unnormalize=False)

    # Extract segmentation mask and save raw data if the atlas contains learned radial offsets
    if model.pc_norm_offset:
        # Only build segmentation mask
        atlas_mask = atlas.get_segmask()

        # Save atlas raw data representation
        atlas_raw_path = f'{opt.output_path}/atlas_raw.npz'
        np.savez(atlas_raw_path, centerlines=atlas_centerlines, log_radii=atlas_log_radii)

    # Save atlas point cloud representation
    atlas_pc_path = f'{opt.output_path}/atlas_point_cloud.npy'
    np.save(atlas_pc_path, atlas_pc)

    # Save atlas segmentation mask representation
    if model.pc_norm_offset:
        atlas_mask_path = f'{opt.output_path}/atlas_mask.png'
        Image.fromarray(np.stack(3 * [255 * atlas_mask], axis=-1).astype(np.uint8)).save(atlas_mask_path)

    # Visualize the atlas
    if model.pc_norm_offset:
        VANTAGE_Atlas.visualize_atlas(atlas_pc, atlas_mask=atlas_mask, mean_disc_mask=mean_disc)
    else:
        VANTAGE_Atlas.visualize_atlas(atlas_pc, mean_disc_mask=mean_disc, figsize=(8, 4.5))
    plt.savefig(f'{opt.output_path}/visualize.png')
    print('\nDone. Successfully built atlas.\n')

if __name__ == '__main__':
    main()