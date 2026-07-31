# Vascular Atlas based on Neural Template-Aligned Graph Encodings

<img width="600" height="513" alt="image" src="https://github.com/user-attachments/assets/426231e7-2d25-4831-8fd4-9516816dad19" />

This is the author's official PyTorch implementation of "**V**ascular **A**tlas based on **N**eural **T**emplate-**A**ligned **G**raph **E**ncodings" (**VANTAGE**). VANTAGE is an AI-based framework for constructing the first learned spatial atlas of the retinal vasculature visualized in fundus imaging. Inspired by the success of AlphaFold at predicting the 3D structure of proteins, leverages high-similarity healthy vascular templates to construct an atlas, based on the idea that the optimal atlas should minimize the amount of energy required to be deformed onto each healthy subject in a dataset.

---

## Table of Contents
- [Installation](#installation)
  - [Requirements](#requirements)  
  - [Setup](#setup) 
- [Dataset and Pre-trained Models](#dataset-and-pre-trained-models)
  - [Data](#data)  
  - [Model Weights and Atlas](#model-weights-and-atlas) 
- [Usage](#usage)  
  - [Training](#training)  
  - [Build Atlas](#build-atlas)
  - [Vessel Segmentation GUI](#vessel-segmentation-gui)
  - [Evaluate Atlas](#evaluate-atlas)
- [Citation](#citation)

---

## Installation
### Requirements
The required dependencies are listed in `requirements.txt`, which consists of:
```
numpy
scipy
matplotlib
pandas
opencv-python
pillow
torch
torchvision
torch-geometric
networkx
pvbm
scikit-learn
scikit-image
fpsample
tqdm
seaborn
mpl-tools
```

### Setup
To use, clone the repository and install the dependencies in `requirements.txt`.
```bash
# Clone repository
git clone https://github.com/JHU-RAIL/VANTAGE.git
cd VANTAGE

# Install dependencies
pip install -r requirements.txt
```

## Dataset and Pre-trained Models
### Data
In our experiments, VANTAGE was trained and evaluated on the [FIVES dataset](https://www.nature.com/articles/s41597-022-01564-3), which is [publicly available for download](https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation). The FIVES dataset (150 training, 50 testing) consists of fundus images, vessel segmentations, and quality assessment scores of healthy, diabetic retinopathy (DR), age-related macular degeneration (AMD), and glaucoma subjects.

As part of our atlas evaluation, we selected the 33/50 healthy test dataset scans that have high quality assessment scores in all three criteria and annotated the major vessel structures. The annotations are located in `/results/example/labeled_healthy_test_fives/`. Additionally, we also graded the train and test dataset DR samples with high quality assessment scores in all three criteria with DR severity scores (1 = _Mild NPDR_, 2 = _Moderate-severe NPDR_, 3 = _PDR_) based on [standard clinical guidelines](https://eyewiki.org/Diabetic_Retinopathy#Clinical_Diagnosis). The DR grades are in `/data/FIVES_DR_grades.csv`.

**Disclaimer:** DR severity grading was performed by a trained grader, but not a board-certified ophthalmologist.

### Model Weights and Atlas
The pre-trained model weights used in our experiments (trained on 150 healthy FIVES training samples) can be downloaded [here](https://doi.org/10.5281/zenodo.21670682), which includes all the trained encoder, decoder, and learned atlas parameter. The vascular atlas subsequently built from the trained VANTAGE model is located in `/results/example/atlas_fives/`. If you use our pre-trained model or atlas, please refer to the [citation section](#citation) for guidance on how to cite our work.

## Usage
### Training
To train VANTAGE on the FIVES dataset:

```bash
python3 train.py --fundus_train ./path/to/FIVES/train/Original/*_N.png --vessel_train ./path/to/FIVES/train/Ground\ truth/*_N.png --gpu 0
```

The training progress and outputs will be saved in `/results/train/` or the folder specified by `--output_dir`, and the latest model checkpoint will be saved as `vantage.pth` and degree tensor (for the PNA encoder) will be saved as `deg.pt`.

Output file `compare_XXXXX.png` shows you a ground-truth dataset sample on the left, your VANTAGE learned atlas on the right, and the atlas deformed to match the sample in the middle. Upon successful convergence, the model's training output should look something like this (taken at 20k epochs, trained on FIVES).

<img width="2100" height="700" alt="image" src="https://github.com/user-attachments/assets/f901541c-1db2-41ee-bfe3-8025b9ca74be" />

### Build Atlas

After training VANTAGE, the next step is to build the atlas vessel point cloud, vessel segmentation mask, and optic disc segmentation. First, segment the optic disc of the training dataset using

```bash
python3 segment_disc.py --fundus_path ./path/to/FIVES/train/Original/*.png --vessel_seg ./path/to/FIVES/train/Ground\ truth/*.png
```

The optic disc segmentation masks are created in `/results/disc_seg/disc/`, or the output path specified. The atlas can then be built using

```bash
python3 build_atlas.py --model_ckpt ./results/train/vantage.pth --deg_path ./results/train/deg.pt --disc_seg ./results/disc_seg/disc/*_N.png
```

The built atlas should be located in `/results/atlas/`, where `atlas_point_cloud.npy` is the vessel point cloud, `atlas_mask.png` is the vessel segmentation mask, and `optic_disc_mask.png` is the optic disc atlas. The optic disc should overlap at an anatomically reasonable location with respect to the vasculature and `visualize.png` is expected to look something like this:

<img width="1867" height="700" alt="image" src="https://github.com/user-attachments/assets/0848a4d4-206c-47fc-bfa3-a5f116645cad" />

### Vessel Segmentation GUI

The major vessel structures of the atlas can also be segmented for future evaluation. While we currently do not have an automated tool for segmenting the vessel structures, the atlas point cloud can be segmented manually without too much effort using our simple Python Matplotlib GUI using the script

```bash
python3 ./utils/label_point_cloud.py --pc ./results/atlas/atlas_point_cloud.npy --names ./results/example/atlas_fives/labeled/label_names.txt
```

A new window should pop up that resembles the following.

<img width="556" height="500" alt="image" src="https://github.com/user-attachments/assets/2a64b665-bb6e-43b4-a8e1-4f8686ca822b" />

Type the `Label Number` on your keyboard, then hit the `Enter` key to switch labels. The number next to "Active label" at the top of the window change to the label number you typed in. Label 1 corresponds to the first line in the file specified by `--names`, label 2 corresponds to the second line, etc. To annotate, draw a circle around the points you wish to label. Closing the window automatically saves the annotations as `[file_name]_labeled.npz`. For an example of a labeled point cloud, see `/results/example/atlas_fives/labeled` or `/results/example/labeled_healthy_test_fives`.

**Note:** Python GUI must be enabled for the point cloud segmentation tool to work. Hence, the tool may not work for users running the code on compute clusters and SSH servers.

### Evaluate Atlas
To evaluate the atlas, use the script

```bash
python3 eval_atlas.py --fundus_train ./path/to/FIVES/train/Original/*_N.png --vessel_train ./path/to/FIVES/train/Ground\ truth/*_N.png --fundus_test ./path/to/FIVES/test/Original/*_N.png --vessel_test ./path/to/FIVES/test/Ground\ truth/*_N.png --disease_fundus_test ./path/to/FIVES/test/Original/*_D.png --disease_vessel_test ./path/to/FIVES/test/Ground\ truth/*_D.png --disease_label DR --model_ckpt ./results/train/vantage.pth --deg_path ./results/train/deg.pt --atlas_seg ./results/example/atlas_fives/atlas_mask.png --disc_seg ./results/example/atlas_fives/optic_disc_mask.png --pc_labels ./results/example/atlas_fives/labeled/atlas_point_cloud_labeled.npz --output_path ./results/eval_atlas/ --gpu 0
```

and specify your healthy training dataset, healthy testing dataset, and diseased testing dataset. The script outputs chamfer distance distribution comparisons, biological quantification metrics across the atlas, per-point average distance heatmaps, segmentation label transfer contours, geometric centricity, and more, in `/results/eval_atlas/` or otherwise specified.

To compare the atlas biological quantification metrics against the training dataset, use

```bash
python3 quantify_vasculature.py --vessel_seg ./results/disc_seg/vessel/*_N.png --disc_seg ./results/disc_seg/disc/*_N.png
```

to compute statistics and vizualize the distributions, the outputs of which are located in `/results/quantification/` unless otherwise specified.

To further evaluate segmentation label transfer and computing performance metrics (i.e. IoU, HD95), call

```bash
python3 eval_seg.py --labels ./results/example/labeled_healthy_test_fives/*.npz --pred_contours ./results/eval_atlas/point_cloud/test/inference/raw_results/contours_*.pkl --n_labels 4
```

where `--pred_contours` should specify the files output by `eval_atlas.py` when predicting the VANTAGE model. These are the atlas segmentation label transfer contour files and should be located in `/results/eval_atlas/point_cloud/[dataset]/inference/raw_results/contours_X.pkl` after running `eval_atlas.py`. The argument `--n_labels 4` limits evaluation to the first four segmentation labels in our experiments, thus excluding `First Major Bifurcation of STA` and `First Major Bifurcation of ITA` due to substantial intersubject variability. The results will be output in `/results/eval_seg/` unless otherwise specified.

## Citation

If you find this repository or our pre-trained models useful, please cite our work:

- Stay tuned for a publication!
