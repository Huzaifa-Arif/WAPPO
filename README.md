# WAPPO: Weather Adversarial Perturbation for AI Weather Prediction

This repository contains the code for **WAPPO** - a framework for analyzing adversarial robustness of AI-based weather forecasting models, specifically targeting [FourCastNet](https://arxiv.org/abs/2202.11214).

## Overview

WAPPO investigates the vulnerability of deep learning weather prediction models to adversarial perturbations. We demonstrate that small, carefully crafted perturbations to initial conditions can significantly degrade forecast accuracy, raising important questions about the reliability of AI weather models in operational settings.

### Key Features

- **Adversarial Attack Framework**: Optimizes perturbations to maximize forecast error
- **Multiple Loss Functions**: MSE, LPIPS (perceptual), and Fourier-domain losses
- **Constrained Perturbations**: Channel masking, patch masking, Total Variation, and L-infinity constraints
- **Comprehensive Metrics**: Latitude-weighted RMSE, ACC, PSNR, SSIM, and more
- **Visualization Tools**: Loss curves, perturbation trajectories, and forecast comparisons

## Installation

```bash
git clone https://github.com/Huzaifa-Arif/WAPPO.git
cd WAPPO
pip install -r requirements.txt
```

## Data & Model Weights

### Download Pre-trained Weights
Download the FourCastNet model weights from:
- [NERSC Web Download](https://portal.nersc.gov/project/m4134/FCN_weights_v0/)
- [Globus Download](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fmodel_weights%2F)

### Download Normalization Statistics
Download from [Globus - Additional](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fadditional%2F)

### Download Test Data
Download ERA5 out-of-sample data from [Globus - Data](https://app.globus.org/file-manager?origin_id=945b3c9e-0f8c-11ed-8daf-9f359c660fbd&origin_path=%2F~%2Fdata%2F)

### Directory Structure
Place downloaded files as follows:
```
WAPPO/
├── FourCast_Dataset/
│   ├── additional/stats_v0/
│   │   ├── global_means.npy
│   │   ├── global_stds.npy
│   │   ├── time_means.npy
│   │   ├── land_sea_mask.npy
│   │   ├── latitude.npy
│   │   └── longitude.npy
│   ├── model_weights/FCN_weights_v0/
│   │   ├── backbone.ckpt
│   │   └── precip.ckpt
│   └── data/FCN_ERA5_data_v0/
│       └── out_of_sample/
│           └── 2018.h5
```

## Usage

### Run Adversarial Attack

```bash
python main.py \
    --field t2m \
    --prediction_length 4 \
    --gamma 1.0 \
    --lamda 0.0 \
    --output_path ./results
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--field` | Target variable (e.g., t2m, sp, u10) | `sp` |
| `--prediction_length` | Forecast steps (6h each) | `4` |
| `--gamma` | MSE loss weight | `1.0` |
| `--lamda` | LPIPS loss weight | `0.0` |
| `--epsilon` | L-infinity constraint per channel | see code |
| `--tau` | Total Variation constraint per channel | see code |
| `--channel_perturb` | Channels to perturb | all 20 |
| `--patch_start_L`, `--patch_size_L` | Longitude patch bounds | full |
| `--patch_start_M`, `--patch_size_M` | Latitude patch bounds | full |

### Supported Variables (20 Channels)

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | u10 | 10m U wind component |
| 1 | v10 | 10m V wind component |
| 2 | t2m | 2m temperature |
| 3 | sp | Surface pressure |
| 4 | msl | Mean sea level pressure |
| 5 | t850 | Temperature at 850 hPa |
| 6-8 | u1000, v1000, z1000 | Wind & geopotential at 1000 hPa |
| 9-11 | u850, v850, z850 | Wind & geopotential at 850 hPa |
| 12-14 | u500, v500, z500 | Wind & geopotential at 500 hPa |
| 15 | t500 | Temperature at 500 hPa |
| 16 | z50 | Geopotential at 50 hPa |
| 17-18 | r500, r850 | Relative humidity |
| 19 | tcwv | Total column water vapor |

## Project Structure

```
WAPPO/
├── main.py                      # Entry point for adversarial attacks
├── model_inference.py           # Inference and perturbation optimization
├── adversarial.py               # Core adversarial perturbation functions
├── adversarial_model_utils.py   # Model loading, LPIPS, masking utilities
├── adversarial_data_utils.py    # Data loading and preprocessing
├── eval_metrics.py              # Evaluation metrics (RMSE, ACC, SSIM, etc.)
├── plot_utils.py                # Visualization utilities
├── variables.py                 # Variable name definitions
│
├── config/
│   └── AFNO.yaml               # Model hyperparameters
│
├── networks/
│   └── afnonet.py              # AFNO architecture (from FourCastNet)
│
├── utils/                       # Original FourCastNet utilities
│   ├── YParams.py
│   ├── data_loader_multifiles.py
│   ├── img_utils.py
│   └── weighted_acc_rmse.py
│
├── inference/                   # Standard inference scripts
│   ├── inference.py
│   └── inference_ensemble.py
│
├── train.py                     # FourCastNet training (original)
├── data_process/                # ERA5 data preprocessing
└── copernicus/                  # Copernicus data download scripts
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{arif2024wappo,
  title={WAPPO: Weather Adversarial Perturbation for AI Weather Prediction},
  author={Arif, Huzaifa and Chen, Pin-Yu and Gittens, Alex and Diffenderfer, James and Kailkhura, Bhavya},
  journal={arXiv preprint arXiv:2512.08832},
  year={2024},
  url={https://arxiv.org/pdf/2512.08832}
}
```

Also cite the original FourCastNet paper:

```bibtex
@article{pathak2022fourcastnet,
  title={FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators},
  author={Pathak, Jaideep and Subramanian, Shashank and Harrington, Peter and Raja, Sanjeev and Chattopadhyay, Ashesh and Mardani, Morteza and Kurth, Thorsten and Hall, David and Li, Zongyi and Azizzadenesheli, Kamyar and Hassanzadeh, Pedram and Kashinath, Karthik and Anandkumar, Animashree},
  journal={arXiv preprint arXiv:2202.11214},
  year={2022}
}
```

## License

This project builds upon [FourCastNet](https://github.com/NVlabs/FourCastNet) which is released under the BSD-3-Clause License.

## Acknowledgments

- NVIDIA and NERSC for the original FourCastNet implementation
- ERA5 data from the Copernicus Climate Change Service (C3S)
