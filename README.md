# Adaptive NGPS: Neighbor-Guided Patch Sampling

Python implementation and experimental study of adaptive neighbor selection for self-supervised denoising of BrainWeb T1 phantom volumes, modifications on existing NGPS by Jaehyun Cho and YoungJoon Yoo.

Adaptive NGPS trains a 2D slice denoiser using noisy neighboring slices. During training, it:

 1.Creates bilateral- and median-filtered guides from noisy slices.
 2.Identifies static and dynamic regions from differences between guides.
 3.Searches each candidate slice for similar local guide patches.
 4.Retrieves and averages raw noisy-neighbor values at the four best matching locations.
 5.Accepts each candidate at each location only when it is available and its normalized matching cost is at most 0.05.

Computes candidate-specific static, dynamic, and regional consistency loss contributions.

Radius 1 considers the immediate neighbors at offsets −1 and +1. Radius 2 also considers −2 and +2. The threshold is an addition to the Original NGPS implementation used in this project. At inference, the trained model denoises slices with a standard forward pass; the adaptive neighbor search is part of training, not inference.

# Repository Contents

configs/      Training and model configuration files
datasets/     Volume loaders
ngps/         Guides, patch sampler, loss functions, and NAFNet
scripts/      Training, validation, inference, evaluation, and plotting
runs/         To save model checkpoints and validation results
outputs/      To save denoised volumes, metrics, and comparison figures

Directory contents and paths are based on the project files and commands shared for this study.

# Installation

Open Windows Command Prompt in your project directory:

cd "complete path for NGPS-master"
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Use the Python environment in which the project dependencies are installed. CUDA requires a compatible PyTorch installation and GPU. Use --device cpu if CUDA is unavailable.

# Data Preparation
The study used ten BrainWeb T1 phantoms split by subject:
Subjects:
Training
18, 41, 44, 48, 49, 51
Validation
42, 54
Test
04, 06

Prepared volumes had shape (181, 256, 256) and 1 mm voxel spacing. Intensities were scaled to [0, 1], and Gaussian noise was generated at σ = 0.03, 0.06, and 0.10. The prepared data were organized by noise level, including paths such as:

data/brainweb/sigma_03/
  train/
  val/
  test/
    noisy/subject04.npy
    noisy/subject06.npy
    clean/subject04.npy
    clean/subject06.npy

# Training Adaptive NGPS

Example: radius 1, σ = 0.03. Each command below is a single Windows CMD line.

Go to your project directory in the command prompt using the command:

cd "complete path for NGPS-Master"
python scripts/train_adaptive_ngps.py --data-root data/brainweb/sigma_03 --config configs/adaptive_radius1_ngps_nafnet_default.yaml --output-dir runs/adaptive_sigma03_radius1_new --device cuda --save-every 1

For radius 2, use configs/adaptive_radius2_ngps_nafnet_default.yaml or configs/synthetic_10epoch_radius_2.yaml (for smaller network and quick training) and a separate output directory. Inspect configuration values before a new run. 

Configuratons used in the adaptive checkpoints radius2:

patch size           7 × 7
spatial search       15 × 15
matches per candidate 4
match threshold      0.05
region-mask threshold 0.05
regional-consistency weight 0.5
NAFNet width         16
encoder blocks       [1, 1, 1, 1]
middle blocks        1
decoder blocks       [1, 1, 1, 1]
batch size           2
crop size            64 × 64
configured epochs    10
optimizer            AdamW, learning rate 0.0002, weight decay 0.00001

The smaller network and crop size made adaptive training more manageable on the local computer. 

# Comparison Models

The experiments included Original NGPS, NS-N2N, adaptive radius 1, adaptive radius 2 and an unmasked same-location adjacent target model. Original NGPS and NS-N2N checkpoints inspected across all three noise levels used NAFNet width 32, batch size 1, and crop size 256. The adaptive radius 2 used same NAFNet width and batch size but crop size was 64. Same-location checkpoints used width 16, batch size 1, and crop size 64. The adaptive radius 2 and same-location models shared the inspected backbone architecture, but not batch size. 

Example same-location training command, using the project-specific script and configuration:

python scripts/train_samelocation.py --data-root data/brainweb/sigma_03 --config configs/samelocation_matched.yaml --output-dir runs/samelocation_sigma03 --device cuda --save-every 1


# Validation

Validate saved adaptive checkpoints on the validation split:

python scripts/validate_adaptive_ngps.py --data-root data/brainweb/sigma_06 --checkpoint-dir runs/Adaptive_brainweb_sigma_06_radius_1 --device cuda

The script evaluates available model_E*.pth checkpoints, saves validation_adaptive_loss.csv, and copies the lowest loss checkpoint to model_best.pth. 
Clean-reference test scores were also examined after models were fixed.

# Inference

Example: apply the selected adaptive radius 1 model to held-out subject 04 at σ = 0.06:

python scripts/infer_volume.py --checkpoint runs/Adaptive_brainweb_sigma_06_radius_1/model_best.pth --input data/brainweb/sigma_06/test/noisy/subject04.npy --output outputs/radius1_sigma06/subject04_denoised.npy --device cuda

The inference script saves a denoised .npy volume. Use the corresponding noise level checkpoint and input for each experiment. 

# Evaluation

Evaluate the saved denoised volume against its matching clean test reference:

python scripts/evaluate_volume.py --pred outputs/radius1_sigma06/subject04_denoised.npy --clean data/brainweb/sigma_06/test/clean/subject04.npy --output-csv outputs/radius1_sigma06/subject04_metrics.csv

The project calculates PSNR and SSIM with data range 1, along with HFEN and GMSD. The code also reports a value labeled FSIM.


# Recorded Adaptive Results

Mean over held-out subjects 04 and 06:
| σ |Radius|PSNR(dB)↑| SSIM ↑| HFEN ↓ | GMSD ↓ |

| 0.03 | 1 | 32.787 | 0.8656 | 0.3333 | 0.0308 |
| 0.03 | 2 | 32.405 | 0.8557 | 0.3533 | 0.0357 |
| 0.06 | 1 | 30.444 | 0.7700 | 0.3737 | 0.0410 |
| 0.06 | 2 | 30.556 | 0.7778 | 0.3928 | 0.0446 |
| 0.10 | 1 | 28.240 | 0.6857 | 0.4431 | 0.0570 |
| 0.10 | 2 | 28.044 | 0.6800 | 0.4494 | 0.0582 |

Radius 1 generally performed better than radius 2, but radius 2 had higher PSNR and SSIM at σ = 0.06. Adaptive NGPS did not consistently outperform the comparison methods. The small test set and unmatched training configurations limit interpretation; phantom experiments do not establish clinical effectiveness.

# Method-Code Mapping

datasets/triplet_dataset.py: noisy volume and guide preparation.

ngps/guide.py: guide filtering and static/dynamic mask.

ngps/sampler.py: guide-patch search, raw target retrieval, and candidate acceptance.

ngps/losses.py: candidate-specific hybrid loss.

scripts/train_adaptive_ngps.py: candidate assembly and training.

scripts/validate_adaptive_ngps.py: validation and checkpoint selection.

scripts/infer_volume.py and scripts/evaluate_volume.py: inference and evaluation.

# References and Attribution

Jaehyun Cho and YoungJoon Yoo, NGPS: Structure-Preserving Self-Supervised Denoising via Neighbor-Guided Patch Sampling (2026): https://arxiv.org/abs/2606.23200

Original authors' repository: https://github.com/cv-cho/NGPS
