RA-FINE
Reliability-Aware Cross-Modal Alignment and Fine-Grained Distribution Refinement for RGB-T UAV Object Detection

Official PyTorch implementation of RA-FINE.

Model Weights • Releases • D-FINE

Introduction
RGB-T UAV detection benefits from the complementary appearance and thermal information provided by visible and infrared modalities. In practice, however, performance is still limited by residual cross-modal misalignment, spatially varying modality reliability, and uncertain localization of tiny targets.

RA-FINE is built on D-FINE and introduces three complementary components:

FS-CDAE — Frequency-Spatial Cross-Modal Deformable Alignment Enhancement
Exploits high-frequency structural cues to estimate bounded local displacements and selectively align RGB features to the infrared reference. In the final configuration, FS-CDAE is applied at P3 and P4.

IGMRF — Improved Guided Multi-Resolution Fusion
Estimates spatially resolved modality reliability and injects reliability-aware priors into RGB-IR feature competition and fusion.

MA-FDR — Modality-Adaptive Fine-Grained Distribution Refinement
Refines D-FINE boundary distributions using modality-specific localization evidence, local reliability, boundary uncertainty, and cross-modal disagreement.

Together, these modules target the three main difficulties of RGB-T UAV detection: misalignment, unreliable modality contributions, and uncertain tiny-target boundaries.

News
2026-08-19 — Initial public release of RA-FINE.
2026-08-19 — Released v1.0.0 with trained RA-FINE and D-FINE-M checkpoints.
Repository Structure
RA-FINE/
├── configs/                         # Dataset, model, and training configurations
├── reference/                       # Reference/upstream materials retained by the project
├── src/                             # Core model and training implementation
│   ├── data/
│   └── zoo/
├── tools/                           # Utility, deployment, testing, and visualization tools
├── weights/                         # Weight documentation
├── Dockerfile
├── LICENSE
├── README.md
├── benchmark_fps_dual_stream.py
├── eval_rgbtdroneperson_official_dual_stream.py
├── requirements.txt
├── rgbtdroneperson_cocoeval.py
└── train.py
The main final P34 configuration for RGBTDronePerson is:

configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml
Installation
Recommended environment
The experiments reported for RA-FINE were conducted with:

Python 3.11
PyTorch 2.2.1
CUDA 12.1.x
Create a Python environment and install the dependencies:

conda create -n rafine python=3.11 -y
conda activate rafine

pip install -r requirements.txt
pip install pycocotools
Make sure that your PyTorch installation is compatible with the CUDA version available on your machine.

Datasets
RA-FINE is evaluated on RGBTDronePerson and VTUAV-det.

RGBTDronePerson
Please obtain RGBTDronePerson from its original public source:

Project page: https://nnnnerd.github.io/RGBTDronePerson/
Repository: https://github.com/NNNNerd/RGBTDronePerson
The paired data loader expects RGB and infrared images together with COCO-style annotations. A typical layout is:

RGBTDronePerson/
├── images/
│   ├── train/
│   │   ├── RGB/
│   │   └── IR/
│   └── val/
│       ├── RGB/
│       └── IR/
└── annotations/
    ├── instances_train.json
    ├── instances_val.json
    └── val_thermal.json
The dual-stream input is organized as:

[RGB (3 channels), IR (3 channels)]
Before training or evaluation, update the local dataset paths in:

configs/dataset/rgbtdroneperson_paired.yml
For example:

train_dataloader:
  dataset:
    rgb_img_folder: /path/to/RGBTDronePerson/images/train/RGB
    ir_img_folder: /path/to/RGBTDronePerson/images/train/IR
    ann_file: /path/to/RGBTDronePerson/annotations/instances_train.json

val_dataloader:
  dataset:
    rgb_img_folder: /path/to/RGBTDronePerson/images/val/RGB
    ir_img_folder: /path/to/RGBTDronePerson/images/val/IR
    ann_file: /path/to/RGBTDronePerson/annotations/instances_val.json
VTUAV-det
VTUAV-det is used as a second RGB-T pedestrian-detection benchmark and as the target dataset in the direct cross-dataset evaluation.

Please obtain VTUAV-det from its original source. The raw dataset is not redistributed in this repository.

The current public repository primarily provides the final RA-FINE implementation and the RGBTDronePerson training/evaluation pipeline. The released VTUAV-det checkpoint is provided through GitHub Releases.

Model Zoo
Trained checkpoints are distributed through:

RA-FINE v1.0.0:
https://github.com/fengzihao2002/RA-FINE/releases/tag/v1.0.0

Checkpoint	Training Dataset	Description
RA-FINE_RGBTDronePerson.pth	RGBTDronePerson	Final RA-FINE model for the main RGBTDronePerson experiments; also used for direct RGBTDronePerson → VTUAV-det evaluation
RA-FINE_VTUAV-det.pth	VTUAV-det	Final RA-FINE model trained and evaluated on VTUAV-det
D-FINE-M_RGBTDronePerson.pth	RGBTDronePerson	Dual-stream D-FINE-M baseline trained on RGBTDronePerson
After downloading, the checkpoints may be organized as:

RA-FINE/
└── weights/
    ├── RA-FINE_RGBTDronePerson.pth
    ├── RA-FINE_VTUAV-det.pth
    └── D-FINE-M_RGBTDronePerson.pth
The original D-FINE-M Objects365 pretrained checkpoint is not redistributed here. Please obtain it from the official D-FINE repository.

Training
RA-FINE is initialized from the D-FINE-M Objects365 pretrained checkpoint.

Official D-FINE repository:

https://github.com/Peterande/D-FINE

The final RGBTDronePerson configuration uses:

D-FINE-M / HGNetv2-B2 backbone
dual-stream RGB-T input
FS-CDAE on P3 and P4
IGMRF for reliability-guided multi-resolution fusion
MA-FDR in the decoder
input resolution: 512 × 640
training epochs: 56
AMP enabled
random seed: 0
Example single-GPU training command:

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  -c configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  -t /path/to/dfine_m_obj365.pth \
  --use-amp \
  --seed 0 \
  --output-dir ./output/rafine_rgbtdroneperson
Evaluation
COCO-style evaluation
Evaluate the released RGBTDronePerson checkpoint with:

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  -c configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  -r weights/RA-FINE_RGBTDronePerson.pth \
  --test-only
Official RGBTDronePerson evaluation
python eval_rgbtdroneperson_official_dual_stream.py \
  --config configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  --checkpoint weights/RA-FINE_RGBTDronePerson.pth \
  --official-ann /path/to/RGBTDronePerson/annotations/val_thermal.json \
  --official-cocoeval ./rgbtdroneperson_cocoeval.py \
  --output-dir ./output/official_eval
The evaluator exports:

official_predictions.json
official_metrics.json
Cross-Dataset Evaluation
For direct RGBTDronePerson-to-VTUAV-det evaluation, use:

RA-FINE_RGBTDronePerson.pth
as the source-domain checkpoint and evaluate it directly on VTUAV-det.

No target-domain training or fine-tuning is performed before cross-dataset evaluation.

This experiment is designed to evaluate whether the improvement of RA-FINE is retained under dataset shift rather than after target-domain adaptation.

Inference Speed
The repository provides:

benchmark_fps_dual_stream.py
The reported timing protocol uses:

batch size: 1
input shape: 6 × 512 × 640
100 warm-up iterations
1000 timed iterations
CUDA synchronization
image loading excluded
Example:

python benchmark_fps_dual_stream.py \
  --config configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  --checkpoint weights/RA-FINE_RGBTDronePerson.pth \
  --device cuda:0 \
  --height 512 \
  --width 640
FPS should only be compared under the same hardware, input resolution, precision mode, batch size, and timing protocol.

Main Configuration
Key settings of the final P34 model include:

input_channels: 6
eval_spatial_size: [512, 640]
epochs: 56

DFINE:
  use_fs_cdae: True
  fs_enabled_levels: [0, 1]
  use_igmrf: True
  use_ma_fdr: True

  fs_max_offset_cells: 1.5
  igmrf_reliability_bias_scale: 0.50

DFINETransformer:
  num_layers: 4
  ma_temperature: 1.5
  ma_residual_clip: 3.0
  ma_max_gamma: 0.50
  ma_reliability_prior_scale: 0.50
For the complete configuration, see:

configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml
Results
RGBTDronePerson
Method	mAP50_all (%)	mAP25_all (%)	COCO AP (%)	AP75 (%)	APs (%)
D-FINE-M	50.36	51.97	18.20	7.44	18.21
RA-FINE	53.06	54.60	20.27	9.48	20.46
Improvement	+2.70	+2.63	+2.07	+2.04	+2.25
VTUAV-det
The final RA-FINE model reaches a COCO AP of:

37.74%
under in-domain training and evaluation on VTUAV-det.

Direct Cross-Dataset Transfer
When the RGBTDronePerson-trained models are evaluated directly on VTUAV-det without target-domain training or fine-tuning, RA-FINE retains a:

+0.92 AP
advantage over the dual-stream D-FINE-M baseline.

Computational Cost
Method	Params	GFLOPs	FPS
D-FINE-M	31.41 M	69.72	32.8
RA-FINE	41.65 M	85.09	22.9
Citation
The formal bibliographic information for the RA-FINE paper will be added after publication.

If this repository is useful for your research, please also cite:

the original D-FINE work;
the RGBTDronePerson benchmark;
the VTUAV / VTUAV-det benchmark as appropriate.
Acknowledgements
RA-FINE is developed on top of D-FINE:

https://github.com/Peterande/D-FINE

We thank the D-FINE authors for making their implementation and pretrained models publicly available.

We also thank the authors of RGBTDronePerson and VTUAV for releasing the RGB-T UAV data and evaluation resources used in this study.

License
This repository retains the upstream Apache License 2.0 license and applicable notices from D-FINE.

Please review:

LICENSE
before reuse or redistribution, and comply with the licenses and attribution requirements of all upstream components and datasets.

Contact
For questions regarding the code or reproduction:

Zihao Feng
School of Railway Intelligent Engineering, Dalian Jiaotong University
Email: fengzihao2002@163.com

For correspondence regarding the paper:

Xinying Chen
Email: chenxy1979@163.com
