# RA-FINE

**RA-FINE: Reliability-Aware Cross-Modal Alignment and Fine-Grained Distribution Refinement for RGB-T UAV Object Detection**

Official implementation of **RA-FINE**, a dual-stream RGB-T UAV object detector built on D-FINE.

**Authors:** Zihao Feng, Xinying Chen  
**Affiliation:** School of Railway Intelligent Engineering, Dalian Jiaotong University, Dalian 116052, China  
**Code contact:** fengzihao2002@163.com  
**Correspondence:** chenxy1979@163.com

---

## Overview

RGB-T UAV object detection benefits from the complementary appearance and thermal information provided by visible and infrared modalities, but its performance can be degraded by residual cross-modal misalignment, spatially varying modality reliability, and uncertain localization of tiny objects.

RA-FINE extends D-FINE with three components:

- **FS-CDAE — Frequency-Spatial Cross-Modal Deformable Alignment Enhancement**  
  Uses high-frequency structural cues to estimate bounded local offsets and selectively aligns RGB features to the infrared reference. In the final P34 configuration, FS-CDAE is applied to **P3 and P4**, while P5 bypasses geometric warping.

- **IGMRF — Improved Guided Multi-Resolution Fusion**  
  Estimates spatially varying modality reliability and introduces reliability-aware priors into RGB-IR feature competition and fusion.

- **MA-FDR — Modality-Adaptive Fine-Grained Distribution Refinement**  
  Refines D-FINE boundary distributions in the decoder using modality-specific localization evidence, reliability information, boundary uncertainty, and cross-modal disagreement.

The current public code package contains the **final P34 implementation and RGBTDronePerson training/evaluation pipeline**.

---

## Repository Structure

```text
RA-FINE/
├── configs/
│   ├── dataset/
│   │   └── rgbtdroneperson_paired.yml
│   └── dfine/
│       └── custom/
│           └── full/
│               └── dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml
├── src/
│   ├── data/
│   │   └── dataset/
│   │       └── paired_coco_dataset.py
│   └── zoo/
│       └── dfine/
│           ├── multimodal_full.py
│           ├── dfine.py
│           ├── dfine_decoder.py
│           └── ...
├── tools/
│   ├── full_model_smoke_test.py
│   └── test_fs_cdae_direction.py
├── train.py
├── benchmark_fps_dual_stream.py
├── eval_rgbtdroneperson_official_dual_stream.py
├── rgbtdroneperson_cocoeval.py
├── requirements.txt
└── LICENSE
```

The main RA-FINE configuration is:

```text
configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml
```

---

## Installation

### Recommended environment

The experiments reported in the manuscript were conducted with a Python/PyTorch/CUDA environment consistent with:

```text
Python 3.11
PyTorch 2.2.1
CUDA 12.1.x
```

Create an environment and install the repository dependencies:

```bash
conda create -n rafine python=3.11 -y
conda activate rafine

pip install -r requirements.txt
pip install pycocotools
```

> GPU/CUDA installations may require a PyTorch build matched to the CUDA version available on your system.

---

## Dataset Preparation

### RGBTDronePerson

Download **RGBTDronePerson** from its official project page:

- Project: https://nnnnerd.github.io/RGBTDronePerson/
- Code/data repository: https://github.com/NNNNerd/RGBTDronePerson

This implementation expects paired RGB and infrared images and COCO-style annotations. A typical directory layout is:

```text
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
```

The paired loader uses a 6-channel input ordered as:

```text
[RGB (3 channels), IR (3 channels)]
```

### Update dataset paths

Before training or evaluation, edit:

```text
configs/dataset/rgbtdroneperson_paired.yml
```

and replace the local paths with paths on your machine:

```yaml
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
```

Also change the default `output_dir` in the final configuration or override it from the command line.

> The current initial release does not include the original dataset images. Users should obtain RGBTDronePerson from its official source and prepare the COCO-style annotation files used by this implementation.

---

## Pretrained Weights

RA-FINE is initialized from the **D-FINE-M Objects365 pretrained checkpoint**.

Please obtain the corresponding checkpoint from the official D-FINE repository/model zoo:

- D-FINE: https://github.com/Peterande/D-FINE

Place the downloaded checkpoint anywhere convenient, for example:

```text
weights/dfine_m_obj365.pth
```

Model weights trained specifically for RA-FINE are not included in this initial source-code package.

---

## Training

The final RGBTDronePerson configuration uses:

- D-FINE-M / HGNetv2-B2 backbone
- RGB-T dual-stream input
- FS-CDAE on P3 and P4
- IGMRF on multi-scale dual-modal features
- MA-FDR in the decoder
- input resolution: `512 x 640`
- training epochs: `56`
- AMP enabled
- random seed: `0`

Example single-GPU training command:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  -c configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  -t /path/to/dfine_m_obj365.pth \
  --use-amp \
  --seed 0 \
  --output-dir ./output/rafine_rgbtdroneperson
```

Training checkpoints and logs will be written to the specified output directory.

---

## Evaluation

### COCO-style evaluation

After editing the dataset paths in the YAML configuration, evaluate a trained checkpoint with:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  -c configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  -r /path/to/RA-FINE.pth \
  --test-only
```

### Official RGBTDronePerson protocol

The repository also includes an evaluation script for the official tiny-person protocol:

```bash
python eval_rgbtdroneperson_official_dual_stream.py \
  --config configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  --checkpoint /path/to/RA-FINE.pth \
  --official-ann /path/to/RGBTDronePerson/annotations/val_thermal.json \
  --official-cocoeval ./rgbtdroneperson_cocoeval.py \
  --output-dir ./output/official_eval
```

The script exports:

```text
official_predictions.json
official_metrics.json
```

and prints the official RGBTDronePerson evaluation metrics.

---

## Inference Speed

`benchmark_fps_dual_stream.py` measures dual-stream inference speed using:

- batch size: 1
- input shape: `6 x 512 x 640`
- 100 warm-up iterations
- 1000 timed iterations
- CUDA synchronization
- image loading excluded

Example:

```bash
python benchmark_fps_dual_stream.py \
  --config configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  --checkpoint /path/to/RA-FINE.pth \
  --device cuda:0 \
  --height 512 \
  --width 640
```

The script reports both pure model-forward FPS and detector FPS (model forward + post-processing).

---

## Smoke Test

A synthetic forward/backward test is provided for the final P34 model:

```bash
PYTHONPATH=. python tools/full_model_smoke_test.py
```

The test verifies that the final model can be constructed, produces finite losses, and propagates gradients through FS-CDAE, IGMRF, and MA-FDR related paths.

A separate synthetic test is also provided for the FS-CDAE offset convention:

```bash
PYTHONPATH=. python tools/test_fs_cdae_direction.py
```

---

## Main Configuration

The final P34 model is configured with the following principal settings:

```yaml
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
```

For the complete configuration, see:

```text
configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml
```

---

## Experimental Results

Selected results reported for RA-FINE on RGBTDronePerson are:

| Method | mAP50_all (%) | mAP25_all (%) | COCO AP (%) | AP75 (%) | APs (%) |
|---|---:|---:|---:|---:|---:|
| RA-FINE | **53.06** | **54.60** | **20.27** | **9.48** | **20.46** |

Computational statistics reported for the final RA-FINE configuration:

| Params | GFLOPs | FPS |
|---:|---:|---:|
| 41.65 M | 85.09 | 22.9 |

> FPS should be compared only under the same hardware, input resolution, precision, batch size, and timing protocol.

---

## Citation

The bibliographic entry for the RA-FINE paper will be updated when the paper is publicly available.

If this repository is useful for your research, please also cite the original **D-FINE** work and the **RGBTDronePerson** benchmark.

---

## Acknowledgements

This project is developed on top of **D-FINE**:

- https://github.com/Peterande/D-FINE

We thank the D-FINE authors for releasing their implementation and pretrained models.

We also thank the authors of **RGBTDronePerson** for making the RGB-T UAV benchmark and its evaluation resources publicly available:

- https://github.com/NNNNerd/RGBTDronePerson
- https://nnnnerd.github.io/RGBTDronePerson/

---

## License

This repository is based on D-FINE and retains the upstream **Apache License 2.0** license file and applicable notices.

Please review `LICENSE` and the license/attribution requirements of all upstream components and datasets before redistribution.

---

## Contact

For questions regarding the code or reproduction:

**Zihao Feng**  
School of Railway Intelligent Engineering, Dalian Jiaotong University  
Email: `fengzihao2002@163.com`

For correspondence regarding the paper:

**Xinying Chen**  
Email: `chenxy1979@163.com`
