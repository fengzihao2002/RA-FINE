# Model Weights

This directory documents the released checkpoints for **RA-FINE**.

The checkpoint files themselves are distributed through the **GitHub Releases** page of this repository rather than stored directly in the Git repository.

## Available Checkpoints

| Checkpoint | Training Dataset | Purpose |
|---|---|---|
| `RA-FINE_RGBTDronePerson.pth` | RGBTDronePerson | Final RA-FINE model used for the main RGBTDronePerson experiments and direct cross-dataset evaluation on VTUAV-det |
| `RA-FINE_VTUAV-det.pth` | VTUAV-det | Final RA-FINE model trained and evaluated on VTUAV-det |
| `D-FINE-M_RGBTDronePerson.pth` | RGBTDronePerson | Dual-stream D-FINE-M baseline used for comparison with RA-FINE |

## Download

Please download the checkpoints from the **Releases** section of this repository:

https://github.com/fengzihao2002/RA-FINE/releases

After downloading, you may place the files in this directory:

```text
RA-FINE/
└── weights/
    ├── RA-FINE_RGBTDronePerson.pth
    ├── RA-FINE_VTUAV-det.pth
    └── D-FINE-M_RGBTDronePerson.pth
```

## Usage

### RA-FINE on RGBTDronePerson

```bash
CUDA_VISIBLE_DEVICES=0 python -u train.py \
  -c configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml \
  -r weights/RA-FINE_RGBTDronePerson.pth \
  --test-only
```

### Cross-dataset evaluation

`RA-FINE_RGBTDronePerson.pth` is also the source-domain checkpoint used for direct RGBTDronePerson-to-VTUAV-det transfer evaluation.

No target-domain training or fine-tuning should be performed before this evaluation.

### D-FINE-M baseline on RGBTDronePerson

Use `D-FINE-M_RGBTDronePerson.pth` with the corresponding dual-stream D-FINE-M baseline configuration.

## Notes

- `RA-FINE_RGBTDronePerson.pth` and `RA-FINE_VTUAV-det.pth` are the final RA-FINE checkpoints corresponding to the paper experiments.
- `D-FINE-M_RGBTDronePerson.pth` is provided to facilitate direct baseline comparison.
- The original D-FINE Objects365 pretrained checkpoint is **not redistributed here**. Please obtain it from the official D-FINE repository.
- Dataset files are not included in this directory. Please obtain RGBTDronePerson and VTUAV-det from their original sources and follow the dataset preparation instructions in the main repository README.

## Citation

If these checkpoints are useful for your research, please cite the RA-FINE paper once the bibliographic information is available, together with the corresponding dataset and D-FINE references.
