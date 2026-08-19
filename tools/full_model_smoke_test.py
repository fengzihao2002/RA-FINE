#!/usr/bin/env python3
"""Build the Full-P34 model and run a synthetic forward/backward smoke test.

This script is intended for the actual D-FINE environment after dependencies in
requirements.txt are installed. It does not require a dataset.
"""

from __future__ import annotations

import argparse

import torch

from src.core import YAMLConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/dfine/custom/full/dfine_hgnetv2_m_obj2rgbtdroneperson_full_p34.yml",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=160)
    args = parser.parse_args()

    cfg = YAMLConfig(args.config)
    model = cfg.model
    criterion = cfg.criterion

    # Synthetic sizes differ from the fixed evaluation resolution, therefore
    # force dynamic positional embeddings and anchors for this test only.
    model.encoder.eval_spatial_size = None
    model.decoder.eval_spatial_size = None
    model.train()
    criterion.train()

    x = torch.randn(1, 6, args.height, args.width)
    targets = [
        {
            "labels": torch.tensor([0, 1], dtype=torch.long),
            "boxes": torch.tensor(
                [[0.40, 0.40, 0.15, 0.20], [0.70, 0.60, 0.10, 0.15]],
                dtype=torch.float32,
            ),
        }
    ]

    assert tuple(model.fs_enabled_levels) == (0, 1), model.fs_enabled_levels
    assert tuple(sorted(int(k) for k in model.fs_cdae_layers.keys())) == (0, 1)
    assert "2" not in model.fs_cdae_layers

    outputs = model(x, targets=targets)
    losses = criterion(outputs, targets, epoch=0)
    non_finite = [name for name, value in losses.items() if not torch.isfinite(value).all()]
    if non_finite:
        raise RuntimeError(f"Non-finite losses: {non_finite}")
    total = sum(losses.values())
    total.backward()

    checks = {
        "FS P3 offset head": model.fs_cdae_layers["0"].offset_predictor.weight.grad,
        "FS P4 offset head": model.fs_cdae_layers["1"].offset_predictor.weight.grad,
        "IGMRF reliability head": model.igmrf_layers[0].rgb_reliability.net[-1].weight.grad,
        "IGMRF fusion strength": model.igmrf_layers[0].fusion_strength.grad,
        "MA-FDR gamma": model.decoder.decoder.ma_gamma.grad,
    }
    missing = [name for name, grad in checks.items() if grad is None]
    if missing:
        raise RuntimeError(f"Missing gradients: {missing}")

    print(f"PASS: {len(losses)} finite losses; total={float(total.detach()):.6f}")
    for name, grad in checks.items():
        print(f"  {name:<28}: mean|grad|={float(grad.abs().mean()):.8e}")
    print(
        "Note: MA adapter weights are gated by zero-initialized gamma and normally "
        "receive gradients after gamma moves away from zero on the first optimizer step."
    )


if __name__ == "__main__":
    main()
