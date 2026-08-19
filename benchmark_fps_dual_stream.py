#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark D-FINE Dual-stream inference speed.

Default protocol:
- batch size: 1
- input: 6 x 512 x 640, ordered as [RGB(3), IR(3)]
- 100 warm-up iterations
- 1000 timed iterations
- CUDA synchronization before/after timing
- FP32 by default
- excludes file reading and dataloader time

The script reports:
1) pure model-forward FPS
2) detector FPS = model forward + D-FINE post-processing
"""

from __future__ import annotations

import argparse
import contextlib
import os
import time
from typing import Dict

import torch

from src.core import YAMLConfig


def load_checkpoint(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if "ema" in checkpoint:
        ema = checkpoint["ema"]
        state = ema["module"] if isinstance(ema, dict) and "module" in ema else ema
        source = "EMA"
    elif "model" in checkpoint:
        state = checkpoint["model"]
        source = "model"
    else:
        state = checkpoint
        source = "raw state_dict"

    state = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    print(f"Loaded checkpoint source: {source}")
    return state


def autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=(
            "configs/dfine/custom/objects365/"
            "dfine_hgnetv2_m_obj2rgbtdroneperson_dual_stream.yml"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/hy-tmp/output/dfine_m_rgbtdroneperson_dual_stream/"
            "best_stg2.pth"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use FP16 autocast. Default is FP32.",
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Convert supported modules to deploy form before timing.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; FPS must be measured on the GPU.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    cfg = YAMLConfig(args.config, resume=None)
    model = cfg.model
    postprocessor = cfg.postprocessor

    state = load_checkpoint(args.checkpoint)
    incompatible = model.load_state_dict(state, strict=False)

    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)
        raise RuntimeError(
            "Checkpoint and Dual-stream model do not match. "
            "Do not use the resulting FPS."
        )

    model = model.to(device).eval()
    postprocessor = postprocessor.to(device).eval()

    if args.deploy:
        model = model.deploy()

    sample = torch.randn(
        1, 6, args.height, args.width,
        device=device,
        dtype=torch.float32,
    )
    # D-FINE uses [original_width, original_height].
    orig_size = torch.tensor(
        [[args.width, args.height]],
        device=device,
        dtype=torch.float32,
    )

    print("=" * 72)
    print("D-FINE Dual-stream FPS benchmark")
    print("=" * 72)
    print(f"GPU          : {torch.cuda.get_device_name(device)}")
    print(f"Checkpoint   : {args.checkpoint}")
    print(f"Input        : batch=1, shape={tuple(sample.shape)}")
    print(f"Precision    : {'FP16 autocast' if args.amp else 'FP32'}")
    print(f"Deploy form  : {args.deploy}")
    print(f"Warm-up      : {args.warmup}")
    print(f"Timed iters  : {args.iters}")
    print("Image I/O    : excluded")
    print("=" * 72)

    # Warm-up includes the full detector path.
    for _ in range(args.warmup):
        with autocast_context(args.amp):
            outputs = model(sample)
            _ = postprocessor(outputs, orig_size)
    torch.cuda.synchronize(device)

    # Pure forward.
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        with autocast_context(args.amp):
            _ = model(sample)
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - start

    # Model + postprocessor.
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.iters):
        with autocast_context(args.amp):
            outputs = model(sample)
            _ = postprocessor(outputs, orig_size)
    torch.cuda.synchronize(device)
    detector_seconds = time.perf_counter() - start

    forward_latency_ms = forward_seconds / args.iters * 1000.0
    detector_latency_ms = detector_seconds / args.iters * 1000.0

    print()
    print(f"Pure forward latency : {forward_latency_ms:.3f} ms/image")
    print(f"Pure forward FPS     : {1000.0 / forward_latency_ms:.2f}")
    print(f"Detector latency     : {detector_latency_ms:.3f} ms/image")
    print(f"Detector FPS         : {1000.0 / detector_latency_ms:.2f}")
    print()
    print(
        "Recommended paper value: Detector FPS "
        "(model forward + post-processing, batch=1, excluding image I/O)."
    )


if __name__ == "__main__":
    main()
