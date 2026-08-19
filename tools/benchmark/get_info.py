"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import torch
import torch.nn as nn
from calflops import calculate_flops

from src.core import YAMLConfig


def custom_repr(self):
    return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


def main(args):
    cfg = YAMLConfig(args.config, resume=None)

    class ModelForFlops(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()

        def forward(self, images):
            return self.model(images)

    model = ModelForFlops().eval()
    eval_h, eval_w = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    input_channels = cfg.yaml_cfg.get("input_channels", 6)

    flops, macs, _ = calculate_flops(
        model=model,
        input_shape=(1, input_channels, eval_h, eval_w),
        output_as_string=True,
        output_precision=4,
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    print("Model FLOPs:%s   MACs:%s   Params:%s\n" % (flops, macs, params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", default="configs/dfine/dfine_hgnetv2_l_coco.yml", type=str
    )
    main(parser.parse_args())
