"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

from calflops import calculate_flops


def stats(
    cfg,
    input_shape: Tuple = (1, 6, 640, 640),
) -> Tuple[int, dict]:
    eval_h, eval_w = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    input_channels = cfg.yaml_cfg.get("input_channels", 6)
    input_shape = (1, input_channels, eval_h, eval_w)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
