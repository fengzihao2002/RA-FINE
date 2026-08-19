#!/usr/bin/env python3
"""Synthetic test for the grid_sample offset convention used by FS-CDAE.

A positive x offset means "sample from a location to the right". Consequently,
an impulse in the source appears one cell to the left in the warped output.
This convention is important when interpreting learned offset visualizations.
"""

import torch
import torch.nn.functional as F


def base_grid(height: int, width: int) -> torch.Tensor:
    ys = (torch.arange(height, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    xs = (torch.arange(width, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1).unsqueeze(0)


def main() -> None:
    h = w = 9
    source = torch.zeros(1, 1, h, w)
    source[0, 0, 4, 5] = 1.0

    grid = base_grid(h, w)
    positive_one_cell_x = torch.zeros_like(grid)
    positive_one_cell_x[..., 0] = 2.0 / w
    warped = F.grid_sample(
        source,
        grid + positive_one_cell_x,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

    peak = torch.nonzero(warped[0, 0] == warped[0, 0].max(), as_tuple=False)[0]
    expected = torch.tensor([4, 4])
    if not torch.equal(peak.cpu(), expected):
        raise AssertionError(f"Unexpected warped peak {peak.tolist()}, expected {expected.tolist()}")
    print("PASS: positive dx samples from the right and moves source content left by one cell.")


if __name__ == "__main__":
    main()
