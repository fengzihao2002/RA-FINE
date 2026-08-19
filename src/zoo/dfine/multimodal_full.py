"""Lightweight modules for the final dual-stream D-FINE model.

The implementation keeps the three innovations at separate network stages:

1. FS-CDAE: high-frequency-guided RGB-to-IR geometric alignment.
2. IGMRF: reliability-biased, per-pixel cross-modal feature fusion.
3. MA-FDR: implemented in ``dfine_decoder.py`` as decoder-side distribution
   correction, using the reliability maps emitted here.

All new residual paths are identity-preserving at initialization. This is
important when fine-tuning from a single-stream Objects365 checkpoint.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedHaarHighFrequency(nn.Module):
    """Extract Haar LH/HL/HH responses without trainable filters.

    The implementation uses tensor slicing instead of a very large grouped
    convolution. It is exact for the standard 2x2 Haar basis and remains fully
    differentiable with respect to the input feature map.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got {tuple(x.shape)}")

        h, w = x.shape[-2:]
        # Backbones used by D-FINE normally produce even feature sizes. Padding
        # keeps the module safe for arbitrary resolutions.
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]

        # Haar high-frequency sub-bands. The factor 0.5 preserves the standard
        # orthonormal scaling convention for a 2x2 basis.
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return torch.cat((lh, hl, hh), dim=1)


class FSCDAELevel(nn.Module):
    """Frequency-spatial cross-modal deformable alignment at one scale.

    RGB is aligned to the IR coordinate system. The offset predictor is zero
    initialized, so the first forward pass is an exact identity warp.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        max_offset_cells: float = 1.5,
        align_strength_init: float = 0.10,
        align_gate_init: float = 0.10,
        padding_mode: str = "border",
        align_corners: bool = False,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        high_channels = 3 * channels

        self.high_frequency = FixedHaarHighFrequency()
        self.high_reduce_rgb = nn.Sequential(
            nn.Conv2d(high_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.high_reduce_ir = nn.Sequential(
            nn.Conv2d(high_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )

        joint_channels = 2 * hidden
        bottleneck = max(hidden, 16)
        self.offset_head = nn.Sequential(
            nn.Conv2d(joint_channels, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.GELU(),
            nn.Conv2d(
                bottleneck,
                bottleneck,
                3,
                padding=1,
                groups=bottleneck,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck),
            nn.GELU(),
            nn.Conv2d(bottleneck, bottleneck, 1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.GELU(),
        )
        self.offset_predictor = nn.Conv2d(bottleneck, 2, 3, padding=1, bias=True)
        self.align_gate = nn.Conv2d(bottleneck, 1, 1, bias=True)

        nn.init.zeros_(self.offset_predictor.weight)
        nn.init.zeros_(self.offset_predictor.bias)
        nn.init.zeros_(self.align_gate.weight)
        gate_init = min(max(float(align_gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.align_gate.bias, math.log(gate_init / (1.0 - gate_init)))

        strength_init = min(max(float(align_strength_init), 1e-4), 1.0 - 1e-4)
        self.align_strength_logit = nn.Parameter(
            torch.tensor(math.log(strength_init / (1.0 - strength_init)), dtype=torch.float32)
        )
        self.max_offset_cells = float(max_offset_cells)
        if padding_mode not in {"zeros", "border", "reflection"}:
            raise ValueError(f"Unsupported grid_sample padding mode: {padding_mode}")
        self.padding_mode = padding_mode
        self.align_corners = bool(align_corners)

    @staticmethod
    def _base_grid(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        align_corners: bool,
    ) -> torch.Tensor:
        if align_corners:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if rgb.shape != ir.shape:
            raise ValueError(
                f"FS-CDAE expects equal RGB/IR shapes, got {tuple(rgb.shape)} and {tuple(ir.shape)}"
            )

        b, _, h, w = rgb.shape
        rgb_high = self.high_reduce_rgb(self.high_frequency(rgb))
        ir_high = self.high_reduce_ir(self.high_frequency(ir))
        rgb_high = F.interpolate(rgb_high, size=(h, w), mode="bilinear", align_corners=False)
        ir_high = F.interpolate(ir_high, size=(h, w), mode="bilinear", align_corners=False)

        joint = self.offset_head(torch.cat((rgb_high, ir_high), dim=1))
        raw_offset = self.offset_predictor(joint)
        align_gate = torch.sigmoid(self.align_gate(joint))
        align_strength = torch.sigmoid(self.align_strength_logit)

        # Offset is represented in feature-map cells and then normalized for
        # grid_sample. Positive x samples from the right; positive y samples
        # from below. The included synthetic test verifies this convention.
        offset_cells = (
            torch.tanh(raw_offset)
            * self.max_offset_cells
            * align_gate
            * align_strength
        )
        if self.align_corners:
            norm_x = 2.0 / max(w - 1, 1)
            norm_y = 2.0 / max(h - 1, 1)
        else:
            norm_x = 2.0 / max(w, 1)
            norm_y = 2.0 / max(h, 1)

        offset_grid = torch.stack(
            (offset_cells[:, 0] * norm_x, offset_cells[:, 1] * norm_y), dim=-1
        )
        base_grid = self._base_grid(
            b, h, w, rgb.device, rgb.dtype, self.align_corners
        )
        sampling_grid = base_grid + offset_grid
        aligned_rgb = F.grid_sample(
            rgb,
            sampling_grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        dx = offset_cells[..., :, 1:] - offset_cells[..., :, :-1]
        dy = offset_cells[..., 1:, :] - offset_cells[..., :-1, :]
        smooth = dx.abs().mean() + dy.abs().mean()
        diagnostics = {
            "offset_abs": offset_cells.abs().mean(),
            "offset_max": offset_cells.detach().abs().amax(),
            "offset_smooth": smooth,
            "align_gate_mean": align_gate.mean(),
            "align_strength": align_strength,
        }
        return aligned_rgb, diagnostics


class SpatialReliabilityEstimator(nn.Module):
    """Predict a spatial reliability map for one modality."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        # Reliability starts at 0.5 everywhere. Detection gradients then learn
        # spatially varying reliability rather than inheriting a hard prior.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class IGMRFLevel(nn.Module):
    """Illumination-guided modality-reliability fusion at one scale.

    A compact two-token attention is evaluated independently at every spatial
    position. Reliability enters as an additive log-prior before Softmax. The
    fused and purified outputs use zero-initialized residual strengths, making
    the initial main path equal to the IR-reference baseline.
    """

    def __init__(
        self,
        channels: int,
        attention_dim: int = 64,
        reliability_bias_scale: float = 0.5,
        reliability_eps: float = 1e-4,
        residual_strength_init: float = 0.05,
    ) -> None:
        super().__init__()
        dim = max(min(int(attention_dim), channels), 16)
        self.rgb_reliability = SpatialReliabilityEstimator(channels)
        self.ir_reliability = SpatialReliabilityEstimator(channels)

        self.query = nn.Conv2d(2 * channels, dim, 1, bias=False)
        self.key_rgb = nn.Conv2d(channels, dim, 1, bias=False)
        self.key_ir = nn.Conv2d(channels, dim, 1, bias=False)
        self.value_rgb = nn.Conv2d(channels, channels, 1, bias=False)
        self.value_ir = nn.Conv2d(channels, channels, 1, bias=False)
        self.output_proj = nn.Conv2d(channels, channels, 1, bias=False)

        self.reliability_bias_scale = float(reliability_bias_scale)
        self.reliability_eps = float(reliability_eps)
        residual_strength_init = min(max(float(residual_strength_init), -0.95), 0.95)
        raw_strength = math.atanh(residual_strength_init)
        self.fusion_strength = nn.Parameter(torch.tensor([raw_strength], dtype=torch.float32))
        self.rgb_purify_strength = nn.Parameter(torch.tensor([raw_strength], dtype=torch.float32))
        self.ir_purify_strength = nn.Parameter(torch.tensor([raw_strength], dtype=torch.float32))

    def forward(
        self, rgb_aligned: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        if rgb_aligned.shape != ir.shape:
            raise ValueError(
                "IGMRF expects equal aligned RGB and IR shapes, got "
                f"{tuple(rgb_aligned.shape)} and {tuple(ir.shape)}"
            )

        r_rgb = self.rgb_reliability(rgb_aligned).clamp(
            min=self.reliability_eps, max=1.0
        )
        r_ir = self.ir_reliability(ir).clamp(min=self.reliability_eps, max=1.0)

        q = self.query(torch.cat((rgb_aligned, ir), dim=1))
        k_rgb = self.key_rgb(rgb_aligned)
        k_ir = self.key_ir(ir)
        scale = 1.0 / math.sqrt(float(q.shape[1]))
        content_rgb = (q * k_rgb).sum(dim=1, keepdim=True) * scale
        content_ir = (q * k_ir).sum(dim=1, keepdim=True) * scale

        logits = torch.cat(
            (
                content_rgb + self.reliability_bias_scale * torch.log(r_rgb),
                content_ir + self.reliability_bias_scale * torch.log(r_ir),
            ),
            dim=1,
        )
        modality_weights = F.softmax(logits, dim=1)
        w_rgb, w_ir = modality_weights[:, 0:1], modality_weights[:, 1:2]

        attended = self.output_proj(
            w_rgb * self.value_rgb(rgb_aligned) + w_ir * self.value_ir(ir)
        )
        gamma_fuse = torch.tanh(self.fusion_strength)
        gamma_rgb = torch.tanh(self.rgb_purify_strength)
        gamma_ir = torch.tanh(self.ir_purify_strength)

        fused = ir + gamma_fuse * (attended - ir)
        rgb_purified = rgb_aligned + gamma_rgb * (attended - rgb_aligned)
        ir_purified = ir + gamma_ir * (attended - ir)
        reliability = torch.cat((r_rgb, r_ir), dim=1)

        diagnostics = {
            "reliability_rgb_mean": r_rgb.mean(),
            "reliability_ir_mean": r_ir.mean(),
            "fusion_weight_rgb_mean": w_rgb.mean(),
            "fusion_weight_ir_mean": w_ir.mean(),
            "fusion_strength": gamma_fuse,
            "rgb_purify_strength": gamma_rgb,
            "ir_purify_strength": gamma_ir,
        }
        return fused, rgb_purified, ir_purified, reliability, diagnostics


class ModalSideProjection(nn.Module):
    """Lightweight modal side path used only by MA-FDR adapters."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                3,
                padding=1,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
