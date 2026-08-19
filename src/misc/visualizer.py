"""Sample visualization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision


torchvision.disable_beta_transforms_warning()

__all__ = ["show_sample", "save_samples"]


def _draw_target(image, target_boxes, target_labels, normalized, box_fmt, title=None):
    from PIL import ImageDraw, ImageFont
    from torchvision.ops import box_convert

    box_colors = [
        "red", "blue", "green", "orange", "purple", "cyan", "magenta", "yellow",
        "lime", "pink", "teal", "lavender", "brown", "beige", "maroon", "navy",
        "olive", "coral", "turquoise", "gold",
    ]
    font = ImageFont.load_default()
    width, height = image.size
    boxes = target_boxes.clone()

    if normalized:
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

    boxes = box_convert(boxes, in_fmt=box_fmt, out_fmt="xyxy")
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, width)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, height)

    draw = ImageDraw.Draw(image)
    if title:
        draw.rectangle([0, 0, max(45, 8 * len(title)), 15], fill="black")
        draw.text((3, 2), title, fill="white", font=font)

    for box, label in zip(boxes.numpy().astype(np.int32), target_labels.numpy().astype(np.int32)):
        x1, y1, x2, y2 = box
        color = box_colors[int(label) % len(box_colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = str(label)
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        padding = 2
        text_top = max(0, y1 - text_height - 2 * padding)
        draw.rectangle(
            [x1, text_top, x1 + text_width + 2 * padding, text_top + text_height + 2 * padding],
            fill=color,
        )
        draw.text((x1 + padding, text_top + padding), label_text, fill="white", font=font)
    return image


def save_samples(
    samples: torch.Tensor,
    targets: List[Dict],
    output_dir: str,
    split: str,
    normalized: bool,
    box_fmt: str,
):
    """Save annotated samples.

    Three-channel inputs are saved normally. Six-channel paired inputs are split
    into RGB and IR and saved side-by-side, making synchronized augmentation easy
    to inspect visually.
    """
    from PIL import Image
    from torchvision.transforms.functional import to_pil_image

    save_dir = Path(output_dir) / f"{split}_samples"
    save_dir.mkdir(parents=True, exist_ok=True)

    for sample, target in zip(samples, targets):
        tensor = sample.detach().clone().cpu()
        boxes = target["boxes"].detach().clone().cpu()
        labels = target["labels"].detach().clone().cpu()
        image_id = target["image_id"].item()
        image_path = target.get("image_path", "sample")
        stem = Path(image_path).stem

        if tensor.shape[0] == 6:
            rgb_image = _draw_target(
                to_pil_image(tensor[:3]), boxes, labels, normalized, box_fmt, title="RGB"
            )
            ir_image = _draw_target(
                to_pil_image(tensor[3:6]), boxes, labels, normalized, box_fmt, title="IR"
            )
            canvas = Image.new("RGB", (rgb_image.width + ir_image.width, rgb_image.height))
            canvas.paste(rgb_image, (0, 0))
            canvas.paste(ir_image, (rgb_image.width, 0))
            visualization = canvas
        elif tensor.shape[0] == 3:
            visualization = _draw_target(
                to_pil_image(tensor), boxes, labels, normalized, box_fmt
            )
        else:
            raise ValueError(
                f"Visualization supports 3 or 6 channels, got {tensor.shape[0]}"
            )

        visualization.save(save_dir / f"{image_id}_{stem}.webp")


def show_sample(sample):
    """Display one dataset sample."""
    import matplotlib.pyplot as plt
    from torchvision.transforms.v2 import functional as F
    from torchvision.utils import draw_bounding_boxes

    image, target = sample
    if isinstance(image, PIL.Image.Image):
        image = F.to_image(image)
    if image.shape[0] == 6:
        image = image[3:6]  # display the IR/reference modality by default

    image = F.convert_dtype(image, torch.uint8)
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    fig, ax = plt.subplots()
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    fig.show()
    plt.show()
