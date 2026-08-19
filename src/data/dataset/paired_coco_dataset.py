"""
Paired RGB-T COCO dataset for D-FINE.

The annotation coordinates are defined in the infrared/thermal reference frame.
RGB and IR images are transformed together so every geometric augmentation uses
exactly the same random parameters for both modalities and the target boxes.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch
from PIL import Image
from torchvision.transforms.v2 import functional as F

from .._misc import Image as TVImage

from ...core import register
from .coco_dataset import CocoDetection

__all__ = ["PairedCocoDetection"]


@register()
class PairedCocoDetection(CocoDetection):
    """COCO detection dataset that returns a six-channel ``[RGB, IR]`` tensor.

    Args:
        rgb_img_folder: Directory containing visible/RGB images.
        ir_img_folder: Directory containing infrared/thermal images.
        ann_file: COCO JSON. Boxes must be in the IR reference coordinate system.
        transforms: D-FINE/torchvision-v2 transform pipeline.
        return_masks: Keep the original D-FINE mask option.
        remap_mscoco_category: Keep the original D-FINE category-remapping option.
        strict_pairing: Raise immediately when a paired file or image size differs.
    """

    __inject__ = ["transforms"]
    __share__ = ["remap_mscoco_category"]

    def __init__(
        self,
        rgb_img_folder: str,
        ir_img_folder: str,
        ann_file: str,
        transforms,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
        strict_pairing: bool = True,
    ) -> None:
        # The parent dataset loads annotations and the reference image from IR.
        super().__init__(
            img_folder=ir_img_folder,
            ann_file=ann_file,
            transforms=transforms,
            return_masks=return_masks,
            remap_mscoco_category=remap_mscoco_category,
        )
        self.rgb_img_folder = rgb_img_folder
        self.ir_img_folder = ir_img_folder
        self.strict_pairing = strict_pairing

    def load_item(self, idx) -> Tuple[Image.Image, Image.Image, dict]:
        ir_image, target = super().load_item(idx)

        image_id = self.ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        file_name = image_info["file_name"]
        rgb_path = os.path.join(self.rgb_img_folder, file_name)
        ir_path = os.path.join(self.ir_img_folder, file_name)

        if not os.path.isfile(rgb_path):
            raise FileNotFoundError(
                f"Missing paired RGB image for image_id={image_id}: {rgb_path}"
            )

        # Force both modalities to three channels. This keeps a fixed six-channel
        # input layout: channels [0:3] are RGB and channels [3:6] are IR.
        rgb_image = Image.open(rgb_path).convert("RGB")
        if ir_image.mode != "RGB":
            ir_image = ir_image.convert("RGB")

        if self.strict_pairing and rgb_image.size != ir_image.size:
            raise ValueError(
                "Paired image size mismatch: "
                f"image_id={image_id}, RGB={rgb_image.size}, IR={ir_image.size}, "
                f"file_name={file_name}"
            )

        target["rgb_image_path"] = rgb_path
        target["ir_image_path"] = ir_path
        # Preserve the parent's image_path as the reference/IR path.
        target["image_path"] = ir_path

        return rgb_image, ir_image, target

    def __getitem__(self, idx):
        rgb_image, ir_image, target = self.load_item(idx)

        # Convert and concatenate before augmentation so torchvision-v2 sees one
        # six-channel image plus one target dictionary. This preserves the original
        # D-FINE transform contract (image, target, dataset), while every geometric
        # operation is intrinsically synchronized across RGB and IR channels.
        rgb_tensor = F.pil_to_tensor(rgb_image).float() / 255.0
        ir_tensor = F.pil_to_tensor(ir_image).float() / 255.0
        paired_image = TVImage(torch.cat((rgb_tensor, ir_tensor), dim=0))

        if self._transforms is not None:
            paired_image, target, _ = self._transforms(paired_image, target, self)

        if not isinstance(paired_image, torch.Tensor):
            raise TypeError("Paired transforms must return a tensor-like image.")
        if paired_image.shape[0] != 6:
            raise RuntimeError(
                f"Expected six channels [RGB(3), IR(3)], got {paired_image.shape[0]}"
            )
        return paired_image, target

    def extra_repr(self) -> str:
        s = f" rgb_img_folder: {self.rgb_img_folder}\n"
        s += f" ir_img_folder: {self.ir_img_folder}\n ann_file: {self.ann_file}\n"
        s += f" return_masks: {self.return_masks}\n strict_pairing: {self.strict_pairing}\n"
        if hasattr(self, "_transforms") and self._transforms is not None:
            s += f" transforms:\n   {repr(self._transforms)}"
        return s
