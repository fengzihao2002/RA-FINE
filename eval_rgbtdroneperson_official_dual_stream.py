#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate D-FINE Dual-stream with the official RGBTDronePerson tiny protocol.

Required:
1) Official original validation annotation JSON, preferably val_thermal.json.
2) Official modified cocoeval.py from NNNNerd/mmdet-rgbtdroneperson.

Outputs:
- official_predictions.json
- official_metrics.json
- the seven paper metrics printed as percentages
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from pycocotools.coco import COCO

from src.core import YAMLConfig


CLASS_NAMES = ("person", "rider", "crowd")


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


def load_official_evaluator(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "Official cocoeval.py not found:\n"
            f"  {path}\n"
            "Download the official file before running this script."
        )

    spec = importlib.util.spec_from_file_location(
        "rgbtdroneperson_official_cocoeval", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official evaluator: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COCOeval, module.Params


def get_dataset_categories(dataset) -> List[dict]:
    if not hasattr(dataset, "coco"):
        raise AttributeError("Validation dataset does not expose a COCO API.")
    return list(dataset.coco.dataset.get("categories", []))


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
    parser.add_argument(
        "--official-ann",
        default=(
            "/hy-tmp/RGBTDronePerson/annotations/"
            "val_thermal.json"
        ),
        help="Original official validation annotation, not the cleaned 3-class JSON.",
    )
    parser.add_argument(
        "--official-cocoeval",
        default=(
            "/hy-tmp/D-FINE-Dual-Stream/"
            "rgbtdroneperson_cocoeval.py"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/hy-tmp/output/dfine_m_rgbtdroneperson_dual_stream/"
            "official_eval"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-dets",
        type=int,
        default=1000,
        help="Official MMDetection wrapper uses the last value of (100,300,1000).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not os.path.isfile(args.official_ann):
        raise FileNotFoundError(
            "Official annotation was not found:\n"
            f"  {args.official_ann}\n"
            "Locate the original val_thermal.json and pass it with --official-ann."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    COCOeval, Params = load_official_evaluator(args.official_cocoeval)

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    cfg = YAMLConfig(args.config, resume=None)
    model = cfg.model
    postprocessor = cfg.postprocessor
    val_loader = cfg.val_dataloader
    dataset = val_loader.dataset

    state = load_checkpoint(args.checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)
        raise RuntimeError("Checkpoint and model do not match.")

    model = model.to(device).eval()
    postprocessor = postprocessor.to(device).eval()

    current_categories = get_dataset_categories(dataset)
    current_id_to_name = {
        int(category["id"]): str(category["name"]).lower()
        for category in current_categories
    }

    coco_gt = COCO(args.official_ann)
    official_categories = coco_gt.loadCats(coco_gt.getCatIds())
    official_name_to_id = {
        str(category["name"]).lower(): int(category["id"])
        for category in official_categories
    }

    missing_names = [
        name for name in CLASS_NAMES if name not in official_name_to_id
    ]
    if missing_names:
        raise KeyError(
            f"Official annotation is missing categories: {missing_names}"
        )

    official_cat_ids = [
        official_name_to_id[name] for name in CLASS_NAMES
    ]

    loader_ids = set(int(x) for x in getattr(dataset, "ids", []))
    official_ids = set(int(x) for x in coco_gt.getImgIds())
    if loader_ids and loader_ids != official_ids:
        raise RuntimeError(
            "Image IDs in the D-FINE validation loader and official annotation "
            "do not match. Do not evaluate until the split is consistent.\n"
            f"D-FINE IDs: {len(loader_ids)}, official IDs: {len(official_ids)}, "
            f"intersection: {len(loader_ids & official_ids)}"
        )

    predictions = []
    total_batches = len(val_loader)

    for batch_index, (samples, targets) in enumerate(val_loader, start=1):
        samples = samples.to(device)
        targets = [
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]

        outputs = model(samples)
        orig_sizes = torch.stack(
            [target["orig_size"] for target in targets], dim=0
        )
        results = postprocessor(outputs, orig_sizes)

        for target, result in zip(targets, results):
            image_id = int(target["image_id"].item())
            boxes = result["boxes"].detach().cpu()
            scores = result["scores"].detach().cpu()
            labels = result["labels"].detach().cpu()

            for box, score, label in zip(boxes, scores, labels):
                raw_label = int(label.item())

                # remap_mscoco_category=False in this project, so labels normally
                # equal the current JSON category IDs (0,1,2).
                if raw_label in current_id_to_name:
                    class_name = current_id_to_name[raw_label]
                elif 0 <= raw_label < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[raw_label]
                else:
                    raise KeyError(
                        f"Cannot map predicted label {raw_label} to a class name."
                    )

                if class_name not in official_name_to_id:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.tolist()]
                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)

                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": official_name_to_id[class_name],
                        "bbox": [x1, y1, width, height],
                        "score": float(score.item()),
                    }
                )

        if batch_index == 1 or batch_index % 10 == 0 or batch_index == total_batches:
            print(
                f"Inference: {batch_index}/{total_batches} batches, "
                f"{len(predictions)} detections"
            )

    prediction_path = output_dir / "official_predictions.json"
    with prediction_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file)

    if not predictions:
        raise RuntimeError("No predictions were generated.")

    coco_dt = coco_gt.loadRes(predictions)

    # This reproduces the official wrapper:
    # Params.EVAL_STRANDARD='tiny';
    # IoU=[0.25,0.50,0.75];
    # maxDets=(100,300,1000);
    # ignore uncertain GT and use IoD for ignored regions.
    Params.EVAL_STRANDARD = "tiny"
    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox",
        ignore_uncertain=True,
        use_iod_for_ignore=True,
    )
    evaluator.params.catIds = official_cat_ids
    evaluator.params.imgIds = sorted(official_ids)
    evaluator.params.maxDets = [100, 300, args.max_dets]
    evaluator.params.iouThrs = np.array([0.25, 0.50, 0.75])

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    if len(stats) < 21:
        raise RuntimeError(
            f"Official evaluator returned only {len(stats)} statistics; "
            "the tiny protocol was not activated correctly."
        )

    metrics = {
        "mAP50_all": float(stats[7]),
        "mAP50_tiny": float(stats[8]),
        "mAP50_tiny1": float(stats[9]),
        "mAP50_tiny2": float(stats[10]),
        "mAP50_tiny3": float(stats[11]),
        "mAP50_small": float(stats[12]),
        "mAP25_all": float(stats[0]),
    }

    metrics_path = output_dir / "official_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "official_annotation": args.official_ann,
                "max_dets": args.max_dets,
                "metrics_decimal": metrics,
                "metrics_percent": {
                    key: value * 100.0 for key, value in metrics.items()
                },
            },
            file,
            indent=2,
        )

    print()
    print("=" * 94)
    print("Official RGBTDronePerson metrics")
    print("=" * 94)
    print(
        f"{'mAP50_all':>12} "
        f"{'mAP50_tiny':>12} "
        f"{'mAP50_tiny1':>13} "
        f"{'mAP50_tiny2':>13} "
        f"{'mAP50_tiny3':>13} "
        f"{'mAP50_small':>13} "
        f"{'mAP25_all':>12}"
    )
    print(
        f"{metrics['mAP50_all'] * 100:12.2f} "
        f"{metrics['mAP50_tiny'] * 100:12.2f} "
        f"{metrics['mAP50_tiny1'] * 100:13.2f} "
        f"{metrics['mAP50_tiny2'] * 100:13.2f} "
        f"{metrics['mAP50_tiny3'] * 100:13.2f} "
        f"{metrics['mAP50_small'] * 100:13.2f} "
        f"{metrics['mAP25_all'] * 100:12.2f}"
    )
    print("=" * 94)
    print(f"Predictions saved to: {prediction_path}")
    print(f"Metrics saved to    : {metrics_path}")


if __name__ == "__main__":
    main()
