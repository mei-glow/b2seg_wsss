from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ImageLevelDataset,
    pil_to_normalized_tensor,
)
from scripts.evaluate_crf import build_model_from_checkpoint, logits_from_outputs


def parse_thresholds(value: str, num_classes: int) -> torch.Tensor:
    items = [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if len(items) == 1:
        items = items * num_classes
    if len(items) != num_classes:
        raise ValueError(f"Expected 1 or {num_classes} thresholds, got {items}")
    return torch.tensor(items, dtype=torch.float32)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate partial pseudo masks from a WSSS checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-head", default="coarse", choices=["auto", "coarse", "refined"])
    parser.add_argument("--thresholds", default="0.70", help="Global or per-class confidence thresholds, e.g. 0.75,0.75,0.55,0.65.")
    parser.add_argument("--score", default="softmax", choices=["softmax", "sigmoid"], help="Confidence score used to keep pseudo labels.")
    parser.add_argument("--restrict-present", action="store_true", help="Only allow classes present in the image-level label.")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.restrict_present and args.score == "softmax":
        raise ValueError(
            "--score softmax with --restrict-present is not valid for partial pseudo masks: "
            "single-present-class images can get softmax confidence near 1 everywhere. "
            "Use --score sigmoid with --restrict-present instead."
        )

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    data_root = args.data_root or saved_args.get("data_root", "data")
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    image_std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    class_names = CLASS_NAMES[dataset]
    num_classes = len(class_names)
    thresholds = parse_thresholds(args.thresholds, num_classes)
    device = torch.device(args.device)

    model = build_model_from_checkpoint(checkpoint, ckpt_path, dataset, device)
    model.eval()

    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = ImageLevelDataset(data_root, dataset, transform=transform)
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_device = thresholds.to(device).view(1, num_classes, 1, 1)

    kept_pixels = np.zeros(num_classes, dtype=np.float64)
    total_kept = 0.0
    total_pixels = 0.0
    image_rows: list[dict[str, object]] = []

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        names = batch["name"]
        heights = batch["height"]
        widths = batch["width"]
        height = int(heights[0]) if not isinstance(heights, int) else heights
        width = int(widths[0]) if not isinstance(widths, int) else widths

        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            logits = logits_from_outputs(outputs, (height, width), args.prediction_head)

        if args.restrict_present:
            absent = labels <= 0
            logits = logits.masked_fill(absent.view(labels.shape[0], num_classes, 1, 1), -1e4)
        pred = logits.argmax(dim=1)
        if args.score == "softmax":
            scores = logits.softmax(dim=1)
        else:
            scores = logits.sigmoid()
        conf = scores.gather(dim=1, index=pred.unsqueeze(1)).squeeze(1)
        class_threshold = thresholds_device.expand(logits.shape[0], -1, height, width).gather(
            dim=1,
            index=pred.unsqueeze(1),
        ).squeeze(1)
        keep = conf >= class_threshold
        pseudo = pred.masked_fill(~keep, args.ignore_index).detach().cpu().numpy().astype(np.uint8)

        for item_idx, name in enumerate(names):
            mask = pseudo[item_idx]
            Image.fromarray(mask).save(output_dir / str(name))
            valid = mask != args.ignore_index
            total_pixels += float(mask.size)
            total_kept += float(valid.sum())
            row = {"name": str(name), "kept_fraction": float(valid.mean())}
            for class_idx, class_name in enumerate(class_names):
                count = float((mask == class_idx).sum())
                kept_pixels[class_idx] += count
                row[f"{class_name}_pixels"] = int(count)
            image_rows.append(row)

        if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(loader):
            print(f"pseudo batch={batch_idx}/{len(loader)} kept={total_kept / max(total_pixels, 1.0):.4f}", flush=True)

    report = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "split": args.split,
        "output_dir": str(output_dir),
        "score": args.score,
        "restrict_present": args.restrict_present,
        "thresholds": thresholds.tolist(),
        "class_names": class_names,
        "kept_fraction": float(total_kept / max(total_pixels, 1.0)),
        "kept_pixels": kept_pixels.tolist(),
        "kept_class_fraction_of_all_pixels": (kept_pixels / max(total_pixels, 1.0)).tolist(),
        "kept_class_fraction_of_kept_pixels": (kept_pixels / max(total_kept, 1.0)).tolist(),
        "num_images": len(dataset_obj),
    }
    (output_dir / "pseudo_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "pseudo_image_stats.csv").open("w", encoding="utf-8") as handle:
        if image_rows:
            keys = list(image_rows[0].keys())
            handle.write(",".join(keys) + "\n")
            for row in image_rows:
                handle.write(",".join(str(row[key]) for key in keys) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
