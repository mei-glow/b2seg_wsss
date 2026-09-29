from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b2seg_wsss.datasets import (  # noqa: E402
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentationDataset,
    pil_to_normalized_tensor,
)
from scripts.evaluate_crf import (  # noqa: E402
    build_model_from_checkpoint,
    logits_from_outputs,
    parse_tta,
    transform_spatial,
)

IGNORE_INDEX = 255


def parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=/path/seed{seed}/best.pt")
    label, template = (item.strip() for item in value.split("=", 1))
    if not label or not template or "{seed}" not in template:
        raise argparse.ArgumentTypeError("--run requires a label and {seed} placeholder")
    return label, template


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must contain unique integers")
    return seeds


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a full checkpoint containing 'model': {path}")
    return checkpoint


def build_model(checkpoint: dict[str, object], path: Path, dataset: str, device: torch.device) -> torch.nn.Module:
    checkpoint_for_build = dict(checkpoint)
    saved_args = dict(checkpoint.get("args", {}))
    saved_args["checkpoint"] = None
    checkpoint_for_build["args"] = saved_args
    return build_model_from_checkpoint(checkpoint_for_build, path, dataset, device).eval()


@torch.no_grad()
def predict(model: torch.nn.Module, images: torch.Tensor, modes: list[str], amp: bool) -> torch.Tensor:
    logits_sum: torch.Tensor | None = None
    for mode in modes:
        augmented = transform_spatial(images, mode)
        with torch.autocast(device_type=images.device.type, enabled=amp and images.device.type == "cuda"):
            outputs = model(augmented)
            logits_aug = logits_from_outputs(outputs, augmented.shape[-2:], "coarse")
        logits = transform_spatial(logits_aug, mode, inverse=True)
        logits_sum = logits if logits_sum is None else logits_sum + logits
    assert logits_sum is not None
    return (logits_sum / len(modes)).argmax(dim=1)


def binary_boundary(mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """One-pixel inner boundary, excluding pixels adjacent to ignored regions."""
    mask_f = mask[:, None].float()
    eroded = 1.0 - F.max_pool2d(1.0 - mask_f, kernel_size=3, stride=1, padding=1)
    boundary = mask[:, None] & (eroded < 0.5)
    valid_f = valid[:, None].float()
    valid_core = 1.0 - F.max_pool2d(1.0 - valid_f, kernel_size=3, stride=1, padding=1)
    return boundary & (valid_core > 0.5)


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask.float(), kernel_size=kernel, stride=1, padding=radius) > 0.5


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_name: str,
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
    tta_modes: list[str],
    tolerance: int,
) -> dict[str, float]:
    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = dict(checkpoint.get("args", {}))
    mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    model = build_model(checkpoint, checkpoint_path, dataset_name, device)
    dataset = SegmentationDataset(
        data_root,
        dataset_name,
        split=split,
        transform=lambda image: pil_to_normalized_tensor(image, mean=mean, std=std),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    num_classes = len(CLASS_NAMES[dataset_name])
    pred_total = np.zeros(num_classes, dtype=np.float64)
    gt_total = np.zeros(num_classes, dtype=np.float64)
    pred_matched = np.zeros(num_classes, dtype=np.float64)
    gt_matched = np.zeros(num_classes, dtype=np.float64)

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["mask"].to(device, non_blocking=True)
        predictions = predict(model, images, tta_modes, amp)
        valid = targets != IGNORE_INDEX
        for class_index in range(num_classes):
            pred_boundary = binary_boundary((predictions == class_index) & valid, valid)
            gt_boundary = binary_boundary((targets == class_index) & valid, valid)
            pred_total[class_index] += float(pred_boundary.sum().item())
            gt_total[class_index] += float(gt_boundary.sum().item())
            pred_matched[class_index] += float((pred_boundary & dilate(gt_boundary, tolerance)).sum().item())
            gt_matched[class_index] += float((gt_boundary & dilate(pred_boundary, tolerance)).sum().item())
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"boundary batch={batch_index}/{len(loader)} checkpoint={checkpoint_path.name}", flush=True)

    precision = pred_matched / np.maximum(pred_total, 1.0)
    recall = gt_matched / np.maximum(gt_total, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    result: dict[str, float] = {
        "boundary_precision": float(precision.mean()),
        "boundary_recall": float(recall.mean()),
        "boundary_f1": float(f1.mean()),
    }
    for index, name in enumerate(CLASS_NAMES[dataset_name]):
        result[f"boundary_precision_{name}"] = float(precision[index])
        result[f"boundary_recall_{name}"] = float(recall[index])
        result[f"boundary_f1_{name}"] = float(f1[index])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary F1 analysis for raw WSSS predictions.")
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["bcss", "luad", "gcss"], default="bcss")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tta", default="none")
    parser.add_argument("--boundary-tolerance", type=int, default=3)
    parser.add_argument("--all-seeds-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    if args.boundary_tolerance < 0:
        raise ValueError("--boundary-tolerance must be >= 0")
    device = torch.device(args.device)
    tta_modes = parse_tta(args.tta)
    rows: list[dict[str, object]] = []
    for label, template in args.run:
        for seed in args.seeds:
            checkpoint_path = Path(template.format(seed=seed))
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            print(f"analyzing={label} seed={seed} checkpoint={checkpoint_path}", flush=True)
            metrics = evaluate_checkpoint(
                checkpoint_path, args.dataset, args.data_root, args.split,
                args.batch_size, args.num_workers, device, args.amp,
                tta_modes, args.boundary_tolerance,
            )
            rows.append({
                "configuration": label, "seed": seed, "checkpoint": str(checkpoint_path),
                "dataset": args.dataset, "split": args.split,
                "tta": "+".join(tta_modes), "boundary_tolerance_px": args.boundary_tolerance,
                **metrics,
            })
    all_path = Path(args.all_seeds_output)
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [key for key in rows[0] if key.startswith("boundary_") and key != "boundary_tolerance_px"]
    summary_rows: list[dict[str, object]] = []
    for label, _ in args.run:
        group = [row for row in rows if row["configuration"] == label]
        summary: dict[str, object] = {
            "configuration": label, "num_seeds": len(group), "dataset": args.dataset,
            "split": args.split, "tta": "+".join(tta_modes),
            "boundary_tolerance_px": args.boundary_tolerance,
        }
        for metric in metric_names:
            mean, std = mean_std([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_std"] = f"{100 * mean:.2f} +/- {100 * std:.2f}"
        summary_rows.append(summary)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary_path.with_suffix(".json").write_text(json.dumps({
        "definition": "Dataset-level class-wise boundary precision/recall/F1 with symmetric pixel-tolerance matching; macro-average over classes.",
        "raw_prediction": True,
        "inference_crf": False,
        "tta": tta_modes,
        "boundary_tolerance_px": args.boundary_tolerance,
    }, indent=2), encoding="utf-8")
    print(f"all_seeds={all_path}", flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
