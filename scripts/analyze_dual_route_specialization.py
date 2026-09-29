from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b2seg_wsss.datasets import (  # noqa: E402
    CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset,
    pil_to_normalized_tensor,
)
from b2seg_wsss.metrics import SegmentationMeter  # noqa: E402
from scripts.evaluate_crf import (  # noqa: E402
    build_model_from_checkpoint, logits_from_outputs, parse_tta, transform_spatial,
)

IGNORE_INDEX = 255
BRANCH_KEYS = {
    "semantic": "semantic_patch_logits",
    "spatial": "spatial_patch_logits",
    "fused": "patch_logits",
}


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
        raise ValueError(f"Expected full checkpoint containing 'model': {path}")
    return checkpoint


def build_model(checkpoint: dict[str, object], path: Path, dataset: str, device: torch.device) -> torch.nn.Module:
    checkpoint_for_build = dict(checkpoint)
    saved_args = dict(checkpoint.get("args", {}))
    saved_args["checkpoint"] = None
    checkpoint_for_build["args"] = saved_args
    return build_model_from_checkpoint(checkpoint_for_build, path, dataset, device).eval()


@torch.no_grad()
def branch_logits(
    model: torch.nn.Module, images: torch.Tensor, modes: list[str], amp: bool
) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    for mode in modes:
        augmented = transform_spatial(images, mode)
        with torch.autocast(device_type=images.device.type, enabled=amp and images.device.type == "cuda"):
            outputs = model(augmented)
            missing = [key for key in BRANCH_KEYS.values() if key not in outputs]
            if missing:
                raise RuntimeError(f"Checkpoint is not dual-route; missing outputs: {missing}")
            for branch, key in BRANCH_KEYS.items():
                logits_aug = logits_from_outputs({"patch_logits": outputs[key]}, augmented.shape[-2:], "coarse")
                logits = transform_spatial(logits_aug, mode, inverse=True)
                sums[branch] = logits if branch not in sums else sums[branch] + logits
    return {branch: logits / len(modes) for branch, logits in sums.items()}


def fusion_alphas(checkpoint: dict[str, object], class_names: tuple[str, ...]) -> dict[str, float]:
    args = dict(checkpoint.get("args", {}))
    output_mode = str(args.get("output_mode", "spatial"))
    state = checkpoint["model"]
    alpha_keys = [
        key for key in state
        if key.removeprefix("module.").endswith("output_fuse_alpha_logits")
    ]
    if alpha_keys:
        alpha = state[alpha_keys[0]].float().sigmoid().cpu().numpy()
    elif output_mode == "fuse":
        alpha = np.full(len(class_names), float(args.get("output_fuse_alpha", 0.5)))
    elif output_mode == "semantic":
        alpha = np.zeros(len(class_names))
    elif output_mode == "spatial":
        alpha = np.ones(len(class_names))
    else:
        alpha = np.full(len(class_names), np.nan)
    return {f"fusion_alpha_{name}": float(alpha[index]) for index, name in enumerate(class_names)}


@torch.no_grad()
def analyze_checkpoint(
    path: Path, data_root: str, dataset_name: str, split: str,
    batch_size: int, num_workers: int, device: torch.device, amp: bool,
    modes: list[str],
) -> dict[str, object]:
    checkpoint = load_checkpoint(path)
    saved_args = dict(checkpoint.get("args", {}))
    mean = tuple(float(value) for value in saved_args.get("image_mean", IMAGENET_MEAN))
    std = tuple(float(value) for value in saved_args.get("image_std", IMAGENET_STD))
    model = build_model(checkpoint, path, dataset_name, device)
    class_names = CLASS_NAMES[dataset_name]
    dataset = SegmentationDataset(
        data_root, dataset_name, split=split,
        transform=lambda image: pil_to_normalized_tensor(image, mean=mean, std=std),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    meters = {branch: SegmentationMeter(num_classes=len(class_names)) for branch in BRANCH_KEYS}
    disagree = 0
    valid_total = 0
    semantic_only_correct = 0
    spatial_only_correct = 0
    either_correct = 0
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        logits = branch_logits(model, images, modes, amp)
        predictions = {branch: value.argmax(dim=1) for branch, value in logits.items()}
        for branch, prediction in predictions.items():
            meters[branch].update(prediction, target)
        valid = target != IGNORE_INDEX
        sem_correct = (predictions["semantic"] == target) & valid
        spa_correct = (predictions["spatial"] == target) & valid
        disagree += int(((predictions["semantic"] != predictions["spatial"]) & valid).sum().item())
        valid_total += int(valid.sum().item())
        semantic_only_correct += int((sem_correct & ~spa_correct).sum().item())
        spatial_only_correct += int((spa_correct & ~sem_correct).sum().item())
        either_correct += int((sem_correct | spa_correct).sum().item())
        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(f"dual_route batch={batch_index}/{len(loader)} checkpoint={path.name}", flush=True)

    result: dict[str, object] = {
        "output_mode": str(saved_args.get("output_mode", "spatial")),
        "disagreement_rate": disagree / max(valid_total, 1),
        "semantic_only_correct_rate": semantic_only_correct / max(valid_total, 1),
        "spatial_only_correct_rate": spatial_only_correct / max(valid_total, 1),
        "either_route_correct_rate": either_correct / max(valid_total, 1),
        **fusion_alphas(checkpoint, class_names),
    }
    for branch, meter in meters.items():
        metrics = meter.compute()
        for metric in ("miou", "mdice", "fwiou"):
            result[f"{branch}_{metric}"] = float(metrics[metric])
        for index, class_name in enumerate(class_names):
            result[f"{branch}_iou_{class_name}"] = float(metrics["iou"][index])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify semantic/spatial dual-route specialization.")
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
    parser.add_argument("--all-seeds-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    modes = parse_tta(args.tta)
    rows: list[dict[str, object]] = []
    for label, template in args.run:
        for seed in args.seeds:
            path = Path(template.format(seed=seed))
            if not path.is_file():
                raise FileNotFoundError(path)
            print(f"analyzing={label} seed={seed} checkpoint={path}", flush=True)
            rows.append({
                "configuration": label, "seed": seed, "checkpoint": str(path),
                "dataset": args.dataset, "split": args.split, "tta": "+".join(modes),
                **analyze_checkpoint(path, args.data_root, args.dataset, args.split,
                                     args.batch_size, args.num_workers, device, args.amp, modes),
            })
    all_path = Path(args.all_seeds_output)
    all_path.parent.mkdir(parents=True, exist_ok=True)
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    nonmetrics = {"configuration", "seed", "checkpoint", "dataset", "split", "tta", "output_mode"}
    metric_names = [key for key, value in rows[0].items() if key not in nonmetrics and isinstance(value, (int, float))]
    summary_rows: list[dict[str, object]] = []
    for label, _ in args.run:
        group = [row for row in rows if row["configuration"] == label]
        summary: dict[str, object] = {
            "configuration": label, "num_seeds": len(group), "dataset": args.dataset,
            "split": args.split, "tta": "+".join(modes), "output_mode": group[0]["output_mode"],
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
        "definitions": {
            "disagreement_rate": "fraction of valid pixels where semantic and spatial argmax differ",
            "semantic_only_correct_rate": "valid pixels corrected only by semantic route",
            "spatial_only_correct_rate": "valid pixels corrected only by spatial route",
            "either_route_correct_rate": "oracle accuracy when either route is correct",
            "fusion_alpha": "spatial coefficient in (1-alpha)*semantic + alpha*spatial",
        },
        "raw_prediction": True, "inference_crf": False, "tta": modes,
    }, indent=2), encoding="utf-8")
    print(f"all_seeds={all_path}", flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
