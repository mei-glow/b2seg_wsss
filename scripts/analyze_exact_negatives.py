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
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentationDataset,
    pil_to_normalized_tensor,
)
from scripts.evaluate_crf import build_model_from_checkpoint  # noqa: E402


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a full training checkpoint with a 'model' key: {path}")
    return checkpoint


def build_final_model(
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    dataset: str,
    device: torch.device,
) -> torch.nn.Module:
    # Final checkpoints contain all learned weights. Avoid reopening a stale
    # machine-specific DeiT pretraining path stored in checkpoint args.
    checkpoint_for_build = dict(checkpoint)
    saved_args = dict(checkpoint.get("args", {}))
    saved_args["checkpoint"] = None
    checkpoint_for_build["args"] = saved_args
    return build_model_from_checkpoint(
        checkpoint_for_build, checkpoint_path, dataset, device
    ).eval()


def parse_run(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--run must use LABEL=/path/with/{seed}/best.pt"
        )
    label, template = value.split("=", 1)
    label, template = label.strip(), template.strip()
    if not label or not template or "{seed}" not in template:
        raise argparse.ArgumentTypeError(
            "--run must contain a label and a {seed} placeholder"
        )
    return label, template


def parse_explicit_checkpoint(value: str) -> tuple[str, int, str]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--checkpoint must use LABEL=SEED=/absolute/path/best.pt"
        )
    label, seed_text, path = (part.strip() for part in parts)
    if not label or not path:
        raise argparse.ArgumentTypeError("--checkpoint label/path cannot be empty")
    try:
        seed = int(seed_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--checkpoint SEED must be an integer") from exc
    return label, seed, path


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must contain unique integers")
    return seeds


@torch.no_grad()
def analyze_checkpoint(
    checkpoint_path: Path,
    data_root: str,
    dataset_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    amp: bool,
    threshold: float,
    topk_frac: float,
) -> dict[str, object]:
    checkpoint = load_checkpoint(checkpoint_path)
    saved_args = checkpoint.get("args", {})
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    image_std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    model = build_final_model(checkpoint, checkpoint_path, dataset_name, device)
    dataset = SegmentationDataset(
        data_root,
        dataset_name,
        split=split,
        transform=lambda image: pil_to_normalized_tensor(
            image, mean=image_mean, std=image_std
        ),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    num_classes = len(CLASS_NAMES[dataset_name])
    fp_tokens = np.zeros(num_classes, dtype=np.int64)
    absent_tokens = np.zeros(num_classes, dtype=np.int64)
    conf_sum = np.zeros(num_classes, dtype=np.float64)
    absent_pairs = np.zeros(num_classes, dtype=np.int64)
    resolved_k: int | None = None

    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        # Validation/test filenames do not necessarily encode weak labels.
        # Dense masks provide an exact and protocol-independent definition of
        # whether each class is present in an image; ignore pixels never match
        # any valid class index.
        labels = torch.stack(
            [
                (masks == class_idx).flatten(1).any(dim=1)
                for class_idx in range(num_classes)
            ],
            dim=1,
        ).to(images.dtype)
        with torch.autocast(
            device_type=device.type, enabled=amp and device.type == "cuda"
        ):
            outputs = model(images)
        probabilities = outputs["patch_logits"].float().sigmoid()
        num_tokens = int(probabilities.shape[1])
        k = max(1, min(num_tokens, int(round(num_tokens * topk_frac))))
        if resolved_k is None:
            resolved_k = k
        elif resolved_k != k:
            raise RuntimeError(f"Token count changed across batches: K={resolved_k} vs {k}")

        absent = labels == 0
        for class_idx in range(num_classes):
            class_absent = absent[:, class_idx]
            if not bool(class_absent.any()):
                continue
            values = probabilities[class_absent, :, class_idx]
            fp_tokens[class_idx] += int((values > threshold).sum().item())
            absent_tokens[class_idx] += int(values.numel())
            conf_sum[class_idx] += float(values.topk(k, dim=1).values.mean(dim=1).sum().item())
            absent_pairs[class_idx] += int(values.shape[0])

        if batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader):
            print(
                f"exact_negative batch={batch_index}/{len(loader)} "
                f"checkpoint={checkpoint_path.name}",
                flush=True,
            )

    if absent_tokens.sum() == 0 or absent_pairs.sum() == 0 or resolved_k is None:
        raise RuntimeError("No absent image-class pairs were found")
    per_class_afpr = fp_tokens / np.maximum(absent_tokens, 1)
    per_class_confk = conf_sum / np.maximum(absent_pairs, 1)
    result: dict[str, object] = {
        "afpr": float(fp_tokens.sum() / absent_tokens.sum()),
        "confk": float(conf_sum.sum() / absent_pairs.sum()),
        "threshold": threshold,
        "topk_frac": topk_frac,
        "topk_tokens": resolved_k,
        "num_images": len(dataset),
        "num_absent_pairs": int(absent_pairs.sum()),
    }
    for class_idx, class_name in enumerate(CLASS_NAMES[dataset_name]):
        result[f"afpr_{class_name}"] = float(per_class_afpr[class_idx])
        result[f"confk_{class_name}"] = float(per_class_confk[class_idx])
        result[f"absent_pairs_{class_name}"] = int(absent_pairs[class_idx])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure absent-class AFPR and top-K confidence from WSSS checkpoints."
    )
    parser.add_argument("--run", action="append", type=parse_run, default=[])
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_explicit_checkpoint,
        default=[],
        help="Explicit checkpoint as LABEL=SEED=/absolute/path/best.pt",
    )
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad", "gcss"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--topk-frac", type=float, default=0.05)
    parser.add_argument("--all-seeds-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()
    if not args.run and not args.checkpoint:
        parser.error("provide at least one --run or --checkpoint")
    if not 0 < args.topk_frac <= 1:
        raise ValueError("--topk-frac must be in (0, 1]")
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be in [0, 1]")
    device = torch.device(args.device)
    rows: list[dict[str, object]] = []

    resolved_checkpoints: list[tuple[str, int, Path]] = []
    for label, template in args.run:
        for seed in args.seeds:
            resolved_checkpoints.append((label, seed, Path(template.format(seed=seed))))
    for label, seed, path in args.checkpoint:
        resolved_checkpoints.append((label, seed, Path(path)))

    for label, seed, checkpoint_path in resolved_checkpoints:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        print(f"analyzing={label} seed={seed} checkpoint={checkpoint_path}", flush=True)
        metrics = analyze_checkpoint(
            checkpoint_path,
            args.data_root,
            args.dataset,
            args.split,
            args.batch_size,
            args.num_workers,
            device,
            args.amp,
            args.threshold,
            args.topk_frac,
        )
        rows.append(
            {
                "configuration": label,
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "dataset": args.dataset,
                "split": args.split,
                **metrics,
            }
        )

    all_output = Path(args.all_seeds_output)
    all_output.parent.mkdir(parents=True, exist_ok=True)
    with all_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        key
        for key in rows[0]
        if key in {"afpr", "confk"} or key.startswith("afpr_") or key.startswith("confk_")
    ]
    summary_rows: list[dict[str, object]] = []
    labels = list(dict.fromkeys(label for label, _, _ in resolved_checkpoints))
    for label in labels:
        group = [row for row in rows if row["configuration"] == label]
        summary: dict[str, object] = {
            "configuration": label,
            "num_seeds": len(group),
            "dataset": args.dataset,
            "split": args.split,
            "threshold": args.threshold,
            "topk_frac": args.topk_frac,
            "topk_tokens": group[0]["topk_tokens"],
        }
        for metric in metric_names:
            mean, std = mean_std([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_mean_std"] = f"{mean:.6f} +/- {std:.6f}"
        summary_rows.append(summary)

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    metadata = {
        "definition": {
            "AFPR": "fraction of absent-class tokens with sigmoid probability > threshold",
            "ConfK": "mean top-K sigmoid probability per absent image-class pair",
            "class_absence": "class has zero ground-truth pixels in the evaluated dense mask",
        },
        "threshold": args.threshold,
        "topk_frac": args.topk_frac,
        "runs": args.run,
        "explicit_checkpoints": args.checkpoint,
        "seeds": args.seeds,
    }
    summary_output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"all_seeds={all_output}", flush=True)
    print(f"summary={summary_output}", flush=True)


if __name__ == "__main__":
    main()
