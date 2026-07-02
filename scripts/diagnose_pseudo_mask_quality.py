from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.metrics import SegmentationMeter
from scripts.evaluate_crf import build_model_from_checkpoint, logits_from_outputs


def parse_layers(value: str, num_layers: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, num_layers + 1))
    layers = [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    bad = [layer for layer in layers if layer < 1 or layer > num_layers]
    if bad:
        raise ValueError(f"Layers out of range 1..{num_layers}: {bad}")
    return sorted(set(layers))


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def parse_threshold_specs(value: str, num_classes: int) -> list[tuple[str, torch.Tensor]]:
    specs = []
    for spec_idx, raw_spec in enumerate(value.replace("|", ";").split(";")):
        raw_spec = raw_spec.strip()
        if not raw_spec:
            continue
        items = parse_float_list(raw_spec)
        if len(items) == 1:
            items = items * num_classes
        if len(items) != num_classes:
            raise ValueError(f"Expected 1 or {num_classes} thresholds in '{raw_spec}', got {items}")
        name = "t" + "_".join(f"{item:.3f}".replace(".", "") for item in items)
        if len(name) > 80:
            name = f"threshold_{spec_idx}"
        specs.append((name, torch.tensor(items, dtype=torch.float32)))
    return specs


def logits_from_patch_logits(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def get_timm_vit(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise ValueError("Model has no backbone attribute.")
    vit = getattr(backbone, "vit", backbone)
    if not hasattr(vit, "blocks") or not hasattr(vit, "patch_embed"):
        raise ValueError("Layer probing currently expects a timm-style ViT backbone.")
    return vit


@torch.no_grad()
def forward_layer_tokens(vit: torch.nn.Module, images: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    wanted = {layer - 1 for layer in layers}
    x = vit.patch_embed(images)
    if hasattr(vit, "_pos_embed"):
        x = vit._pos_embed(x)
    else:
        raise ValueError("Expected timm ViT with _pos_embed.")
    x = vit.patch_drop(x)
    x = vit.norm_pre(x)
    out: dict[int, torch.Tensor] = {}
    for block_idx, block in enumerate(vit.blocks):
        x = block(x)
        if block_idx in wanted:
            out[block_idx + 1] = vit.norm(x)[:, 1:, :].detach()
    return out


def logits_from_tokens(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "head"):
        sims = model.head.prototype_similarity(tokens)
        return model.head.aggregate_prototypes(sims)
    if hasattr(model, "classifier"):
        return model.classifier(tokens)
    raise ValueError("Layer-level diagnostic requires either a prototype head or linear classifier.")


def update_full_metrics(
    meters: dict[str, SegmentationMeter],
    source: str,
    logits: torch.Tensor,
    masks: torch.Tensor,
) -> None:
    pred = logits.argmax(dim=1)
    meters[source].update(pred, masks)


def threshold_stats(
    logits: torch.Tensor,
    masks: torch.Tensor,
    thresholds: torch.Tensor,
    score_mode: str,
    num_classes: int,
    ignore_index: int,
) -> dict[str, object]:
    pred = logits.argmax(dim=1)
    scores = logits.softmax(dim=1) if score_mode == "softmax" else logits.sigmoid()
    conf = scores.gather(dim=1, index=pred.unsqueeze(1)).squeeze(1)
    thresholds = thresholds.to(logits.device).view(1, num_classes, 1, 1)
    class_threshold = thresholds.expand(logits.shape[0], -1, logits.shape[-2], logits.shape[-1]).gather(
        dim=1,
        index=pred.unsqueeze(1),
    ).squeeze(1)
    keep = conf >= class_threshold
    valid_gt = (masks != ignore_index) & (masks >= 0) & (masks < num_classes)
    keep_valid = keep & valid_gt
    correct = (pred == masks) & keep_valid

    total_valid = valid_gt.sum().item()
    total_kept = keep_valid.sum().item()
    total_correct = correct.sum().item()
    pred_pixels = []
    gt_pixels = []
    tp = []
    for class_idx in range(num_classes):
        pred_c = keep_valid & (pred == class_idx)
        gt_c = valid_gt & (masks == class_idx)
        tp_c = pred_c & gt_c
        pred_pixels.append(float(pred_c.sum().item()))
        gt_pixels.append(float(gt_c.sum().item()))
        tp.append(float(tp_c.sum().item()))

    pred_np = np.asarray(pred_pixels, dtype=np.float64)
    gt_np = np.asarray(gt_pixels, dtype=np.float64)
    tp_np = np.asarray(tp, dtype=np.float64)
    precision = np.divide(tp_np, pred_np, out=np.zeros_like(tp_np), where=pred_np > 0)
    coverage = np.divide(tp_np, gt_np, out=np.zeros_like(tp_np), where=gt_np > 0)
    partial_iou = np.divide(tp_np, pred_np + gt_np - tp_np, out=np.zeros_like(tp_np), where=(pred_np + gt_np - tp_np) > 0)
    partial_dice = np.divide(2 * tp_np, pred_np + gt_np, out=np.zeros_like(tp_np), where=(pred_np + gt_np) > 0)

    return {
        "kept_pixels": float(total_kept),
        "valid_pixels": float(total_valid),
        "correct_pixels": float(total_correct),
        "pred_pixels": pred_pixels,
        "gt_pixels": gt_pixels,
        "tp": tp,
        "kept_fraction": float(total_kept / max(total_valid, 1)),
        "kept_accuracy": float(total_correct / max(total_kept, 1)),
        "precision": precision.tolist(),
        "coverage": coverage.tolist(),
        "partial_iou": partial_iou.tolist(),
        "partial_dice": partial_dice.tolist(),
    }


def add_stats(acc: dict[str, dict[str, object]], key: str, stats: dict[str, object], num_classes: int) -> None:
    if key not in acc:
        acc[key] = {
            "kept_pixels": 0.0,
            "valid_pixels": 0.0,
            "correct_pixels": 0.0,
            "pred_pixels": np.zeros(num_classes, dtype=np.float64),
            "gt_pixels": np.zeros(num_classes, dtype=np.float64),
            "tp": np.zeros(num_classes, dtype=np.float64),
        }
    item = acc[key]
    item["kept_pixels"] = float(item["kept_pixels"]) + float(stats["kept_pixels"])
    item["valid_pixels"] = float(item["valid_pixels"]) + float(stats["valid_pixels"])
    item["correct_pixels"] = float(item["correct_pixels"]) + float(stats["correct_pixels"])
    item["pred_pixels"] = item["pred_pixels"] + np.asarray(stats["pred_pixels"], dtype=np.float64)
    item["gt_pixels"] = item["gt_pixels"] + np.asarray(stats["gt_pixels"], dtype=np.float64)
    item["tp"] = item["tp"] + np.asarray(stats["tp"], dtype=np.float64)


def finalize_threshold_rows(
    acc: dict[str, dict[str, object]],
    class_names: tuple[str, ...],
) -> list[dict[str, object]]:
    rows = []
    num_classes = len(class_names)
    for key, item in sorted(acc.items()):
        source, score_mode, threshold_name = key.split("::", 2)
        pred_pixels = np.asarray(item["pred_pixels"], dtype=np.float64)
        gt_pixels = np.asarray(item["gt_pixels"], dtype=np.float64)
        tp = np.asarray(item["tp"], dtype=np.float64)
        precision = np.divide(tp, pred_pixels, out=np.zeros(num_classes), where=pred_pixels > 0)
        coverage = np.divide(tp, gt_pixels, out=np.zeros(num_classes), where=gt_pixels > 0)
        partial_iou = np.divide(tp, pred_pixels + gt_pixels - tp, out=np.zeros(num_classes), where=(pred_pixels + gt_pixels - tp) > 0)
        partial_dice = np.divide(2 * tp, pred_pixels + gt_pixels, out=np.zeros(num_classes), where=(pred_pixels + gt_pixels) > 0)
        row: dict[str, object] = {
            "source": source,
            "score": score_mode,
            "threshold": threshold_name,
            "kept_fraction": float(item["kept_pixels"] / max(float(item["valid_pixels"]), 1.0)),
            "kept_accuracy": float(item["correct_pixels"] / max(float(item["kept_pixels"]), 1.0)),
            "mean_precision": float(precision.mean()),
            "mean_coverage": float(coverage.mean()),
            "mean_partial_iou": float(partial_iou.mean()),
            "mean_partial_dice": float(partial_dice.mean()),
        }
        for idx, name in enumerate(class_names):
            row[f"{name}_precision"] = float(precision[idx])
            row[f"{name}_coverage"] = float(coverage[idx])
            row[f"{name}_partial_iou"] = float(partial_iou[idx])
            row[f"{name}_partial_dice"] = float(partial_dice[idx])
            row[f"{name}_pred_fraction"] = float(pred_pixels[idx] / max(float(item["valid_pixels"]), 1.0))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose which layer/fusion output makes better pseudo masks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="4,8,12", help="Layers to probe through the trained prototype head, or 'all'.")
    parser.add_argument("--include-model-output", action="store_true", help="Also score the checkpoint's normal forward output.")
    parser.add_argument("--score-modes", default="sigmoid,softmax", help="Comma-separated confidence scores for threshold pseudo masks.")
    parser.add_argument("--thresholds", default="0.70;0.80;0.90;0.95;0.97", help="Semicolon-separated global/per-class thresholds.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-head", default="coarse", choices=["auto", "coarse", "refined"])
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--max-batches", type=int, default=0, help="Optional quick smoke-test limit.")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    data_root = args.data_root or saved_args.get("data_root", "data")
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    image_std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    class_names = CLASS_NAMES[dataset]
    num_classes = len(class_names)
    device = torch.device(args.device)

    model = build_model_from_checkpoint(checkpoint, ckpt_path, dataset, device)
    model.eval()
    vit = get_timm_vit(model)
    layers = parse_layers(args.layers, len(vit.blocks))
    score_modes = [item.strip() for item in args.score_modes.replace(";", ",").split(",") if item.strip()]
    bad_scores = sorted(set(score_modes) - {"sigmoid", "softmax"})
    if bad_scores:
        raise ValueError(f"Unknown score modes: {bad_scores}")
    threshold_specs = parse_threshold_specs(args.thresholds, num_classes)

    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = SegmentationDataset(data_root, dataset, split=args.split, transform=transform)
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_names = [f"layer{layer}" for layer in layers]
    if args.include_model_output:
        source_names.append("model_output")
    full_meters = {source: SegmentationMeter(num_classes=num_classes, ignore_index=args.ignore_index) for source in source_names}
    threshold_acc: dict[str, dict[str, object]] = {}

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        size = masks.shape[-2:]
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            layer_tokens = forward_layer_tokens(vit, images, layers)
            logits_by_source = {}
            for layer, tokens in layer_tokens.items():
                patch_logits = logits_from_tokens(model, tokens)
                logits_by_source[f"layer{layer}"] = logits_from_patch_logits(patch_logits, size)
            if args.include_model_output:
                outputs = model(images)
                logits_by_source["model_output"] = logits_from_outputs(outputs, size, args.prediction_head)

        for source, logits in logits_by_source.items():
            update_full_metrics(full_meters, source, logits, masks)
            for score_mode in score_modes:
                for threshold_name, thresholds in threshold_specs:
                    stats = threshold_stats(logits, masks, thresholds, score_mode, num_classes, args.ignore_index)
                    add_stats(threshold_acc, f"{source}::{score_mode}::{threshold_name}", stats, num_classes)

        if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(loader):
            print(f"pseudo-diagnostic batch={batch_idx}/{len(loader)}", flush=True)
        if args.max_batches and batch_idx >= args.max_batches:
            break

    full_rows = []
    full_report = {}
    for source in source_names:
        metrics = full_meters[source].compute()
        full_report[source] = {key: value for key, value in metrics.items() if key != "confusion"}
        row: dict[str, object] = {
            "source": source,
            "miou": metrics["miou"],
            "mdice": metrics["mdice"],
            "mrecall": metrics["mrecall"],
            "mprecision": metrics["mprecision"],
            "fwiou": metrics["fwiou"],
        }
        for idx, name in enumerate(class_names):
            row[f"{name}_iou"] = metrics["iou"][idx]
            row[f"{name}_dice"] = metrics["dice"][idx]
            row[f"{name}_recall"] = metrics["recall"][idx]
            row[f"{name}_precision"] = metrics["precision"][idx]
        full_rows.append(row)

    threshold_rows = finalize_threshold_rows(threshold_acc, class_names)
    write_csv(output_dir / "full_argmax_metrics.csv", full_rows)
    write_csv(output_dir / "threshold_pseudo_metrics.csv", threshold_rows)
    report = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "split": args.split,
        "layers": layers,
        "class_names": class_names,
        "score_modes": score_modes,
        "thresholds": {name: values.tolist() for name, values in threshold_specs},
        "full_argmax": full_report,
        "threshold_rows": threshold_rows,
    }
    (output_dir / "pseudo_mask_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Full argmax summary:")
    for row in full_rows:
        print(
            f"{row['source']}: mIoU={row['miou']:.4f} mDice={row['mdice']:.4f} "
            f"lymIoU={row.get('lymphocyte_iou', float('nan')):.4f}",
            flush=True,
        )
    best_threshold = max(threshold_rows, key=lambda row: row["mean_partial_iou"]) if threshold_rows else None
    if best_threshold:
        print(
            "Best threshold pseudo by mean_partial_iou: "
            f"{best_threshold['source']} {best_threshold['score']} {best_threshold['threshold']} "
            f"kept={best_threshold['kept_fraction']:.4f} "
            f"acc={best_threshold['kept_accuracy']:.4f} "
            f"mPIoU={best_threshold['mean_partial_iou']:.4f}",
            flush=True,
        )
    print(f"wrote={output_dir}", flush=True)


if __name__ == "__main__":
    main()
