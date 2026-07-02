from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import CLASS_NAMES, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.models import LinearWSSSModel, PrototypeWSSSModel
from scripts.evaluate import parse_prototype_counts


def logits_to_map(patch_logits: torch.Tensor) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)


def build_model(checkpoint: dict[str, object], dataset: str, device: torch.device) -> torch.nn.Module:
    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get("model", "deit_base_patch16_224")
    pretrain_checkpoint = saved_args.get("checkpoint")
    num_classes = len(CLASS_NAMES[dataset])
    patch_stride = saved_args.get("patch_stride")
    patch_stride = None if patch_stride is None else int(patch_stride)
    patch_padding = int(saved_args.get("patch_padding", 0))
    fusion_layers = saved_args.get("fusion_layers")
    fusion_mode = saved_args.get("fusion_mode", "weighted_sum")
    fusion_init = saved_args.get("fusion_init", "average")

    if saved_args.get("model_type") == "linear_wsss":
        model = LinearWSSSModel(
            model_name=model_name,
            checkpoint_path=pretrain_checkpoint,
            num_classes=num_classes,
            grad_checkpointing=False,
            fusion_layers=fusion_layers,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            pooling=saved_args.get("pooling", "max"),
            topk_frac=float(saved_args.get("topk_frac", 0.05)),
        )
    else:
        prototype_counts = parse_prototype_counts(
            saved_args.get("resolved_prototype_counts", saved_args.get("prototype_counts")),
            num_classes,
        )
        model = PrototypeWSSSModel(
            model_name=model_name,
            checkpoint_path=pretrain_checkpoint,
            num_classes=num_classes,
            prototypes_per_class=int(saved_args.get("prototypes_per_class", 4)),
            prototype_counts=prototype_counts,
            prototype_gating=bool(saved_args.get("prototype_gating", False)),
            gate_init=float(saved_args.get("gate_init", 2.0)),
            prototype_dropout=float(saved_args.get("prototype_dropout", 0.0)),
            prototype_aggregation=saved_args.get("prototype_aggregation", "max"),
            lse_tau=float(saved_args.get("prototype_lse_tau", saved_args.get("lse_tau", 1.0))),
            refine_head=bool(saved_args.get("refine_head", False)),
            refine_dim=int(saved_args.get("refine_dim", 256)),
            refine_scale=int(saved_args.get("refine_scale", 2)),
            refine_pooling=saved_args.get("refine_pooling", "topk"),
            refine_topk_frac=float(saved_args.get("refine_topk_frac", 0.05)),
            grad_checkpointing=False,
            fusion_layers=fusion_layers,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
            dense_prototype_scale=int(saved_args.get("dense_prototype_scale", 1)),
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            prototype_pooling=saved_args.get("prototype_pooling", "max"),
            prototype_topk_frac=float(saved_args.get("prototype_topk_frac", 0.05)),
            prototype_mix_alpha=float(saved_args.get("prototype_mix_alpha", 0.5)),
            prototype_multiscale=bool(saved_args.get("prototype_multiscale", False)),
            prototype_scale_branches=saved_args.get("prototype_scale_branches", "identity,local,coarse"),
            prototype_scale_init=saved_args.get("prototype_scale_init", "identity"),
            prototype_scale_residual_init=float(saved_args.get("prototype_scale_residual_init", 0.05)),
            prototype_scale_mode=saved_args.get("prototype_scale_mode", "mixture"),
            prototype_scale_alpha_init=float(saved_args.get("prototype_scale_alpha_init", 0.02)),
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def downsample_binary_mask(mask: torch.Tensor, class_idx: int, grid: int) -> torch.Tensor:
    binary = (mask == class_idx).float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_max_pool2d(binary, (grid, grid))[0, 0]
    return pooled > 0.0


def connected_components(binary: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or seen[y, x]:
                continue
            comp: list[tuple[int, int]] = []
            queue: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while queue:
                cy, cx = queue.popleft()
                comp.append((cy, cx))
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            comps.append(comp)
    return comps


def topk_mask(score: torch.Tensor, frac: float) -> torch.Tensor:
    flat = score.flatten()
    k = max(1, int(round(flat.numel() * frac)))
    idx = flat.topk(k).indices
    out = torch.zeros_like(flat, dtype=torch.bool)
    out[idx] = True
    return out.reshape_as(score)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def component_hit_rate(gt_patch: torch.Tensor, pred_patch: torch.Tensor) -> tuple[int, int, float, float]:
    gt_np = gt_patch.detach().cpu().numpy().astype(bool)
    pred_np = pred_patch.detach().cpu().numpy().astype(bool)
    comps = connected_components(gt_np)
    if not comps:
        return 0, 0, 0.0, 0.0
    hits = 0
    weighted_hits = 0
    total_pixels = 0
    for comp in comps:
        comp_hit = any(pred_np[y, x] for y, x in comp)
        hits += int(comp_hit)
        size = len(comp)
        total_pixels += size
        weighted_hits += size * int(comp_hit)
    return len(comps), hits, safe_div(hits, len(comps)), safe_div(weighted_hits, total_pixels)


def color_grid(gt: torch.Tensor, pred: torch.Tensor, score: torch.Tensor, scale: int = 16) -> Image.Image:
    gt_np = gt.detach().cpu().numpy().astype(bool)
    pred_np = pred.detach().cpu().numpy().astype(bool)
    score_np = score.detach().cpu().numpy()
    score_np = (score_np - score_np.min()) / max(float(score_np.max() - score_np.min()), 1e-6)
    height, width = gt_np.shape
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[..., 0] = (score_np * 255).astype(np.uint8)
    canvas[..., 1] = (pred_np * 220).astype(np.uint8)
    canvas[..., 2] = (gt_np * 220).astype(np.uint8)
    return Image.fromarray(canvas, mode="RGB").resize((width * scale, height * scale), Image.Resampling.NEAREST)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose lymphocyte coverage from patch logits.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--topk-fracs", default="0.01,0.03,0.05,0.10")
    parser.add_argument("--score", default="logit", choices=["logit", "sigmoid", "softmax"])
    parser.add_argument("--visual-count", type=int, default=24)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    data_root = args.data_root or saved_args.get("data_root", "data")
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    class_names = CLASS_NAMES[dataset]
    if "lymphocyte" not in class_names:
        raise ValueError(f"Dataset has no lymphocyte class: {class_names}")
    lymph_idx = class_names.index("lymphocyte")
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", (0.485, 0.456, 0.406)))
    image_std = tuple(float(x) for x in saved_args.get("image_std", (0.229, 0.224, 0.225)))
    device = torch.device(args.device)
    topk_fracs = [float(x) for x in args.topk_fracs.replace(";", ",").split(",") if x.strip()]

    model = build_model(checkpoint, dataset, device)
    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = SegmentationDataset(data_root, dataset, split=args.split, transform=transform)
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    metric_lists: dict[str, list[float]] = {}
    visual_written = 0
    total_lym_images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"]
            names = batch["name"]
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                logit_map = logits_to_map(outputs["patch_logits"])
            if args.score == "sigmoid":
                score_map = logit_map.sigmoid()
            elif args.score == "softmax":
                score_map = logit_map.softmax(dim=1)
            else:
                score_map = logit_map
            lym_scores = score_map[:, lymph_idx].detach().cpu()
            lym_logits = logit_map[:, lymph_idx].detach().cpu()
            grid = lym_scores.shape[-1]

            for item_idx in range(images.shape[0]):
                gt_patch = downsample_binary_mask(masks[item_idx], lymph_idx, grid)
                gt_count = int(gt_patch.sum().item())
                if gt_count == 0:
                    continue
                total_lym_images += 1
                score = lym_scores[item_idx]
                logits = lym_logits[item_idx]
                prob = logits.sigmoid()
                softmax_mass = torch.softmax(logits.flatten(), dim=0).reshape_as(logits)
                top1_mass = float(softmax_mass.max().item())
                top5_count = max(1, int(round(softmax_mass.numel() * 0.05)))
                top5_mass = float(softmax_mass.flatten().topk(top5_count).values.sum().item())

                row: dict[str, object] = {
                    "name": names[item_idx],
                    "gt_patch_count": gt_count,
                    "mean_gt_score": float(score[gt_patch].mean().item()),
                    "mean_bg_score": float(score[~gt_patch].mean().item()),
                    "max_gt_score": float(score[gt_patch].max().item()),
                    "max_all_score": float(score.max().item()),
                    "top1_softmax_mass": top1_mass,
                    "top5pct_softmax_mass": top5_mass,
                }
                for frac in topk_fracs:
                    pred_patch = topk_mask(score, frac)
                    overlap = (pred_patch & gt_patch).sum().item()
                    pred_count = pred_patch.sum().item()
                    comp_count, comp_hits, comp_hit_rate, comp_hit_weighted = component_hit_rate(gt_patch, pred_patch)
                    prefix = f"top{frac:g}"
                    row[f"{prefix}_patch_recall"] = safe_div(float(overlap), float(gt_count))
                    row[f"{prefix}_patch_precision"] = safe_div(float(overlap), float(pred_count))
                    row[f"{prefix}_components"] = comp_count
                    row[f"{prefix}_component_hits"] = comp_hits
                    row[f"{prefix}_component_hit_rate"] = comp_hit_rate
                    row[f"{prefix}_component_hit_weighted"] = comp_hit_weighted
                rows.append(row)
                for key, value in row.items():
                    if isinstance(value, (float, int)) and key != "gt_patch_count":
                        metric_lists.setdefault(key, []).append(float(value))

                if visual_written < args.visual_count:
                    pred_patch = topk_mask(score, 0.05)
                    grid_img = color_grid(gt_patch, pred_patch, score)
                    grid_img.save(visual_dir / f"{visual_written + 1:03d}_{Path(str(names[item_idx])).stem}_grid.png")
                    visual_written += 1

            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(loader):
                print(f"diagnose batch={batch_idx}/{len(loader)} lym_images={total_lym_images}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "lymphocyte_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "split": args.split,
        "num_lymphocyte_images": total_lym_images,
        "score": args.score,
        "topk_fracs": topk_fracs,
        "metrics": {key: summarize(values) for key, values in metric_lists.items()},
        "visual_dir": str(visual_dir),
    }
    (output_dir / "lymphocyte_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
