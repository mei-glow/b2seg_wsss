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

from jepa_wsss.datasets import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.metrics import SegmentationMeter
from scripts.evaluate_crf import build_model_from_checkpoint, logits_from_outputs


HE_DAB_STAIN_MATRIX = np.array(
    [
        [0.650, 0.072, 0.268],
        [0.704, 0.990, 0.570],
        [0.286, 0.105, 0.776],
    ],
    dtype=np.float32,
)


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    half = max(1, int(round(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_axis_reflect(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(array, ((pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(array, dtype=np.float32)
        for idx, weight in enumerate(kernel):
            out += float(weight) * padded[idx : idx + array.shape[0], :]
        return out
    if axis == 1:
        padded = np.pad(array, ((0, 0), (pad, pad)), mode="reflect")
        out = np.zeros_like(array, dtype=np.float32)
        for idx, weight in enumerate(kernel):
            out += float(weight) * padded[:, idx : idx + array.shape[1]]
        return out
    raise ValueError(f"Unsupported axis: {axis}")


def gaussian_blur(array: np.ndarray, radius: float) -> np.ndarray:
    kernel = gaussian_kernel1d(radius)
    src = array.astype(np.float32, copy=False)
    return convolve_axis_reflect(convolve_axis_reflect(src, kernel, axis=1), kernel, axis=0)


def normalize_positive(array: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    score = np.clip(array, 0.0, None).astype(np.float32)
    scale = float(np.percentile(score, percentile))
    if scale <= 1e-6:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip(score / scale, 0.0, 1.0).astype(np.float32)


def hematoxylin_density(image_rgb: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    rgb = image_rgb.astype(np.float32)
    od = -np.log((rgb + 1.0) / 255.0)
    concentrations = od.reshape(-1, 3) @ np.linalg.inv(HE_DAB_STAIN_MATRIX).T
    h = concentrations[:, 0].reshape(rgb.shape[:2])
    h = np.clip(h, 0.0, None)
    scale = float(np.percentile(h, percentile))
    if scale <= 1e-6:
        return np.zeros_like(h, dtype=np.float32)
    return np.clip(h / scale, 0.0, 1.0).astype(np.float32)


def local_maxima(score: np.ndarray) -> np.ndarray:
    padded = np.pad(score, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    is_max = np.ones_like(center, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            is_max &= center >= padded[1 + dy : 1 + dy + score.shape[0], 1 + dx : 1 + dx + score.shape[1]]
    return is_max


def denormalize_image(
    image: torch.Tensor,
    mean_values: tuple[float, float, float],
    std_values: tuple[float, float, float],
) -> np.ndarray:
    mean = torch.tensor(mean_values, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(std_values, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def extract_nodes(
    image_rgb: np.ndarray,
    dog_small: float,
    dog_medium: float,
    peak_percentile: float,
    max_nodes: int,
    min_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    h = hematoxylin_density(image_rgb)
    dog = normalize_positive(gaussian_blur(h, dog_small) - gaussian_blur(h, dog_medium))
    threshold = float(np.percentile(dog, peak_percentile))
    peaks = (dog >= threshold) & local_maxima(dog)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        coords = np.argwhere(dog >= threshold)
    if coords.shape[0] > 0:
        coords = suppress_close_nodes(coords, dog[coords[:, 0], coords[:, 1]], max_nodes, min_distance)
    return coords.astype(np.int64), dog


def suppress_close_nodes(coords: np.ndarray, values: np.ndarray, max_nodes: int, min_distance: float) -> np.ndarray:
    if coords.shape[0] == 0:
        return coords.astype(np.int64)
    order = np.argsort(values)[::-1]
    min_dist2 = float(min_distance) * float(min_distance)
    kept: list[tuple[int, int]] = []
    for idx in order:
        y = int(coords[idx, 0])
        x = int(coords[idx, 1])
        if all((y - ky) * (y - ky) + (x - kx) * (x - kx) >= min_dist2 for ky, kx in kept):
            kept.append((y, x))
            if len(kept) >= max_nodes:
                break
    return np.asarray(kept, dtype=np.int64)


def logits_to_map(patch_logits: torch.Tensor) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    batch, num_patches, dim = tokens.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return tokens.transpose(1, 2).reshape(batch, dim, grid, grid)


def sample_node_values(feature_map: torch.Tensor, coords: np.ndarray, height: int, width: int) -> torch.Tensor:
    if coords.shape[0] == 0:
        return feature_map.new_zeros((0, feature_map.shape[0]))
    y = torch.from_numpy(coords[:, 0]).to(feature_map.device, dtype=torch.float32)
    x = torch.from_numpy(coords[:, 1]).to(feature_map.device, dtype=torch.float32)
    grid_x = (x / max(width - 1, 1)) * 2.0 - 1.0
    grid_y = (y / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(feature_map.unsqueeze(0), grid, mode="bilinear", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1).contiguous()


def graph_propagate(
    node_features: torch.Tensor,
    coords: np.ndarray,
    seed_scores: torch.Tensor,
    knn: int,
    semantic_temp: float,
    spatial_sigma: float,
    steps: int,
    restart: float,
) -> torch.Tensor:
    num_nodes = node_features.shape[0]
    if num_nodes == 0:
        return seed_scores
    if num_nodes == 1:
        return seed_scores.clamp(0.0, 1.0)
    coords_t = torch.from_numpy(coords).to(node_features.device, dtype=torch.float32)
    dist = torch.cdist(coords_t, coords_t)
    feat = F.normalize(node_features, dim=-1)
    sim = feat @ feat.t()
    affinity = torch.exp(sim / max(semantic_temp, 1e-6)) * torch.exp(-(dist * dist) / (2.0 * spatial_sigma * spatial_sigma))
    affinity.fill_diagonal_(0.0)
    k = min(max(1, knn), num_nodes - 1)
    top_idx = affinity.topk(k, dim=1).indices
    mask = torch.zeros_like(affinity, dtype=torch.bool)
    mask.scatter_(dim=1, index=top_idx, value=True)
    affinity = affinity.masked_fill(~mask, 0.0)
    transition = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-6)
    seed = seed_scores.clamp(0.0, 1.0)
    score = seed.clone()
    for _ in range(max(0, steps)):
        score = float(restart) * seed + (1.0 - float(restart)) * (transition @ score)
    return score.clamp(0.0, 1.0)


def splat_nodes(coords: np.ndarray, scores: torch.Tensor, height: int, width: int, radius: float) -> torch.Tensor:
    if coords.shape[0] == 0:
        return scores.new_zeros((height, width))
    canvas = scores.new_zeros((height, width))
    weight = scores.new_zeros((height, width))
    yy, xx = torch.meshgrid(
        torch.arange(height, device=scores.device),
        torch.arange(width, device=scores.device),
        indexing="ij",
    )
    rad = max(float(radius), 1.0)
    support = int(round(3.0 * rad))
    for idx, (y_np, x_np) in enumerate(coords):
        y = int(y_np)
        x = int(x_np)
        y0, y1 = max(0, y - support), min(height, y + support + 1)
        x0, x1 = max(0, x - support), min(width, x + support + 1)
        dist2 = (yy[y0:y1, x0:x1].float() - y) ** 2 + (xx[y0:y1, x0:x1].float() - x) ** 2
        w = torch.exp(-dist2 / (2.0 * rad * rad))
        canvas[y0:y1, x0:x1] += scores[idx] * w
        weight[y0:y1, x0:x1] += w
    return canvas / weight.clamp_min(1e-6)


def binary_iou(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    pred = pred.bool()
    target = target.bool()
    tp = float((pred & target).sum().item())
    union = float((pred | target).sum().item())
    pred_count = float(pred.sum().item())
    target_count = float(target.sum().item())
    return (
        tp / max(union, 1.0),
        tp / max(target_count, 1.0),
        tp / max(pred_count, 1.0),
    )


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


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline nuclei-level graph propagation diagnostic for lymphocyte coverage.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-head", default="coarse", choices=["auto", "coarse", "refined"])
    parser.add_argument("--dog-small", type=float, default=1.5)
    parser.add_argument("--dog-medium", type=float, default=3.0)
    parser.add_argument("--peak-percentile", type=float, default=94.0)
    parser.add_argument("--max-nodes", type=int, default=256)
    parser.add_argument("--node-min-distance", type=float, default=3.0)
    parser.add_argument("--knn", type=int, default=8)
    parser.add_argument("--semantic-temp", type=float, default=0.25)
    parser.add_argument("--spatial-sigma", type=float, default=28.0)
    parser.add_argument("--prop-steps", type=int, default=4)
    parser.add_argument("--restart", type=float, default=0.45)
    parser.add_argument("--seed-score", default="sigmoid", choices=["softmax", "sigmoid"])
    parser.add_argument("--seed-threshold", type=float, default=0.70)
    parser.add_argument("--seed-topk-frac", type=float, default=0.0)
    parser.add_argument("--splat-radius", type=float, default=4.0)
    parser.add_argument("--graph-thresholds", default="0.25,0.35,0.45,0.55")
    parser.add_argument("--boosts", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--visual-count", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=25)
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
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    image_std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    num_classes = len(class_names)
    graph_thresholds = [float(x) for x in args.graph_thresholds.replace(";", ",").split(",") if x.strip()]
    boosts = [float(x) for x in args.boosts.replace(";", ",").split(",") if x.strip()]
    device = torch.device(args.device)

    model = build_model_from_checkpoint(checkpoint, ckpt_path, dataset, device)
    model.eval()
    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = SegmentationDataset(data_root, dataset, split=args.split, transform=transform)
    loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    meters = {boost: SegmentationMeter(num_classes=num_classes) for boost in boosts}
    base_meter = SegmentationMeter(num_classes=num_classes)
    graph_stats: dict[str, list[float]] = {}
    node_stats: dict[str, list[float]] = {}
    rows: list[dict[str, object]] = []
    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_written = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        names = batch["name"]
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(images)
            full_logits = logits_from_outputs(outputs, masks.shape[-2:], args.prediction_head)
            token_map = tokens_to_map(outputs["patch_tokens"])
            token_map = F.interpolate(token_map, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            if args.seed_score == "sigmoid":
                lym_score_map = full_logits[:, lymph_idx].sigmoid()
            else:
                lym_score_map = full_logits.softmax(dim=1)[:, lymph_idx]
        base_meter.update(full_logits.argmax(dim=1), masks)

        for item_idx in range(images.shape[0]):
            image_rgb = denormalize_image(images[item_idx], image_mean, image_std)
            height, width = image_rgb.shape[:2]
            coords, dog = extract_nodes(
                image_rgb,
                dog_small=args.dog_small,
                dog_medium=args.dog_medium,
                peak_percentile=args.peak_percentile,
                max_nodes=args.max_nodes,
                min_distance=args.node_min_distance,
            )
            node_features = sample_node_values(token_map[item_idx], coords, height, width)
            node_probs = sample_node_values(lym_score_map[item_idx].unsqueeze(0), coords, height, width).squeeze(1)
            seed_mask = node_probs >= args.seed_threshold
            if args.seed_topk_frac > 0.0 and node_probs.numel() > 0:
                topk = max(1, int(round(float(args.seed_topk_frac) * node_probs.numel())))
                topk = min(topk, node_probs.numel())
                seed_mask[node_probs.topk(topk).indices] = True
            seed_scores = torch.where(seed_mask, node_probs, torch.zeros_like(node_probs))
            graph_node_scores = graph_propagate(
                node_features=node_features,
                coords=coords,
                seed_scores=seed_scores,
                knn=args.knn,
                semantic_temp=args.semantic_temp,
                spatial_sigma=args.spatial_sigma,
                steps=args.prop_steps,
                restart=args.restart,
            )
            graph_map = splat_nodes(coords, graph_node_scores, height, width, args.splat_radius)
            gt_lym = masks[item_idx] == lymph_idx
            seed_count = int(seed_mask.sum().item())
            row: dict[str, object] = {
                "name": names[item_idx],
                "nodes": int(coords.shape[0]),
                "seed_nodes": seed_count,
                "seed_fraction": seed_count / max(float(coords.shape[0]), 1.0),
                "node_score_mean": float(node_probs.mean().item()) if node_probs.numel() else 0.0,
                "node_score_max": float(node_probs.max().item()) if node_probs.numel() else 0.0,
                "graph_mean": float(graph_map.mean().item()),
                "graph_max": float(graph_map.max().item()),
            }
            for stat_key in (
                "nodes",
                "seed_nodes",
                "seed_fraction",
                "node_score_mean",
                "node_score_max",
                "graph_mean",
                "graph_max",
            ):
                node_stats.setdefault(stat_key, []).append(float(row[stat_key]))
            for threshold in graph_thresholds:
                graph_pred = graph_map >= threshold
                iou, recall, precision = binary_iou(graph_pred, gt_lym)
                prefix = f"graph_t{threshold:g}"
                row[f"{prefix}_iou"] = iou
                row[f"{prefix}_recall"] = recall
                row[f"{prefix}_precision"] = precision
                graph_stats.setdefault(f"{prefix}_iou", []).append(iou)
                graph_stats.setdefault(f"{prefix}_recall", []).append(recall)
                graph_stats.setdefault(f"{prefix}_precision", []).append(precision)
            rows.append(row)

            for boost in boosts:
                logits = full_logits[item_idx : item_idx + 1].clone()
                logits[:, lymph_idx] = logits[:, lymph_idx] + float(boost) * graph_map.unsqueeze(0)
                meters[boost].update(logits.argmax(dim=1), masks[item_idx : item_idx + 1])

            if visual_written < args.visual_count:
                base_pred = full_logits[item_idx].argmax(dim=0).detach().cpu().numpy().astype(np.uint8)
                graph_np = (graph_map.detach().cpu().numpy().clip(0.0, 1.0) * 255).astype(np.uint8)
                dog_np = (dog.clip(0.0, 1.0) * 255).astype(np.uint8)
                gt_np = (gt_lym.detach().cpu().numpy().astype(np.uint8) * 255)
                panels = [
                    Image.fromarray(image_rgb, mode="RGB"),
                    Image.fromarray(dog_np, mode="L").convert("RGB"),
                    Image.fromarray(graph_np, mode="L").convert("RGB"),
                    Image.fromarray(gt_np, mode="L").convert("RGB"),
                    Image.fromarray((base_pred == lymph_idx).astype(np.uint8) * 255, mode="L").convert("RGB"),
                ]
                canvas = Image.new("RGB", (width * len(panels), height), "white")
                for panel_idx, panel in enumerate(panels):
                    canvas.paste(panel, (panel_idx * width, 0))
                canvas.save(visual_dir / f"{visual_written + 1:03d}_{Path(str(names[item_idx])).stem}.png")
                visual_written += 1

        if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(loader):
            print(f"graph batch={batch_idx}/{len(loader)}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "nuclei_graph_rows.csv").open("w", encoding="utf-8") as handle:
        if rows:
            keys = list(rows[0].keys())
            handle.write(",".join(keys) + "\n")
            for row in rows:
                handle.write(",".join(str(row[key]) for key in keys) + "\n")

    summary = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "split": args.split,
        "class_names": class_names,
        "lymphocyte_index": lymph_idx,
        "params": {
            "dog_small": args.dog_small,
            "dog_medium": args.dog_medium,
            "peak_percentile": args.peak_percentile,
            "max_nodes": args.max_nodes,
            "node_min_distance": args.node_min_distance,
            "knn": args.knn,
            "semantic_temp": args.semantic_temp,
            "spatial_sigma": args.spatial_sigma,
            "prop_steps": args.prop_steps,
            "restart": args.restart,
            "seed_score": args.seed_score,
            "seed_threshold": args.seed_threshold,
            "seed_topk_frac": args.seed_topk_frac,
            "splat_radius": args.splat_radius,
            "graph_thresholds": graph_thresholds,
            "boosts": boosts,
        },
        "node_summary": {key: summarize(values) for key, values in node_stats.items()},
        "base": {key: value for key, value in base_meter.compute().items() if key != "confusion"},
        "boosted": {
            str(boost): {key: value for key, value in meter.compute().items() if key != "confusion"}
            for boost, meter in meters.items()
        },
        "graph_binary": {key: summarize(values) for key, values in graph_stats.items()},
        "visual_dir": str(visual_dir),
    }
    (output_dir / "nuclei_graph_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
