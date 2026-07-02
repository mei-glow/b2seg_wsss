from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.backbones import load_local_checkpoint
from jepa_wsss.datasets import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset, pil_to_normalized_tensor


@dataclass
class LayerSamples:
    features: list[torch.Tensor]
    labels: list[torch.Tensor]
    purities: list[torch.Tensor]


def parse_layers(value: str, num_layers: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(1, num_layers + 1))
    layers = [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    bad = [layer for layer in layers if layer < 1 or layer > num_layers]
    if bad:
        raise ValueError(f"Layers out of range 1..{num_layers}: {bad}")
    return sorted(set(layers))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_deit(model_name: str, checkpoint: str | None, device: torch.device) -> nn.Module:
    import timm

    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if checkpoint:
        report = load_local_checkpoint(model, checkpoint)
        print(f"checkpoint_loaded_tensors={report['loaded']}", flush=True)
    model.eval().to(device)
    return model


@torch.no_grad()
def forward_layer_tokens(vit: nn.Module, images: torch.Tensor, layers: list[int]) -> dict[int, torch.Tensor]:
    wanted = {layer - 1 for layer in layers}
    x = vit.patch_embed(images)
    x = vit._pos_embed(x)
    x = vit.patch_drop(x)
    x = vit.norm_pre(x)
    out: dict[int, torch.Tensor] = {}
    for block_idx, block in enumerate(vit.blocks):
        x = block(x)
        if block_idx in wanted:
            out[block_idx + 1] = vit.norm(x)[:, 1:, :].detach()
    return out


def patch_labels_from_masks(
    masks: torch.Tensor,
    num_classes: int,
    grid_size: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = masks != ignore_index
    one_hot = []
    for class_idx in range(num_classes):
        one_hot.append(((masks == class_idx) & valid).float())
    class_maps = torch.stack(one_hot, dim=1)
    valid_map = valid.float().unsqueeze(1)
    class_counts = F.adaptive_avg_pool2d(class_maps, (grid_size, grid_size))
    valid_counts = F.adaptive_avg_pool2d(valid_map, (grid_size, grid_size)).clamp_min(1e-6)
    fractions = class_counts / valid_counts
    purity, labels = fractions.max(dim=1)
    has_valid = F.adaptive_avg_pool2d(valid_map, (grid_size, grid_size)).squeeze(1) > 0
    labels = labels.masked_fill(~has_valid, ignore_index)
    purity = purity.masked_fill(~has_valid, 0.0)
    return labels.flatten(1), purity.flatten(1)


def append_balanced_samples(
    store: LayerSamples,
    features: torch.Tensor,
    labels: torch.Tensor,
    purities: torch.Tensor,
    num_classes: int,
    min_purity: float,
    max_per_class: int,
) -> None:
    flat_features = features.detach().cpu()
    flat_labels = labels.detach().cpu()
    flat_purities = purities.detach().cpu()
    for class_idx in range(num_classes):
        mask = (flat_labels == class_idx) & (flat_purities >= min_purity)
        idx = mask.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        take = min(idx.numel(), max_per_class)
        if idx.numel() > take:
            idx = idx[torch.randperm(idx.numel())[:take]]
        store.features.append(flat_features[idx])
        store.labels.append(flat_labels[idx])
        store.purities.append(flat_purities[idx])


def stack_and_cap(store: LayerSamples, num_classes: int, max_per_class: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not store.features:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long), torch.empty(0)
    features = torch.cat(store.features, dim=0)
    labels = torch.cat(store.labels, dim=0)
    purities = torch.cat(store.purities, dim=0)
    keep_all = []
    for class_idx in range(num_classes):
        idx = (labels == class_idx).nonzero(as_tuple=False).flatten()
        if idx.numel() > max_per_class:
            idx = idx[torch.randperm(idx.numel())[:max_per_class]]
        keep_all.append(idx)
    keep = torch.cat(keep_all) if keep_all else torch.empty(0, dtype=torch.long)
    keep = keep[torch.randperm(keep.numel())] if keep.numel() else keep
    return features[keep], labels[keep], purities[keep]


def auc_rank(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().float().cpu()
    targets = targets.detach().bool().cpu()
    num_pos = int(targets.sum().item())
    num_neg = int((~targets).sum().item())
    if num_pos == 0 or num_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores, dtype=torch.float64)
    start = 0
    rank_value = 1.0
    while start < scores.numel():
        end = start + 1
        while end < scores.numel() and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (rank_value + rank_value + (end - start) - 1.0) / 2.0
        ranks[order[start:end]] = avg_rank
        rank_value += end - start
        start = end
    pos_rank_sum = ranks[targets].sum().item()
    return float((pos_rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))


def split_indices(num_items: int, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_items, generator=gen)
    train_count = max(1, min(num_items - 1, int(round(num_items * train_fraction))))
    return perm[:train_count], perm[train_count:]


def centroid_scores(train_x: torch.Tensor, train_y: torch.Tensor, eval_x: torch.Tensor, num_classes: int) -> torch.Tensor:
    train_x = F.normalize(train_x.float(), dim=-1)
    eval_x = F.normalize(eval_x.float(), dim=-1)
    centroids = []
    global_mean = train_x.mean(dim=0)
    for class_idx in range(num_classes):
        selected = train_x[train_y == class_idx]
        centroids.append(selected.mean(dim=0) if selected.numel() else global_mean)
    centroids_t = F.normalize(torch.stack(centroids, dim=0), dim=-1)
    return eval_x @ centroids_t.t()


def train_linear_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    dim = train_x.shape[-1]
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = ((train_x - mean) / std).float()
    eval_x = ((eval_x - mean) / std).float()
    probe = nn.Linear(dim, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    n = train_x.shape[0]
    for _ in range(max(1, epochs)):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            loss = F.cross_entropy(probe(train_x[idx]), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    probe.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, eval_x.shape[0], batch_size):
            outputs.append(probe(eval_x[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs, dim=0)


def metrics_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    class_names: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    num_classes = len(class_names)
    pred = scores.argmax(dim=1)
    accuracy = float((pred == labels).float().mean().item())
    class_rows = []
    for class_idx, name in enumerate(class_names):
        target = labels == class_idx
        class_rows.append(
            {
                "class": name,
                "class_index": class_idx,
                "auc_ovr": auc_rank(scores[:, class_idx], target),
                "acc_one_class": float((pred[target] == class_idx).float().mean().item()) if target.any() else float("nan"),
                "count": int(target.sum().item()),
            }
        )
    pair_rows = []
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            mask = (labels == i) | (labels == j)
            pair_labels = labels[mask] == i
            pair_score = scores[mask, i] - scores[mask, j]
            pair_rows.append(
                {
                    "class_a": class_names[i],
                    "class_b": class_names[j],
                    "auc_a_vs_b": auc_rank(pair_score, pair_labels),
                    "count_a": int((labels == i).sum().item()),
                    "count_b": int((labels == j).sum().item()),
                }
            )
    return class_rows, pair_rows, {"accuracy": accuracy}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose per-layer DeiT patch separability using dense masks for analysis only.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", default="all", help="Comma-separated 1-based layers, or 'all'.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--min-purity", type=float, default=0.60)
    parser.add_argument("--max-samples-per-class", type=int, default=6000)
    parser.add_argument("--sample-per-class-per-batch", type=int, default=256)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--linear-probe-epochs", type=int, default=20)
    parser.add_argument("--linear-probe-lr", type=float, default=1e-3)
    parser.add_argument("--linear-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--probe-batch-size", type=int, default=2048)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = CLASS_NAMES[args.dataset]
    num_classes = len(class_names)

    model = make_deit(args.model, args.checkpoint, device)
    layers = parse_layers(args.layers, len(model.blocks))
    transform = lambda image: pil_to_normalized_tensor(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    dataset = SegmentationDataset(args.data_root, args.dataset, split=args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    stores = {layer: LayerSamples([], [], []) for layer in layers}
    purity_counts = {class_idx: 0 for class_idx in range(num_classes)}
    total_valid_patches = 0
    total_kept_patches = 0

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            layer_tokens = forward_layer_tokens(model, images, layers)
        num_patches = next(iter(layer_tokens.values())).shape[1]
        grid = int(num_patches**0.5)
        if grid * grid != num_patches:
            raise ValueError(f"Expected square token grid, got {num_patches}")
        patch_labels, patch_purities = patch_labels_from_masks(masks, num_classes, grid, ignore_index=255)
        valid = patch_labels != 255
        total_valid_patches += int(valid.sum().item())
        kept = valid & (patch_purities >= args.min_purity)
        total_kept_patches += int(kept.sum().item())
        for class_idx in range(num_classes):
            purity_counts[class_idx] += int((kept & (patch_labels == class_idx)).sum().item())
        flat_labels = patch_labels.reshape(-1)
        flat_purities = patch_purities.reshape(-1)
        for layer, tokens in layer_tokens.items():
            append_balanced_samples(
                stores[layer],
                tokens.reshape(-1, tokens.shape[-1]),
                flat_labels,
                flat_purities,
                num_classes,
                min_purity=args.min_purity,
                max_per_class=args.sample_per_class_per_batch,
            )
        if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == len(loader):
            print(f"layer_diag batch={batch_idx}/{len(loader)} kept={total_kept_patches}/{total_valid_patches}", flush=True)

    summary: dict[str, object] = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "layers": layers,
        "min_purity": args.min_purity,
        "class_names": class_names,
        "total_valid_patches": total_valid_patches,
        "total_kept_patches": total_kept_patches,
        "kept_fraction": total_kept_patches / max(total_valid_patches, 1),
        "kept_by_class": {class_names[idx]: count for idx, count in purity_counts.items()},
        "layer_results": {},
    }
    all_class_rows: list[dict[str, object]] = []
    all_pair_rows: list[dict[str, object]] = []
    all_layer_rows: list[dict[str, object]] = []

    for layer in layers:
        features, labels, purities = stack_and_cap(stores[layer], num_classes, args.max_samples_per_class)
        if features.numel() == 0:
            continue
        train_idx, eval_idx = split_indices(features.shape[0], args.train_fraction, args.seed + layer)
        train_x, train_y = features[train_idx], labels[train_idx]
        eval_x, eval_y = features[eval_idx], labels[eval_idx]
        centroid = centroid_scores(train_x, train_y, eval_x, num_classes)
        linear = train_linear_probe(
            train_x,
            train_y,
            eval_x,
            num_classes,
            epochs=args.linear_probe_epochs,
            lr=args.linear_probe_lr,
            weight_decay=args.linear_probe_weight_decay,
            batch_size=args.probe_batch_size,
            device=device,
        )
        layer_result: dict[str, object] = {
            "num_samples": int(features.shape[0]),
            "train_samples": int(train_idx.numel()),
            "eval_samples": int(eval_idx.numel()),
            "mean_purity": float(purities.mean().item()),
        }
        for probe_name, scores in (("centroid", centroid), ("linear", linear)):
            class_rows, pair_rows, scalar = metrics_from_scores(scores, eval_y, class_names)
            layer_result[f"{probe_name}_accuracy"] = scalar["accuracy"]
            for row in class_rows:
                row = {"layer": layer, "probe": probe_name, **row}
                all_class_rows.append(row)
            for row in pair_rows:
                row = {"layer": layer, "probe": probe_name, **row}
                all_pair_rows.append(row)
        all_layer_rows.append({"layer": layer, **layer_result})
        summary["layer_results"][str(layer)] = layer_result
        print(
            f"layer={layer} samples={features.shape[0]} "
            f"centroid_acc={layer_result['centroid_accuracy']:.4f} "
            f"linear_acc={layer_result['linear_accuracy']:.4f}",
            flush=True,
        )

    write_csv(output_dir / "layer_summary.csv", all_layer_rows)
    write_csv(output_dir / "class_auc.csv", all_class_rows)
    write_csv(output_dir / "pairwise_auc.csv", all_pair_rows)
    (output_dir / "layer_diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
