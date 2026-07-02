from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
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
    ImageLevelDataset,
    SegmentationDataset,
    load_preprocessor_stats,
    parse_image_level_label,
    pil_to_normalized_tensor,
    resolve_dataset_paths,
)
from jepa_wsss.losses import multilabel_loss
from jepa_wsss.metrics import SegmentationMeter
from jepa_wsss.models import DualRouteLinearWSSSModel


class TrainTransform:
    def __init__(self, mean: tuple[float, float, float], std: tuple[float, float, float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return pil_to_normalized_tensor(image, mean=self.mean, std=self.std)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segmentation_from_patch_logits(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


@torch.no_grad()
def evaluate(
    model: DualRouteLinearWSSSModel,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    num_classes: int,
) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes)
    route_sum = None
    route_sq_sum = None
    route_count = 0
    expected_sum = None
    expected_sq_sum = None
    layers = torch.tensor(model.route_layers, dtype=torch.float32, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
        pred = segmentation_from_patch_logits(outputs["patch_logits"], masks.shape[-2:])
        meter.update(pred, masks)
        route = outputs["spatial_route_weights"]
        if route.dim() == 2:
            route = route.unsqueeze(0).expand(images.shape[0], -1, -1)
        route_f = route.detach().float()
        expected = (route_f * layers.view(1, 1, -1)).sum(dim=-1)
        batch_route_sum = route_f.sum(dim=0)
        batch_route_sq_sum = route_f.pow(2).sum(dim=0)
        batch_expected_sum = expected.sum(dim=0)
        batch_expected_sq_sum = expected.pow(2).sum(dim=0)
        if route_sum is None:
            route_sum = batch_route_sum
            route_sq_sum = batch_route_sq_sum
            expected_sum = batch_expected_sum
            expected_sq_sum = batch_expected_sq_sum
        else:
            route_sum += batch_route_sum
            route_sq_sum += batch_route_sq_sum
            expected_sum += batch_expected_sum
            expected_sq_sum += batch_expected_sq_sum
        route_count += images.shape[0]
    metrics = meter.compute()
    if route_sum is not None and route_count > 0:
        route_mean = route_sum / float(route_count)
        route_var = (route_sq_sum / float(route_count) - route_mean.pow(2)).clamp_min(0.0)
        expected_mean = expected_sum / float(route_count)
        expected_var = (expected_sq_sum / float(route_count) - expected_mean.pow(2)).clamp_min(0.0)
        metrics["val_spatial_route_mean"] = route_mean.cpu().tolist()
        metrics["val_spatial_route_std"] = route_var.sqrt().cpu().tolist()
        metrics["val_spatial_expected_layer_mean"] = expected_mean.cpu().tolist()
        metrics["val_spatial_expected_layer_std"] = expected_var.sqrt().cpu().tolist()
    return metrics


def route_summary(model: DualRouteLinearWSSSModel) -> dict[str, object]:
    with torch.no_grad():
        sem = model.semantic_route_logits.softmax(dim=-1).detach().cpu()
        layers = torch.tensor(model.route_layers, dtype=torch.float32)
        sem_expected = (sem * layers.view(1, -1)).sum(dim=-1)
        out: dict[str, object] = {
            "route_layers": model.route_layers,
            "semantic_route_weights": sem.tolist(),
            "semantic_expected_layer": sem_expected.tolist(),
        }
        if model.variant in {"single_semantic", "single_spatial", "dual_param"}:
            if model.variant == "dual_param":
                spa = model.spatial_route_logits.softmax(dim=-1).detach().cpu()
            else:
                spa = sem
            spa_expected = (spa * layers.view(1, -1)).sum(dim=-1)
            out["spatial_route_weights"] = spa.tolist()
            out["spatial_expected_layer"] = spa_expected.tolist()
        elif model.variant == "dual_shift":
            spa = model._shift_spatial_weights(model.semantic_route_logits.softmax(dim=-1)).detach().cpu()
            spa_expected = (spa * layers.view(1, -1)).sum(dim=-1)
            out["spatial_route_weights"] = spa.tolist()
            out["spatial_expected_layer"] = spa_expected.tolist()
            out["spatial_shift"] = (F.softplus(model.spatial_shift_raw).detach().cpu() + 0.25).tolist()
            out["spatial_width"] = (F.softplus(model.spatial_width_raw).detach().cpu() + 0.75).tolist()
        elif hasattr(model, "spatial_quality_bias"):
            bias = model.spatial_quality_bias.softmax(dim=-1).detach().cpu()
            bias_expected = (bias * layers.view(1, -1)).sum(dim=-1)
            out["spatial_quality_bias_weights"] = bias.tolist()
            out["spatial_quality_bias_expected_layer"] = bias_expected.tolist()
        return out


def write_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {key: json.dumps(value) if isinstance(value, (list, dict, tuple)) else value for key, value in row.items()}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if scheduler == "none":
        return None
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * float(cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_fast_pos_weight(
    data_root: str | Path,
    dataset: str,
    max_weight: float,
    device: torch.device,
) -> torch.Tensor:
    paths = resolve_dataset_paths(data_root, dataset)
    image_paths = sorted(paths.train_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No training PNG files found in {paths.train_dir}")
    labels = torch.stack([parse_image_level_label(path.name, dataset) for path in image_paths]).float()
    pos = labels.sum(dim=0).clamp_min(1.0)
    neg = labels.shape[0] - pos
    return (neg / pos).clamp(min=1.0, max=max_weight).to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaptive dual semantic-spatial layer routing from a raw DeiT backbone.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", required=True, choices=["single_semantic", "single_spatial", "dual_param", "dual_quality", "dual_shift"])
    parser.add_argument("--route-layers", default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--scheduler", default="warmup_cosine", choices=["none", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=3.0)
    parser.add_argument("--topk-frac", type=float, default=0.05)
    parser.add_argument("--spatial-weight", type=float, default=0.5)
    parser.add_argument("--quality-hidden-dim", type=int, default=32)
    parser.add_argument("--semantic-init", default="final", choices=["uniform", "final", "middle"])
    parser.add_argument("--spatial-init", default="uniform", choices=["uniform", "final", "middle"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES[args.dataset])
    image_mean, image_std = load_preprocessor_stats(args.checkpoint)
    print(f"image_mean={image_mean}", flush=True)
    print(f"image_std={image_std}", flush=True)
    print(f"variant={args.variant} route_layers={args.route_layers}", flush=True)

    train_set = ImageLevelDataset(args.data_root, args.dataset, transform=TrainTransform(image_mean, image_std))
    val_set = SegmentationDataset(
        args.data_root,
        args.dataset,
        split="val",
        transform=lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = DualRouteLinearWSSSModel(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        num_classes=num_classes,
        route_layers=args.route_layers,
        variant=args.variant,
        topk_frac=args.topk_frac,
        spatial_weight=args.spatial_weight,
        quality_hidden_dim=args.quality_hidden_dim,
        semantic_init=args.semantic_init,
        spatial_init=args.spatial_init,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    print(f"resolved_route_layers={model.route_layers}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        args.scheduler,
        total_steps=len(train_loader) * args.epochs,
        warmup_steps=int(round(len(train_loader) * args.warmup_epochs)),
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    pos_weight = compute_fast_pos_weight(args.data_root, args.dataset, args.max_pos_weight, device) if args.class_balance else None
    if pos_weight is not None:
        print(f"pos_weight={pos_weight.detach().cpu().tolist()}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.csv"
    best_miou = -1.0
    best_epoch = 0
    global_step = 0
    saved_args = vars(args).copy()
    saved_args["model_type"] = "dual_route_linear"
    saved_args["image_mean"] = list(image_mean)
    saved_args["image_std"] = list(image_std)
    saved_args["class_names"] = CLASS_NAMES[args.dataset]

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        sem_loss_sum = 0.0
        spa_loss_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                if args.variant == "single_semantic":
                    sem_loss = multilabel_loss(outputs["semantic_image_logits"], labels, pos_weight=pos_weight)
                    spa_loss = outputs["semantic_image_logits"].sum() * 0.0
                    loss = sem_loss
                elif args.variant == "single_spatial":
                    sem_loss = outputs["spatial_image_logits"].sum() * 0.0
                    spa_loss = multilabel_loss(outputs["spatial_image_logits"], labels, pos_weight=pos_weight)
                    loss = spa_loss
                else:
                    sem_loss = multilabel_loss(outputs["semantic_image_logits"], labels, pos_weight=pos_weight)
                    spa_loss = multilabel_loss(outputs["spatial_image_logits"], labels, pos_weight=pos_weight)
                    loss = sem_loss + float(args.spatial_weight) * spa_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            loss_sum += float(loss.detach().cpu())
            sem_loss_sum += float(sem_loss.detach().cpu())
            spa_loss_sum += float(spa_loss.detach().cpu())
            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - start
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} sem={sem_loss.item():.4f} spa={spa_loss.item():.4f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s peak_mem={peak:.2f}GB",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, args.amp, num_classes)
        sec_epoch = time.perf_counter() - start
        routes = route_summary(model)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "train_sem_loss": sem_loss_sum / max(1, len(train_loader)),
            "train_spatial_loss": spa_loss_sum / max(1, len(train_loader)),
            "sec_epoch": sec_epoch,
            "val_miou": val_metrics["miou"],
            "val_mdice": val_metrics["mdice"],
            "val_mrecall": val_metrics["mrecall"],
            "val_mprecision": val_metrics["mprecision"],
            "val_fwiou": val_metrics["fwiou"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "val_spatial_route_mean": val_metrics.get("val_spatial_route_mean"),
            "val_spatial_route_std": val_metrics.get("val_spatial_route_std"),
            "val_spatial_expected_layer_mean": val_metrics.get("val_spatial_expected_layer_mean"),
            "val_spatial_expected_layer_std": val_metrics.get("val_spatial_expected_layer_std"),
            **routes,
        }
        write_log(log_path, row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_miou={val_metrics['miou']:.4f} val_mdice={val_metrics['mdice']:.4f} "
            f"val_fwiou={val_metrics['fwiou']:.4f}",
            flush=True,
        )
        checkpoint = {
            "model": model.state_dict(),
            "args": saved_args,
            "metrics": val_metrics,
            "epoch": epoch,
            "routes": routes,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if float(val_metrics["miou"]) > best_miou:
            best_miou = float(val_metrics["miou"])
            best_epoch = epoch
            torch.save(checkpoint, output_dir / "best.pt")

    print(f"best_epoch={best_epoch} best_val_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
