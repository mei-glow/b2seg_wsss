from __future__ import annotations

import argparse
import copy
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

from b2seg_wsss.datasets import (
    CLASS_NAMES,
    ImageLevelDataset,
    SegmentationDataset,
    load_preprocessor_stats,
    parse_image_level_label,
    pil_to_normalized_tensor,
    resolve_dataset_paths,
)
from b2seg_wsss.losses import multilabel_loss
from b2seg_wsss.metrics import SegmentationMeter
from b2seg_wsss.models import DualRouteLinearWSSSModel
from scripts.patch_embed_adapt import adapt_vit_patch_embed


# Ruifrok-Johnston H&E-DAB stain matrix (same convention as skimage.color.rgb2hed).
RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]],
    dtype=np.float32,
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)

ROT90_TRANSPOSES = (
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
)


def hed_jitter(image: Image.Image, sigma_alpha: float, sigma_beta: float) -> Image.Image:
    """Tellez-style H&E stain augmentation: perturb haematoxylin/eosin in optical-density space.

    Models the real cross-institution variation in H&E staining, which plain
    brightness/contrast jitter cannot express.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    optical_density = -np.log((rgb + 1.0) / 256.0)
    stains = optical_density @ HED_FROM_RGB
    alpha = np.random.uniform(1.0 - sigma_alpha, 1.0 + sigma_alpha, 3).astype(np.float32)
    beta = np.random.uniform(-sigma_beta, sigma_beta, 3).astype(np.float32)
    # H&E slides carry no DAB, so the third channel is residual: leave it untouched.
    alpha[2] = 1.0
    beta[2] = 0.0
    jittered = (stains * alpha + beta) @ RGB_FROM_HED
    rgb = 256.0 * np.exp(-jittered) - 1.0
    return Image.fromarray(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


class TrainTransform:
    def __init__(
        self,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        rot90: bool = False,
        hed_prob: float = 0.0,
        hed_sigma_alpha: float = 0.05,
        hed_sigma_beta: float = 0.05,
    ) -> None:
        self.mean = mean
        self.std = std
        self.rot90 = bool(rot90)
        self.hed_prob = float(hed_prob)
        self.hed_sigma_alpha = float(hed_sigma_alpha)
        self.hed_sigma_beta = float(hed_sigma_beta)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        if self.rot90:
            # Flips plus a random quarter turn cover the full D4 symmetry group.
            turns = random.randint(0, 3)
            if turns:
                image = image.transpose(ROT90_TRANSPOSES[turns - 1])
        if self.hed_prob > 0.0 and random.random() < self.hed_prob:
            image = hed_jitter(image, self.hed_sigma_alpha, self.hed_sigma_beta)
        return pil_to_normalized_tensor(image, mean=self.mean, std=self.std)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ramp_value(epoch: int, warmup_epochs: float, ramp_epochs: float) -> float:
    """0 during warmup, then linear ramp to 1 over `ramp_epochs`."""
    if epoch <= float(warmup_epochs):
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return max(0.0, min(1.0, (float(epoch) - float(warmup_epochs)) / float(ramp_epochs)))


def patch_logits_to_map(patch_logits: torch.Tensor) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)


def apply_equivariance_transform(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(images, dims=(3,))
    if mode == "vflip":
        return torch.flip(images, dims=(2,))
    if mode == "hvflip":
        return torch.flip(images, dims=(2, 3))
    if mode == "rot90":
        return torch.rot90(images, k=1, dims=(2, 3))
    if mode == "rot180":
        return torch.rot90(images, k=2, dims=(2, 3))
    if mode == "rot270":
        return torch.rot90(images, k=3, dims=(2, 3))
    if mode == "transpose":
        return images.transpose(2, 3)
    raise ValueError(f"Unknown equivariance mode: {mode}")


def invert_equivariance_logits(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "hflip":
        return torch.flip(logits, dims=(-1,))
    if mode == "vflip":
        return torch.flip(logits, dims=(-2,))
    if mode == "hvflip":
        return torch.flip(logits, dims=(-2, -1))
    if mode == "rot90":
        return torch.rot90(logits, k=-1, dims=(-2, -1))
    if mode == "rot180":
        return torch.rot90(logits, k=-2, dims=(-2, -1))
    if mode == "rot270":
        return torch.rot90(logits, k=-3, dims=(-2, -1))
    if mode == "transpose":
        return logits.transpose(-2, -1)
    raise ValueError(f"Unknown equivariance mode: {mode}")


def parse_equivariance_modes(value: str) -> list[str]:
    valid = {"hflip", "vflip", "hvflip", "rot90", "rot180", "rot270", "transpose"}
    modes = [item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()]
    if not modes:
        raise ValueError("Expected at least one equivariance mode.")
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"Unknown equivariance modes: {unknown}. Valid: {sorted(valid)}")
    return modes


def equivariance_kl_loss(
    reference_patch_logits: torch.Tensor,
    transformed_patch_logits: torch.Tensor,
    mode: str,
    temperature: float,
) -> torch.Tensor:
    """Force f(T(x)) to match T(f(x)) on the patch map, so the argmax token stops drifting."""
    reference_map = patch_logits_to_map(reference_patch_logits.detach()).float()
    transformed_map = invert_equivariance_logits(patch_logits_to_map(transformed_patch_logits).float(), mode)
    temp = max(float(temperature), 1e-6)
    target = F.softmax(reference_map / temp, dim=1).clamp_min(1e-6)
    log_prob = F.log_softmax(transformed_map / temp, dim=1)
    return (target * (target.log() - log_prob)).sum(dim=1).mean() * (temp * temp)


def absent_token_hard_negative_loss(
    patch_logits: torch.Tensor,
    labels: torch.Tensor,
    topk_frac: float,
) -> tuple[torch.Tensor, float, float]:
    """Exact negatives: for an image-level absent class every token is a true negative.

    Penalizes only the highest-scoring tokens of absent classes, which turns the
    1-token-per-class max-MIL signal into ~topk tokens per absent class with no label noise.
    """
    probs = torch.sigmoid(patch_logits.float())
    absent = labels <= 0.0
    batch, num_patches, _ = probs.shape
    topk = max(1, min(num_patches, int(round(num_patches * float(topk_frac)))))
    losses: list[torch.Tensor] = []
    selected_score_sum = 0.0
    selected_count = 0
    for batch_idx in range(batch):
        absent_classes = torch.nonzero(absent[batch_idx], as_tuple=False).flatten()
        if absent_classes.numel() == 0:
            continue
        candidate_scores = probs[batch_idx, :, absent_classes]
        hard_scores = candidate_scores.topk(topk, dim=0).values
        losses.append(-torch.log1p(-hard_scores.clamp(max=1.0 - 1e-6)).mean())
        selected_score_sum += float(hard_scores.detach().sum().cpu())
        selected_count += int(hard_scores.numel())
    if not losses:
        zero = patch_logits.sum() * 0.0
        return zero, 0.0, 0.0
    loss = torch.stack(losses).mean()
    absent_fraction = float(absent.float().mean().detach().cpu())
    selected_score_mean = selected_score_sum / float(max(1, selected_count))
    return loss, absent_fraction, selected_score_mean


def build_param_groups(
    model: torch.nn.Module,
    lr: float,
    backbone_lr: float,
    layer_decay: float,
    weight_decay: float,
    route_lr: float | None = None,
    no_decay_1d: bool = True,
) -> list[dict[str, object]]:
    """Layer-wise LR decay for the pretrained ViT, full LR for the newly added modules.

    Routing over all 12 layers is only meaningful while the pretrained layer
    hierarchy survives, so early blocks are updated far more slowly than late ones.
    Norms, biases and route logits are excluded from weight decay.

    `route_lr` gives the route logits their own learning rate. The softmax Jacobian
    scales each layer's route gradient by its own weight, so with `--semantic-init final`
    the non-final layers receive ~55x less gradient than the final one and can never
    grow; a larger route LR compensates for that factor directly.
    """
    num_blocks = len(model.vit.blocks)
    groups: dict[tuple[float, float], dict[str, object]] = {}
    summary: dict[str, float] = {}
    for name, param in model.named_parameters():
        is_route = "route_logits" in name
        decay = 0.0 if (is_route or (no_decay_1d and param.ndim <= 1)) else float(weight_decay)
        if is_route and route_lr is not None:
            group_lr = float(route_lr)
            key = (round(group_lr, 12), decay)
            if key not in groups:
                groups[key] = {"params": [], "lr": group_lr, "weight_decay": decay}
            groups[key]["params"].append(param)  # type: ignore[union-attr]
            summary[name] = group_lr
            continue
        if name.startswith("vit."):
            if name.startswith("vit.blocks."):
                block_idx = int(name.split(".")[2])
                scale = float(layer_decay) ** max(0, num_blocks - 1 - block_idx)
            elif name.startswith("vit.norm"):
                scale = 1.0
            else:  # patch_embed, cls_token, pos_embed
                scale = float(layer_decay) ** num_blocks
            group_lr = float(backbone_lr) * scale
        else:
            group_lr = float(lr)
        key = (round(group_lr, 12), decay)
        if key not in groups:
            groups[key] = {"params": [], "lr": group_lr, "weight_decay": decay}
        groups[key]["params"].append(param)  # type: ignore[union-attr]
        summary[name.split(".")[0] if not name.startswith("vit.blocks.") else f"block{name.split('.')[2]}"] = group_lr
    print(f"param_group_lrs={ {k: round(v, 8) for k, v in sorted(summary.items()) } }", flush=True)
    return list(groups.values())


@torch.no_grad()
def update_ema_model(model: torch.nn.Module, ema_model: torch.nn.Module, decay: float) -> None:
    model_state = model.state_dict()
    ema_state = ema_model.state_dict()
    decay = float(decay)
    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value.to(dtype=ema_value.dtype), alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


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


# Only the pos-weight cap differs across datasets; everything else in the F recipe is shared.
# luad lymphocyte is present in ~4% of images, so its neg/pos ratio needs a higher cap.
DATASET_MAX_POS_WEIGHT = {"bcss": 3.0, "gcss": 3.0, "luad": 8.0}


def apply_preset(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Resolve dataset-aware defaults and, for --preset f, the full Stage-1 F recipe.

    A user-supplied flag always wins: preset values are only written where the argument
    is still at its parser default, so the baseline stays reproducible with --preset none.
    """
    if args.max_pos_weight is None:
        args.max_pos_weight = DATASET_MAX_POS_WEIGHT.get(args.dataset, 3.0)

    if args.preset == "none":
        return

    f_recipe = {
        "variant": "single_semantic",
        "route_layers": "all",
        "epochs": 15,
        "semantic_init": "final",
        "semantic_pooling": "lse",
        "lse_tau": 1.0,
        "clip_grad": 1.0,
        "class_balance": True,
        "ema_teacher": True,
        "ema_decay": 0.999,
        "equivariance_weight": 0.03,
        "equivariance_warmup_epochs": 1.0,
        "equivariance_ramp_epochs": 2.0,
        "equivariance_modes": "hflip,vflip,hvflip",
        "absent_token_weight": 0.03,
        "absent_token_topk_frac": 0.05,
        "absent_token_warmup_epochs": 1.0,
        "absent_token_ramp_epochs": 2.0,
        "augment_rot90": True,
        "hed_jitter_prob": 0.5,
        "hed_sigma_alpha": 0.05,
        "hed_sigma_beta": 0.05,
        "amp": True,
        "grad_checkpointing": True,
    }
    overridden = []
    for key, value in f_recipe.items():
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, value)
        elif getattr(args, key) != value:
            overridden.append(key)
    if overridden:
        print(f"preset=f: kept user overrides for {overridden}", flush=True)
    print(
        f"preset=f applied: dataset={args.dataset} max_pos_weight={args.max_pos_weight} "
        f"pooling={args.semantic_pooling} ema_decay={args.ema_decay}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaptive dual semantic-spatial layer routing from a raw DeiT backbone.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad", "gcss"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--preset",
        default="none",
        choices=["none", "f"],
        help="'f' applies the full Stage-1 F recipe (LSE + EMA + equivariance + absent-token + D4 + HED) "
             "and a dataset-aware pos-weight, so only --dataset/--data-root/--output-dir/--seed are needed.",
    )
    parser.add_argument("--variant", default="single_semantic", choices=["single_semantic", "single_spatial", "dual_param", "dual_quality", "dual_shift"])
    parser.add_argument("--route-layers", default="all")
    parser.add_argument("--patch-kernel", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--patch-padding", type=int, default=0)
    parser.add_argument("--patch-resample-scale", default="area", choices=["area", "none"])
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
    parser.add_argument(
        "--max-pos-weight",
        type=float,
        default=None,
        help="Cap on BCE pos_weight. Unset resolves per dataset: luad=8 (lymphocyte is very rare), bcss/gcss=3.",
    )
    parser.add_argument("--topk-frac", type=float, default=0.05)
    parser.add_argument(
        "--semantic-pooling",
        default="max",
        choices=["max", "lse", "topk"],
        help="MIL pooling for the semantic image logits. 'lse' is the smooth max that removes argmax jitter.",
    )
    parser.add_argument("--lse-tau", type=float, default=1.0, help="Temperature for --semantic-pooling lse.")
    parser.add_argument("--clip-grad", type=float, default=0.0, help="Max grad norm; 0 disables clipping.")
    parser.add_argument(
        "--param-groups",
        action="store_true",
        help="Layer-wise LR decay for the ViT plus no weight decay on norms/biases/route logits.",
    )
    parser.add_argument("--backbone-lr", type=float, default=None, help="Base LR for the ViT when --param-groups is set.")
    parser.add_argument("--layer-decay", type=float, default=0.75, help="Per-block LR decay factor, applied late-to-early.")
    parser.add_argument("--route-freeze-epochs", type=int, default=0, help="Keep route logits fixed for the first N epochs.")
    parser.add_argument(
        "--route-lr",
        type=float,
        default=None,
        help="Separate LR for the route logits. Compensates the softmax Jacobian that otherwise freezes them at init.",
    )
    parser.add_argument("--augment-rot90", action="store_true", help="Add random quarter turns (full D4 group with the flips).")
    parser.add_argument("--hed-jitter-prob", type=float, default=0.0, help="Probability of applying H&E stain jitter.")
    parser.add_argument("--hed-sigma-alpha", type=float, default=0.05)
    parser.add_argument("--hed-sigma-beta", type=float, default=0.05)
    parser.add_argument("--ema-teacher", action="store_true", help="Track an EMA copy and evaluate/save it as the teacher.")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--equivariance-weight", type=float, default=0.0)
    parser.add_argument("--equivariance-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--equivariance-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--equivariance-modes", default="hflip,vflip,hvflip")
    parser.add_argument("--equivariance-temperature", type=float, default=1.0)
    parser.add_argument("--absent-token-weight", type=float, default=0.0)
    parser.add_argument("--absent-token-topk-frac", type=float, default=0.05)
    parser.add_argument("--absent-token-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--absent-token-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--spatial-weight", type=float, default=0.5)
    parser.add_argument("--quality-hidden-dim", type=int, default=32)
    parser.add_argument("--semantic-init", default="final", choices=["uniform", "final", "middle"])
    parser.add_argument("--spatial-init", default="uniform", choices=["uniform", "final", "middle"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    apply_preset(args, parser)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES[args.dataset])
    image_mean, image_std = load_preprocessor_stats(args.checkpoint)
    print(f"image_mean={image_mean}", flush=True)
    print(f"image_std={image_std}", flush=True)
    print(f"variant={args.variant} route_layers={args.route_layers}", flush=True)

    train_transform = TrainTransform(
        image_mean,
        image_std,
        rot90=args.augment_rot90,
        hed_prob=args.hed_jitter_prob,
        hed_sigma_alpha=args.hed_sigma_alpha,
        hed_sigma_beta=args.hed_sigma_beta,
    )
    print(
        f"augment_rot90={bool(args.augment_rot90)} hed_jitter_prob={args.hed_jitter_prob} "
        f"hed_sigma_alpha={args.hed_sigma_alpha} hed_sigma_beta={args.hed_sigma_beta}",
        flush=True,
    )
    train_set = ImageLevelDataset(args.data_root, args.dataset, transform=train_transform)
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
        semantic_pooling=args.semantic_pooling,
        lse_tau=args.lse_tau,
        spatial_weight=args.spatial_weight,
        quality_hidden_dim=args.quality_hidden_dim,
        semantic_init=args.semantic_init,
        spatial_init=args.spatial_init,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    patch_info = adapt_vit_patch_embed(
        model,
        patch_kernel=args.patch_kernel,
        patch_stride=args.patch_stride,
        patch_padding=args.patch_padding,
        resample_scale=args.patch_resample_scale,
    )
    if patch_info["patch_adapted"]:
        print(f"patch_embed_adapt={patch_info}", flush=True)
    print(f"resolved_route_layers={model.route_layers}", flush=True)

    if args.param_groups or args.route_lr is not None:
        # --route-lr stays orthogonal to --param-groups: without the latter the backbone
        # keeps a single LR, no layer decay and the original weight-decay behaviour.
        backbone_lr = (args.backbone_lr if (args.param_groups and args.backbone_lr is not None) else args.lr)
        optimizer = torch.optim.AdamW(
            build_param_groups(
                model,
                args.lr,
                backbone_lr,
                args.layer_decay if args.param_groups else 1.0,
                args.weight_decay,
                route_lr=args.route_lr,
                no_decay_1d=args.param_groups,
            ),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
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

    ema_model = None
    if args.ema_teacher:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for param in ema_model.parameters():
            param.requires_grad_(False)
    equivariance_modes = parse_equivariance_modes(args.equivariance_modes)
    print(
        f"semantic_pooling={args.semantic_pooling} lse_tau={args.lse_tau} clip_grad={args.clip_grad} "
        f"ema_teacher={bool(args.ema_teacher)} ema_decay={args.ema_decay} "
        f"equivariance_weight={args.equivariance_weight} modes={equivariance_modes} "
        f"absent_token_weight={args.absent_token_weight} absent_token_topk_frac={args.absent_token_topk_frac}",
        flush=True,
    )

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
    saved_args["patch_embed_info"] = patch_info

    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        sem_loss_sum = 0.0
        eq_loss_sum = 0.0
        absent_loss_sum = 0.0
        absent_score_sum = 0.0
        grad_norm_sum = 0.0
        grad_norm_count = 0
        grad_overflow_count = 0
        current_equivariance_weight = float(args.equivariance_weight) * ramp_value(
            epoch, args.equivariance_warmup_epochs, args.equivariance_ramp_epochs
        )
        current_absent_token_weight = float(args.absent_token_weight) * ramp_value(
            epoch, args.absent_token_warmup_epochs, args.absent_token_ramp_epochs
        )
        # Let the classifier settle at the initial routing before the gate is allowed to move.
        route_trainable = epoch > int(args.route_freeze_epochs)
        model.semantic_route_logits.requires_grad_(route_trainable)
        if getattr(model, "spatial_route_logits", None) is not None:
            model.spatial_route_logits.requires_grad_(route_trainable)
        if int(args.route_freeze_epochs) > 0:
            print(f"epoch={epoch} route_trainable={route_trainable}", flush=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                sem_loss = multilabel_loss(outputs["semantic_image_logits"], labels, pos_weight=pos_weight)
                loss = sem_loss
                if current_absent_token_weight > 0.0:
                    absent_loss, _absent_fraction, absent_score = absent_token_hard_negative_loss(
                        outputs["patch_logits"],
                        labels,
                        args.absent_token_topk_frac,
                    )
                    loss = loss + current_absent_token_weight * absent_loss
                else:
                    absent_loss = outputs["patch_logits"].sum() * 0.0
                    absent_score = 0.0
                if current_equivariance_weight > 0.0:
                    eq_mode = equivariance_modes[global_step % len(equivariance_modes)]
                    equiv_outputs = model(apply_equivariance_transform(images, eq_mode))
                    eq_loss = equivariance_kl_loss(
                        outputs["patch_logits"],
                        equiv_outputs["patch_logits"],
                        eq_mode,
                        args.equivariance_temperature,
                    )
                    loss = loss + current_equivariance_weight * eq_loss
                else:
                    eq_loss = outputs["patch_logits"].sum() * 0.0
            scaler.scale(loss).backward()
            if args.clip_grad > 0.0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.clip_grad))
                grad_norm_value = float(grad_norm.detach().cpu())
                # AMP overflow makes the norm inf and GradScaler skips the step; do not
                # let those steps poison the epoch average.
                if grad_norm_value == grad_norm_value and grad_norm_value != float("inf"):
                    grad_norm_sum += grad_norm_value
                    grad_norm_count += 1
                else:
                    grad_overflow_count += 1
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema_model is not None:
                update_ema_model(model, ema_model, args.ema_decay)
            global_step += 1
            loss_sum += float(loss.detach().cpu())
            sem_loss_sum += float(sem_loss.detach().cpu())
            eq_loss_sum += float(eq_loss.detach().cpu())
            absent_loss_sum += float(absent_loss.detach().cpu())
            absent_score_sum += float(absent_score)
            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                lr = max(float(group["lr"]) for group in optimizer.param_groups)
                elapsed = time.perf_counter() - start
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} sem={sem_loss.item():.4f} "
                    f"abs={absent_loss.item():.4f} eq={eq_loss.item():.4f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s peak_mem={peak:.2f}GB",
                    flush=True,
                )

        # With EMA enabled the EMA weights are the teacher we ship, so they drive model selection.
        eval_model = ema_model if ema_model is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, args.amp, num_classes)
        raw_metrics = evaluate(model, val_loader, device, args.amp, num_classes) if ema_model is not None else None
        sec_epoch = time.perf_counter() - start
        routes = route_summary(eval_model)
        # The EMA lags the live weights, so always report the trained model's route too.
        raw_routes = route_summary(model)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "train_sem_loss": sem_loss_sum / max(1, len(train_loader)),
            "train_equivariance_loss": eq_loss_sum / max(1, len(train_loader)),
            "train_absent_token_loss": absent_loss_sum / max(1, len(train_loader)),
            "train_absent_token_score": absent_score_sum / max(1, len(train_loader)),
            "train_grad_norm": grad_norm_sum / max(1, grad_norm_count),
            "train_grad_overflow_steps": grad_overflow_count,
            "equivariance_weight": current_equivariance_weight,
            "absent_token_weight": current_absent_token_weight,
            "eval_model": "ema" if ema_model is not None else "raw",
            "val_miou_raw": None if raw_metrics is None else raw_metrics["miou"],
            "val_fwiou_raw": None if raw_metrics is None else raw_metrics["fwiou"],
            "raw_semantic_expected_layer": raw_routes["semantic_expected_layer"],
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
            "semantic_route_weights": routes["semantic_route_weights"],
            "semantic_expected_layer": routes["semantic_expected_layer"],
        }
        write_log(log_path, row)
        raw_note = "" if raw_metrics is None else f" val_miou_raw={raw_metrics['miou']:.4f}"
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_miou={val_metrics['miou']:.4f} val_mdice={val_metrics['mdice']:.4f} "
            f"val_fwiou={val_metrics['fwiou']:.4f}{raw_note}",
            flush=True,
        )
        checkpoint = {
            "model": eval_model.state_dict(),
            "raw_model": model.state_dict() if ema_model is not None else None,
            "args": saved_args,
            "metrics": val_metrics,
            "raw_metrics": raw_metrics,
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
