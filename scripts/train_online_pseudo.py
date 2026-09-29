from __future__ import annotations

import argparse
import copy
import csv
import json
import os
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

from b2seg_wsss.datasets import (  # noqa: E402
    CLASS_NAMES,
    ImageLevelDataset,
    SegmentationDataset,
    load_preprocessor_stats,
    parse_image_level_label,
    pil_to_normalized_tensor,
    resolve_dataset_paths,
)
from b2seg_wsss.jepa import JEPAPredictor, gather_tokens  # noqa: E402
from b2seg_wsss.losses import multilabel_loss  # noqa: E402
from b2seg_wsss.metrics import SegmentationMeter  # noqa: E402
from b2seg_wsss.models import DualRouteLinearWSSSModel  # noqa: E402
from scripts.patch_embed_adapt import adapt_vit_patch_embed  # noqa: E402
from scripts.evaluate_crf import build_model_from_checkpoint  # noqa: E402


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


# Ruifrok-Johnston H&E-DAB stain matrix, same convention as skimage.color.rgb2hed.
RGB_FROM_HED = np.array(
    [[0.65, 0.70, 0.29],
     [0.07, 0.99, 0.11],
     [0.27, 0.57, 0.78]],
    dtype=np.float32,
)
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED).astype(np.float32)


def hed_jitter_tensor(
    images: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    prob: float,
    sigma_alpha: float,
    sigma_beta: float,
) -> torch.Tensor:
    """Per-image H&E stain jitter on an already-normalized batch.

    Brightness/contrast/noise are generic perturbations the model is largely invariant
    to already (consistency agreement saturates near 1.0). Stain variation is the real
    cross-institution shift in H&E, so it is what the consistency branch should defend against.
    """
    if prob <= 0.0:
        return images
    batch = images.shape[0]
    device, dtype = images.device, images.dtype
    mean_t = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
    hed_from_rgb = torch.from_numpy(HED_FROM_RGB).to(device=device, dtype=dtype)
    rgb_from_hed = torch.from_numpy(RGB_FROM_HED).to(device=device, dtype=dtype)

    rgb = ((images * std_t + mean_t).clamp(0.0, 1.0) * 255.0)
    optical_density = -torch.log((rgb + 1.0) / 256.0)
    stains = torch.einsum("bchw,cd->bdhw", optical_density, hed_from_rgb)

    alpha = torch.empty(batch, 3, 1, 1, device=device, dtype=dtype).uniform_(
        1.0 - float(sigma_alpha), 1.0 + float(sigma_alpha)
    )
    beta = torch.empty(batch, 3, 1, 1, device=device, dtype=dtype).uniform_(
        -float(sigma_beta), float(sigma_beta)
    )
    # H&E carries no DAB, so leave the residual channel alone.
    alpha[:, 2] = 1.0
    beta[:, 2] = 0.0
    apply = (torch.rand(batch, 1, 1, 1, device=device) < float(prob)).to(dtype)
    alpha = alpha * apply + torch.ones_like(alpha) * (1.0 - apply)
    beta = beta * apply

    jittered = torch.einsum("bchw,cd->bdhw", stains * alpha + beta, rgb_from_hed)
    rgb_out = (256.0 * torch.exp(-jittered) - 1.0).clamp(0.0, 255.0) / 255.0
    return (rgb_out - mean_t) / std_t


def strong_augment_tensor(
    images: torch.Tensor,
    brightness: float,
    contrast: float,
    noise: float,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    hed_prob: float = 0.0,
    hed_sigma_alpha: float = 0.05,
    hed_sigma_beta: float = 0.05,
) -> torch.Tensor:
    out = images
    if hed_prob > 0.0:
        if mean is None or std is None:
            raise ValueError("HED stain jitter needs the normalization statistics.")
        out = hed_jitter_tensor(out, mean, std, hed_prob, hed_sigma_alpha, hed_sigma_beta)
    if brightness > 0.0:
        shift = torch.empty(out.shape[0], 1, 1, 1, device=out.device, dtype=out.dtype)
        shift.uniform_(-float(brightness), float(brightness))
        out = out + shift
    if contrast > 0.0:
        mean = out.mean(dim=(2, 3), keepdim=True)
        scale = torch.empty(out.shape[0], 1, 1, 1, device=out.device, dtype=out.dtype)
        scale.uniform_(1.0 - float(contrast), 1.0 + float(contrast))
        out = (out - mean) * scale + mean
    if noise > 0.0:
        out = out + torch.randn_like(out) * float(noise)
    return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def ramp_value(epoch: int, warmup_epochs: float, ramp_epochs: float) -> float:
    if epoch <= float(warmup_epochs):
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return max(0.0, min(1.0, (float(epoch) - float(warmup_epochs)) / float(ramp_epochs)))


def interpolate_thresholds(start: torch.Tensor, end: torch.Tensor, scale: float) -> torch.Tensor:
    scale = max(0.0, min(1.0, float(scale)))
    return start.to(end.device, end.dtype) * (1.0 - scale) + end * scale


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


def parse_thresholds(value: str, num_classes: int) -> torch.Tensor:
    items = [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if len(items) == 1:
        items = items * num_classes
    if len(items) != num_classes:
        raise ValueError(f"Expected 1 or {num_classes} thresholds, got {items}")
    return torch.tensor(items, dtype=torch.float32)


def parse_class_values(value: str, num_classes: int, name: str) -> torch.Tensor:
    items = [float(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if len(items) == 1:
        items = items * num_classes
    if len(items) != num_classes:
        raise ValueError(f"Expected 1 or {num_classes} values for {name}, got {items}")
    return torch.tensor(items, dtype=torch.float32)


def apply_interclass_inhibition(
    probs: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    strength: float,
    margin: float,
    temperature: float,
    restrict_present: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if mode == "none":
        zeros = {
            "inhibition_ambiguous": 0.0,
            "inhibition_changed_winner": 0.0,
            "inhibition_score_drop": 0.0,
            "inhibition_seed_fraction": 0.0,
            "inhibition_centroid_class_fraction": 0.0,
        }
        return probs, zeros
    if mode == "feature_centroid":
        zeros = {
            "inhibition_ambiguous": 0.0,
            "inhibition_changed_winner": 0.0,
            "inhibition_score_drop": 0.0,
            "inhibition_seed_fraction": 0.0,
            "inhibition_centroid_class_fraction": 0.0,
        }
        return probs, zeros
    if mode not in {"margin", "subtract", "soft_margin"}:
        raise ValueError(f"Unknown pseudo inhibition mode: {mode}")

    raw_conf, raw_winner = probs.max(dim=-1)
    sorted_probs = probs.sort(dim=-1, descending=True).values
    top1 = sorted_probs[..., 0]
    top2 = sorted_probs[..., 1] if probs.shape[-1] > 1 else torch.zeros_like(top1)
    top_margin = top1 - top2

    other_max = []
    for class_idx in range(probs.shape[-1]):
        if probs.shape[-1] == 1:
            other_max.append(torch.zeros_like(probs[..., class_idx]))
            continue
        mask = torch.ones(probs.shape[-1], dtype=torch.bool, device=probs.device)
        mask[class_idx] = False
        other_max.append(probs[..., mask].max(dim=-1).values)
    other = torch.stack(other_max, dim=-1)
    class_margin = probs - other

    if mode == "margin":
        inhibited = probs.masked_fill(class_margin < float(margin), 0.0)
    elif mode == "subtract":
        inhibited = (probs - float(strength) * other).clamp_min(0.0)
    else:
        temp = max(float(temperature), 1e-4)
        gate = torch.sigmoid((class_margin - float(margin)) / temp)
        inhibited = probs * ((1.0 - float(strength)) + float(strength) * gate)

    if restrict_present:
        inhibited = inhibited.masked_fill(labels[:, None, :] <= 0, 0.0)

    inhibited_conf, inhibited_winner = inhibited.max(dim=-1)
    present_token = labels.gather(dim=1, index=raw_winner.clamp_min(0)).bool() if restrict_present else torch.ones_like(raw_winner, dtype=torch.bool)
    denom = present_token.float().sum().clamp_min(1.0)
    ambiguous = ((top_margin < float(margin)) & present_token).float().sum() / denom
    changed = ((inhibited_winner != raw_winner) & present_token).float().sum() / denom
    score_drop = ((raw_conf - inhibited_conf).clamp_min(0.0) * present_token.float()).sum() / denom
    stats = {
        "inhibition_ambiguous": float(ambiguous.detach().cpu()),
        "inhibition_changed_winner": float(changed.detach().cpu()),
        "inhibition_score_drop": float(score_drop.detach().cpu()),
        "inhibition_seed_fraction": 0.0,
        "inhibition_centroid_class_fraction": 0.0,
    }
    return inhibited, stats


def apply_feature_centroid_inhibition(
    probs: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
    tokens: torch.Tensor | None,
    strength: float,
    margin: float,
    temperature: float,
    restrict_present: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    if tokens is None:
        raise ValueError("pseudo-inhibition=feature_centroid requires inhibition tokens.")
    if tokens.shape[:2] != probs.shape[:2]:
        raise ValueError(f"Feature tokens shape {tuple(tokens.shape)} does not match probs {tuple(probs.shape)}")

    raw_conf, raw_winner = probs.max(dim=-1)
    base_thresholds = thresholds.to(probs.device).view(1, 1, -1).expand(probs.shape).gather(
        dim=-1,
        index=raw_winner.unsqueeze(-1),
    ).squeeze(-1)
    seed = raw_conf >= base_thresholds
    if restrict_present:
        seed = seed & labels.gather(dim=1, index=raw_winner.clamp_min(0)).bool()

    norm_tokens = F.normalize(tokens.detach().float(), dim=-1)
    batch, _num_tokens, num_classes = probs.shape
    centroids = torch.zeros(batch, num_classes, norm_tokens.shape[-1], device=probs.device, dtype=norm_tokens.dtype)
    has_centroid = torch.zeros(batch, num_classes, device=probs.device, dtype=torch.bool)
    for batch_idx in range(batch):
        present_classes = torch.nonzero(labels[batch_idx] > 0, as_tuple=False).flatten()
        for class_idx in present_classes.tolist():
            class_seed = seed[batch_idx] & (raw_winner[batch_idx] == class_idx)
            if bool(class_seed.any()):
                centroids[batch_idx, class_idx] = F.normalize(norm_tokens[batch_idx, class_seed].mean(dim=0), dim=0)
                has_centroid[batch_idx, class_idx] = True

    affinity = torch.einsum("bnd,bcd->bnc", norm_tokens, centroids)
    affinity = affinity.masked_fill(~has_centroid[:, None, :], -1.0)
    if restrict_present:
        affinity = affinity.masked_fill(labels[:, None, :] <= 0, -1.0)

    other_max = []
    for class_idx in range(num_classes):
        if num_classes == 1:
            other_max.append(torch.full_like(affinity[..., class_idx], -1.0))
            continue
        mask = torch.ones(num_classes, dtype=torch.bool, device=probs.device)
        mask[class_idx] = False
        other_max.append(affinity[..., mask].max(dim=-1).values)
    other = torch.stack(other_max, dim=-1)
    affinity_margin = affinity - other

    temp = max(float(temperature), 1e-4)
    gate = torch.sigmoid((affinity_margin - float(margin)) / temp)
    adjusted = probs * ((1.0 - float(strength)) + float(strength) * gate)
    adjusted = torch.where(has_centroid[:, None, :], adjusted, probs)
    if restrict_present:
        adjusted = adjusted.masked_fill(labels[:, None, :] <= 0, 0.0)

    inhibited_conf, inhibited_winner = adjusted.max(dim=-1)
    present_token = labels.gather(dim=1, index=raw_winner.clamp_min(0)).bool() if restrict_present else torch.ones_like(raw_winner, dtype=torch.bool)
    raw_has = has_centroid.gather(dim=1, index=raw_winner.clamp_min(0)).bool()
    valid = present_token & raw_has
    denom = valid.float().sum().clamp_min(1.0)
    raw_affinity_margin = affinity_margin.gather(dim=-1, index=raw_winner.unsqueeze(-1)).squeeze(-1)
    ambiguous = ((raw_affinity_margin < float(margin)) & valid).float().sum() / denom
    changed = ((inhibited_winner != raw_winner) & valid).float().sum() / denom
    score_drop = ((raw_conf - inhibited_conf).clamp_min(0.0) * valid.float()).sum() / denom
    stats = {
        "inhibition_ambiguous": float(ambiguous.detach().cpu()),
        "inhibition_changed_winner": float(changed.detach().cpu()),
        "inhibition_score_drop": float(score_drop.detach().cpu()),
        "inhibition_seed_fraction": float(seed.float().mean().detach().cpu()),
        "inhibition_centroid_class_fraction": float(has_centroid.float().mean().detach().cpu()),
    }
    return adjusted, stats


def logits_to_segmentation(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def route_summary(model: DualRouteLinearWSSSModel) -> dict[str, object]:
    with torch.no_grad():
        layers = torch.tensor(model.route_layers, dtype=torch.float32)
        sem = model.semantic_route_logits.softmax(dim=-1).detach().cpu()
        spa = model.spatial_route_logits.softmax(dim=-1).detach().cpu()
        out: dict[str, object] = {
            "route_layers": model.route_layers,
            "semantic_route_weights": sem.tolist(),
            "semantic_expected_layer": (sem * layers.view(1, -1)).sum(dim=-1).tolist(),
            "spatial_route_weights": spa.tolist(),
            "spatial_expected_layer": (spa * layers.view(1, -1)).sum(dim=-1).tolist(),
        }
        if getattr(model, "output_fuse_alpha_logits", None) is not None:
            out["output_fuse_alpha"] = model.output_fuse_alpha_logits.sigmoid().detach().cpu().tolist()
        return out


def lock_fixed_spatial_route(model: DualRouteLinearWSSSModel, fixed_layer: int) -> None:
    if fixed_layer not in model.route_layers:
        raise ValueError(f"fixed spatial layer {fixed_layer} not in route layers {model.route_layers}")
    layer_idx = model.route_layers.index(fixed_layer)
    with torch.no_grad():
        model.spatial_route_logits.fill_(-20.0)
        model.spatial_route_logits[:, layer_idx] = 20.0
    model.spatial_route_logits.requires_grad_(False)


def copy_single_semantic_teacher_to_dual_student(
    student: DualRouteLinearWSSSModel,
    teacher_checkpoint_path: str | Path,
    init_spatial_route: bool = False,
) -> None:
    checkpoint = torch.load(teacher_checkpoint_path, map_location="cpu")
    state = checkpoint["model"]
    own = student.state_dict()
    copied = []
    update = {}
    for key, value in state.items():
        if key.startswith("vit.") or key.startswith("layer_norms.") or key == "semantic_route_logits":
            if key in own and own[key].shape == value.shape:
                update[key] = value
                copied.append(key)
    if init_spatial_route and "semantic_route_logits" in state:
        target = "spatial_route_logits"
        if target in own and own[target].shape == state["semantic_route_logits"].shape:
            update[target] = state["semantic_route_logits"]
            copied.append(target)
    if "shared_classifier.weight" in state:
        for target in ("semantic_classifier.weight", "spatial_classifier.weight"):
            if target in own and own[target].shape == state["shared_classifier.weight"].shape:
                update[target] = state["shared_classifier.weight"]
                copied.append(target)
    if "shared_classifier.bias" in state:
        for target in ("semantic_classifier.bias", "spatial_classifier.bias"):
            if target in own and own[target].shape == state["shared_classifier.bias"].shape:
                update[target] = state["shared_classifier.bias"]
                copied.append(target)
    missing, unexpected = student.load_state_dict(update, strict=False)
    print(f"init_from_teacher_copied={len(copied)}", flush=True)
    print(f"init_from_teacher_missing_ignored={len(missing)} unexpected_ignored={len(unexpected)}", flush=True)


@torch.no_grad()
def update_ema(student: torch.nn.Module, teacher: torch.nn.Module, decay: float) -> None:
    student_state = student.state_dict()
    teacher_state = teacher.state_dict()
    for key, teacher_value in teacher_state.items():
        student_value = student_state[key]
        if teacher_value.dtype.is_floating_point:
            teacher_value.mul_(decay).add_(student_value.detach(), alpha=1.0 - decay)
        else:
            teacher_value.copy_(student_value)


@torch.no_grad()
def cap_pseudo_keep(
    keep: torch.Tensor,
    pseudo: torch.Tensor,
    probs: torch.Tensor,
    labels: torch.Tensor,
    max_frac: float,
    mode: str,
    target_slack: float,
    protect_floor_ratio: float,
    rescue_frac: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    max_frac = float(max_frac)
    if max_frac >= 1.0:
        return keep, {"pseudo_cap_removed": 0.0, "pseudo_cap_target": 1.0}
    if max_frac <= 0.0:
        removed = keep.float().mean()
        return torch.zeros_like(keep), {"pseudo_cap_removed": float(removed.detach().cpu()), "pseudo_cap_target": 0.0}
    if mode not in {"global", "target", "protected", "protected_rescue"}:
        raise ValueError(f"Unknown pseudo kept cap mode: {mode}")

    capped = keep.clone()
    selected_conf = probs.gather(dim=-1, index=pseudo.unsqueeze(-1)).squeeze(-1)
    batch, num_tokens = keep.shape
    max_keep_total = max(1, int(round(num_tokens * max_frac)))

    if mode == "global":
        for batch_idx in range(batch):
            valid_idx = torch.nonzero(capped[batch_idx], as_tuple=False).flatten()
            if valid_idx.numel() <= max_keep_total:
                continue
            scores = selected_conf[batch_idx, valid_idx]
            keep_local = scores.topk(max_keep_total).indices
            next_mask = torch.zeros_like(capped[batch_idx])
            next_mask[valid_idx[keep_local]] = True
            capped[batch_idx] = next_mask
    elif mode == "target":
        class_fraction = []
        for class_idx in range(labels.shape[1]):
            class_fraction.append(((pseudo == class_idx) & capped).float().mean())
        class_fraction_tensor = torch.stack(class_fraction)
        target_fraction = target_fraction_from_labels(labels, class_fraction_tensor, 0.0)
        class_caps = ((target_fraction + float(target_slack)).clamp_min(0.0) * num_tokens).ceil().to(torch.long)
        class_caps = class_caps.clamp_min(1)
        for batch_idx in range(batch):
            next_mask = capped[batch_idx].clone()
            present_classes = torch.nonzero(labels[batch_idx] > 0, as_tuple=False).flatten()
            for class_idx in present_classes.tolist():
                class_idx_tensor = torch.nonzero(
                    capped[batch_idx] & (pseudo[batch_idx] == class_idx),
                    as_tuple=False,
                ).flatten()
                cap = int(class_caps[class_idx].item())
                if class_idx_tensor.numel() <= cap:
                    continue
                scores = selected_conf[batch_idx, class_idx_tensor]
                keep_local = scores.topk(cap).indices
                drop_mask = torch.ones(class_idx_tensor.numel(), dtype=torch.bool, device=keep.device)
                drop_mask[keep_local] = False
                next_mask[class_idx_tensor[drop_mask]] = False
            valid_idx = torch.nonzero(next_mask, as_tuple=False).flatten()
            if valid_idx.numel() > max_keep_total:
                scores = selected_conf[batch_idx, valid_idx]
                keep_local = scores.topk(max_keep_total).indices
                final_mask = torch.zeros_like(next_mask)
                final_mask[valid_idx[keep_local]] = True
                next_mask = final_mask
            capped[batch_idx] = next_mask
    else:
        class_fraction = []
        for class_idx in range(labels.shape[1]):
            class_fraction.append(((pseudo == class_idx) & capped).float().mean())
        class_fraction_tensor = torch.stack(class_fraction)
        target_fraction = target_fraction_from_labels(labels, class_fraction_tensor, 0.0)
        floor_ratio = max(0.0, min(1.0, float(protect_floor_ratio)))
        class_floors = (target_fraction.clamp_min(0.0) * floor_ratio * num_tokens).ceil().to(torch.long).clamp_min(1)
        for batch_idx in range(batch):
            present_classes = torch.nonzero(labels[batch_idx] > 0, as_tuple=False).flatten()
            next_mask = capped[batch_idx].clone()
            while int(next_mask.sum().item()) > max_keep_total:
                class_counts = torch.stack(
                    [((pseudo[batch_idx] == class_idx) & next_mask).sum() for class_idx in range(labels.shape[1])]
                ).to(torch.float32)
                pressure = class_counts / float(num_tokens) - target_fraction.to(class_counts.device)
                eligible = torch.zeros_like(pressure, dtype=torch.bool)
                for class_idx in present_classes.tolist():
                    eligible[class_idx] = class_counts[class_idx] > class_floors[class_idx].to(class_counts.device)
                if not bool(eligible.any()):
                    valid_idx = torch.nonzero(next_mask, as_tuple=False).flatten()
                    drop_idx = valid_idx[selected_conf[batch_idx, valid_idx].argmin()]
                    next_mask[drop_idx] = False
                    continue
                pressure = pressure.masked_fill(~eligible, -1e6)
                drop_class = int(pressure.argmax().item())
                class_idx_tensor = torch.nonzero(
                    next_mask & (pseudo[batch_idx] == drop_class),
                    as_tuple=False,
                ).flatten()
                if class_idx_tensor.numel() == 0:
                    valid_idx = torch.nonzero(next_mask, as_tuple=False).flatten()
                    drop_idx = valid_idx[selected_conf[batch_idx, valid_idx].argmin()]
                    next_mask[drop_idx] = False
                    continue
                drop_idx = class_idx_tensor[selected_conf[batch_idx, class_idx_tensor].argmin()]
                next_mask[drop_idx] = False

            if mode == "protected_rescue" and rescue_frac > 0.0:
                rescue_budget = int(round(num_tokens * float(rescue_frac)))
                rescue_budget = max(0, rescue_budget)
                for _ in range(rescue_budget):
                    class_counts = torch.stack(
                        [((pseudo[batch_idx] == class_idx) & next_mask).sum() for class_idx in range(labels.shape[1])]
                    ).to(torch.float32)
                    class_frac = class_counts / float(num_tokens)
                    deficit = target_fraction.to(class_frac.device) - class_frac
                    present_mask = torch.zeros_like(deficit, dtype=torch.bool)
                    for class_idx in present_classes.tolist():
                        present_mask[class_idx] = True
                    deficit = deficit.masked_fill(~present_mask, -1e6)
                    rescue_class = int(deficit.argmax().item())
                    if float(deficit[rescue_class].item()) <= float(target_slack):
                        break
                    add_idx = torch.nonzero(
                        (~next_mask) & keep[batch_idx] & (pseudo[batch_idx] == rescue_class),
                        as_tuple=False,
                    ).flatten()
                    if add_idx.numel() == 0:
                        break
                    add_token = add_idx[selected_conf[batch_idx, add_idx].argmax()]

                    surplus = class_frac - target_fraction.to(class_frac.device)
                    surplus[rescue_class] = -1e6
                    surplus = surplus.masked_fill(~present_mask, -1e6)
                    drop_class = int(surplus.argmax().item())
                    if float(surplus[drop_class].item()) <= 0.0:
                        break
                    drop_idx = torch.nonzero(
                        next_mask & (pseudo[batch_idx] == drop_class),
                        as_tuple=False,
                    ).flatten()
                    if drop_idx.numel() == 0:
                        break
                    drop_token = drop_idx[selected_conf[batch_idx, drop_idx].argmin()]
                    next_mask[drop_token] = False
                    next_mask[add_token] = True
            capped[batch_idx] = next_mask

    before = keep.float().mean()
    after = capped.float().mean()
    return capped, {
        "pseudo_cap_removed": float((before - after).clamp_min(0.0).detach().cpu()),
        "pseudo_cap_target": max_frac,
        "pseudo_cap_protect_floor": float(protect_floor_ratio),
        "pseudo_cap_rescue": float(rescue_frac if mode == "protected_rescue" else 0.0),
    }


@torch.no_grad()
def compute_affinity_walk_scores(
    probs: torch.Tensor,
    labels: torch.Tensor,
    pseudo: torch.Tensor,
    keep: torch.Tensor,
    tokens: torch.Tensor,
    steps: int,
    gamma: float,
    self_loop: float,
    contrast_beta: float,
    restrict_present: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    steps = max(1, int(steps))
    gamma = max(0.1, float(gamma))
    self_loop = max(0.0, float(self_loop))
    contrast_beta = max(0.0, float(contrast_beta))

    batch, num_tokens, num_classes = probs.shape
    norm_tokens = F.normalize(tokens.detach().float(), dim=-1)
    walked_batches = []
    walk_score_sum = torch.zeros((), device=probs.device, dtype=torch.float32)
    walk_score_count = torch.zeros((), device=probs.device, dtype=torch.float32)
    for batch_idx in range(batch):
        present_classes = torch.nonzero(labels[batch_idx] > 0, as_tuple=False).flatten()
        if present_classes.numel() == 0:
            walked_batches.append(torch.zeros(num_tokens, num_classes, device=probs.device, dtype=torch.float32))
            continue

        affinity = torch.mm(norm_tokens[batch_idx], norm_tokens[batch_idx].t()).clamp_min(0.0)
        if gamma != 1.0:
            affinity = affinity.pow(gamma)
        if self_loop > 0.0:
            affinity = affinity + torch.eye(num_tokens, device=affinity.device, dtype=affinity.dtype) * self_loop
        transition = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        seeds = torch.zeros(num_tokens, num_classes, device=probs.device, dtype=torch.float32)
        for class_idx in present_classes.tolist():
            seed_mask = (pseudo[batch_idx] == class_idx) & keep[batch_idx]
            if bool(seed_mask.any()):
                seeds[seed_mask, class_idx] = probs[batch_idx, seed_mask, class_idx].float().clamp_min(1e-6)
        if float(seeds.sum().item()) <= 0.0:
            walked_batches.append(torch.zeros(num_tokens, num_classes, device=probs.device, dtype=torch.float32))
            continue

        walked = seeds
        for _ in range(steps):
            walked = torch.mm(transition, walked)
        walked = walked / walked.amax(dim=0, keepdim=True).clamp_min(1e-6)
        if contrast_beta > 0.0 and num_classes > 1:
            other = []
            for class_idx in range(num_classes):
                mask = torch.ones(num_classes, dtype=torch.bool, device=probs.device)
                mask[class_idx] = False
                other.append(walked[:, mask].max(dim=-1).values)
            other_walked = torch.stack(other, dim=-1)
            walked = (walked - contrast_beta * other_walked).clamp_min(0.0)
        if restrict_present:
            walked = walked.masked_fill(labels[batch_idx].view(1, -1) <= 0, 0.0)
        walked_batches.append(walked)
        seed_any = seeds.sum(dim=-1) > 0
        if bool(seed_any.any()):
            walk_score_sum = walk_score_sum + walked[seed_any].max(dim=-1).values.sum()
            walk_score_count = walk_score_count + float(seed_any.sum().item())

    walked_scores = torch.stack(walked_batches, dim=0)
    return walked_scores.to(probs.dtype), {
        "affinity_walk_delta": 0.0,
        "affinity_walk_seed": float(keep.float().mean().detach().cpu()),
        "affinity_walk_score": float((walk_score_sum / walk_score_count.clamp_min(1.0)).detach().cpu()),
    }


@torch.no_grad()
def apply_affinity_walk_refinement(
    probs: torch.Tensor,
    labels: torch.Tensor,
    pseudo: torch.Tensor,
    keep: torch.Tensor,
    tokens: torch.Tensor,
    steps: int,
    alpha: float,
    gamma: float,
    self_loop: float,
    contrast_beta: float,
    restrict_present: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    alpha = max(0.0, min(1.0, float(alpha)))
    if alpha <= 0.0:
        return probs, {
            "affinity_walk_delta": 0.0,
            "affinity_walk_seed": float(keep.float().mean().detach().cpu()),
            "affinity_walk_score": 0.0,
        }
    walked_scores, stats = compute_affinity_walk_scores(
        probs,
        labels,
        pseudo,
        keep,
        tokens,
        steps,
        gamma,
        self_loop,
        contrast_beta,
        restrict_present,
    )
    refined = (1.0 - alpha) * probs.float() + alpha * walked_scores.float()
    if restrict_present:
        refined = refined.masked_fill(labels[:, None, :] <= 0, 0.0)
    delta = (refined.float() - probs.float()).abs().mean()
    stats["affinity_walk_delta"] = float(delta.detach().cpu())
    return refined.to(probs.dtype), {
        "affinity_walk_delta": float(delta.detach().cpu()),
        "affinity_walk_seed": stats["affinity_walk_seed"],
        "affinity_walk_score": stats["affinity_walk_score"],
    }


@torch.no_grad()
def make_pseudo(
    logits: torch.Tensor,
    labels: torch.Tensor,
    thresholds: torch.Tensor,
    score: str,
    restrict_present: bool,
    ignore_index: int,
    expand_mode: str = "fixed",
    expand_min_frac: float = 0.0,
    expand_min_score: float = 0.0,
    expand_max_frac: float = 0.06,
    expand_margin_min: float = 0.05,
    expand_under_strength: float = 1.0,
    reliability_tokens: torch.Tensor | None = None,
    jepa_reliability: torch.Tensor | None = None,
    jepa_completed_probs: torch.Tensor | None = None,
    jepa_affinity_tokens: torch.Tensor | None = None,
    jepa_completed_min_score: float = 0.0,
    jepa_completed_mix_alpha: float = 0.5,
    jepa_completed_class_alpha: torch.Tensor | None = None,
    reliability_min: float = 0.0,
    expand_affinity_alpha: float = 0.3,
    expand_affinity_beta: float = 0.2,
    expand_affinity_margin: float = 0.0,
    inhibition_mode: str = "none",
    inhibition_strength: float = 0.5,
    inhibition_margin: float = 0.05,
    inhibition_temperature: float = 0.05,
    inhibition_tokens: torch.Tensor | None = None,
    kept_cap_max_frac: float = 1.0,
    kept_cap_mode: str = "global",
    kept_cap_target_slack: float = 0.02,
    kept_cap_protect_floor_ratio: float = 0.5,
    kept_cap_rescue_frac: float = 0.0,
    affinity_walk_steps: int = 2,
    affinity_walk_alpha: float = 0.4,
    affinity_walk_gamma: float = 2.0,
    affinity_walk_self_loop: float = 1.0,
    affinity_walk_contrast_beta: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    if restrict_present:
        absent = labels <= 0
        logits = logits.masked_fill(absent[:, None, :], -1e4)
    if score == "softmax":
        probs = logits.softmax(dim=-1)
    elif score == "sigmoid":
        probs = logits.sigmoid()
    else:
        raise ValueError(f"Unknown pseudo score: {score}")
    if inhibition_mode == "feature_centroid":
        probs, inhibition_stats = apply_feature_centroid_inhibition(
            probs,
            labels,
            thresholds,
            inhibition_tokens,
            inhibition_strength,
            inhibition_margin,
            inhibition_temperature,
            restrict_present,
        )
    else:
        probs, inhibition_stats = apply_interclass_inhibition(
            probs,
            labels,
            inhibition_mode,
            inhibition_strength,
            inhibition_margin,
            inhibition_temperature,
            restrict_present,
        )
    conf, pseudo = probs.max(dim=-1)
    class_thresholds = thresholds.to(logits.device).view(1, 1, -1).expand(logits.shape).gather(
        dim=-1,
        index=pseudo.unsqueeze(-1),
    ).squeeze(-1)
    keep = conf >= class_thresholds
    if restrict_present:
        keep = keep & labels.gather(dim=1, index=pseudo.clamp_min(0)).bool()
    affinity_walk_stats = {"affinity_walk_delta": 0.0, "affinity_walk_seed": 0.0, "affinity_walk_score": 0.0}
    affinity_candidate_scores = None
    if expand_mode in {"affinity_walk", "affinity_walk_contrast"}:
        if reliability_tokens is None:
            raise ValueError(f"{expand_mode} requires reliability_tokens.")
        contrast_beta = affinity_walk_contrast_beta if expand_mode == "affinity_walk_contrast" else 0.0
        probs, affinity_walk_stats = apply_affinity_walk_refinement(
            probs,
            labels,
            pseudo,
            keep,
            reliability_tokens,
            affinity_walk_steps,
            affinity_walk_alpha,
            affinity_walk_gamma,
            affinity_walk_self_loop,
            contrast_beta,
            restrict_present,
        )
        conf, pseudo = probs.max(dim=-1)
        class_thresholds = thresholds.to(logits.device).view(1, 1, -1).expand(logits.shape).gather(
            dim=-1,
            index=pseudo.unsqueeze(-1),
        ).squeeze(-1)
        keep = conf >= class_thresholds
        if restrict_present:
            keep = keep & labels.gather(dim=1, index=pseudo.clamp_min(0)).bool()
    elif expand_mode in {"affinity_candidate", "affinity_candidate_contrast"}:
        if reliability_tokens is None:
            raise ValueError(f"{expand_mode} requires reliability_tokens.")
        contrast_beta = affinity_walk_contrast_beta if expand_mode == "affinity_candidate_contrast" else 0.0
        affinity_candidate_scores, affinity_walk_stats = compute_affinity_walk_scores(
            probs,
            labels,
            pseudo,
            keep,
            reliability_tokens,
            affinity_walk_steps,
            affinity_walk_gamma,
            affinity_walk_self_loop,
            contrast_beta,
            restrict_present,
        )
    expanded = torch.zeros_like(keep)
    alpha_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    alpha_count = torch.zeros((), device=logits.device, dtype=torch.float32)
    affinity_margin_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    affinity_margin_count = torch.zeros((), device=logits.device, dtype=torch.float32)
    expand_min_frac = max(0.0, float(expand_min_frac))
    if expand_min_frac > 0.0:
        if expand_mode not in {
            "fixed",
            "minimal_teacher",
            "minimal_jepa",
            "minimal_mix",
            "minimal_mix_adaptive",
            "minimal_mix_class_adaptive",
            "minimal_mix_geometric",
            "adaptive_margin",
            "adaptive_margin_reliability",
            "feature_affinity",
            "feature_affinity_margin",
            "affinity_walk",
            "affinity_walk_contrast",
            "affinity_candidate",
            "affinity_candidate_contrast",
            "jepa_affinity",
            "jepa_contrast",
            "online_jepa_affinity",
            "online_jepa_contrast",
            "affinity_walk",
            "affinity_walk_contrast",
            "adaptive_margin_jepa",
            "adaptive_margin_jepa_completed",
        }:
            raise ValueError(f"Unknown pseudo expand mode: {expand_mode}")
        num_tokens = logits.shape[1]
        class_scores = probs if score in {"softmax", "sigmoid"} else logits
        sorted_probs, _sorted_idx = probs.sort(dim=-1, descending=True)
        margins = sorted_probs[..., 0] - sorted_probs[..., 1]
        current_fraction = []
        for class_idx in range(labels.shape[1]):
            current_fraction.append(((pseudo == class_idx) & keep).float().mean())
        current_fraction_tensor = torch.stack(current_fraction)
        target_fraction = target_fraction_from_labels(labels, current_fraction_tensor, 0.0)
        under = (target_fraction / current_fraction_tensor.clamp_min(1e-6)).pow(float(expand_under_strength))
        adaptive_budget = (float(expand_min_frac) * under).clamp(0.0, float(expand_max_frac))
        adaptive_budget = torch.where(current_fraction_tensor < target_fraction, adaptive_budget, torch.zeros_like(adaptive_budget))
        norm_tokens = None
        if expand_mode in {"adaptive_margin_reliability", "feature_affinity", "feature_affinity_margin"}:
            if reliability_tokens is None:
                raise ValueError(f"{expand_mode} requires reliability_tokens.")
            norm_tokens = F.normalize(reliability_tokens.detach().float(), dim=-1)
        if expand_mode in {"jepa_affinity", "jepa_contrast", "online_jepa_affinity", "online_jepa_contrast"}:
            if jepa_affinity_tokens is None:
                raise ValueError(f"{expand_mode} requires jepa_affinity_tokens.")
            norm_tokens = F.normalize(jepa_affinity_tokens.detach().float(), dim=-1)
        if expand_mode == "adaptive_margin_jepa" and jepa_reliability is None:
            raise ValueError("adaptive_margin_jepa requires jepa_reliability.")
        jepa_completed_modes = {
            "adaptive_margin_jepa_completed",
            "minimal_jepa",
            "minimal_mix",
            "minimal_mix_adaptive",
            "minimal_mix_class_adaptive",
            "minimal_mix_geometric",
        }
        if expand_mode in jepa_completed_modes and jepa_completed_probs is None:
            raise ValueError(f"{expand_mode} requires jepa_completed_probs.")
        if expand_mode == "minimal_mix_class_adaptive" and jepa_completed_class_alpha is None:
            raise ValueError("minimal_mix_class_adaptive requires jepa_completed_class_alpha.")
        for batch_idx in range(logits.shape[0]):
            present_classes = torch.nonzero(labels[batch_idx] > 0, as_tuple=False).flatten()
            for class_idx in present_classes.tolist():
                if expand_mode in {
                    "fixed",
                    "minimal_teacher",
                    "minimal_jepa",
                    "minimal_mix",
                    "minimal_mix_adaptive",
                    "minimal_mix_class_adaptive",
                    "minimal_mix_geometric",
                }:
                    target_k = max(1, int(round(num_tokens * expand_min_frac)))
                else:
                    target_k = int(round(num_tokens * float(adaptive_budget[class_idx].detach().cpu())))
                target_k = min(max(0, target_k), num_tokens)
                if target_k <= 0:
                    continue
                current = ((pseudo[batch_idx] == class_idx) & keep[batch_idx]).sum()
                if expand_mode in {
                    "minimal_teacher",
                    "minimal_jepa",
                    "minimal_mix",
                    "minimal_mix_adaptive",
                    "minimal_mix_class_adaptive",
                    "minimal_mix_geometric",
                }:
                    needed = target_k
                else:
                    needed = int(max(0, target_k - int(current.item())))
                if needed <= 0:
                    continue
                candidates = class_scores[batch_idx, :, class_idx]
                candidates = candidates.masked_fill(keep[batch_idx], -1.0)
                if expand_min_score > 0.0 and expand_mode not in {
                    "minimal_teacher",
                    "minimal_jepa",
                    "minimal_mix",
                    "minimal_mix_adaptive",
                    "minimal_mix_class_adaptive",
                    "minimal_mix_geometric",
                }:
                    candidates = candidates.masked_fill(candidates < float(expand_min_score), -1.0)
                if expand_mode in {
                    "adaptive_margin",
                    "adaptive_margin_reliability",
                    "feature_affinity",
                    "feature_affinity_margin",
                    "jepa_affinity",
                    "jepa_contrast",
                    "online_jepa_affinity",
                    "online_jepa_contrast",
                    "affinity_walk",
                    "affinity_walk_contrast",
                    "affinity_candidate",
                    "affinity_candidate_contrast",
                    "adaptive_margin_jepa",
                    "adaptive_margin_jepa_completed",
                }:
                    candidates = candidates.masked_fill(margins[batch_idx] < float(expand_margin_min), -1.0)
                if expand_mode == "adaptive_margin_reliability" and norm_tokens is not None:
                    seed_mask = (pseudo[batch_idx] == class_idx) & keep[batch_idx]
                    if not bool(seed_mask.any()):
                        continue
                    centroid = F.normalize(norm_tokens[batch_idx, seed_mask].mean(dim=0), dim=0)
                    reliability = torch.mv(norm_tokens[batch_idx], centroid)
                    candidates = candidates.masked_fill(reliability < float(reliability_min), -1.0)
                elif expand_mode == "adaptive_margin_jepa" and jepa_reliability is not None:
                    reliability = jepa_reliability[batch_idx].to(candidates.device, candidates.dtype)
                    candidates = candidates.masked_fill(reliability < float(reliability_min), -1.0)
                elif expand_mode == "adaptive_margin_jepa_completed" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    if jepa_completed_min_score > 0.0:
                        candidates = candidates.masked_fill(completed < float(jepa_completed_min_score), -1.0)
                available = int((candidates >= 0.0).sum().item())
                if available <= 0:
                    continue
                topk = min(needed, available)
                if expand_mode in {"adaptive_margin", "adaptive_margin_reliability", "affinity_walk", "affinity_walk_contrast"}:
                    rank_score = candidates * margins[batch_idx].clamp_min(0.0)
                    idx = rank_score.topk(topk).indices
                elif expand_mode in {"affinity_candidate", "affinity_candidate_contrast"} and affinity_candidate_scores is not None:
                    walked = affinity_candidate_scores[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    rank_score = (
                        candidates * margins[batch_idx].clamp_min(0.0)
                        + float(expand_affinity_alpha) * walked
                    )
                    rank_score = rank_score.masked_fill(candidates < 0.0, -1.0)
                    idx = rank_score.topk(topk).indices
                    affinity_margin_sum = affinity_margin_sum + walked[idx].sum().float()
                    affinity_margin_count = affinity_margin_count + float(topk)
                elif expand_mode in {
                    "feature_affinity",
                    "feature_affinity_margin",
                    "jepa_affinity",
                    "jepa_contrast",
                    "online_jepa_affinity",
                    "online_jepa_contrast",
                } and norm_tokens is not None:
                    seed_mask = (pseudo[batch_idx] == class_idx) & keep[batch_idx]
                    if not bool(seed_mask.any()):
                        rank_score = candidates * margins[batch_idx].clamp_min(0.0)
                        idx = rank_score.topk(topk).indices
                    else:
                        token_bank = norm_tokens[batch_idx]
                        centroid = F.normalize(token_bank[seed_mask].mean(dim=0), dim=0)
                        own_affinity = torch.mv(token_bank, centroid).to(candidates.device, candidates.dtype)
                        other_affinity = torch.zeros_like(own_affinity)
                        other_count = 0
                        for other_class in present_classes.tolist():
                            if other_class == class_idx:
                                continue
                            other_seed = (pseudo[batch_idx] == other_class) & keep[batch_idx]
                            if not bool(other_seed.any()):
                                continue
                            other_centroid = F.normalize(token_bank[other_seed].mean(dim=0), dim=0)
                            other_score = torch.mv(token_bank, other_centroid).to(candidates.device, candidates.dtype)
                            other_affinity = torch.maximum(other_affinity, other_score)
                            other_count += 1
                        affinity_margin = own_affinity - other_affinity if other_count > 0 else own_affinity
                        if expand_mode in {"feature_affinity_margin", "jepa_contrast", "online_jepa_contrast"}:
                            candidates = candidates.masked_fill(affinity_margin < float(expand_affinity_margin), -1.0)
                        beta = (
                            float(expand_affinity_beta)
                            if expand_mode in {"feature_affinity", "feature_affinity_margin", "jepa_contrast", "online_jepa_contrast"}
                            else 0.0
                        )
                        rank_score = (
                            candidates * margins[batch_idx].clamp_min(0.0)
                            + float(expand_affinity_alpha) * own_affinity
                            - beta * other_affinity
                        )
                        rank_score = rank_score.masked_fill(candidates < 0.0, -1.0)
                        available = int((candidates >= 0.0).sum().item())
                        if available <= 0:
                            continue
                        topk = min(topk, available)
                        idx = rank_score.topk(topk).indices
                        affinity_margin_sum = affinity_margin_sum + affinity_margin[idx].sum().float()
                        affinity_margin_count = affinity_margin_count + float(topk)
                elif expand_mode == "adaptive_margin_jepa" and jepa_reliability is not None:
                    rank_score = candidates * margins[batch_idx].clamp_min(0.0) * jepa_reliability[batch_idx].to(candidates.device, candidates.dtype)
                    idx = rank_score.topk(topk).indices
                elif expand_mode == "adaptive_margin_jepa_completed" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    rank_score = candidates * margins[batch_idx].clamp_min(0.0) * completed
                    idx = rank_score.topk(topk).indices
                elif expand_mode == "minimal_teacher":
                    idx = candidates.topk(topk).indices
                elif expand_mode == "minimal_jepa" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    idx = completed.masked_fill(candidates < 0.0, -1.0).topk(topk).indices
                elif expand_mode == "minimal_mix" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    alpha = float(jepa_completed_mix_alpha)
                    alpha = max(0.0, min(1.0, alpha))
                    rank_score = (1.0 - alpha) * candidates + alpha * completed
                    idx = rank_score.masked_fill(candidates < 0.0, -1.0).topk(topk).indices
                elif expand_mode == "minimal_mix_adaptive" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    uncertainty = (1.0 - margins[batch_idx]).clamp(0.0, 1.0)
                    alpha = float(jepa_completed_mix_alpha) * uncertainty * completed.clamp(0.0, 1.0)
                    alpha = alpha.clamp(0.0, 1.0)
                    rank_score = (1.0 - alpha) * candidates + alpha * completed
                    rank_score = rank_score.masked_fill(candidates < 0.0, -1.0)
                    idx = rank_score.topk(topk).indices
                    alpha_sum = alpha_sum + alpha[idx].sum().float()
                    alpha_count = alpha_count + float(topk)
                elif expand_mode == "minimal_mix_class_adaptive" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    uncertainty = (1.0 - margins[batch_idx]).clamp(0.0, 1.0)
                    class_alpha = jepa_completed_class_alpha.to(candidates.device, candidates.dtype)[class_idx]
                    alpha = class_alpha.clamp(0.0, 1.0) * uncertainty * completed.clamp(0.0, 1.0)
                    alpha = alpha.clamp(0.0, 1.0)
                    rank_score = (1.0 - alpha) * candidates + alpha * completed
                    rank_score = rank_score.masked_fill(candidates < 0.0, -1.0)
                    idx = rank_score.topk(topk).indices
                    alpha_sum = alpha_sum + alpha[idx].sum().float()
                    alpha_count = alpha_count + float(topk)
                elif expand_mode == "minimal_mix_geometric" and jepa_completed_probs is not None:
                    completed = jepa_completed_probs[batch_idx, :, class_idx].to(candidates.device, candidates.dtype)
                    uncertainty = (1.0 - margins[batch_idx]).clamp(0.0, 1.0)
                    alpha = float(jepa_completed_mix_alpha) * uncertainty * completed.clamp(0.0, 1.0)
                    alpha = alpha.clamp(0.0, 1.0)
                    agreement = (candidates.clamp_min(0.0) * completed.clamp_min(0.0)).sqrt()
                    rank_score = (1.0 - alpha) * candidates + alpha * agreement
                    rank_score = rank_score.masked_fill(candidates < 0.0, -1.0)
                    idx = rank_score.topk(topk).indices
                    alpha_sum = alpha_sum + alpha[idx].sum().float()
                    alpha_count = alpha_count + float(topk)
                else:
                    idx = candidates.topk(topk).indices
                pseudo[batch_idx, idx] = class_idx
                keep[batch_idx, idx] = True
                expanded[batch_idx, idx] = True
    keep, cap_stats = cap_pseudo_keep(
        keep,
        pseudo,
        probs,
        labels,
        kept_cap_max_frac,
        kept_cap_mode,
        kept_cap_target_slack,
        kept_cap_protect_floor_ratio,
        kept_cap_rescue_frac,
    )
    expanded = expanded & keep
    pseudo = pseudo.masked_fill(~keep, ignore_index)
    valid = pseudo != ignore_index
    per_class = []
    for class_idx in range(labels.shape[1]):
        per_class.append(float((pseudo == class_idx).float().mean().detach().cpu()))
    stats = {
        "pseudo_kept": float(valid.float().mean().detach().cpu()),
        "pseudo_class_fraction": per_class,
        "pseudo_expanded": float(expanded.float().mean().detach().cpu()),
        "jepa_expand_alpha": float((alpha_sum / alpha_count.clamp_min(1.0)).detach().cpu()) if expand_min_frac > 0.0 else 0.0,
        "affinity_expand_margin": float((affinity_margin_sum / affinity_margin_count.clamp_min(1.0)).detach().cpu()) if expand_min_frac > 0.0 else 0.0,
        **cap_stats,
        **inhibition_stats,
        **affinity_walk_stats,
    }
    return pseudo, valid, stats


@torch.no_grad()
def crf_refine_teacher_logits(
    teacher_logits: torch.Tensor,
    images: torch.Tensor,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
    alpha: float,
    frac: float,
    crf_iters: int,
    sxy_gaussian: int,
    compat_gaussian: int,
    sxy_bilateral: int,
    srgb_bilateral: int,
    compat_bilateral: int,
) -> tuple[torch.Tensor, float]:
    """Blend the teacher's token probabilities with a dense-CRF refinement of them.

    Every other signal in Stage 2 is derived from the same ViT features. The CRF pairwise
    term is not: it comes from the raw image's colour and spatial affinity, which the 16x16
    token grid destroyed. Distilling it puts that structure into the weights instead of
    only applying it once at inference.
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax

    batch, num_tokens, num_classes = teacher_logits.shape
    grid = int(num_tokens**0.5)
    height, width = images.shape[-2:]
    probs = teacher_logits.float().softmax(dim=-1)
    logit_map = probs.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    upsampled = F.interpolate(logit_map, size=(height, width), mode="bilinear", align_corners=False)

    mean_t = torch.tensor(image_mean, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(image_std, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    rgb = ((images * std_t + mean_t).clamp(0.0, 1.0) * 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    dense = upsampled.cpu().numpy()

    selected = torch.rand(batch) < float(frac)
    refined = probs.clone()
    refined_count = 0
    for batch_idx in range(batch):
        if not bool(selected[batch_idx]):
            continue
        unary = unary_from_softmax(np.ascontiguousarray(dense[batch_idx]))
        crf = dcrf.DenseCRF2D(width, height, num_classes)
        crf.setUnaryEnergy(unary)
        crf.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
        crf.addPairwiseBilateral(
            sxy=sxy_bilateral,
            srgb=srgb_bilateral,
            rgbim=np.ascontiguousarray(rgb[batch_idx]),
            compat=compat_bilateral,
        )
        out = np.asarray(crf.inference(crf_iters), dtype=np.float32).reshape(num_classes, height, width)
        out_t = torch.from_numpy(out).unsqueeze(0).to(teacher_logits.device)
        # Back to token resolution: the pseudo labels live on the 14x14 grid.
        pooled = F.adaptive_avg_pool2d(out_t, (grid, grid)).squeeze(0)
        pooled = pooled / pooled.sum(dim=0, keepdim=True).clamp_min(1e-6)
        token_probs = pooled.reshape(num_classes, num_tokens).t()
        refined[batch_idx] = (1.0 - float(alpha)) * probs[batch_idx] + float(alpha) * token_probs
        refined_count += 1

    refined = refined.clamp_min(1e-6)
    return refined.log().to(teacher_logits.dtype), float(refined_count) / max(1, batch)


@torch.no_grad()
def update_prototype_bank(
    bank: torch.Tensor,
    tokens: torch.Tensor,
    pseudo: torch.Tensor,
    ignore_index: int,
    decay: float,
    initialized: torch.Tensor,
) -> None:
    """EMA class prototypes over the whole dataset.

    The teacher is strictly per-image, so nothing in the pipeline compares a token
    against how the class looks elsewhere. The bank supplies that cross-image view.
    """
    feats = F.normalize(tokens.detach().float(), dim=-1)
    for class_idx in range(bank.shape[0]):
        mask = pseudo == class_idx
        if not bool(mask.any()):
            continue
        mean = F.normalize(feats[mask].mean(dim=0), dim=0)
        if bool(initialized[class_idx]):
            bank[class_idx] = F.normalize(bank[class_idx] * decay + mean * (1.0 - decay), dim=0)
        else:
            bank[class_idx] = mean
            initialized[class_idx] = True


def cross_image_contrastive_loss(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    pseudo: torch.Tensor,
    ignore_index: int,
    bank: torch.Tensor,
    initialized: torch.Tensor,
    temperature: float,
    max_anchors: int,
    max_negatives: int,
    margins: torch.Tensor | None = None,
    margin_min: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """InfoNCE with negatives that are exact rather than model-derived.

    For class c, every token of an image whose image-level label says c is absent is a
    guaranteed negative. BCSS has 11470 images that contain stroma and provably contain no
    lymphocyte, which is exactly the confusion that survives extent calibration.
    """
    num_classes = bank.shape[0]
    feats = F.normalize(tokens.float(), dim=-1)
    flat_feats = feats.reshape(-1, feats.shape[-1])
    flat_pseudo = pseudo.reshape(-1)
    temp = max(float(temperature), 1e-4)
    bank_n = F.normalize(bank.detach().float(), dim=-1)

    losses: list[torch.Tensor] = []
    anchor_total = 0
    for class_idx in range(num_classes):
        if not bool(initialized[class_idx]):
            continue
        anchor_mask = flat_pseudo == class_idx
        if margins is not None and margin_min > 0.0:
            anchor_mask = anchor_mask & (margins.reshape(-1) >= float(margin_min))
        anchor_idx = torch.nonzero(anchor_mask, as_tuple=False).flatten()
        if anchor_idx.numel() == 0:
            continue
        # Images that provably do not contain this class -> every token is a true negative.
        absent_images = labels[:, class_idx] <= 0
        if not bool(absent_images.any()):
            continue
        negative_pool = feats[absent_images].reshape(-1, feats.shape[-1])
        if negative_pool.shape[0] == 0:
            continue
        if anchor_idx.numel() > max_anchors:
            anchor_idx = anchor_idx[torch.randperm(anchor_idx.numel(), device=anchor_idx.device)[:max_anchors]]
        if negative_pool.shape[0] > max_negatives:
            pick = torch.randperm(negative_pool.shape[0], device=negative_pool.device)[:max_negatives]
            negative_pool = negative_pool[pick]

        anchors = flat_feats[anchor_idx]
        positive = (anchors @ bank_n[class_idx]).unsqueeze(1) / temp
        other = torch.tensor(
            [c for c in range(num_classes) if c != class_idx and bool(initialized[c])],
            device=anchors.device,
            dtype=torch.long,
        )
        negatives = [anchors @ negative_pool.detach().t() / temp]
        if other.numel() > 0:
            negatives.append(anchors @ bank_n[other].t() / temp)
        logits = torch.cat([positive] + negatives, dim=1)
        target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        losses.append(F.cross_entropy(logits, target))
        anchor_total += int(anchor_idx.numel())

    if not losses:
        return tokens.sum() * 0.0, 0.0
    return torch.stack(losses).mean(), float(anchor_total)


def partial_ce_loss(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    ignore_index: int,
    class_weights: torch.Tensor | None = None,
    normalize_by_weight: bool = True,
) -> torch.Tensor:
    valid = pseudo != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_pseudo = pseudo.reshape(-1)
    if class_weights is None:
        return F.cross_entropy(flat_logits, flat_pseudo, ignore_index=ignore_index)
    losses = F.cross_entropy(flat_logits, flat_pseudo.clamp_max(logits.shape[-1] - 1), reduction="none")
    flat_valid = flat_pseudo != ignore_index
    weights = class_weights.to(logits.device).gather(dim=0, index=flat_pseudo[flat_valid].clamp_min(0))
    if normalize_by_weight:
        denom = weights.sum().clamp_min(1e-6)
    else:
        denom = flat_valid.float().sum().clamp_min(1e-6)
    return (losses[flat_valid] * weights).sum() / denom


def mask_pool_image_logits(
    patch_logits: torch.Tensor,
    size_weight: float,
    size_lambda: float,
    detach_mask: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = patch_logits.float().softmax(dim=-1)
    weights = probs.detach() if detach_mask else probs
    denom = weights.sum(dim=1).clamp_min(1e-6)
    pooled = (weights * patch_logits.float()).sum(dim=1) / denom
    area = probs.mean(dim=1)
    if size_weight != 0.0:
        size_term = torch.log(area.clamp_min(1e-6) + float(size_lambda))
        pooled = pooled + float(size_weight) * size_term
    return pooled.to(patch_logits.dtype), area


def select_patch_logits_for_mask_pool(outputs: dict[str, torch.Tensor], source: str) -> torch.Tensor:
    if source == "output":
        return outputs["patch_logits"]
    if source == "spatial":
        return outputs.get("spatial_patch_logits", outputs["patch_logits"])
    if source == "semantic":
        return outputs.get("semantic_patch_logits", outputs["patch_logits"])
    raise ValueError(f"Unknown mask pool source: {source}")


@torch.no_grad()
def consistency_weights_from_stability(
    teacher_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    pseudo: torch.Tensor,
    ignore_index: int,
    score: str,
    base_weights: torch.Tensor,
    mode: str,
    min_weight: float,
    max_weight: float,
    adaptive_floor: torch.Tensor | None = None,
    adaptive_cap: torch.Tensor | None = None,
) -> torch.Tensor:
    base = base_weights.to(strong_logits.device, strong_logits.dtype)
    if mode == "fixed":
        return base
    if score == "softmax":
        teacher_probs = teacher_logits.float().softmax(dim=-1)
        strong_probs = strong_logits.float().softmax(dim=-1)
    elif score == "sigmoid":
        teacher_probs = teacher_logits.float().sigmoid()
        strong_probs = strong_logits.float().sigmoid()
    else:
        raise ValueError(f"Unknown pseudo score: {score}")

    weights = []
    for class_idx in range(strong_logits.shape[-1]):
        mask = pseudo == class_idx
        if not bool(mask.any()):
            weights.append(base[class_idx])
            continue
        strong_mean = strong_probs[..., class_idx][mask].mean()
        if mode == "strong_prob":
            value = strong_mean
        elif mode in {"ratio", "stability_ratio", "stability_cap"}:
            teacher_mean = teacher_probs[..., class_idx][mask].mean().clamp_min(1e-6)
            value = strong_mean / teacher_mean
        else:
            raise ValueError(f"Unknown consistency weight mode: {mode}")
        class_min = float(min_weight)
        if adaptive_floor is not None:
            class_min = float(adaptive_floor.to(value.device)[class_idx].detach().cpu())
        class_max = float(max_weight)
        if adaptive_cap is not None:
            class_max = float(adaptive_cap.to(value.device)[class_idx].detach().cpu())
        class_max = max(class_min, class_max)
        value = value.clamp(class_min, class_max).to(base.dtype)
        weights.append(base[class_idx] * value)
    return torch.stack(weights).clamp(0.0, float(max_weight))


@torch.no_grad()
def class_pseudo_agreement(
    teacher_logits: torch.Tensor,
    strong_logits: torch.Tensor,
    pseudo: torch.Tensor,
    ignore_index: int,
    score: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if score == "softmax":
        teacher_pred = teacher_logits.float().argmax(dim=-1)
        strong_pred = strong_logits.float().argmax(dim=-1)
    elif score == "sigmoid":
        teacher_pred = teacher_logits.float().sigmoid().argmax(dim=-1)
        strong_pred = strong_logits.float().sigmoid().argmax(dim=-1)
    else:
        raise ValueError(f"Unknown pseudo score: {score}")

    num_classes = strong_logits.shape[-1]
    agreement = torch.zeros(num_classes, device=strong_logits.device, dtype=torch.float32)
    observed = torch.zeros(num_classes, device=strong_logits.device, dtype=torch.float32)
    valid = pseudo != ignore_index
    for class_idx in range(num_classes):
        mask = valid & (pseudo == class_idx)
        if bool(mask.any()):
            agree = (teacher_pred[mask] == strong_pred[mask]).float().mean()
            agreement[class_idx] = agree
            observed[class_idx] = 1.0
    return agreement, observed


@torch.no_grad()
def stability_floor_from_ema(
    stability: torch.Tensor,
    target: float,
    temperature: float,
    min_floor: float,
    max_floor: float,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    floor = torch.sigmoid((stability.float() - float(target)) / temp)
    return floor.clamp(float(min_floor), float(max_floor))


def adaptive_thresholds_from_stats(
    base_thresholds: torch.Tensor,
    pseudo_class_fraction: torch.Tensor,
    labels: torch.Tensor,
    strength: float,
    min_threshold: float,
    max_threshold: float,
    target_min_frac: float,
    target_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_override is not None:
        # Anchor mode: hold the class-area distribution where the teacher put it instead of
        # pushing it toward presence frequency, which is what drives the self-training drift.
        target = target_override.to(pseudo_class_fraction.device, pseudo_class_fraction.dtype)
        target = target / target.sum().clamp_min(1e-6) * pseudo_class_fraction.sum().clamp_min(1e-6)
        delta = float(strength) * (pseudo_class_fraction - target)
        thresholds = (base_thresholds.to(labels.device) + delta).clamp(float(min_threshold), float(max_threshold))
        return thresholds, target
    present = labels.float().mean(dim=0)
    if float(present.sum().detach().cpu()) <= 0.0:
        target = pseudo_class_fraction.detach()
    else:
        target = present / present.sum().clamp_min(1e-6) * pseudo_class_fraction.sum().clamp_min(1e-6)
    if target_min_frac > 0.0:
        target = target.clamp_min(float(target_min_frac))
        target = target / target.sum().clamp_min(1e-6) * pseudo_class_fraction.sum().clamp_min(1e-6)
    delta = float(strength) * (pseudo_class_fraction - target)
    thresholds = (base_thresholds.to(labels.device) + delta).clamp(float(min_threshold), float(max_threshold))
    return thresholds, target


def target_fraction_from_labels(
    labels: torch.Tensor,
    total_fraction: torch.Tensor,
    target_min_frac: float,
) -> torch.Tensor:
    present = labels.float().mean(dim=0)
    if float(present.sum().detach().cpu()) <= 0.0:
        target = total_fraction.detach()
    else:
        target = present / present.sum().clamp_min(1e-6) * total_fraction.sum().clamp_min(1e-6)
    if target_min_frac > 0.0:
        target = target.clamp_min(float(target_min_frac))
        target = target / target.sum().clamp_min(1e-6) * total_fraction.sum().clamp_min(1e-6)
    return target


def calibration_from_stats(
    observed_fraction: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
    bias_strength: float,
    bias_max: float,
    temp_strength: float,
    temp_min: float,
    temp_max: float,
    target_min_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mode == "none":
        zeros = torch.zeros_like(observed_fraction)
        ones = torch.ones_like(observed_fraction)
        return zeros, ones, observed_fraction.detach()
    target = target_fraction_from_labels(labels, observed_fraction, target_min_frac)
    diff = target - observed_fraction
    bias = (float(bias_strength) * diff).clamp(-float(bias_max), float(bias_max))
    temp = torch.ones_like(observed_fraction)
    if mode == "bias_temp":
        temp = torch.exp(float(temp_strength) * (observed_fraction - target)).clamp(float(temp_min), float(temp_max))
    elif mode != "bias":
        raise ValueError(f"Unknown pseudo calibration mode: {mode}")
    return bias, temp, target


def apply_logit_calibration(
    logits: torch.Tensor,
    bias: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    return logits / temperature.to(logits.device).view(1, 1, -1).clamp_min(1e-6) + bias.to(logits.device).view(1, 1, -1)


def select_reliability_tokens(outputs: dict[str, torch.Tensor], mode: str) -> torch.Tensor | None:
    layer_tokens = outputs.get("layer_tokens")
    if layer_tokens is None:
        return None
    if mode == "final":
        return layer_tokens[:, -1]
    if mode == "semantic":
        weights = outputs.get("semantic_route_weights")
    elif mode == "spatial":
        weights = outputs.get("spatial_route_weights")
    else:
        raise ValueError(f"Unknown reliability token mode: {mode}")
    if weights is None:
        return layer_tokens[:, -1]
    if weights.dim() == 2:
        layer_weights = weights.mean(dim=0)
        return torch.einsum("blnd,l->bnd", layer_tokens, layer_weights.to(layer_tokens.device, layer_tokens.dtype))
    layer_weights = weights.mean(dim=1)
    return torch.einsum("blnd,bl->bnd", layer_tokens, layer_weights.to(layer_tokens.device, layer_tokens.dtype))


def build_jepa_predictor_from_checkpoint(
    checkpoint_path: str | Path,
    dim: int,
    num_patches: int,
    device: torch.device,
) -> JEPAPredictor:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    predictor = JEPAPredictor(
        dim=dim,
        hidden_dim=int(saved_args.get("predictor_dim", 384)),
        num_patches=num_patches,
        predictor_type=saved_args.get("predictor_type", "mean_mlp"),
        num_heads=int(saved_args.get("predictor_heads", 6)),
        depth=int(saved_args.get("predictor_depth", 2)),
        dropout=float(saved_args.get("predictor_dropout", 0.0)),
    ).to(device)
    state = checkpoint.get("predictor")
    if state is None:
        raise KeyError(f"Checkpoint has no predictor state: {checkpoint_path}")
    missing, unexpected = predictor.load_state_dict(state, strict=False)
    if missing:
        print(f"jepa_predictor_missing={missing}", flush=True)
    if unexpected:
        print(f"jepa_predictor_unexpected={unexpected}", flush=True)
    predictor.eval()
    for param in predictor.parameters():
        param.requires_grad_(False)
    print(f"jepa_predictor_loaded={checkpoint_path}", flush=True)
    return predictor


@torch.no_grad()
def compute_jepa_reliability(
    patch_tokens: torch.Tensor,
    predictor: JEPAPredictor,
    chunks: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_patches, _dim = patch_tokens.shape
    chunks = max(1, min(int(chunks), num_patches))
    indices = torch.arange(num_patches, device=patch_tokens.device)
    error = torch.zeros(batch, num_patches, device=patch_tokens.device, dtype=torch.float32)
    for part in torch.chunk(indices, chunks):
        mask_indices = part.view(1, -1).expand(batch, -1)
        pred = predictor(patch_tokens.detach(), mask_indices)
        target = gather_tokens(patch_tokens.detach(), mask_indices)
        part_error = F.smooth_l1_loss(pred.float(), target.float(), reduction="none").mean(dim=-1)
        error.scatter_(dim=1, index=mask_indices, src=part_error)
    reliability = torch.exp(-float(scale) * error).clamp(0.0, 1.0)
    return reliability, error


@torch.no_grad()
def compute_jepa_completed_probs(
    patch_tokens: torch.Tensor,
    predictor: JEPAPredictor,
    classifier: torch.nn.Module,
    chunks: int,
) -> torch.Tensor:
    batch, num_patches, _dim = patch_tokens.shape
    chunks = max(1, min(int(chunks), num_patches))
    indices = torch.arange(num_patches, device=patch_tokens.device)
    logits = torch.zeros(batch, num_patches, classifier.out_features, device=patch_tokens.device, dtype=torch.float32)
    for part in torch.chunk(indices, chunks):
        mask_indices = part.view(1, -1).expand(batch, -1)
        pred = predictor(patch_tokens.detach(), mask_indices)
        part_logits = classifier(pred).float()
        logits.scatter_(dim=1, index=mask_indices.unsqueeze(-1).expand_as(part_logits), src=part_logits)
    return logits.softmax(dim=-1)


@torch.no_grad()
def compute_jepa_completed_tokens(
    patch_tokens: torch.Tensor,
    predictor: JEPAPredictor,
    chunks: int,
) -> torch.Tensor:
    batch, num_patches, dim = patch_tokens.shape
    chunks = max(1, min(int(chunks), num_patches))
    indices = torch.arange(num_patches, device=patch_tokens.device)
    completed = torch.zeros(batch, num_patches, dim, device=patch_tokens.device, dtype=patch_tokens.dtype)
    for part in torch.chunk(indices, chunks):
        mask_indices = part.view(1, -1).expand(batch, -1)
        pred = predictor(patch_tokens.detach(), mask_indices)
        completed.scatter_(dim=1, index=mask_indices.unsqueeze(-1).expand_as(pred), src=pred.to(completed.dtype))
    return completed


def online_jepa_prediction_loss(
    patch_tokens: torch.Tensor,
    predictor: JEPAPredictor,
    mask_ratio: float,
    train_backbone: bool,
) -> torch.Tensor:
    batch, num_patches, _dim = patch_tokens.shape
    mask_count = max(1, min(num_patches, int(round(num_patches * float(mask_ratio)))))
    scores = torch.rand(batch, num_patches, device=patch_tokens.device)
    mask_indices = scores.topk(mask_count, dim=1).indices
    context_tokens = patch_tokens if train_backbone else patch_tokens.detach()
    pred = predictor(context_tokens, mask_indices)
    target = gather_tokens(patch_tokens.detach(), mask_indices)
    return F.smooth_l1_loss(pred.float(), target.float())


def pseudo_class_weights_from_stats(
    pseudo_class_fraction: torch.Tensor,
    mode: str,
    strength: float,
    max_weight: float,
) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "inverse_fraction":
        raise ValueError(f"Unknown pseudo class weight mode: {mode}")
    frac = pseudo_class_fraction.clamp_min(1e-6)
    active = frac > 1e-6
    mean_frac = frac[active].mean() if bool(active.any()) else frac.mean()
    weights = (mean_frac / frac).pow(float(strength)).clamp(1.0 / float(max_weight), float(max_weight))
    return weights / weights.mean().clamp_min(1e-6)


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
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
        pred = logits_to_segmentation(outputs["patch_logits"], masks.shape[-2:])
        meter.update(pred, masks)
    return meter.compute()


def teacher_logits_from_mode(
    teacher_mode: str,
    student_outputs: dict[str, torch.Tensor],
    fixed_teacher_outputs: dict[str, torch.Tensor] | None,
    ema_outputs: dict[str, torch.Tensor] | None,
    teacher_logit_source: str,
    teacher_source_temperature: float,
    teacher_source_topk_frac: float,
    teacher_source_area_penalty: float,
    output_fuse_alpha: float,
) -> torch.Tensor:
    if teacher_mode in {"fixed", "periodic_hard", "periodic_ema"}:
        if fixed_teacher_outputs is None:
            raise ValueError(f"teacher_mode={teacher_mode} requires teacher outputs.")
        return select_teacher_logits(
            fixed_teacher_outputs,
            teacher_logit_source,
            teacher_source_temperature,
            teacher_source_topk_frac,
            teacher_source_area_penalty,
            output_fuse_alpha,
        )
    if teacher_mode == "current_detached":
        return select_teacher_logits(
            student_outputs,
            teacher_logit_source,
            teacher_source_temperature,
            teacher_source_topk_frac,
            teacher_source_area_penalty,
            output_fuse_alpha,
        ).detach()
    if teacher_mode == "semantic_detached":
        return student_outputs["semantic_patch_logits"].detach()
    if teacher_mode == "ema":
        if ema_outputs is None:
            raise ValueError("teacher_mode=ema requires EMA outputs.")
        return select_teacher_logits(
            ema_outputs,
            teacher_logit_source,
            teacher_source_temperature,
            teacher_source_topk_frac,
            teacher_source_area_penalty,
            output_fuse_alpha,
        )
    raise ValueError(f"Unknown teacher mode: {teacher_mode}")


def select_teacher_logits(
    outputs: dict[str, torch.Tensor],
    source: str,
    temperature: float,
    topk_frac: float,
    area_penalty: float,
    output_fuse_alpha: float,
) -> torch.Tensor:
    if source == "output":
        return outputs["patch_logits"]
    if source == "semantic":
        return outputs.get("semantic_patch_logits", outputs["patch_logits"])
    if source == "spatial":
        return outputs.get("spatial_patch_logits", outputs["patch_logits"])
    if source == "fuse":
        if "semantic_patch_logits" in outputs and "spatial_patch_logits" in outputs:
            alpha = max(0.0, min(1.0, float(output_fuse_alpha)))
            return (1.0 - alpha) * outputs["semantic_patch_logits"] + alpha * outputs["spatial_patch_logits"]
        return outputs["patch_logits"]
    if source not in {"adaptive_topk", "adaptive_topk_area"}:
        raise ValueError(f"Unknown teacher logit source: {source}")

    per_layer_logits = outputs.get("per_layer_logits")
    if per_layer_logits is None:
        per_layer_logits = outputs.get("semantic_per_layer_logits")
    if per_layer_logits is None:
        return outputs["patch_logits"]
    topk = max(1, int(round(per_layer_logits.shape[2] * float(topk_frac))))
    score = per_layer_logits.topk(topk, dim=2).values.mean(dim=2)
    if source == "adaptive_topk_area":
        area = (per_layer_logits.sigmoid() > 0.5).to(per_layer_logits.dtype).mean(dim=2)
        score = score - float(area_penalty) * area
    weights = (score / max(float(temperature), 1e-4)).permute(0, 2, 1).softmax(dim=-1)
    return torch.einsum("blnc,bcl->bnc", per_layer_logits, weights)


def apply_n2_preset(args: argparse.Namespace) -> None:
    """Stage-2 N2 recipe: teacher-initialized EMA self-distillation with a dense-CRF teacher.

    Class count is derived from --dataset, and consistency-class-weights is left at "1.0"
    which expands to num_classes ones, so the same recipe covers bcss/luad (4) and gcss (6)
    with no per-dataset flags. Only paths, dataset and seed differ between runs.
    """
    if args.preset != "n2":
        return
    recipe = {
        # schedule / optim
        "epochs": 10, "batch_size": 24, "val_batch_size": 64,
        "lr": 1e-4, "scheduler": "warmup_cosine", "warmup_epochs": 1.0, "min_lr_ratio": 0.1,
        "route_layers": "all", "amp": True, "grad_checkpointing": True,
        # teacher-initialized student + EMA teacher
        "init_from_teacher": True, "semantic_pooling": "lse", "lse_tau": 1.0,
        "teacher_mode": "ema", "ema_decay": 0.999, "select_on": "ema",
        "init_spatial_route_from_semantic": True,
        "teacher_logit_source": "adaptive_topk",
        "teacher_source_temperature": 0.5, "teacher_source_topk_frac": 0.05,
        "output_mode": "learned_fuse", "output_fuse_alpha": 0.35,
        # pseudo labels
        "pseudo_logit_target": "output", "pseudo_score": "softmax", "pseudo_thresholds": "0.85",
        "adaptive_pseudo_thresholds": True, "adaptive_threshold_strength": 0.5,
        "adaptive_threshold_min": 0.50, "adaptive_threshold_max": 0.90,
        "pseudo_weight": 0.2, "pseudo_expand_mode": "fixed", "pseudo_expand_min_frac": 0.03,
        # strong-augmentation consistency
        "strong_consistency_weight": 0.1, "consistency_class_weights": "1.0",
        "consistency_weight_mode": "ratio", "consistency_weight_min": 0.3, "consistency_weight_max": 1.0,
        "strong_brightness": 0.15, "strong_contrast": 0.25, "strong_noise": 0.03,
        "restrict_present": True,
        # the new ingredient: dense-CRF teacher
        "crf_teacher_alpha": 0.5, "crf_teacher_frac": 0.25,
        "crf_teacher_warmup_epochs": 1.0, "crf_teacher_iters": 5,
    }
    for key, value in recipe.items():
        setattr(args, key, value)
    print(f"preset=n2 applied: dataset={args.dataset} num_classes={len(CLASS_NAMES[args.dataset])} "
          f"crf_alpha={args.crf_teacher_alpha} crf_frac={args.crf_teacher_frac}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual semantic-spatial routing with online partial pseudo supervision.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad", "gcss"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None, help="Raw DeiT checkpoint used to initialize the student backbone.")
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional fixed/initial teacher checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-layers", default="all")
    parser.add_argument("--patch-kernel", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--patch-padding", type=int, default=0)
    parser.add_argument("--patch-resample-scale", default="area", choices=["area", "none"])
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--image-logit-mode", default="semantic", choices=["semantic", "mask_pool", "mix"])
    parser.add_argument("--mask-pool-source", default="output", choices=["output", "spatial", "semantic"])
    parser.add_argument("--mask-pool-loss-weight", type=float, default=0.5)
    parser.add_argument("--mask-pool-size-weight", type=float, default=0.1)
    parser.add_argument("--mask-pool-size-lambda", type=float, default=0.05)
    parser.add_argument("--mask-pool-detach-mask", action="store_true")
    parser.add_argument("--pseudo-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--pseudo-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--pseudo-logit-target", default="spatial", choices=["spatial", "output"])
    parser.add_argument("--pseudo-score", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--pseudo-thresholds", default="0.70")
    parser.add_argument("--pseudo-thresholds-start", default="")
    parser.add_argument("--pseudo-threshold-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--pseudo-calibration", default="none", choices=["none", "bias", "bias_temp"])
    parser.add_argument("--pseudo-calibration-bias-strength", type=float, default=2.0)
    parser.add_argument("--pseudo-calibration-bias-max", type=float, default=0.5)
    parser.add_argument("--pseudo-calibration-temp-strength", type=float, default=2.0)
    parser.add_argument("--pseudo-calibration-temp-min", type=float, default=0.7)
    parser.add_argument("--pseudo-calibration-temp-max", type=float, default=1.5)
    parser.add_argument("--pseudo-calibration-target-min-frac", type=float, default=0.0)
    parser.add_argument("--adaptive-pseudo-thresholds", action="store_true")
    parser.add_argument("--adaptive-threshold-strength", type=float, default=0.5)
    parser.add_argument("--adaptive-threshold-min", type=float, default=0.50)
    parser.add_argument("--adaptive-threshold-max", type=float, default=0.90)
    parser.add_argument("--adaptive-target-min-frac", type=float, default=0.0)
    parser.add_argument("--pseudo-inhibition", default="none", choices=["none", "margin", "subtract", "soft_margin", "feature_centroid"])
    parser.add_argument("--pseudo-inhibition-strength", type=float, default=0.5)
    parser.add_argument("--pseudo-inhibition-margin", type=float, default=0.05)
    parser.add_argument("--pseudo-inhibition-temperature", type=float, default=0.05)
    parser.add_argument("--pseudo-inhibition-layer", default="spatial", choices=["semantic", "spatial", "final"])
    parser.add_argument(
        "--pseudo-expand-mode",
        default="fixed",
        choices=[
            "fixed",
            "minimal_teacher",
            "minimal_jepa",
            "minimal_mix",
            "minimal_mix_adaptive",
            "minimal_mix_class_adaptive",
            "minimal_mix_geometric",
            "adaptive_margin",
            "adaptive_margin_reliability",
            "feature_affinity",
            "feature_affinity_margin",
            "affinity_walk",
            "affinity_walk_contrast",
            "affinity_candidate",
            "affinity_candidate_contrast",
            "jepa_affinity",
            "jepa_contrast",
            "online_jepa_affinity",
            "online_jepa_contrast",
            "adaptive_margin_jepa",
            "adaptive_margin_jepa_completed",
        ],
    )
    parser.add_argument("--pseudo-expand-min-frac", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-min-score", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-max-frac", type=float, default=0.06)
    parser.add_argument("--pseudo-expand-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-margin-min", type=float, default=0.05)
    parser.add_argument("--pseudo-expand-under-strength", type=float, default=1.0)
    parser.add_argument("--pseudo-expand-reliability-layer", default="semantic", choices=["semantic", "spatial", "final"])
    parser.add_argument("--pseudo-expand-reliability-min", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-affinity-alpha", type=float, default=0.3)
    parser.add_argument("--pseudo-expand-affinity-beta", type=float, default=0.2)
    parser.add_argument("--pseudo-expand-affinity-margin", type=float, default=0.0)
    parser.add_argument("--affinity-walk-steps", type=int, default=2)
    parser.add_argument("--affinity-walk-alpha", type=float, default=0.4)
    parser.add_argument("--affinity-walk-gamma", type=float, default=2.0)
    parser.add_argument("--affinity-walk-self-loop", type=float, default=1.0)
    parser.add_argument("--affinity-walk-contrast-beta", type=float, default=0.2)
    parser.add_argument("--jepa-reliability-checkpoint", default=None)
    parser.add_argument("--jepa-reliability-chunks", type=int, default=4)
    parser.add_argument("--jepa-reliability-scale", type=float, default=10.0)
    parser.add_argument("--jepa-completed-classifier", default="spatial", choices=["semantic", "spatial", "output"])
    parser.add_argument("--jepa-completed-min-score", type=float, default=0.0)
    parser.add_argument("--jepa-completed-mix-alpha", type=float, default=0.5)
    parser.add_argument("--jepa-completed-class-alpha", default="0.5")
    parser.add_argument("--online-jepa-weight", type=float, default=0.0)
    parser.add_argument("--online-jepa-mask-ratio", type=float, default=0.25)
    parser.add_argument("--online-jepa-predictor-dim", type=int, default=384)
    parser.add_argument("--online-jepa-predictor-type", default="mean_mlp")
    parser.add_argument("--online-jepa-predictor-heads", type=int, default=6)
    parser.add_argument("--online-jepa-predictor-depth", type=int, default=2)
    parser.add_argument("--online-jepa-predictor-dropout", type=float, default=0.0)
    parser.add_argument("--online-jepa-train-backbone", action="store_true")
    parser.add_argument("--online-jepa-loss-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--online-jepa-loss-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--online-jepa-expand-warmup-epochs", type=float, default=2.0)
    parser.add_argument("--pseudo-class-weight-mode", default="none", choices=["none", "inverse_fraction"])
    parser.add_argument("--pseudo-class-weight-strength", type=float, default=0.5)
    parser.add_argument("--pseudo-class-weight-max", type=float, default=3.0)
    parser.add_argument("--pseudo-kept-max-frac", type=float, default=1.0)
    parser.add_argument("--pseudo-kept-cap-mode", default="global", choices=["global", "target", "protected", "protected_rescue"])
    parser.add_argument("--pseudo-kept-cap-target-slack", type=float, default=0.02)
    parser.add_argument("--pseudo-kept-cap-protect-floor-ratio", type=float, default=0.5)
    parser.add_argument("--pseudo-kept-cap-rescue-frac", type=float, default=0.0)
    parser.add_argument("--restrict-present", action="store_true")
    parser.add_argument(
        "--teacher-mode",
        default="fixed",
        choices=["fixed", "current_detached", "semantic_detached", "ema", "periodic_hard", "periodic_ema"],
    )
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--periodic-ema-decay", type=float, default=0.9)
    parser.add_argument("--teacher-logit-source", default="output", choices=["output", "semantic", "spatial", "fuse", "adaptive_topk", "adaptive_topk_area"])
    parser.add_argument("--teacher-source-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-source-topk-frac", type=float, default=0.05)
    parser.add_argument("--teacher-source-area-penalty", type=float, default=0.5)
    parser.add_argument("--strong-consistency-weight", type=float, default=0.0)
    parser.add_argument("--consistency-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--consistency-ramp-epochs", type=float, default=0.0)
    parser.add_argument("--consistency-class-weights", default="1.0")
    parser.add_argument("--consistency-weight-mode", default="fixed", choices=["fixed", "strong_prob", "ratio", "stability_ratio", "stability_cap"])
    parser.add_argument("--consistency-weight-min", type=float, default=0.0)
    parser.add_argument("--consistency-weight-max", type=float, default=1.0)
    parser.add_argument("--consistency-stability-ema", type=float, default=0.95)
    parser.add_argument("--consistency-stability-target", type=float, default=0.70)
    parser.add_argument("--consistency-stability-temp", type=float, default=0.10)
    parser.add_argument("--consistency-floor-min", type=float, default=0.0)
    parser.add_argument("--consistency-floor-max", type=float, default=1.0)
    parser.add_argument("--strong-brightness", type=float, default=0.15)
    parser.add_argument("--strong-contrast", type=float, default=0.25)
    parser.add_argument("--strong-noise", type=float, default=0.03)
    parser.add_argument("--init-from-teacher", action="store_true")
    parser.add_argument("--spatial-route-mode", default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--fixed-spatial-layer", type=int, default=11)
    parser.add_argument("--topk-frac", type=float, default=0.05)
    parser.add_argument(
        "--semantic-pooling",
        default="max",
        choices=["max", "lse", "topk"],
        help="Must match the Stage-1 teacher, otherwise the student inherits weights trained under a different objective.",
    )
    parser.add_argument("--lse-tau", type=float, default=1.0)
    parser.add_argument(
        "--adaptive-target-mode",
        default="presence",
        choices=["presence", "anchor"],
        help="'presence' targets class areas proportional to label frequency (drives drift). "
             "'anchor' freezes the target at the class areas observed during the first epochs.",
    )
    parser.add_argument("--adaptive-target-anchor-epochs", type=float, default=1.0)
    parser.add_argument(
        "--clean-single-class-weight",
        type=float,
        default=0.0,
        help="Weight of the exact dense CE on images whose image-level label names exactly one class. "
             "For those images every token is that class, so the label needs no teacher and no threshold. "
             "When > 0 they are removed from the noisy pseudo path to avoid double counting.",
    )
    parser.add_argument(
        "--contrast-weight",
        type=float,
        default=0.0,
        help="Cross-image InfoNCE whose negatives are tokens from images the label says lack the class.",
    )
    parser.add_argument("--contrast-temperature", type=float, default=0.1)
    parser.add_argument("--contrast-bank-decay", type=float, default=0.99)
    parser.add_argument("--contrast-max-anchors", type=int, default=128)
    parser.add_argument("--contrast-max-negatives", type=int, default=256)
    parser.add_argument("--contrast-margin-min", type=float, default=0.0, help="Skip anchors whose teacher top1-top2 margin is below this.")
    parser.add_argument("--contrast-token-layer", default="final", choices=["final", "semantic", "spatial"])
    parser.add_argument("--contrast-warmup-epochs", type=float, default=1.0)
    parser.add_argument("--contrast-ramp-epochs", type=float, default=1.0)
    parser.add_argument(
        "--crf-teacher-alpha",
        type=float,
        default=0.0,
        help="Blend weight of the dense-CRF refinement into the teacher probabilities. 0 disables it.",
    )
    parser.add_argument("--crf-teacher-frac", type=float, default=1.0, help="Fraction of each batch to refine; lowers the CPU cost.")
    parser.add_argument("--crf-teacher-iters", type=int, default=5)
    parser.add_argument("--crf-teacher-sxy-gaussian", type=int, default=3)
    parser.add_argument("--crf-teacher-compat-gaussian", type=int, default=3)
    parser.add_argument("--crf-teacher-sxy-bilateral", type=int, default=40)
    parser.add_argument("--crf-teacher-srgb-bilateral", type=int, default=8)
    parser.add_argument("--crf-teacher-compat-bilateral", type=int, default=5)
    parser.add_argument("--crf-teacher-warmup-epochs", type=float, default=0.0)
    parser.add_argument("--strong-hed-prob", type=float, default=0.0, help="Probability of H&E stain jitter in the strong view.")
    parser.add_argument("--strong-hed-sigma-alpha", type=float, default=0.05)
    parser.add_argument("--strong-hed-sigma-beta", type=float, default=0.05)
    parser.add_argument(
        "--init-spatial-route-from-semantic",
        action="store_true",
        help="Copy the teacher route into the spatial route as well. Without this the spatial branch keeps a uniform "
             "12-layer mixture it can never unlearn, while contributing to the fused output from step 0.",
    )
    parser.add_argument(
        "--select-on",
        default="student",
        choices=["student", "ema"],
        help="Which model drives best.pt. 'ema' also stores the EMA weights under 'model'.",
    )
    parser.add_argument("--output-mode", default="spatial", choices=["spatial", "semantic", "fuse", "learned_fuse"])
    parser.add_argument("--output-fuse-alpha", type=float, default=0.5)
    parser.add_argument("--semantic-init", default="final", choices=["uniform", "final", "middle"])
    parser.add_argument("--spatial-init", default="uniform", choices=["uniform", "final", "middle"])
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Compute device: 'cuda' (default, GPU index via --gpu), 'cuda:N', or 'cpu'. "
             "Refuses to run if 'cuda' is requested but unavailable.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--preset",
        default="none",
        choices=["none", "n2"],
        help="'n2' applies the full Stage-2 CRF-teacher recipe, so only "
             "--dataset/--data-root/--checkpoint/--teacher-checkpoint/--output-dir/--seed are needed.",
    )
    args = parser.parse_args()
    apply_n2_preset(args)

    # Pin the GPU before any CUDA context is created (seed_everything touches CUDA).
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA requested (--device cuda) but torch.cuda.is_available() is False — "
            "refusing to train on CPU. Check nvidia-smi / CUDA_VISIBLE_DEVICES / torch build, "
            "or pass --device cpu to override."
        )
    device = torch.device(args.device)
    print(f"device={device}"
          + (f" (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})" if args.gpu is not None else ""),
          flush=True)

    seed_everything(args.seed)
    num_classes = len(CLASS_NAMES[args.dataset])
    thresholds = parse_thresholds(args.pseudo_thresholds, num_classes)
    threshold_start = parse_thresholds(args.pseudo_thresholds_start, num_classes) if args.pseudo_thresholds_start.strip() else thresholds.clone()
    jepa_completed_class_alpha = parse_class_values(args.jepa_completed_class_alpha, num_classes, "--jepa-completed-class-alpha").to(device)
    consistency_class_weights = parse_class_values(args.consistency_class_weights, num_classes, "--consistency-class-weights").to(device)
    image_mean, image_std = load_preprocessor_stats(args.checkpoint)
    print(f"image_mean={image_mean}", flush=True)
    print(f"image_std={image_std}", flush=True)
    print(f"teacher_mode={args.teacher_mode} pseudo_score={args.pseudo_score} thresholds={thresholds.tolist()}", flush=True)
    print(f"threshold_start={threshold_start.tolist()}", flush=True)
    print(f"consistency_class_weights={consistency_class_weights.detach().cpu().tolist()}", flush=True)

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
        variant="dual_param",
        topk_frac=args.topk_frac,
        semantic_pooling=args.semantic_pooling,
        lse_tau=args.lse_tau,
        spatial_weight=0.0,
        semantic_init=args.semantic_init,
        spatial_init=args.spatial_init,
        output_mode=args.output_mode,
        output_fuse_alpha=args.output_fuse_alpha,
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
    if args.init_from_teacher:
        if args.teacher_checkpoint is None:
            raise ValueError("--init-from-teacher requires --teacher-checkpoint")
        copy_single_semantic_teacher_to_dual_student(
            model,
            args.teacher_checkpoint,
            init_spatial_route=args.init_spatial_route_from_semantic,
        )
    if args.spatial_route_mode == "fixed":
        lock_fixed_spatial_route(model, args.fixed_spatial_layer)
        print(f"fixed_spatial_layer={args.fixed_spatial_layer}", flush=True)
    print(f"resolved_route_layers={model.route_layers}", flush=True)

    jepa_predictor = None
    external_jepa_modes = {
        "adaptive_margin_jepa",
        "adaptive_margin_jepa_completed",
        "minimal_jepa",
        "minimal_mix",
        "minimal_mix_adaptive",
        "minimal_mix_class_adaptive",
        "minimal_mix_geometric",
        "jepa_affinity",
        "jepa_contrast",
    }
    online_jepa_modes = {"online_jepa_affinity", "online_jepa_contrast"}
    online_jepa_enabled = args.pseudo_expand_mode in online_jepa_modes or (
        args.online_jepa_weight > 0.0 and args.pseudo_expand_mode not in external_jepa_modes
    )
    if args.pseudo_expand_mode in external_jepa_modes:
        if args.jepa_reliability_checkpoint is None:
            raise ValueError(f"--pseudo-expand-mode {args.pseudo_expand_mode} requires --jepa-reliability-checkpoint")
        jepa_predictor = build_jepa_predictor_from_checkpoint(
            args.jepa_reliability_checkpoint,
            model.num_features,
            model.num_patches,
            device,
        )
    elif online_jepa_enabled:
        jepa_predictor = JEPAPredictor(
            dim=model.num_features,
            hidden_dim=int(args.online_jepa_predictor_dim),
            num_patches=model.num_patches,
            predictor_type=args.online_jepa_predictor_type,
            num_heads=int(args.online_jepa_predictor_heads),
            depth=int(args.online_jepa_predictor_depth),
            dropout=float(args.online_jepa_predictor_dropout),
        ).to(device)
        print(
            "online_jepa_predictor=created "
            f"type={args.online_jepa_predictor_type} dim={args.online_jepa_predictor_dim} "
            f"mask_ratio={args.online_jepa_mask_ratio}",
            flush=True,
        )

    fixed_teacher = None
    if args.teacher_mode == "fixed":
        if args.teacher_checkpoint is None:
            raise ValueError("--teacher-mode fixed requires --teacher-checkpoint")
        teacher_ckpt_path = Path(args.teacher_checkpoint)
        teacher_ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
        fixed_teacher = build_model_from_checkpoint(teacher_ckpt, teacher_ckpt_path, args.dataset, device)
        fixed_teacher.eval()
        for param in fixed_teacher.parameters():
            param.requires_grad_(False)
    elif args.teacher_mode in {"periodic_hard", "periodic_ema"}:
        fixed_teacher = copy.deepcopy(model).to(device)
        fixed_teacher.eval()
        for param in fixed_teacher.parameters():
            param.requires_grad_(False)
        print(f"periodic_teacher_init=student update_mode={args.teacher_mode}", flush=True)

    ema_teacher = None
    if args.teacher_mode == "ema":
        ema_teacher = copy.deepcopy(model).to(device)
        ema_teacher.eval()
        for param in ema_teacher.parameters():
            param.requires_grad_(False)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if jepa_predictor is not None and online_jepa_enabled:
        trainable_params += [param for param in jepa_predictor.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
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
    saved_args = vars(args).copy()
    saved_args["model_type"] = "dual_route_linear"
    saved_args["variant"] = "dual_param"
    saved_args["spatial_weight"] = 0.0
    saved_args["image_mean"] = list(image_mean)
    saved_args["image_std"] = list(image_std)
    saved_args["class_names"] = CLASS_NAMES[args.dataset]
    saved_args["patch_embed_info"] = patch_info
    (output_dir / "config.json").write_text(json.dumps(saved_args, indent=2), encoding="utf-8")

    best_miou = -1.0
    best_epoch = 0
    global_step = 0
    log_path = output_dir / "log.csv"
    threshold_device = thresholds.to(device)
    threshold_start_device = threshold_start.to(device)
    consistency_stability_ema = torch.full((num_classes,), float(args.consistency_stability_target), device=device)
    anchor_target = None
    anchor_accum = torch.zeros(num_classes, device=device, dtype=torch.float64)
    anchor_steps = 0
    prototype_bank = torch.zeros(num_classes, model.num_features, device=device)
    prototype_initialized = torch.zeros(num_classes, dtype=torch.bool, device=device)
    for epoch in range(1, args.epochs + 1):
        pseudo_scale = ramp_value(epoch, args.pseudo_warmup_epochs, args.pseudo_ramp_epochs)
        consistency_scale = ramp_value(epoch, args.consistency_warmup_epochs, args.consistency_ramp_epochs)
        online_jepa_loss_scale = ramp_value(epoch, args.online_jepa_loss_warmup_epochs, args.online_jepa_loss_ramp_epochs)
        expand_scale = ramp_value(epoch, args.pseudo_expand_warmup_epochs, args.pseudo_expand_ramp_epochs)
        threshold_scale = ramp_value(epoch, args.pseudo_warmup_epochs, args.pseudo_threshold_ramp_epochs)
        scheduled_pseudo_weight = float(args.pseudo_weight) * pseudo_scale
        scheduled_consistency_weight = float(args.strong_consistency_weight) * consistency_scale
        scheduled_contrast_weight = float(args.contrast_weight) * ramp_value(
            epoch, args.contrast_warmup_epochs, args.contrast_ramp_epochs
        )
        scheduled_online_jepa_weight = float(args.online_jepa_weight) * online_jepa_loss_scale
        scheduled_thresholds = interpolate_thresholds(threshold_start_device, threshold_device, threshold_scale)
        scheduled_expand_min_frac = float(args.pseudo_expand_min_frac) * expand_scale
        scheduled_expand_max_frac = float(args.pseudo_expand_max_frac) * expand_scale
        active_expand_mode = args.pseudo_expand_mode
        if args.pseudo_expand_mode in online_jepa_modes and epoch <= float(args.online_jepa_expand_warmup_epochs):
            active_expand_mode = "fixed"
        model.train()
        if jepa_predictor is not None:
            jepa_predictor.train(online_jepa_enabled)
        start = time.perf_counter()
        loss_sum = sem_sum = semantic_cls_sum = mask_pool_cls_sum = pseudo_sum = consistency_sum = kept_sum = expanded_sum = 0.0
        clean_sum = 0.0
        clean_fraction_sum = 0.0
        contrast_sum = 0.0
        contrast_anchor_sum = 0.0
        crf_frac_sum = 0.0
        online_jepa_loss_sum = 0.0
        online_jepa_weight_sum = 0.0
        cap_removed_sum = 0.0
        cap_target_sum = 0.0
        cap_floor_sum = 0.0
        cap_rescue_sum = 0.0
        class_fraction_sum = torch.zeros(num_classes, dtype=torch.float64)
        mask_pool_area_sum = torch.zeros(num_classes, dtype=torch.float64)
        threshold_sum = torch.zeros(num_classes, dtype=torch.float64)
        target_fraction_sum = torch.zeros(num_classes, dtype=torch.float64)
        pseudo_weight_sum = torch.zeros(num_classes, dtype=torch.float64)
        calibration_bias_sum = torch.zeros(num_classes, dtype=torch.float64)
        calibration_temp_sum = torch.zeros(num_classes, dtype=torch.float64)
        calibration_target_sum = torch.zeros(num_classes, dtype=torch.float64)
        consistency_weight_sum = torch.zeros(num_classes, dtype=torch.float64)
        consistency_effective_weight_sum = torch.zeros(num_classes, dtype=torch.float64)
        consistency_stability_sum = torch.zeros(num_classes, dtype=torch.float64)
        consistency_floor_sum = torch.zeros(num_classes, dtype=torch.float64)
        consistency_agreement_sum = torch.zeros(num_classes, dtype=torch.float64)
        jepa_reliability_sum = 0.0
        jepa_error_sum = 0.0
        jepa_completed_conf_sum = 0.0
        jepa_completed_class_sum = torch.zeros(num_classes, dtype=torch.float64)
        jepa_expand_alpha_sum = 0.0
        affinity_expand_margin_sum = 0.0
        affinity_walk_delta_sum = 0.0
        affinity_walk_seed_sum = 0.0
        affinity_walk_score_sum = 0.0
        inhibition_ambiguous_sum = 0.0
        inhibition_changed_sum = 0.0
        inhibition_drop_sum = 0.0
        inhibition_seed_sum = 0.0
        inhibition_centroid_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                online_jepa_loss = outputs["patch_logits"].sum() * 0.0
                if jepa_predictor is not None and online_jepa_enabled and scheduled_online_jepa_weight > 0.0:
                    online_jepa_tokens = select_reliability_tokens(outputs, args.pseudo_expand_reliability_layer)
                    if online_jepa_tokens is None:
                        raise RuntimeError("Could not select tokens for online JEPA loss.")
                    online_jepa_loss = online_jepa_prediction_loss(
                        online_jepa_tokens,
                        jepa_predictor,
                        args.online_jepa_mask_ratio,
                        args.online_jepa_train_backbone,
                    )
                fixed_outputs = fixed_teacher(images) if fixed_teacher is not None else None
                ema_outputs = ema_teacher(images) if ema_teacher is not None else None
                teacher_logits = teacher_logits_from_mode(
                    args.teacher_mode,
                    outputs,
                    fixed_outputs,
                    ema_outputs,
                    args.teacher_logit_source,
                    args.teacher_source_temperature,
                    args.teacher_source_topk_frac,
                    args.teacher_source_area_penalty,
                    args.output_fuse_alpha,
                )
                teacher_logits = teacher_logits.detach().float()
                crf_refined_frac = 0.0
                if args.crf_teacher_alpha > 0.0 and epoch > float(args.crf_teacher_warmup_epochs):
                    teacher_logits, crf_refined_frac = crf_refine_teacher_logits(
                        teacher_logits,
                        images,
                        image_mean,
                        image_std,
                        args.crf_teacher_alpha,
                        args.crf_teacher_frac,
                        args.crf_teacher_iters,
                        args.crf_teacher_sxy_gaussian,
                        args.crf_teacher_compat_gaussian,
                        args.crf_teacher_sxy_bilateral,
                        args.crf_teacher_srgb_bilateral,
                        args.crf_teacher_compat_bilateral,
                    )
                inhibition_tokens = None
                if args.pseudo_inhibition == "feature_centroid":
                    inhibition_source = fixed_outputs if fixed_outputs is not None else (ema_outputs if ema_outputs is not None else outputs)
                    inhibition_tokens = select_reliability_tokens(inhibition_source, args.pseudo_inhibition_layer)
                    if inhibition_tokens is None:
                        raise RuntimeError("Could not select tokens for feature-centroid inhibition.")
                calibration_bias = torch.zeros(num_classes, device=device)
                calibration_temp = torch.ones(num_classes, device=device)
                calibration_target = torch.zeros(num_classes, device=device)
                if args.pseudo_calibration != "none":
                    raw_pseudo, _raw_valid, raw_stats = make_pseudo(
                        teacher_logits,
                        labels,
                        scheduled_thresholds,
                        args.pseudo_score,
                        args.restrict_present,
                        args.ignore_index,
                        inhibition_mode=args.pseudo_inhibition,
                        inhibition_strength=args.pseudo_inhibition_strength,
                        inhibition_margin=args.pseudo_inhibition_margin,
                        inhibition_temperature=args.pseudo_inhibition_temperature,
                        inhibition_tokens=inhibition_tokens,
                    )
                    del raw_pseudo, _raw_valid
                    raw_fraction = torch.tensor(raw_stats["pseudo_class_fraction"], device=device)
                    calibration_bias, calibration_temp, calibration_target = calibration_from_stats(
                        raw_fraction,
                        labels,
                        args.pseudo_calibration,
                        args.pseudo_calibration_bias_strength,
                        args.pseudo_calibration_bias_max,
                        args.pseudo_calibration_temp_strength,
                        args.pseudo_calibration_temp_min,
                        args.pseudo_calibration_temp_max,
                        args.pseudo_calibration_target_min_frac,
                    )
                    teacher_logits = apply_logit_calibration(teacher_logits, calibration_bias, calibration_temp)
                effective_thresholds = scheduled_thresholds
                target_fraction = torch.zeros(num_classes, device=device)
                if args.adaptive_pseudo_thresholds:
                    prelim_pseudo, _prelim_valid, prelim_stats = make_pseudo(
                        teacher_logits,
                        labels,
                        scheduled_thresholds,
                        args.pseudo_score,
                        args.restrict_present,
                        args.ignore_index,
                        inhibition_mode=args.pseudo_inhibition,
                        inhibition_strength=args.pseudo_inhibition_strength,
                        inhibition_margin=args.pseudo_inhibition_margin,
                        inhibition_temperature=args.pseudo_inhibition_temperature,
                        inhibition_tokens=inhibition_tokens,
                    )
                    del prelim_pseudo, _prelim_valid
                    prelim_fraction = torch.tensor(prelim_stats["pseudo_class_fraction"], device=device)
                    if args.adaptive_target_mode == "anchor":
                        if epoch <= float(args.adaptive_target_anchor_epochs):
                            # Still measuring: accumulate the teacher's own class areas.
                            anchor_accum += prelim_fraction.detach().double()
                            anchor_steps += 1
                        elif anchor_target is None and anchor_steps > 0:
                            anchor_target = (anchor_accum / float(anchor_steps)).float()
                            print(f"anchor_target={anchor_target.detach().cpu().tolist()}", flush=True)
                    effective_thresholds, target_fraction = adaptive_thresholds_from_stats(
                        scheduled_thresholds,
                        prelim_fraction,
                        labels,
                        args.adaptive_threshold_strength,
                        args.adaptive_threshold_min,
                        args.adaptive_threshold_max,
                        args.adaptive_target_min_frac,
                        target_override=anchor_target,
                    )
                reliability_tokens = None
                jepa_reliability = None
                jepa_completed_probs = None
                jepa_affinity_tokens = None
                jepa_error_mean = torch.zeros((), device=device)
                if active_expand_mode in {
                    "adaptive_margin_reliability",
                    "feature_affinity",
                    "feature_affinity_margin",
                    "affinity_walk",
                    "affinity_walk_contrast",
                    "affinity_candidate",
                    "affinity_candidate_contrast",
                }:
                    reliability_source = fixed_outputs if fixed_outputs is not None else (ema_outputs if ema_outputs is not None else outputs)
                    reliability_tokens = select_reliability_tokens(reliability_source, args.pseudo_expand_reliability_layer)
                    if reliability_tokens is None:
                        raise RuntimeError("Could not select tokens for feature-affinity expansion.")
                elif active_expand_mode == "adaptive_margin_jepa":
                    if jepa_predictor is None:
                        raise RuntimeError("JEPA predictor was not initialized.")
                    jepa_tokens = select_reliability_tokens(outputs, args.pseudo_expand_reliability_layer)
                    if jepa_tokens is None:
                        raise RuntimeError("Could not select tokens for JEPA reliability.")
                    jepa_reliability, jepa_error = compute_jepa_reliability(
                        jepa_tokens,
                        jepa_predictor,
                        args.jepa_reliability_chunks,
                        args.jepa_reliability_scale,
                    )
                    jepa_error_mean = jepa_error.mean()
                elif active_expand_mode in {"jepa_affinity", "jepa_contrast", "online_jepa_affinity", "online_jepa_contrast"}:
                    if jepa_predictor is None:
                        raise RuntimeError("JEPA predictor was not initialized.")
                    jepa_tokens = select_reliability_tokens(outputs, args.pseudo_expand_reliability_layer)
                    if jepa_tokens is None:
                        raise RuntimeError("Could not select tokens for JEPA affinity expansion.")
                    jepa_affinity_tokens = compute_jepa_completed_tokens(
                        jepa_tokens,
                        jepa_predictor,
                        args.jepa_reliability_chunks,
                    )
                elif active_expand_mode in {
                    "adaptive_margin_jepa_completed",
                    "minimal_jepa",
                    "minimal_mix",
                    "minimal_mix_adaptive",
                    "minimal_mix_class_adaptive",
                    "minimal_mix_geometric",
                }:
                    if jepa_predictor is None:
                        raise RuntimeError("JEPA predictor was not initialized.")
                    jepa_tokens = select_reliability_tokens(outputs, args.pseudo_expand_reliability_layer)
                    if jepa_tokens is None:
                        raise RuntimeError("Could not select tokens for JEPA completed expansion.")
                    if args.jepa_completed_classifier == "semantic":
                        completed_classifier = model.semantic_classifier
                    elif args.jepa_completed_classifier == "spatial":
                        completed_classifier = model.spatial_classifier
                    else:
                        completed_classifier = (
                            model.spatial_classifier
                            if args.pseudo_logit_target == "spatial"
                            else model.semantic_classifier
                        )
                    jepa_completed_probs = compute_jepa_completed_probs(
                        jepa_tokens,
                        jepa_predictor,
                        completed_classifier,
                        args.jepa_reliability_chunks,
                    )
                pseudo, _valid, pseudo_stats = make_pseudo(
                    teacher_logits,
                    labels,
                    effective_thresholds,
                    args.pseudo_score,
                    args.restrict_present,
                    args.ignore_index,
                    active_expand_mode,
                    scheduled_expand_min_frac,
                    args.pseudo_expand_min_score,
                    scheduled_expand_max_frac,
                    args.pseudo_expand_margin_min,
                    args.pseudo_expand_under_strength,
                    reliability_tokens,
                    jepa_reliability,
                    jepa_completed_probs,
                    jepa_affinity_tokens,
                    args.jepa_completed_min_score,
                    args.jepa_completed_mix_alpha,
                    jepa_completed_class_alpha,
                    args.pseudo_expand_reliability_min,
                    args.pseudo_expand_affinity_alpha,
                    args.pseudo_expand_affinity_beta,
                    args.pseudo_expand_affinity_margin,
                    args.pseudo_inhibition,
                    args.pseudo_inhibition_strength,
                    args.pseudo_inhibition_margin,
                    args.pseudo_inhibition_temperature,
                    inhibition_tokens,
                    args.pseudo_kept_max_frac,
                    args.pseudo_kept_cap_mode,
                    args.pseudo_kept_cap_target_slack,
                    args.pseudo_kept_cap_protect_floor_ratio,
                    args.pseudo_kept_cap_rescue_frac,
                    args.affinity_walk_steps,
                    args.affinity_walk_alpha,
                    args.affinity_walk_gamma,
                    args.affinity_walk_self_loop,
                    args.affinity_walk_contrast_beta,
                )
                semantic_cls_loss = multilabel_loss(outputs["semantic_image_logits"], labels, pos_weight=pos_weight)
                mask_pool_loss = semantic_cls_loss * 0.0
                mask_pool_area = torch.zeros(num_classes, device=device)
                if args.image_logit_mode in {"mask_pool", "mix"}:
                    mask_pool_logits, mask_pool_area = mask_pool_image_logits(
                        select_patch_logits_for_mask_pool(outputs, args.mask_pool_source),
                        args.mask_pool_size_weight,
                        args.mask_pool_size_lambda,
                        args.mask_pool_detach_mask,
                    )
                    mask_pool_loss = multilabel_loss(mask_pool_logits, labels, pos_weight=pos_weight)
                if args.image_logit_mode == "semantic":
                    sem_loss = semantic_cls_loss
                elif args.image_logit_mode == "mask_pool":
                    sem_loss = mask_pool_loss
                else:
                    sem_loss = semantic_cls_loss + float(args.mask_pool_loss_weight) * mask_pool_loss
                pseudo_logits = outputs["patch_logits"] if args.pseudo_logit_target == "output" else outputs["spatial_patch_logits"]
                pseudo_fraction = torch.tensor(pseudo_stats["pseudo_class_fraction"], device=device)
                pseudo_class_weights = pseudo_class_weights_from_stats(
                    pseudo_fraction,
                    args.pseudo_class_weight_mode,
                    args.pseudo_class_weight_strength,
                    args.pseudo_class_weight_max,
                )
                clean_loss = pseudo_logits.sum() * 0.0
                clean_fraction = 0.0
                if args.clean_single_class_weight > 0.0:
                    # Exactly one class named at image level => every token is that class.
                    # No teacher, no threshold, no EMA: this label is exact by definition.
                    single_class = labels.sum(dim=1) == 1
                    clean_fraction = float(single_class.float().mean().detach().cpu())
                    if bool(single_class.any()):
                        clean_target = labels.argmax(dim=1)[:, None].expand(-1, pseudo.shape[1])
                        clean_loss = F.cross_entropy(
                            pseudo_logits[single_class].reshape(-1, num_classes),
                            clean_target[single_class].reshape(-1),
                        )
                        # Drop them from the noisy path so they are not counted twice.
                        pseudo = pseudo.masked_fill(single_class[:, None], args.ignore_index)
                pseudo_loss = partial_ce_loss(pseudo_logits, pseudo, args.ignore_index, pseudo_class_weights)
                contrast_loss = pseudo_logits.sum() * 0.0
                contrast_anchors = 0.0
                if scheduled_contrast_weight > 0.0:
                    contrast_tokens = select_reliability_tokens(outputs, args.contrast_token_layer)
                    if contrast_tokens is None:
                        raise RuntimeError("Could not select tokens for the contrastive loss.")
                    teacher_probs = teacher_logits.float().softmax(dim=-1)
                    sorted_probs = teacher_probs.sort(dim=-1, descending=True).values
                    teacher_margin = sorted_probs[..., 0] - sorted_probs[..., 1]
                    update_prototype_bank(
                        prototype_bank,
                        contrast_tokens,
                        pseudo,
                        args.ignore_index,
                        args.contrast_bank_decay,
                        prototype_initialized,
                    )
                    contrast_loss, contrast_anchors = cross_image_contrastive_loss(
                        contrast_tokens,
                        labels,
                        pseudo,
                        args.ignore_index,
                        prototype_bank,
                        prototype_initialized,
                        args.contrast_temperature,
                        args.contrast_max_anchors,
                        args.contrast_max_negatives,
                        margins=teacher_margin,
                        margin_min=args.contrast_margin_min,
                    )
                consistency_loss = pseudo_loss * 0.0
                consistency_effective_weights = consistency_class_weights
                consistency_agreement = torch.zeros(num_classes, device=device)
                consistency_floor = torch.zeros(num_classes, device=device)
                if scheduled_consistency_weight > 0.0:
                    strong_images = strong_augment_tensor(
                        images,
                        args.strong_brightness,
                        args.strong_contrast,
                        args.strong_noise,
                        mean=image_mean,
                        std=image_std,
                        hed_prob=args.strong_hed_prob,
                        hed_sigma_alpha=args.strong_hed_sigma_alpha,
                        hed_sigma_beta=args.strong_hed_sigma_beta,
                    )
                    strong_outputs = model(strong_images)
                    strong_logits = (
                        strong_outputs["patch_logits"]
                        if args.pseudo_logit_target == "output"
                        else strong_outputs["spatial_patch_logits"]
                    )
                    adaptive_floor = None
                    adaptive_cap = None
                    if args.consistency_weight_mode in {"stability_ratio", "stability_cap"}:
                        agreement, observed = class_pseudo_agreement(
                            teacher_logits,
                            strong_logits,
                            pseudo,
                            args.ignore_index,
                            args.pseudo_score,
                        )
                        decay = max(0.0, min(1.0, float(args.consistency_stability_ema)))
                        consistency_stability_ema = torch.where(
                            observed > 0,
                            consistency_stability_ema * decay + agreement * (1.0 - decay),
                            consistency_stability_ema,
                        )
                        adaptive_bound = stability_floor_from_ema(
                            consistency_stability_ema,
                            args.consistency_stability_target,
                            args.consistency_stability_temp,
                            args.consistency_floor_min,
                            args.consistency_floor_max,
                        )
                        if args.consistency_weight_mode == "stability_ratio":
                            adaptive_floor = adaptive_bound
                        else:
                            adaptive_cap = adaptive_bound
                        consistency_agreement = agreement
                        consistency_floor = adaptive_bound
                    consistency_effective_weights = consistency_weights_from_stability(
                        teacher_logits,
                        strong_logits,
                        pseudo,
                        args.ignore_index,
                        args.pseudo_score,
                        consistency_class_weights,
                        args.consistency_weight_mode,
                        args.consistency_weight_min,
                        args.consistency_weight_max,
                        adaptive_floor,
                        adaptive_cap,
                    )
                    consistency_loss = partial_ce_loss(
                        strong_logits,
                        pseudo,
                        args.ignore_index,
                        consistency_effective_weights,
                        normalize_by_weight=False,
                    )
                loss = (
                    sem_loss
                    + scheduled_pseudo_weight * pseudo_loss
                    + float(args.clean_single_class_weight) * clean_loss
                    + scheduled_contrast_weight * contrast_loss
                    + scheduled_consistency_weight * consistency_loss
                    + scheduled_online_jepa_weight * online_jepa_loss
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            if ema_teacher is not None:
                update_ema(model, ema_teacher, args.ema_decay)
            global_step += 1
            loss_sum += float(loss.detach().cpu())
            sem_sum += float(sem_loss.detach().cpu())
            semantic_cls_sum += float(semantic_cls_loss.detach().cpu())
            mask_pool_cls_sum += float(mask_pool_loss.detach().cpu())
            pseudo_sum += float(pseudo_loss.detach().cpu())
            clean_sum += float(clean_loss.detach().cpu())
            clean_fraction_sum += float(clean_fraction)
            contrast_sum += float(contrast_loss.detach().cpu())
            contrast_anchor_sum += float(contrast_anchors)
            crf_frac_sum += float(crf_refined_frac)
            consistency_sum += float(consistency_loss.detach().cpu())
            online_jepa_loss_sum += float(online_jepa_loss.detach().cpu())
            online_jepa_weight_sum += float(scheduled_online_jepa_weight)
            kept_sum += float(pseudo_stats["pseudo_kept"])
            expanded_sum += float(pseudo_stats["pseudo_expanded"])
            cap_removed_sum += float(pseudo_stats.get("pseudo_cap_removed", 0.0))
            cap_target_sum += float(pseudo_stats.get("pseudo_cap_target", 1.0))
            cap_floor_sum += float(pseudo_stats.get("pseudo_cap_protect_floor", 0.0))
            cap_rescue_sum += float(pseudo_stats.get("pseudo_cap_rescue", 0.0))
            jepa_expand_alpha_sum += float(pseudo_stats.get("jepa_expand_alpha", 0.0))
            affinity_expand_margin_sum += float(pseudo_stats.get("affinity_expand_margin", 0.0))
            affinity_walk_delta_sum += float(pseudo_stats.get("affinity_walk_delta", 0.0))
            affinity_walk_seed_sum += float(pseudo_stats.get("affinity_walk_seed", 0.0))
            affinity_walk_score_sum += float(pseudo_stats.get("affinity_walk_score", 0.0))
            inhibition_ambiguous_sum += float(pseudo_stats.get("inhibition_ambiguous", 0.0))
            inhibition_changed_sum += float(pseudo_stats.get("inhibition_changed_winner", 0.0))
            inhibition_drop_sum += float(pseudo_stats.get("inhibition_score_drop", 0.0))
            inhibition_seed_sum += float(pseudo_stats.get("inhibition_seed_fraction", 0.0))
            inhibition_centroid_sum += float(pseudo_stats.get("inhibition_centroid_class_fraction", 0.0))
            class_fraction_sum += torch.tensor(pseudo_stats["pseudo_class_fraction"], dtype=torch.float64)
            if mask_pool_area.ndim == 2:
                mask_pool_area_sum += mask_pool_area.detach().mean(dim=0).cpu().double()
            else:
                mask_pool_area_sum += mask_pool_area.detach().cpu().double()
            threshold_sum += effective_thresholds.detach().cpu().double()
            target_fraction_sum += target_fraction.detach().cpu().double()
            calibration_bias_sum += calibration_bias.detach().cpu().double()
            calibration_temp_sum += calibration_temp.detach().cpu().double()
            calibration_target_sum += calibration_target.detach().cpu().double()
            consistency_weight_sum += consistency_class_weights.detach().cpu().double()
            consistency_effective_weight_sum += consistency_effective_weights.detach().cpu().double()
            consistency_stability_sum += consistency_stability_ema.detach().cpu().double()
            consistency_floor_sum += consistency_floor.detach().cpu().double()
            consistency_agreement_sum += consistency_agreement.detach().cpu().double()
            if jepa_reliability is not None:
                jepa_reliability_sum += float(jepa_reliability.mean().detach().cpu())
                jepa_error_sum += float(jepa_error_mean.detach().cpu())
            if jepa_completed_probs is not None:
                jepa_completed_conf_sum += float(jepa_completed_probs.max(dim=-1).values.mean().detach().cpu())
                jepa_completed_class_sum += jepa_completed_probs.mean(dim=(0, 1)).detach().cpu().double()
            if pseudo_class_weights is not None:
                pseudo_weight_sum += pseudo_class_weights.detach().cpu().double()
            else:
                pseudo_weight_sum += torch.ones(num_classes, dtype=torch.float64)
            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                peak = torch.cuda.max_memory_allocated() / (1024**3) if device.type == "cuda" else 0.0
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - start
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.4f} sem={sem_loss.item():.4f} "
                    f"sem_cls={semantic_cls_loss.item():.4f} mask_cls={mask_pool_loss.item():.4f} "
                    f"mask_area={(mask_pool_area.detach().mean(dim=0) if mask_pool_area.ndim == 2 else mask_pool_area.detach()).cpu().tolist()} "
                    f"pseudo={pseudo_loss.item():.4f} "
                    f"cons={consistency_loss.item():.4f} "
                    f"online_jepa={online_jepa_loss.item():.4f} "
                    f"w_pseudo={scheduled_pseudo_weight:.3f} w_cons={scheduled_consistency_weight:.3f} "
                    f"w_online_jepa={scheduled_online_jepa_weight:.3f} "
                    f"expand_mode={active_expand_mode} "
                    f"thr={scheduled_thresholds.detach().cpu().tolist()} "
                    f"exp_min={scheduled_expand_min_frac:.3f} exp_max={scheduled_expand_max_frac:.3f} "
                    f"cons_w={consistency_effective_weights.detach().cpu().tolist()} "
                    f"cons_floor={consistency_floor.detach().cpu().tolist()} "
                    f"cons_stab={consistency_stability_ema.detach().cpu().tolist()} "
                    f"kept={pseudo_stats['pseudo_kept']:.3f} expand={pseudo_stats['pseudo_expanded']:.3f} "
                    f"cap_rm={float(pseudo_stats.get('pseudo_cap_removed', 0.0)):.3f} "
                    f"cap_floor={float(pseudo_stats.get('pseudo_cap_protect_floor', 0.0)):.2f} "
                    f"cap_rescue={float(pseudo_stats.get('pseudo_cap_rescue', 0.0)):.3f} "
                    f"inh_amb={float(pseudo_stats.get('inhibition_ambiguous', 0.0)):.3f} "
                    f"inh_chg={float(pseudo_stats.get('inhibition_changed_winner', 0.0)):.3f} "
                    f"inh_drop={float(pseudo_stats.get('inhibition_score_drop', 0.0)):.3f} "
                    f"inh_seed={float(pseudo_stats.get('inhibition_seed_fraction', 0.0)):.3f} "
                    f"inh_cent={float(pseudo_stats.get('inhibition_centroid_class_fraction', 0.0)):.3f} "
                    f"exp_alpha={float(pseudo_stats.get('jepa_expand_alpha', 0.0)):.3f} "
                    f"aff_margin={float(pseudo_stats.get('affinity_expand_margin', 0.0)):.3f} "
                    f"affwalk_delta={float(pseudo_stats.get('affinity_walk_delta', 0.0)):.3f} "
                    f"affwalk_score={float(pseudo_stats.get('affinity_walk_score', 0.0)):.3f} "
                    f"jepa_rel={(float(jepa_reliability.mean().detach().cpu()) if jepa_reliability is not None else 0.0):.3f} "
                    f"jepa_comp={(float(jepa_completed_probs.max(dim=-1).values.mean().detach().cpu()) if jepa_completed_probs is not None else 0.0):.3f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s peak_mem={peak:.2f}GB",
                    flush=True,
                )

        student_metrics = evaluate(model, val_loader, device, args.amp, num_classes)
        # The EMA teacher supplies every pseudo label but was never evaluated or saved.
        ema_metrics = (
            evaluate(ema_teacher, val_loader, device, args.amp, num_classes)
            if ema_teacher is not None
            else None
        )
        use_ema_for_selection = args.select_on == "ema" and ema_metrics is not None
        val_metrics = ema_metrics if use_ema_for_selection else student_metrics
        if args.teacher_mode == "periodic_hard":
            fixed_teacher.load_state_dict(model.state_dict(), strict=True)
            fixed_teacher.eval()
            print(f"periodic_teacher_update=hard epoch={epoch}", flush=True)
        elif args.teacher_mode == "periodic_ema":
            update_ema(model, fixed_teacher, args.periodic_ema_decay)
            fixed_teacher.eval()
            print(
                f"periodic_teacher_update=ema epoch={epoch} decay={args.periodic_ema_decay:.4f}",
                flush=True,
            )
        routes = route_summary(model)
        sec_epoch = time.perf_counter() - start
        # Only the columns the N2 recipe actually exercises: image BCE + pseudo-CE +
        # consistency + the CRF-teacher refinement, plus pseudo/consistency diagnostics.
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "sec_epoch": sec_epoch,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "train_sem_loss": sem_sum / max(1, len(train_loader)),
            "train_pseudo_loss": pseudo_sum / max(1, len(train_loader)),
            "train_consistency_loss": consistency_sum / max(1, len(train_loader)),
            "train_crf_refined_frac": crf_frac_sum / max(1, len(train_loader)),
            "scheduled_pseudo_weight": scheduled_pseudo_weight,
            "scheduled_consistency_weight": scheduled_consistency_weight,
            "pseudo_kept": kept_sum / max(1, len(train_loader)),
            "pseudo_expanded": expanded_sum / max(1, len(train_loader)),
            "pseudo_class_fraction": (class_fraction_sum / max(1, len(train_loader))).tolist(),
            "pseudo_thresholds_effective": (threshold_sum / max(1, len(train_loader))).tolist(),
            "pseudo_target_fraction": (target_fraction_sum / max(1, len(train_loader))).tolist(),
            "consistency_stability_ema": (consistency_stability_sum / max(1, len(train_loader))).tolist(),
            "consistency_batch_agreement": (consistency_agreement_sum / max(1, len(train_loader))).tolist(),
            "selected_model": "ema" if use_ema_for_selection else "student",
            "val_miou_student": student_metrics["miou"],
            "val_fwiou_student": student_metrics["fwiou"],
            "val_miou_ema": None if ema_metrics is None else ema_metrics["miou"],
            "val_fwiou_ema": None if ema_metrics is None else ema_metrics["fwiou"],
            "val_iou_ema": None if ema_metrics is None else ema_metrics["iou"],
            "val_miou": val_metrics["miou"],
            "val_mdice": val_metrics["mdice"],
            "val_mrecall": val_metrics["mrecall"],
            "val_mprecision": val_metrics["mprecision"],
            "val_fwiou": val_metrics["fwiou"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "output_fuse_alpha": routes.get("output_fuse_alpha"),
        }
        write_log(log_path, row)
        ema_note = "" if ema_metrics is None else f" val_miou_ema={ema_metrics['miou']:.4f}"
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} pseudo_kept={row['pseudo_kept']:.3f} "
            f"val_miou={val_metrics['miou']:.4f} val_mdice={val_metrics['mdice']:.4f} "
            f"val_fwiou={val_metrics['fwiou']:.4f} val_miou_student={student_metrics['miou']:.4f}{ema_note}",
            flush=True,
        )
        state = {
            # best.pt["model"] must be the model the metrics describe, so downstream eval matches.
            "model": ema_teacher.state_dict() if use_ema_for_selection else model.state_dict(),
            "student_model": model.state_dict() if use_ema_for_selection else None,
            "ema_model": None if ema_teacher is None else ema_teacher.state_dict(),
            "args": saved_args,
            "metrics": val_metrics,
            "student_metrics": student_metrics,
            "ema_metrics": ema_metrics,
            "epoch": epoch,
            "routes": routes,
        }
        if jepa_predictor is not None:
            state["predictor"] = jepa_predictor.state_dict()
        torch.save(state, output_dir / "last.pt")
        if float(val_metrics["miou"]) > best_miou:
            best_miou = float(val_metrics["miou"])
            best_epoch = epoch
            torch.save(state, output_dir / "best.pt")

    print(f"best_epoch={best_epoch} best_val_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
