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

from jepa_wsss.datasets import (  # noqa: E402
    CLASS_NAMES,
    ImageLevelDataset,
    SegmentationDataset,
    load_preprocessor_stats,
    parse_image_level_label,
    pil_to_normalized_tensor,
    resolve_dataset_paths,
)
from jepa_wsss.jepa import JEPAPredictor, gather_tokens  # noqa: E402
from jepa_wsss.losses import multilabel_loss  # noqa: E402
from jepa_wsss.metrics import SegmentationMeter  # noqa: E402
from jepa_wsss.models import DualRouteLinearWSSSModel  # noqa: E402
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
    jepa_completed_min_score: float = 0.0,
    jepa_completed_mix_alpha: float = 0.5,
    jepa_completed_class_alpha: torch.Tensor | None = None,
    reliability_min: float = 0.0,
    inhibition_mode: str = "none",
    inhibition_strength: float = 0.5,
    inhibition_margin: float = 0.05,
    inhibition_temperature: float = 0.05,
    inhibition_tokens: torch.Tensor | None = None,
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
    expanded = torch.zeros_like(keep)
    alpha_sum = torch.zeros((), device=logits.device, dtype=torch.float32)
    alpha_count = torch.zeros((), device=logits.device, dtype=torch.float32)
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
        if expand_mode == "adaptive_margin_reliability":
            if reliability_tokens is None:
                raise ValueError("adaptive_margin_reliability requires reliability_tokens.")
            norm_tokens = F.normalize(reliability_tokens.detach().float(), dim=-1)
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
                if expand_mode in {"adaptive_margin", "adaptive_margin_reliability"}:
                    rank_score = candidates * margins[batch_idx].clamp_min(0.0)
                    idx = rank_score.topk(topk).indices
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
        **inhibition_stats,
    }
    return pseudo, valid, stats


def partial_ce_loss(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    ignore_index: int,
    class_weights: torch.Tensor | None = None,
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
    return (losses[flat_valid] * weights).sum() / weights.sum().clamp_min(1e-6)


def adaptive_thresholds_from_stats(
    base_thresholds: torch.Tensor,
    pseudo_class_fraction: torch.Tensor,
    labels: torch.Tensor,
    strength: float,
    min_threshold: float,
    max_threshold: float,
    target_min_frac: float,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dual semantic-spatial routing with online partial pseudo supervision.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None, help="Raw DeiT checkpoint used to initialize the student backbone.")
    parser.add_argument("--teacher-checkpoint", default=None, help="Optional fixed/initial teacher checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--route-layers", default="all")
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
    parser.add_argument("--pseudo-weight", type=float, default=0.2)
    parser.add_argument("--pseudo-logit-target", default="spatial", choices=["spatial", "output"])
    parser.add_argument("--pseudo-score", default="softmax", choices=["softmax", "sigmoid"])
    parser.add_argument("--pseudo-thresholds", default="0.70")
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
            "adaptive_margin_jepa",
            "adaptive_margin_jepa_completed",
        ],
    )
    parser.add_argument("--pseudo-expand-min-frac", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-min-score", type=float, default=0.0)
    parser.add_argument("--pseudo-expand-max-frac", type=float, default=0.06)
    parser.add_argument("--pseudo-expand-margin-min", type=float, default=0.05)
    parser.add_argument("--pseudo-expand-under-strength", type=float, default=1.0)
    parser.add_argument("--pseudo-expand-reliability-layer", default="semantic", choices=["semantic", "spatial", "final"])
    parser.add_argument("--pseudo-expand-reliability-min", type=float, default=0.0)
    parser.add_argument("--jepa-reliability-checkpoint", default=None)
    parser.add_argument("--jepa-reliability-chunks", type=int, default=4)
    parser.add_argument("--jepa-reliability-scale", type=float, default=10.0)
    parser.add_argument("--jepa-completed-classifier", default="spatial", choices=["semantic", "spatial", "output"])
    parser.add_argument("--jepa-completed-min-score", type=float, default=0.0)
    parser.add_argument("--jepa-completed-mix-alpha", type=float, default=0.5)
    parser.add_argument("--jepa-completed-class-alpha", default="0.5")
    parser.add_argument("--pseudo-class-weight-mode", default="none", choices=["none", "inverse_fraction"])
    parser.add_argument("--pseudo-class-weight-strength", type=float, default=0.5)
    parser.add_argument("--pseudo-class-weight-max", type=float, default=3.0)
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
    parser.add_argument("--init-from-teacher", action="store_true")
    parser.add_argument("--spatial-route-mode", default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--fixed-spatial-layer", type=int, default=11)
    parser.add_argument("--topk-frac", type=float, default=0.05)
    parser.add_argument("--output-mode", default="spatial", choices=["spatial", "semantic", "fuse", "learned_fuse"])
    parser.add_argument("--output-fuse-alpha", type=float, default=0.5)
    parser.add_argument("--semantic-init", default="final", choices=["uniform", "final", "middle"])
    parser.add_argument("--spatial-init", default="uniform", choices=["uniform", "final", "middle"])
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(CLASS_NAMES[args.dataset])
    thresholds = parse_thresholds(args.pseudo_thresholds, num_classes)
    jepa_completed_class_alpha = parse_class_values(args.jepa_completed_class_alpha, num_classes, "--jepa-completed-class-alpha").to(device)
    image_mean, image_std = load_preprocessor_stats(args.checkpoint)
    print(f"image_mean={image_mean}", flush=True)
    print(f"image_std={image_std}", flush=True)
    print(f"teacher_mode={args.teacher_mode} pseudo_score={args.pseudo_score} thresholds={thresholds.tolist()}", flush=True)

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
        spatial_weight=0.0,
        semantic_init=args.semantic_init,
        spatial_init=args.spatial_init,
        output_mode=args.output_mode,
        output_fuse_alpha=args.output_fuse_alpha,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    if args.init_from_teacher:
        if args.teacher_checkpoint is None:
            raise ValueError("--init-from-teacher requires --teacher-checkpoint")
        copy_single_semantic_teacher_to_dual_student(model, args.teacher_checkpoint)
    if args.spatial_route_mode == "fixed":
        lock_fixed_spatial_route(model, args.fixed_spatial_layer)
        print(f"fixed_spatial_layer={args.fixed_spatial_layer}", flush=True)
    print(f"resolved_route_layers={model.route_layers}", flush=True)

    jepa_predictor = None
    if args.pseudo_expand_mode in {
        "adaptive_margin_jepa",
        "adaptive_margin_jepa_completed",
        "minimal_jepa",
        "minimal_mix",
        "minimal_mix_adaptive",
        "minimal_mix_class_adaptive",
        "minimal_mix_geometric",
    }:
        if args.jepa_reliability_checkpoint is None:
            raise ValueError(f"--pseudo-expand-mode {args.pseudo_expand_mode} requires --jepa-reliability-checkpoint")
        jepa_predictor = build_jepa_predictor_from_checkpoint(
            args.jepa_reliability_checkpoint,
            model.num_features,
            model.num_patches,
            device,
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

    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
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
    (output_dir / "config.json").write_text(json.dumps(saved_args, indent=2), encoding="utf-8")

    best_miou = -1.0
    best_epoch = 0
    global_step = 0
    log_path = output_dir / "log.csv"
    threshold_device = thresholds.to(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = sem_sum = pseudo_sum = kept_sum = expanded_sum = 0.0
        class_fraction_sum = torch.zeros(num_classes, dtype=torch.float64)
        threshold_sum = torch.zeros(num_classes, dtype=torch.float64)
        target_fraction_sum = torch.zeros(num_classes, dtype=torch.float64)
        pseudo_weight_sum = torch.zeros(num_classes, dtype=torch.float64)
        calibration_bias_sum = torch.zeros(num_classes, dtype=torch.float64)
        calibration_temp_sum = torch.zeros(num_classes, dtype=torch.float64)
        calibration_target_sum = torch.zeros(num_classes, dtype=torch.float64)
        jepa_reliability_sum = 0.0
        jepa_error_sum = 0.0
        jepa_completed_conf_sum = 0.0
        jepa_completed_class_sum = torch.zeros(num_classes, dtype=torch.float64)
        jepa_expand_alpha_sum = 0.0
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
                        threshold_device,
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
                effective_thresholds = threshold_device
                target_fraction = torch.zeros(num_classes, device=device)
                if args.adaptive_pseudo_thresholds:
                    prelim_pseudo, _prelim_valid, prelim_stats = make_pseudo(
                        teacher_logits,
                        labels,
                        threshold_device,
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
                    effective_thresholds, target_fraction = adaptive_thresholds_from_stats(
                        threshold_device,
                        prelim_fraction,
                        labels,
                        args.adaptive_threshold_strength,
                        args.adaptive_threshold_min,
                        args.adaptive_threshold_max,
                        args.adaptive_target_min_frac,
                    )
                reliability_tokens = None
                jepa_reliability = None
                jepa_completed_probs = None
                jepa_error_mean = torch.zeros((), device=device)
                if args.pseudo_expand_mode == "adaptive_margin_reliability":
                    reliability_tokens = select_reliability_tokens(outputs, args.pseudo_expand_reliability_layer)
                elif args.pseudo_expand_mode == "adaptive_margin_jepa":
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
                elif args.pseudo_expand_mode in {
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
                    args.pseudo_expand_mode,
                    args.pseudo_expand_min_frac,
                    args.pseudo_expand_min_score,
                    args.pseudo_expand_max_frac,
                    args.pseudo_expand_margin_min,
                    args.pseudo_expand_under_strength,
                    reliability_tokens,
                    jepa_reliability,
                    jepa_completed_probs,
                    args.jepa_completed_min_score,
                    args.jepa_completed_mix_alpha,
                    jepa_completed_class_alpha,
                    args.pseudo_expand_reliability_min,
                    args.pseudo_inhibition,
                    args.pseudo_inhibition_strength,
                    args.pseudo_inhibition_margin,
                    args.pseudo_inhibition_temperature,
                    inhibition_tokens,
                )
                sem_loss = multilabel_loss(outputs["semantic_image_logits"], labels, pos_weight=pos_weight)
                pseudo_logits = outputs["patch_logits"] if args.pseudo_logit_target == "output" else outputs["spatial_patch_logits"]
                pseudo_fraction = torch.tensor(pseudo_stats["pseudo_class_fraction"], device=device)
                pseudo_class_weights = pseudo_class_weights_from_stats(
                    pseudo_fraction,
                    args.pseudo_class_weight_mode,
                    args.pseudo_class_weight_strength,
                    args.pseudo_class_weight_max,
                )
                pseudo_loss = partial_ce_loss(pseudo_logits, pseudo, args.ignore_index, pseudo_class_weights)
                loss = sem_loss + float(args.pseudo_weight) * pseudo_loss
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
            pseudo_sum += float(pseudo_loss.detach().cpu())
            kept_sum += float(pseudo_stats["pseudo_kept"])
            expanded_sum += float(pseudo_stats["pseudo_expanded"])
            jepa_expand_alpha_sum += float(pseudo_stats.get("jepa_expand_alpha", 0.0))
            inhibition_ambiguous_sum += float(pseudo_stats.get("inhibition_ambiguous", 0.0))
            inhibition_changed_sum += float(pseudo_stats.get("inhibition_changed_winner", 0.0))
            inhibition_drop_sum += float(pseudo_stats.get("inhibition_score_drop", 0.0))
            inhibition_seed_sum += float(pseudo_stats.get("inhibition_seed_fraction", 0.0))
            inhibition_centroid_sum += float(pseudo_stats.get("inhibition_centroid_class_fraction", 0.0))
            class_fraction_sum += torch.tensor(pseudo_stats["pseudo_class_fraction"], dtype=torch.float64)
            threshold_sum += effective_thresholds.detach().cpu().double()
            target_fraction_sum += target_fraction.detach().cpu().double()
            calibration_bias_sum += calibration_bias.detach().cpu().double()
            calibration_temp_sum += calibration_temp.detach().cpu().double()
            calibration_target_sum += calibration_target.detach().cpu().double()
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
                    f"loss={loss.item():.4f} sem={sem_loss.item():.4f} pseudo={pseudo_loss.item():.4f} "
                    f"kept={pseudo_stats['pseudo_kept']:.3f} expand={pseudo_stats['pseudo_expanded']:.3f} "
                    f"inh_amb={float(pseudo_stats.get('inhibition_ambiguous', 0.0)):.3f} "
                    f"inh_chg={float(pseudo_stats.get('inhibition_changed_winner', 0.0)):.3f} "
                    f"inh_drop={float(pseudo_stats.get('inhibition_score_drop', 0.0)):.3f} "
                    f"inh_seed={float(pseudo_stats.get('inhibition_seed_fraction', 0.0)):.3f} "
                    f"inh_cent={float(pseudo_stats.get('inhibition_centroid_class_fraction', 0.0)):.3f} "
                    f"exp_alpha={float(pseudo_stats.get('jepa_expand_alpha', 0.0)):.3f} "
                    f"jepa_rel={(float(jepa_reliability.mean().detach().cpu()) if jepa_reliability is not None else 0.0):.3f} "
                    f"jepa_comp={(float(jepa_completed_probs.max(dim=-1).values.mean().detach().cpu()) if jepa_completed_probs is not None else 0.0):.3f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s peak_mem={peak:.2f}GB",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, args.amp, num_classes)
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
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(1, len(train_loader)),
            "train_sem_loss": sem_sum / max(1, len(train_loader)),
            "train_pseudo_loss": pseudo_sum / max(1, len(train_loader)),
            "pseudo_kept": kept_sum / max(1, len(train_loader)),
            "pseudo_expanded": expanded_sum / max(1, len(train_loader)),
            "pseudo_class_fraction": (class_fraction_sum / max(1, len(train_loader))).tolist(),
            "pseudo_thresholds_effective": (threshold_sum / max(1, len(train_loader))).tolist(),
            "pseudo_target_fraction": (target_fraction_sum / max(1, len(train_loader))).tolist(),
            "pseudo_class_weights": (pseudo_weight_sum / max(1, len(train_loader))).tolist(),
            "pseudo_calibration_bias": (calibration_bias_sum / max(1, len(train_loader))).tolist(),
            "pseudo_calibration_temp": (calibration_temp_sum / max(1, len(train_loader))).tolist(),
            "pseudo_calibration_target": (calibration_target_sum / max(1, len(train_loader))).tolist(),
            "inhibition_ambiguous_mean": inhibition_ambiguous_sum / max(1, len(train_loader)),
            "inhibition_changed_winner_mean": inhibition_changed_sum / max(1, len(train_loader)),
            "inhibition_score_drop_mean": inhibition_drop_sum / max(1, len(train_loader)),
            "inhibition_seed_fraction_mean": inhibition_seed_sum / max(1, len(train_loader)),
            "inhibition_centroid_class_fraction_mean": inhibition_centroid_sum / max(1, len(train_loader)),
            "jepa_reliability_mean": jepa_reliability_sum / max(1, len(train_loader)),
            "jepa_error_mean": jepa_error_sum / max(1, len(train_loader)),
            "jepa_completed_conf_mean": jepa_completed_conf_sum / max(1, len(train_loader)),
            "jepa_completed_class_mean": (jepa_completed_class_sum / max(1, len(train_loader))).tolist(),
            "jepa_expand_alpha_mean": jepa_expand_alpha_sum / max(1, len(train_loader)),
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
            **routes,
        }
        write_log(log_path, row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} pseudo_kept={row['pseudo_kept']:.3f} "
            f"val_miou={val_metrics['miou']:.4f} val_mdice={val_metrics['mdice']:.4f} "
            f"val_fwiou={val_metrics['fwiou']:.4f}",
            flush=True,
        )
        state = {
            "model": model.state_dict(),
            "args": saved_args,
            "metrics": val_metrics,
            "epoch": epoch,
            "routes": routes,
        }
        torch.save(state, output_dir / "last.pt")
        if float(val_metrics["miou"]) > best_miou:
            best_miou = float(val_metrics["miou"])
            best_epoch = epoch
            torch.save(state, output_dir / "best.pt")

    print(f"best_epoch={best_epoch} best_val_miou={best_miou:.4f}", flush=True)


if __name__ == "__main__":
    main()
