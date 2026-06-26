from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import CLASS_NAMES, ImageLevelDataset, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.jepa import (
    JEPAPredictor,
    balanced_entropy_mask_indices,
    class_weighted_entropy_mask_indices,
    entropy_mask_indices,
    frequency_uncertainty_mask_indices,
    class_weighted_jepa_smooth_l1_loss,
    gather_tokens,
    jepa_prototype_affinity_loss,
    jepa_semantic_affinity_consistency_loss,
    jepa_smooth_l1_loss,
    local_context_targets,
    make_ema_backbone,
    propagation_loss,
    soft_frequency_uncertainty_mask_indices,
    update_ema_backbone,
)
from jepa_wsss.metrics import SegmentationMeter
from jepa_wsss.models import PrototypeWSSSModel
from jepa_wsss.losses import AuxiliaryUncertaintyWeighting, compute_frequency_weight, compute_pos_weight, multilabel_loss


def find_pretrain_checkpoint(model_name: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    names = {
        "deit_small_patch16_224": "deit_small_patch16_224-cd65a155.pth",
        "deit_base_patch16_224": "deit_base_patch16_224-b5f2ef4d.pth",
    }
    filename = names[model_name]
    candidates = [Path("pretrained") / filename, Path("jepa_wsss") / "pretrained" / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}; checked: {', '.join(str(p) for p in candidates)}")


class TrainTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return pil_to_normalized_tensor(image)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segmentation_from_patch_logits(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def segmentation_from_outputs(outputs: dict[str, torch.Tensor], size: tuple[int, int]) -> torch.Tensor:
    if "refined_logits" in outputs:
        logits = F.interpolate(outputs["refined_logits"], size=size, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)
    return segmentation_from_patch_logits(outputs["patch_logits"], size)


@torch.no_grad()
def evaluate(model: PrototypeWSSSModel, loader: DataLoader, device: torch.device, amp: bool, num_classes: int) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
        pred = segmentation_from_outputs(outputs, masks.shape[-2:])
        meter.update(pred, masks)
    return meter.compute()


def write_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def load_baseline(model: PrototypeWSSSModel, baseline_checkpoint: str | None, new_gate_init: float = 0.0) -> dict[str, object] | None:
    if baseline_checkpoint is None:
        return None
    checkpoint = torch.load(baseline_checkpoint, map_location="cpu")
    source_state = checkpoint["model"]
    target_state = model.state_dict()
    load_state = {}
    partial = []
    skipped = []
    copied_prototypes: tuple[int, int] | None = None
    for key, value in source_state.items():
        target_key = key
        if target_key not in target_state and key.startswith("backbone."):
            wrapped_key = "backbone.vit." + key[len("backbone.") :]
            if wrapped_key in target_state:
                target_key = wrapped_key
        if target_key not in target_state:
            skipped.append(key)
            continue
        target = target_state[target_key]
        if target.shape == value.shape:
            load_state[target_key] = value
        elif target_key == "head.prototypes" and target.ndim == 3 and value.ndim == 3 and target.shape[2] == value.shape[2]:
            copied = target.clone()
            num_classes = min(target.shape[0], value.shape[0])
            num_prototypes = min(target.shape[1], value.shape[1])
            copied[:num_classes, :num_prototypes] = value[:num_classes, :num_prototypes]
            load_state[target_key] = copied
            partial.append(f"{target_key}: copied {num_classes} classes x {num_prototypes} prototypes")
            copied_prototypes = (num_classes, num_prototypes)
        elif target_key == "head.gate_logits":
            skipped.append(key)
        else:
            skipped.append(key)
    if "head.gate_logits" in target_state and "head.gate_logits" not in load_state:
        gates = torch.full_like(target_state["head.gate_logits"], float(new_gate_init))
        if copied_prototypes is None and "head.prototypes" in source_state:
            num_classes = min(gates.shape[0], source_state["head.prototypes"].shape[0])
            num_prototypes = min(gates.shape[1], source_state["head.prototypes"].shape[1])
        elif copied_prototypes is not None:
            num_classes, num_prototypes = copied_prototypes
        else:
            num_classes, num_prototypes = gates.shape
        gates[:num_classes, :num_prototypes] = 2.0
        load_state["head.gate_logits"] = gates
        partial.append(f"head.gate_logits: warm-started {num_classes} classes x {num_prototypes} copied prototypes")
    missing, unexpected = model.load_state_dict(load_state, strict=False)
    if partial:
        print(f"baseline_partial_load={partial}", flush=True)
    if skipped:
        print(f"baseline_skipped={skipped}", flush=True)
    if missing:
        print(f"baseline_missing={missing}", flush=True)
    if unexpected:
        print(f"baseline_unexpected={unexpected}", flush=True)
    return checkpoint


def load_predictor(predictor: JEPAPredictor, predictor_checkpoint: str | None) -> dict[str, object] | None:
    if predictor_checkpoint is None:
        return None
    checkpoint = torch.load(predictor_checkpoint, map_location="cpu")
    if "predictor" not in checkpoint:
        raise KeyError(f"Checkpoint has no predictor state: {predictor_checkpoint}")
    missing, unexpected = predictor.load_state_dict(checkpoint["predictor"], strict=False)
    if missing:
        print(f"predictor_missing={missing}", flush=True)
    if unexpected:
        print(f"predictor_unexpected={unexpected}", flush=True)
    print(f"predictor_loaded={predictor_checkpoint}", flush=True)
    return checkpoint


def parse_prototype_counts(value: str | None, num_classes: int) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    counts = [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if len(counts) != num_classes:
        raise ValueError(f"--prototype-counts expects {num_classes} comma-separated integers, got {counts}")
    if min(counts) < 1:
        raise ValueError("--prototype-counts values must be positive.")
    return counts


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train JEPA-v0 prototype WSSS.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None, help="ImageNet DeiT checkpoint.")
    parser.add_argument("--baseline-checkpoint", default=None, help="Optional baseline best.pt to initialize from.")
    parser.add_argument("--predictor-checkpoint", default=None, help="Optional JEPA checkpoint to initialize predictor from.")
    parser.add_argument("--output-dir", default="runs/bcss_deit_base_jepa_v0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--scheduler", default="none", choices=["none", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--div-weight", type=float, default=0.01)
    parser.add_argument("--div-type", default="vector", choices=["vector", "spatial", "both"])
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    parser.add_argument("--jepa-weight", type=float, default=0.1)
    parser.add_argument("--class-aware-jepa", action="store_true", help="Weight JEPA latent loss toward rare/present predicted classes.")
    parser.add_argument("--jepa-class-weight-mode", default="sqrt_inv", choices=["sqrt_inv", "inv", "none"])
    parser.add_argument("--max-jepa-class-weight", type=float, default=3.0)
    parser.add_argument("--completion-weight", type=float, default=0.0, help="Image-level BCE on JEPA-completed prototype logits.")
    parser.add_argument("--completion-alpha", type=float, default=0.5, help="Blend predicted-token logits into masked positions for completion.")
    parser.add_argument("--completion-start-epoch", type=int, default=1, help="First epoch to enable JEPA completion loss.")
    parser.add_argument("--propagate-weight", type=float, default=0.0)
    parser.add_argument("--affinity-weight", type=float, default=0.0, help="Prototype-level KL from JEPA-predicted latent assignments to masked patch assignments.")
    parser.add_argument("--affinity-start-epoch", type=int, default=1, help="First epoch to enable JEPA prototype affinity loss.")
    parser.add_argument("--affinity-teacher-temp", type=float, default=1.0)
    parser.add_argument("--affinity-student-temp", type=float, default=1.0)
    parser.add_argument("--affinity-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--affinity-class-confidence-threshold", type=float, default=0.0)
    parser.add_argument("--affinity-agreement-threshold", type=float, default=0.0)
    parser.add_argument("--semantic-affinity-weight", type=float, default=0.0, help="JEPA-guided patch-patch semantic affinity consistency loss.")
    parser.add_argument("--semantic-affinity-start-epoch", type=int, default=1)
    parser.add_argument("--semantic-affinity-temp", type=float, default=0.2)
    parser.add_argument("--semantic-affinity-visible-conf", type=float, default=0.6)
    parser.add_argument("--semantic-affinity-target-conf", type=float, default=0.5)
    parser.add_argument("--semantic-affinity-topk", type=int, default=16)
    parser.add_argument("--loss-weighting", default="fixed", choices=["fixed", "uncertainty_aux", "uncertainty_jepa"])
    parser.add_argument("--mask-ratio", type=float, default=0.25)
    parser.add_argument("--mask-ratio-start", type=float, default=None, help="Optional curriculum start ratio; overrides --mask-ratio when paired with --mask-ratio-end.")
    parser.add_argument("--mask-ratio-end", type=float, default=None, help="Optional curriculum end ratio; overrides --mask-ratio when paired with --mask-ratio-start.")
    parser.add_argument("--masking", default="entropy", choices=["entropy", "balanced_entropy", "class_weighted_entropy", "freq_uncertainty", "soft_freq_uncertainty"])
    parser.add_argument("--minority-mask-boost", type=float, default=2.0)
    parser.add_argument("--freq-weight-mode", default="sqrt_inv", choices=["sqrt_inv", "inv", "none"])
    parser.add_argument("--max-freq-weight", type=float, default=5.0)
    parser.add_argument("--context-radius", type=int, default=1)
    parser.add_argument("--homogeneity-threshold", type=float, default=0.6)
    parser.add_argument("--agreement-threshold", type=float, default=0.7)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--predictor-dim", type=int, default=384)
    parser.add_argument("--predictor-type", default="mean_mlp", choices=["mean_mlp", "cross_attn"])
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-heads", type=int, default=6)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)
    parser.add_argument("--train-predictor-only", action="store_true", help="Freeze WSSS model and train only the JEPA predictor.")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--fusion-layers", default=None, help="Optional 1-based ViT layers to fuse, e.g. 4,8,12.")
    parser.add_argument("--fusion-mode", default="weighted_sum", choices=["weighted_sum", "concat_proj"])
    parser.add_argument("--fusion-init", default="average", choices=["average", "final"], help="Initialize fusion as uniform average or as the final selected layer.")
    parser.add_argument("--prototypes-per-class", type=int, default=10)
    parser.add_argument("--prototype-counts", default=None, help="Optional per-class prototype counts, e.g. 10,10,16,16 for BCSS.")
    parser.add_argument("--prototype-gating", action="store_true", help="Learn soft gates over an overcomplete prototype bank.")
    parser.add_argument("--gate-init", type=float, default=2.0, help="Initial prototype gate logit when --prototype-gating is enabled.")
    parser.add_argument("--new-gate-init", type=float, default=0.0, help="Gate logit for extra prototypes when loading a smaller baseline checkpoint.")
    parser.add_argument("--gate-weight", type=float, default=0.0, help="Sparsity weight for learned prototype gates.")
    parser.add_argument("--gate-warmup-epochs", type=int, default=1, help="Number of initial epochs with no gate sparsity penalty.")
    parser.add_argument("--prototype-usage-weight", type=float, default=0.0, help="Encourage present-class prototype usage diversity without changing max pooling.")
    parser.add_argument("--usage-warmup-epochs", type=int, default=0, help="Number of initial epochs with prototype usage regularization enabled before sparsity dominates.")
    parser.add_argument("--prototype-aggregation", default="max", choices=["max", "logmeanexp"])
    parser.add_argument("--prototype-lse-tau", type=float, default=1.0)
    parser.add_argument("--refine-head", action="store_true", help="Enable lightweight decoder on top of patch tokens and prototype logits.")
    parser.add_argument("--refine-dim", type=int, default=256)
    parser.add_argument("--refine-scale", type=int, default=2)
    parser.add_argument("--refine-pooling", default="topk", choices=["max", "topk"])
    parser.add_argument("--refine-topk-frac", type=float, default=0.05)
    parser.add_argument("--refine-cls-weight", type=float, default=1.0)
    parser.add_argument("--refine-consistency-weight", type=float, default=0.1)
    parser.add_argument("--train-refine-only", action="store_true", help="Freeze backbone/prototypes and train only the refinement decoder.")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()
    if args.train_predictor_only and args.train_refine_only:
        raise ValueError("--train-predictor-only and --train-refine-only are mutually exclusive.")
    if (args.mask_ratio_start is None) != (args.mask_ratio_end is None):
        raise ValueError("--mask-ratio-start and --mask-ratio-end must be provided together.")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretrain_checkpoint = find_pretrain_checkpoint(args.model, args.checkpoint)
    device = torch.device(args.device)
    num_classes = len(CLASS_NAMES[args.dataset])
    prototype_counts = parse_prototype_counts(args.prototype_counts, num_classes)

    train_set = ImageLevelDataset(args.data_root, args.dataset, transform=TrainTransform())
    val_set = SegmentationDataset(args.data_root, args.dataset, split="val")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = PrototypeWSSSModel(
        model_name=args.model,
        checkpoint_path=str(pretrain_checkpoint),
        num_classes=num_classes,
        prototypes_per_class=args.prototypes_per_class,
        prototype_counts=prototype_counts,
        prototype_gating=args.prototype_gating,
        gate_init=args.gate_init,
        prototype_aggregation=args.prototype_aggregation,
        lse_tau=args.prototype_lse_tau,
        refine_head=args.refine_head,
        refine_dim=args.refine_dim,
        refine_scale=args.refine_scale,
        refine_pooling=args.refine_pooling,
        refine_topk_frac=args.refine_topk_frac,
        grad_checkpointing=args.grad_checkpointing,
        fusion_layers=args.fusion_layers,
        fusion_mode=args.fusion_mode,
        fusion_init=args.fusion_init,
    )
    baseline_state = load_baseline(model, args.baseline_checkpoint, args.new_gate_init)
    if args.train_predictor_only:
        for param in model.parameters():
            param.requires_grad_(False)
    if args.train_refine_only:
        if not args.refine_head:
            raise ValueError("--train-refine-only requires --refine-head.")
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith("refine_decoder."))
    model.to(device)
    predictor = JEPAPredictor(
        model.backbone.num_features,
        hidden_dim=args.predictor_dim,
        predictor_type=args.predictor_type,
        num_heads=args.predictor_heads,
        depth=args.predictor_depth,
        dropout=args.predictor_dropout,
    ).to(device)
    predictor_state = load_predictor(predictor, args.predictor_checkpoint)
    ema_backbone = make_ema_backbone(model.backbone).to(device)
    aux_weighter = None
    if args.loss_weighting == "uncertainty_aux":
        aux_weighter = AuxiliaryUncertaintyWeighting().to(device)
    elif args.loss_weighting == "uncertainty_jepa":
        aux_weighter = AuxiliaryUncertaintyWeighting(names=("jepa",)).to(device)

    params = [param for param in model.parameters() if param.requires_grad]
    if not args.train_refine_only:
        params += list(predictor.parameters())
    if aux_weighter is not None:
        params += list(aux_weighter.parameters())
    if not params:
        raise ValueError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_update_steps = math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)) * args.epochs
    warmup_update_steps = int(round(math.ceil(len(train_loader) / max(args.grad_accum_steps, 1)) * args.warmup_epochs))
    scheduler = None
    if args.scheduler == "warmup_cosine":
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_update_steps,
            warmup_steps=warmup_update_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    pos_weight = compute_pos_weight(args.data_root, args.dataset, args.max_pos_weight, device) if args.class_balance else None
    if pos_weight is not None:
        print(f"pos_weight={pos_weight.detach().cpu().tolist()}", flush=True)
    mask_class_weights = compute_frequency_weight(args.data_root, args.dataset, args.freq_weight_mode, args.max_freq_weight, device)
    print(f"mask_class_weights={mask_class_weights.detach().cpu().tolist()}", flush=True)
    jepa_class_weights = compute_frequency_weight(args.data_root, args.dataset, args.jepa_class_weight_mode, args.max_jepa_class_weight, device)
    print(f"jepa_class_weights={jepa_class_weights.detach().cpu().tolist()}", flush=True)

    config = vars(args).copy()
    config["checkpoint"] = str(pretrain_checkpoint)
    config["class_names"] = CLASS_NAMES[args.dataset]
    config["resolved_prototype_counts"] = prototype_counts or [args.prototypes_per_class] * num_classes
    config["baseline_epoch"] = None if baseline_state is None else baseline_state.get("epoch")
    config["predictor_epoch"] = None if predictor_state is None else predictor_state.get("epoch")
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_miou = -1.0
    global_step = 0
    log_path = output_dir / "log.csv"
    for epoch in range(1, args.epochs + 1):
        if args.mask_ratio_start is None:
            current_mask_ratio = args.mask_ratio
        else:
            progress = 0.0 if args.epochs <= 1 else float(epoch - 1) / float(args.epochs - 1)
            current_mask_ratio = args.mask_ratio_start + progress * (args.mask_ratio_end - args.mask_ratio_start)
        current_mask_ratio = max(0.0, min(1.0, float(current_mask_ratio)))
        if args.train_predictor_only:
            model.eval()
        else:
            model.train()
        if args.train_refine_only:
            model.backbone.eval()
            model.head.eval()
            if model.refine_decoder is not None:
                model.refine_decoder.train()
        predictor.train()
        if aux_weighter is not None:
            aux_weighter.train()
        ema_backbone.eval()
        start = time.perf_counter()
        loss_sum = cls_sum = proto_cls_sum = refine_cls_sum = refine_cons_sum = div_sum = jepa_sum = jepa_token_weight_sum = affinity_sum = affinity_gate_sum = affinity_weight_sum = sem_affinity_sum = sem_affinity_gate_sum = sem_affinity_weight_sum = completion_sum = prop_sum = gate_sum = proto_gate_sum = usage_sum = div_uw_sum = jepa_uw_sum = prop_uw_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                proto_cls_loss = multilabel_loss(outputs["prototype_image_logits"], labels, pos_weight)
                if args.refine_head:
                    refine_cls_loss = multilabel_loss(outputs["refined_image_logits"], labels, pos_weight)
                    refine_consistency_loss = model.refine_consistency_loss(outputs["refined_logits"], outputs["patch_logits"])
                    cls_loss = proto_cls_loss + args.refine_cls_weight * refine_cls_loss
                else:
                    refine_cls_loss = outputs["patch_logits"].sum() * 0.0
                    refine_consistency_loss = outputs["patch_logits"].sum() * 0.0
                    cls_loss = proto_cls_loss
                if args.train_refine_only:
                    div_loss = outputs["patch_logits"].sum() * 0.0
                    proto_gate_loss = outputs["patch_logits"].sum() * 0.0
                    usage_loss = outputs["patch_logits"].sum() * 0.0
                    jepa_loss = outputs["patch_logits"].sum() * 0.0
                    jepa_token_weight = torch.ones((), device=device)
                    completion_loss = outputs["patch_logits"].sum() * 0.0
                    affinity_loss = outputs["patch_logits"].sum() * 0.0
                    affinity_gate = torch.zeros((), device=device)
                    affinity_token_weight = torch.ones((), device=device)
                    sem_affinity_loss = outputs["patch_logits"].sum() * 0.0
                    sem_affinity_gate = torch.zeros((), device=device)
                    sem_affinity_token_weight = torch.ones((), device=device)
                    active_completion_weight = 0.0
                    prop_loss = outputs["patch_logits"].sum() * 0.0
                    prop_gate = torch.zeros_like(labels, dtype=torch.bool)
                    active_gate_weight = 0.0
                    active_usage_weight = 0.0
                    loss = (
                        args.refine_cls_weight * refine_cls_loss
                        + args.refine_consistency_weight * refine_consistency_loss
                    ) / args.grad_accum_steps
                    aux_weights = {
                        "div_uw": 0.0,
                        "jepa_uw": 0.0,
                        "propagate_uw": 0.0,
                    }
                else:
                    if args.train_predictor_only:
                        div_loss = outputs["patch_logits"].sum() * 0.0
                        proto_gate_loss = outputs["patch_logits"].sum() * 0.0
                        usage_loss = outputs["patch_logits"].sum() * 0.0
                        active_gate_weight = 0.0
                        active_usage_weight = 0.0
                    else:
                        vector_div_loss = model.diversity_loss()
                        spatial_div_loss = model.spatial_diversity_loss(outputs["prototype_sims"], labels)
                        if args.div_type == "spatial":
                            div_loss = spatial_div_loss
                        elif args.div_type == "both":
                            div_loss = vector_div_loss + spatial_div_loss
                        else:
                            div_loss = vector_div_loss
                        proto_gate_loss = model.gate_loss()
                        active_gate_weight = 0.0 if epoch <= args.gate_warmup_epochs else args.gate_weight
                        usage_loss = model.usage_loss(outputs["prototype_sims"], labels)
                        active_usage_weight = args.prototype_usage_weight if args.usage_warmup_epochs <= 0 or epoch <= args.usage_warmup_epochs else 0.0
                    with torch.no_grad():
                        if args.masking == "soft_freq_uncertainty":
                            mask_indices = soft_frequency_uncertainty_mask_indices(
                                outputs["patch_logits"].detach(),
                                labels,
                                current_mask_ratio,
                                mask_class_weights,
                            )
                        elif args.masking == "freq_uncertainty":
                            mask_indices = frequency_uncertainty_mask_indices(
                                outputs["patch_logits"].detach(),
                                labels,
                                current_mask_ratio,
                                mask_class_weights,
                            )
                        elif args.masking == "balanced_entropy":
                            mask_indices = balanced_entropy_mask_indices(
                                outputs["patch_logits"].detach(),
                                labels,
                                current_mask_ratio,
                                minority_boost=args.minority_mask_boost,
                            )
                        elif args.masking == "class_weighted_entropy":
                            mask_indices = class_weighted_entropy_mask_indices(
                                outputs["patch_logits"].detach(),
                                labels,
                                current_mask_ratio,
                                mask_class_weights,
                            )
                        else:
                            mask_indices = entropy_mask_indices(outputs["patch_logits"].detach(), current_mask_ratio)
                        target_tokens = ema_backbone.forward_features(images)[:, 1:, :]
                        target_tokens = gather_tokens(target_tokens, mask_indices)
                    pred_tokens = predictor(outputs["patch_tokens"], mask_indices)
                    if args.class_aware_jepa:
                        jepa_loss, jepa_token_weight = class_weighted_jepa_smooth_l1_loss(
                            pred_tokens,
                            target_tokens,
                            outputs["patch_logits"],
                            mask_indices,
                            labels,
                            jepa_class_weights,
                        )
                    else:
                        jepa_loss = jepa_smooth_l1_loss(pred_tokens, target_tokens)
                        jepa_token_weight = torch.ones((), device=device)
                    pred_prototype_sims = model.head.prototype_similarity(pred_tokens)
                    pred_patch_logits = model.head.aggregate_prototypes(pred_prototype_sims)
                    active_affinity_weight = args.affinity_weight if epoch >= args.affinity_start_epoch else 0.0
                    if active_affinity_weight > 0:
                        affinity_loss, affinity_gate, affinity_token_weight = jepa_prototype_affinity_loss(
                            outputs["prototype_sims"],
                            pred_prototype_sims,
                            mask_indices,
                            labels,
                            model.head.prototype_valid,
                            class_weights=jepa_class_weights,
                            teacher_temp=args.affinity_teacher_temp,
                            student_temp=args.affinity_student_temp,
                            confidence_threshold=args.affinity_confidence_threshold,
                            class_confidence_threshold=args.affinity_class_confidence_threshold,
                            agreement_threshold=args.affinity_agreement_threshold,
                        )
                    else:
                        affinity_loss = outputs["patch_logits"].sum() * 0.0
                        affinity_gate = torch.zeros((), device=device)
                        affinity_token_weight = torch.ones((), device=device)
                    active_sem_affinity_weight = args.semantic_affinity_weight if epoch >= args.semantic_affinity_start_epoch else 0.0
                    if active_sem_affinity_weight > 0:
                        sem_affinity_loss, sem_affinity_gate, sem_affinity_token_weight = jepa_semantic_affinity_consistency_loss(
                            outputs["patch_tokens"],
                            pred_tokens,
                            outputs["patch_logits"],
                            mask_indices,
                            labels,
                            class_weights=jepa_class_weights,
                            affinity_temp=args.semantic_affinity_temp,
                            visible_confidence_threshold=args.semantic_affinity_visible_conf,
                            target_confidence_threshold=args.semantic_affinity_target_conf,
                            topk=args.semantic_affinity_topk,
                        )
                    else:
                        sem_affinity_loss = outputs["patch_logits"].sum() * 0.0
                        sem_affinity_gate = torch.zeros((), device=device)
                        sem_affinity_token_weight = torch.ones((), device=device)
                    active_completion_weight = args.completion_weight if epoch >= args.completion_start_epoch else 0.0
                    if active_completion_weight > 0:
                        gather_idx = mask_indices.unsqueeze(-1).expand(-1, -1, outputs["patch_logits"].shape[-1])
                        original_masked_logits = outputs["patch_logits"].gather(dim=1, index=gather_idx)
                        alpha = max(0.0, min(1.0, float(args.completion_alpha)))
                        blended_logits = (1.0 - alpha) * original_masked_logits + alpha * pred_patch_logits
                        completed_logits = outputs["patch_logits"].clone()
                        completed_logits.scatter_(dim=1, index=gather_idx, src=blended_logits)
                        completion_image_logits = completed_logits.amax(dim=1)
                        completion_loss = multilabel_loss(completion_image_logits, labels, pos_weight)
                    else:
                        completion_loss = outputs["patch_logits"].sum() * 0.0
                    if args.train_predictor_only:
                        prop_loss = outputs["patch_logits"].sum() * 0.0
                        prop_gate = torch.zeros_like(labels, dtype=torch.bool)
                        aux_weights = {
                            "div_uw": 0.0,
                            "jepa_uw": args.jepa_weight,
                            "propagate_uw": 0.0,
                        }
                        loss = (args.jepa_weight * jepa_loss + active_completion_weight * completion_loss) / args.grad_accum_steps
                    else:
                        prop_target, prop_gate, prop_conf = local_context_targets(
                            outputs["patch_logits"].detach(),
                            mask_indices,
                            labels,
                            radius=args.context_radius,
                            confidence_threshold=args.homogeneity_threshold,
                            agreement_threshold=args.agreement_threshold,
                        )
                        prop_loss = propagation_loss(pred_patch_logits, prop_target, prop_gate)
                    if (not args.train_predictor_only) and aux_weighter is not None:
                        if args.loss_weighting == "uncertainty_jepa":
                            aux_losses = {"jepa": jepa_loss}
                        else:
                            aux_losses = {"div": div_loss, "jepa": jepa_loss}
                        if args.propagate_weight > 0 and args.loss_weighting == "uncertainty_aux":
                            aux_losses["propagate"] = prop_loss
                        aux_loss, aux_weights = aux_weighter(aux_losses)
                        fixed_aux = (
                            args.div_weight * div_loss
                            + active_gate_weight * proto_gate_loss
                            + active_usage_weight * usage_loss
                            + args.refine_consistency_weight * refine_consistency_loss
                            + active_completion_weight * completion_loss
                            + active_affinity_weight * affinity_loss
                            + active_sem_affinity_weight * sem_affinity_loss
                        )
                        if args.loss_weighting == "uncertainty_jepa":
                            fixed_aux = fixed_aux + args.propagate_weight * prop_loss
                        loss = (cls_loss + fixed_aux + aux_loss) / args.grad_accum_steps
                    elif not args.train_predictor_only:
                        aux_weights = {
                            "div_uw": args.div_weight,
                            "jepa_uw": args.jepa_weight,
                            "propagate_uw": args.propagate_weight,
                        }
                        loss = (
                            cls_loss
                            + args.div_weight * div_loss
                            + args.jepa_weight * jepa_loss
                            + args.propagate_weight * prop_loss
                            + active_gate_weight * proto_gate_loss
                            + active_usage_weight * usage_loss
                            + args.refine_consistency_weight * refine_consistency_loss
                            + active_completion_weight * completion_loss
                            + active_affinity_weight * affinity_loss
                            + active_sem_affinity_weight * sem_affinity_loss
                        ) / args.grad_accum_steps

            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if not (args.train_refine_only or args.train_predictor_only):
                    update_ema_backbone(ema_backbone, model.backbone, args.ema_momentum)
                global_step += 1

            loss_sum += float(loss.detach().cpu()) * args.grad_accum_steps
            cls_sum += float(cls_loss.detach().cpu())
            proto_cls_sum += float(proto_cls_loss.detach().cpu())
            refine_cls_sum += float(refine_cls_loss.detach().cpu())
            refine_cons_sum += float(refine_consistency_loss.detach().cpu())
            div_sum += float(div_loss.detach().cpu())
            jepa_sum += float(jepa_loss.detach().cpu())
            jepa_token_weight_sum += float(jepa_token_weight.detach().cpu())
            affinity_sum += float(affinity_loss.detach().cpu())
            affinity_gate_sum += float(affinity_gate.detach().cpu())
            affinity_weight_sum += float(affinity_token_weight.detach().cpu())
            sem_affinity_sum += float(sem_affinity_loss.detach().cpu())
            sem_affinity_gate_sum += float(sem_affinity_gate.detach().cpu())
            sem_affinity_weight_sum += float(sem_affinity_token_weight.detach().cpu())
            completion_sum += float(completion_loss.detach().cpu())
            prop_sum += float(prop_loss.detach().cpu())
            gate_sum += float(prop_gate.float().mean().detach().cpu())
            proto_gate_sum += float(proto_gate_loss.detach().cpu())
            usage_sum += float(usage_loss.detach().cpu())
            div_uw_sum += float(aux_weights.get("div_uw", 0.0))
            jepa_uw_sum += float(aux_weights.get("jepa_uw", 0.0))
            prop_uw_sum += float(aux_weights.get("propagate_uw", 0.0))
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                elapsed = time.perf_counter() - start
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / step:.4f} cls={cls_sum / step:.4f} "
                    f"proto_cls={proto_cls_sum / step:.4f} ref_cls={refine_cls_sum / step:.4f} "
                    f"ref_cons={refine_cons_sum / step:.4f} "
                    f"div={div_sum / step:.4f} jepa={jepa_sum / step:.4f} "
                    f"jepa_tok_w={jepa_token_weight_sum / step:.3f} "
                    f"aff={affinity_sum / step:.4f} aff_gate={affinity_gate_sum / step:.3f} "
                    f"aff_w={affinity_weight_sum / step:.3f} "
                    f"sem_aff={sem_affinity_sum / step:.4f} sem_gate={sem_affinity_gate_sum / step:.3f} "
                    f"sem_w={sem_affinity_weight_sum / step:.3f} "
                    f"completion={completion_sum / step:.4f} comp_w={active_completion_weight:.2f} "
                    f"prop={prop_sum / step:.4f} gate={gate_sum / step:.3f} "
                    f"proto_gate={proto_gate_sum / step:.4f} usage={usage_sum / step:.4f} "
                    f"gate_w={active_gate_weight:.1e} usage_w={active_usage_weight:.1e} "
                    f"w_div={div_uw_sum / step:.3f} w_jepa={jepa_uw_sum / step:.3f} "
                    f"w_prop={prop_uw_sum / step:.3f} "
                    f"mask_ratio={current_mask_ratio:.3f} "
                    f"lr={lr:.2e} "
                    f"elapsed={elapsed:.1f}s"
                )
                if device.type == "cuda":
                    msg += f" peak_mem={torch.cuda.max_memory_allocated(device) / (1024**3):.2f}GB"
                print(msg, flush=True)

        row: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "train_cls_loss": cls_sum / max(len(train_loader), 1),
            "train_proto_cls_loss": proto_cls_sum / max(len(train_loader), 1),
            "train_refine_cls_loss": refine_cls_sum / max(len(train_loader), 1),
            "train_refine_consistency_loss": refine_cons_sum / max(len(train_loader), 1),
            "train_div_loss": div_sum / max(len(train_loader), 1),
            "train_jepa_loss": jepa_sum / max(len(train_loader), 1),
            "train_jepa_token_weight": jepa_token_weight_sum / max(len(train_loader), 1),
            "train_affinity_loss": affinity_sum / max(len(train_loader), 1),
            "train_affinity_gate": affinity_gate_sum / max(len(train_loader), 1),
            "train_affinity_token_weight": affinity_weight_sum / max(len(train_loader), 1),
            "train_semantic_affinity_loss": sem_affinity_sum / max(len(train_loader), 1),
            "train_semantic_affinity_gate": sem_affinity_gate_sum / max(len(train_loader), 1),
            "train_semantic_affinity_token_weight": sem_affinity_weight_sum / max(len(train_loader), 1),
            "train_completion_loss": completion_sum / max(len(train_loader), 1),
            "train_propagate_loss": prop_sum / max(len(train_loader), 1),
            "train_propagate_gate": gate_sum / max(len(train_loader), 1),
            "train_proto_gate_loss": proto_gate_sum / max(len(train_loader), 1),
            "train_usage_loss": usage_sum / max(len(train_loader), 1),
            "mask_ratio": current_mask_ratio,
            "active_prototypes": model.effective_prototypes().detach().cpu().tolist(),
            "train_div_uw": div_uw_sum / max(len(train_loader), 1),
            "train_jepa_uw": jepa_uw_sum / max(len(train_loader), 1),
            "train_propagate_uw": prop_uw_sum / max(len(train_loader), 1),
            "sec_epoch": time.perf_counter() - start,
        }

        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, args.amp, num_classes)
            row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "confusion"})
            print(
                f"epoch={epoch} train_loss={row['train_loss']:.4f} "
                f"val_miou={val_metrics['miou']:.4f} val_mrecall={val_metrics['mrecall']:.4f} "
                f"val_fwiou={val_metrics['fwiou']:.4f}"
            )
            if float(val_metrics["miou"]) > best_miou:
                best_miou = float(val_metrics["miou"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "predictor": predictor.state_dict(),
                        "ema_backbone": ema_backbone.state_dict(),
                        "aux_weighter": None if aux_weighter is None else aux_weighter.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None if scheduler is None else scheduler.state_dict(),
                        "best_miou": best_miou,
                        "args": config,
                        "val_metrics": val_metrics,
                    },
                    output_dir / "best.pt",
                )

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "predictor": predictor.state_dict(),
                "ema_backbone": ema_backbone.state_dict(),
                "aux_weighter": None if aux_weighter is None else aux_weighter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "args": config,
            },
            output_dir / "last.pt",
        )
        write_log(log_path, row)


if __name__ == "__main__":
    main()
