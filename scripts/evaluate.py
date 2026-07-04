from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import CLASS_NAMES, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.metrics import SegmentationMeter
from jepa_wsss.models import DualRouteLinearWSSSModel, LinearWSSSModel, PrototypeWSSSModel
from scripts.patch_embed_adapt import adapt_vit_patch_embed


def parse_prototype_counts(value: object, num_classes: int) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, list):
        counts = [int(item) for item in value]
    else:
        text = str(value).strip()
        if not text:
            return None
        counts = [int(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
    if len(counts) != num_classes:
        raise ValueError(f"Expected {num_classes} prototype counts, got {counts}")
    return counts


def segmentation_from_patch_logits(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


def segmentation_from_outputs(outputs: dict[str, torch.Tensor], size: tuple[int, int], prediction_head: str = "auto") -> torch.Tensor:
    if prediction_head == "refined":
        if "refined_logits" not in outputs:
            raise ValueError("Requested --prediction-head refined but checkpoint/model has no refined logits.")
        logits = F.interpolate(outputs["refined_logits"], size=size, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)
    if prediction_head == "auto" and "refined_logits" in outputs:
        logits = F.interpolate(outputs["refined_logits"], size=size, mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)
    if prediction_head not in {"auto", "coarse"}:
        raise ValueError(f"Unknown prediction head: {prediction_head}")
    return segmentation_from_patch_logits(outputs["patch_logits"], size)


@torch.no_grad()
def evaluate(
    model: PrototypeWSSSModel,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    num_classes: int,
    prediction_head: str,
) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes)
    for idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
        pred = segmentation_from_outputs(outputs, masks.shape[-2:], prediction_head)
        meter.update(pred, masks)
        if idx == 1 or idx % 25 == 0 or idx == len(loader):
            print(f"eval batch={idx}/{len(loader)}", flush=True)
    return meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained prototype WSSS checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt from train_baseline.py.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-head", default="auto", choices=["auto", "coarse", "refined"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})

    data_root = args.data_root or saved_args.get("data_root", "data")
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    model_name = saved_args.get("model", "deit_base_patch16_224")
    pretrain_checkpoint = saved_args.get("checkpoint")
    prototypes_per_class = int(saved_args.get("prototypes_per_class", 4))
    num_classes = len(CLASS_NAMES[dataset])
    prototype_counts = parse_prototype_counts(
        saved_args.get("resolved_prototype_counts", saved_args.get("prototype_counts")),
        num_classes,
    )
    prototype_gating = bool(saved_args.get("prototype_gating", False))
    gate_init = float(saved_args.get("gate_init", 2.0))
    prototype_dropout = float(saved_args.get("prototype_dropout", 0.0))
    prototype_aggregation = saved_args.get("prototype_aggregation", "max")
    prototype_lse_tau = float(saved_args.get("prototype_lse_tau", saved_args.get("lse_tau", 1.0)))
    refine_head = bool(saved_args.get("refine_head", False))
    refine_dim = int(saved_args.get("refine_dim", 256))
    refine_scale = int(saved_args.get("refine_scale", 2))
    refine_pooling = saved_args.get("refine_pooling", "topk")
    refine_topk_frac = float(saved_args.get("refine_topk_frac", 0.05))
    fusion_layers = saved_args.get("fusion_layers")
    fusion_mode = saved_args.get("fusion_mode", "weighted_sum")
    fusion_init = saved_args.get("fusion_init", "average")
    dense_prototype_scale = int(saved_args.get("dense_prototype_scale", 1))
    patch_stride = saved_args.get("patch_stride")
    patch_stride = None if patch_stride is None else int(patch_stride)
    patch_kernel = saved_args.get("patch_kernel")
    patch_kernel = None if patch_kernel is None else int(patch_kernel)
    patch_padding = int(saved_args.get("patch_padding", 0))
    prototype_pooling = saved_args.get("prototype_pooling", "max")
    prototype_topk_frac = float(saved_args.get("prototype_topk_frac", 0.05))
    prototype_mix_alpha = float(saved_args.get("prototype_mix_alpha", 0.5))
    prototype_multiscale = bool(saved_args.get("prototype_multiscale", False))
    prototype_scale_branches = saved_args.get("prototype_scale_branches", "identity,local,coarse")
    prototype_scale_init = saved_args.get("prototype_scale_init", "identity")
    prototype_scale_residual_init = float(saved_args.get("prototype_scale_residual_init", 0.05))
    prototype_scale_mode = saved_args.get("prototype_scale_mode", "mixture")
    prototype_scale_alpha_init = float(saved_args.get("prototype_scale_alpha_init", 0.02))
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", (0.485, 0.456, 0.406)))
    image_std = tuple(float(x) for x in saved_args.get("image_std", (0.229, 0.224, 0.225)))
    device = torch.device(args.device)

    if saved_args.get("model_type") == "dual_route_linear":
        model = DualRouteLinearWSSSModel(
            model_name=model_name,
            checkpoint_path=pretrain_checkpoint,
            num_classes=num_classes,
            route_layers=saved_args.get("route_layers", "all"),
            variant=saved_args.get("variant", "dual_quality"),
            topk_frac=float(saved_args.get("topk_frac", 0.05)),
            spatial_weight=float(saved_args.get("spatial_weight", 0.5)),
            quality_hidden_dim=int(saved_args.get("quality_hidden_dim", 32)),
            semantic_init=saved_args.get("semantic_init", "final"),
            spatial_init=saved_args.get("spatial_init", "uniform"),
            output_mode=saved_args.get("output_mode", "spatial"),
            output_fuse_alpha=float(saved_args.get("output_fuse_alpha", 0.5)),
            grad_checkpointing=False,
        )
        adapt_vit_patch_embed(
            model,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            resample_scale=saved_args.get("patch_resample_scale", "area"),
        )
    elif saved_args.get("model_type") == "linear_wsss":
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
        model = PrototypeWSSSModel(
            model_name=model_name,
            checkpoint_path=pretrain_checkpoint,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            prototype_counts=prototype_counts,
            prototype_gating=prototype_gating,
            gate_init=gate_init,
            prototype_dropout=prototype_dropout,
            prototype_aggregation=prototype_aggregation,
            lse_tau=prototype_lse_tau,
            refine_head=refine_head,
            refine_dim=refine_dim,
            refine_scale=refine_scale,
            refine_pooling=refine_pooling,
            refine_topk_frac=refine_topk_frac,
            grad_checkpointing=False,
            fusion_layers=fusion_layers,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
            dense_prototype_scale=dense_prototype_scale,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
            prototype_pooling=prototype_pooling,
            prototype_topk_frac=prototype_topk_frac,
            prototype_mix_alpha=prototype_mix_alpha,
            prototype_multiscale=prototype_multiscale,
            prototype_scale_branches=prototype_scale_branches,
            prototype_scale_init=prototype_scale_init,
            prototype_scale_residual_init=prototype_scale_residual_init,
            prototype_scale_mode=prototype_scale_mode,
            prototype_scale_alpha_init=prototype_scale_alpha_init,
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)

    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = SegmentationDataset(data_root, dataset, split=args.split, transform=transform)
    loader = DataLoader(
        dataset_obj,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    metrics = evaluate(model, loader, device, args.amp, num_classes, args.prediction_head)

    class_names = CLASS_NAMES[dataset]
    print(f"split: {args.split}")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"mDice: {metrics['mdice']:.4f}")
    print(f"mRecall: {metrics['mrecall']:.4f}")
    print(f"mPrecision: {metrics['mprecision']:.4f}")
    print(f"FwIoU: {metrics['fwiou']:.4f}")
    for idx, name in enumerate(class_names):
        print(
            f"{idx}:{name} "
            f"IoU={metrics['iou'][idx]:.4f} "
            f"Dice={metrics['dice'][idx]:.4f} "
            f"Recall={metrics['recall'][idx]:.4f} "
            f"Precision={metrics['precision'][idx]:.4f}"
        )

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v for k, v in metrics.items() if k != "confusion"}
        serializable["split"] = args.split
        serializable["checkpoint"] = str(ckpt_path)
        serializable["prediction_head"] = args.prediction_head
        serializable["class_names"] = class_names
        output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
