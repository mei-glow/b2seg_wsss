from __future__ import annotations

import argparse
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

from b2seg_wsss.datasets import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset, pil_to_normalized_tensor
from b2seg_wsss.metrics import SegmentationMeter
from b2seg_wsss.models import DualRouteLinearWSSSModel, LinearWSSSModel, PrototypeWSSSModel
from scripts.patch_embed_adapt import adapt_vit_patch_embed
from scripts.evaluate import parse_prototype_counts


def parse_tta(value: str) -> list[str]:
    aliases = {
        "none": ["id"],
        "flip": ["id", "hflip", "vflip", "hvflip"],
        "d4": ["id", "hflip", "vflip", "hvflip", "rot90", "rot180", "rot270", "transpose"],
        "flip+d4": ["id", "hflip", "vflip", "hvflip", "rot90", "rot180", "rot270", "transpose"],
    }
    value = value.strip().lower()
    if value in aliases:
        return aliases[value]
    modes = [item.strip().lower() for item in value.replace(";", ",").split(",") if item.strip()]
    valid = {"id", "hflip", "vflip", "hvflip", "rot90", "rot180", "rot270", "transpose"}
    unknown = sorted(set(modes) - valid)
    if unknown:
        raise ValueError(f"Unknown TTA transforms: {unknown}. Valid transforms: {sorted(valid)}")
    if "id" not in modes:
        modes.insert(0, "id")
    return modes


def transform_spatial(tensor: torch.Tensor, mode: str, *, inverse: bool = False) -> torch.Tensor:
    if mode == "id":
        return tensor
    if mode == "hflip":
        return torch.flip(tensor, dims=(3,))
    if mode == "vflip":
        return torch.flip(tensor, dims=(2,))
    if mode == "hvflip":
        return torch.flip(tensor, dims=(2, 3))
    if mode == "rot90":
        return torch.rot90(tensor, k=-1 if inverse else 1, dims=(2, 3))
    if mode == "rot180":
        return torch.rot90(tensor, k=2, dims=(2, 3))
    if mode == "rot270":
        return torch.rot90(tensor, k=1 if inverse else 3, dims=(2, 3))
    if mode == "transpose":
        return tensor.transpose(2, 3)
    raise ValueError(f"Unknown TTA transform: {mode}")


def require_densecrf():
    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax
    except ImportError as exc:
        raise SystemExit(
            "pydensecrf is required for CRF evaluation. Install it in the server venv with:\n"
            "  pip install pydensecrf\n"
            "If that fails, try:\n"
            "  pip install git+https://github.com/lucasb-eyer/pydensecrf.git"
        ) from exc
    return dcrf, unary_from_softmax


def logits_from_patch_logits(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def logits_from_outputs(outputs: dict[str, torch.Tensor], size: tuple[int, int], prediction_head: str) -> torch.Tensor:
    if prediction_head == "refined":
        if "refined_logits" not in outputs:
            raise ValueError("Requested --prediction-head refined but checkpoint/model has no refined logits.")
        return F.interpolate(outputs["refined_logits"], size=size, mode="bilinear", align_corners=False)
    if prediction_head == "auto" and "refined_logits" in outputs:
        return F.interpolate(outputs["refined_logits"], size=size, mode="bilinear", align_corners=False)
    if prediction_head not in {"auto", "coarse"}:
        raise ValueError(f"Unknown prediction head: {prediction_head}")
    return logits_from_patch_logits(outputs["patch_logits"], size)


def denormalize_image(
    image: torch.Tensor,
    mean_values: tuple[float, float, float] = IMAGENET_MEAN,
    std_values: tuple[float, float, float] = IMAGENET_STD,
) -> np.ndarray:
    mean = torch.tensor(mean_values, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std = torch.tensor(std_values, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)


def dense_crf_predict(
    image_rgb: np.ndarray,
    probs: np.ndarray,
    num_classes: int,
    iterations: int,
    sxy_gaussian: int,
    compat_gaussian: int,
    sxy_bilateral: int,
    srgb_bilateral: int,
    compat_bilateral: int,
) -> np.ndarray:
    dcrf, unary_from_softmax = require_densecrf()
    height, width = image_rgb.shape[:2]
    crf = dcrf.DenseCRF2D(width, height, num_classes)
    unary = unary_from_softmax(np.ascontiguousarray(probs))
    crf.setUnaryEnergy(unary)
    crf.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian)
    crf.addPairwiseBilateral(
        sxy=sxy_bilateral,
        srgb=srgb_bilateral,
        rgbim=np.ascontiguousarray(image_rgb),
        compat=compat_bilateral,
    )
    refined = np.asarray(crf.inference(iterations), dtype=np.float32).reshape(num_classes, height, width)
    return refined.argmax(axis=0).astype(np.int64)


def build_model_from_checkpoint(checkpoint: dict[str, object], checkpoint_path: Path, dataset: str, device: torch.device) -> torch.nn.Module:
    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get("model", "deit_base_patch16_224")
    pretrain_checkpoint = saved_args.get("checkpoint")
    num_classes = len(CLASS_NAMES[dataset])
    prototype_counts = parse_prototype_counts(
        saved_args.get("resolved_prototype_counts", saved_args.get("prototype_counts")),
        num_classes,
    )
    patch_stride = saved_args.get("patch_stride")
    patch_stride = None if patch_stride is None else int(patch_stride)
    patch_kernel = saved_args.get("patch_kernel")
    patch_kernel = None if patch_kernel is None else int(patch_kernel)
    patch_padding = int(saved_args.get("patch_padding", 0))
    if saved_args.get("model_type") == "dual_route_linear":
        model = DualRouteLinearWSSSModel(
            model_name=model_name,
            checkpoint_path=pretrain_checkpoint,
            num_classes=num_classes,
            route_layers=saved_args.get("route_layers", "all"),
            variant=saved_args.get("variant", "dual_quality"),
            topk_frac=float(saved_args.get("topk_frac", 0.05)),
            semantic_pooling=saved_args.get("semantic_pooling", "max"),
            lse_tau=float(saved_args.get("lse_tau", 1.0)),
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
            fusion_layers=saved_args.get("fusion_layers"),
            fusion_mode=saved_args.get("fusion_mode", "weighted_sum"),
            fusion_init=saved_args.get("fusion_init", "average"),
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
            fusion_layers=saved_args.get("fusion_layers"),
            fusion_mode=saved_args.get("fusion_mode", "weighted_sum"),
            fusion_init=saved_args.get("fusion_init", "average"),
            dense_prototype_scale=int(saved_args.get("dense_prototype_scale", 1)),
            patch_stride=patch_stride,
            patch_padding=int(saved_args.get("patch_padding", 0)),
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
    print(f"loaded_checkpoint={checkpoint_path}", flush=True)
    return model


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, object]:
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    data_root = args.data_root or saved_args.get("data_root", "data")
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    image_std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    num_classes = len(CLASS_NAMES[dataset])
    device = torch.device(args.device)
    tta_modes = parse_tta(args.tta)

    model = build_model_from_checkpoint(checkpoint, ckpt_path, dataset, device)
    model.eval()

    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = SegmentationDataset(data_root, dataset, split=args.split, transform=transform)
    loader = DataLoader(
        dataset_obj,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    raw_meter = SegmentationMeter(num_classes=num_classes)
    crf_meter = SegmentationMeter(num_classes=num_classes)

    for batch_idx, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits_sum = None
        for mode in tta_modes:
            augmented_images = transform_spatial(images, mode)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(augmented_images)
                augmented_logits = logits_from_outputs(outputs, augmented_images.shape[-2:], args.prediction_head)
            logits = transform_spatial(augmented_logits, mode, inverse=True)
            logits_sum = logits if logits_sum is None else logits_sum + logits
        logits = logits_sum / float(len(tta_modes))

        raw_pred = logits.argmax(dim=1)
        raw_meter.update(raw_pred, masks)

        probs = F.softmax(logits / args.softmax_temp, dim=1).detach().cpu().numpy()
        for item_idx in range(images.shape[0]):
            image_rgb = denormalize_image(images[item_idx], image_mean, image_std)
            crf_pred = dense_crf_predict(
                image_rgb=image_rgb,
                probs=probs[item_idx],
                num_classes=num_classes,
                iterations=args.crf_iters,
                sxy_gaussian=args.sxy_gaussian,
                compat_gaussian=args.compat_gaussian,
                sxy_bilateral=args.sxy_bilateral,
                srgb_bilateral=args.srgb_bilateral,
                compat_bilateral=args.compat_bilateral,
            )
            crf_meter.update(torch.from_numpy(crf_pred).unsqueeze(0), masks[item_idx : item_idx + 1].cpu())

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(loader):
            print(f"eval_tta_crf batch={batch_idx}/{len(loader)} tta={','.join(tta_modes)}", flush=True)

    return {"raw": raw_meter.compute(), "crf": crf_meter.compute(), "tta": tta_modes}


def print_metrics(title: str, metrics: dict[str, object], class_names: tuple[str, ...]) -> None:
    print(title)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with geometric TTA and dense CRF post-processing.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad", "gcss"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-head", default="coarse", choices=["auto", "coarse", "refined"])
    parser.add_argument(
        "--tta",
        default="flip+d4",
        help="none, flip, d4, flip+d4, or a comma-separated list of id,hflip,vflip,hvflip,rot90,rot180,rot270,transpose",
    )
    parser.add_argument("--softmax-temp", type=float, default=1.0)
    parser.add_argument("--crf-iters", type=int, default=5)
    parser.add_argument("--sxy-gaussian", type=int, default=3)
    parser.add_argument("--compat-gaussian", type=int, default=3)
    parser.add_argument("--sxy-bilateral", type=int, default=40)
    parser.add_argument("--srgb-bilateral", type=int, default=8)
    parser.add_argument("--compat-bilateral", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metrics = evaluate(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    class_names = CLASS_NAMES[dataset]

    print(f"split: {args.split}")
    print(f"tta: {','.join(metrics['tta'])}")
    print_metrics("raw:", metrics["raw"], class_names)
    print_metrics("crf:", metrics["crf"], class_names)

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "raw": {k: v for k, v in metrics["raw"].items() if k != "confusion"},
            "crf": {k: v for k, v in metrics["crf"].items() if k != "confusion"},
            "split": args.split,
            "checkpoint": args.checkpoint,
            "prediction_head": args.prediction_head,
            "class_names": class_names,
            "tta": metrics["tta"],
            "crf_params": {
                "softmax_temp": args.softmax_temp,
                "crf_iters": args.crf_iters,
                "sxy_gaussian": args.sxy_gaussian,
                "compat_gaussian": args.compat_gaussian,
                "sxy_bilateral": args.sxy_bilateral,
                "srgb_bilateral": args.srgb_bilateral,
                "compat_bilateral": args.compat_bilateral,
            },
        }
        output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
