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

from jepa_wsss.datasets import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.metrics import SegmentationMeter
from scripts.evaluate_crf import build_model_from_checkpoint, logits_from_outputs


def window_starts(length: int, crop_size: int, stride: int) -> list[int]:
    if crop_size >= length:
        return [0]
    starts = list(range(0, max(length - crop_size + 1, 1), max(1, stride)))
    last = length - crop_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_windows(height: int, width: int, crop_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    ys = window_starts(height, crop_h, stride)
    xs = window_starts(width, crop_w, stride)
    return [(y, x, crop_h, crop_w) for y in ys for x in xs]


def merge_weight(height: int, width: int, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "uniform":
        return torch.ones(1, 1, height, width, device=device)
    if mode == "hann":
        if height <= 2 or width <= 2:
            return torch.ones(1, 1, height, width, device=device)
        wy = torch.hann_window(height, periodic=False, device=device).clamp_min(1e-3)
        wx = torch.hann_window(width, periodic=False, device=device).clamp_min(1e-3)
        return (wy[:, None] * wx[None, :]).view(1, 1, height, width)
    raise ValueError(f"Unknown merge weight mode: {mode}")


@torch.no_grad()
def multicrop_logits(
    model: torch.nn.Module,
    image: torch.Tensor,
    num_classes: int,
    input_size: int,
    crop_size: int,
    stride: int,
    crop_batch_size: int,
    amp: bool,
    prediction_head: str,
    merge_mode: str,
    include_full: bool,
    full_weight: float,
) -> torch.Tensor:
    device = image.device
    _, height, width = image.shape
    logits_sum = torch.zeros(1, num_classes, height, width, device=device)
    weight_sum = torch.zeros(1, 1, height, width, device=device)

    if include_full:
        full_input = F.interpolate(image.unsqueeze(0), size=(input_size, input_size), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(full_input)
            full_logits = logits_from_outputs(outputs, (height, width), prediction_head)
        logits_sum += float(full_weight) * full_logits
        weight_sum += float(full_weight)

    windows = make_windows(height, width, crop_size, stride)
    for start in range(0, len(windows), crop_batch_size):
        chunk = windows[start : start + crop_batch_size]
        crops = []
        for y, x, crop_h, crop_w in chunk:
            crop = image[:, y : y + crop_h, x : x + crop_w].unsqueeze(0)
            crop = F.interpolate(crop, size=(input_size, input_size), mode="bilinear", align_corners=False).squeeze(0)
            crops.append(crop)
        crop_batch = torch.stack(crops, dim=0)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(crop_batch)
            crop_logits = logits_from_outputs(outputs, (input_size, input_size), prediction_head)
        for crop_idx, (y, x, crop_h, crop_w) in enumerate(chunk):
            resized = F.interpolate(
                crop_logits[crop_idx : crop_idx + 1],
                size=(crop_h, crop_w),
                mode="bilinear",
                align_corners=False,
            )
            weight = merge_weight(crop_h, crop_w, merge_mode, device)
            logits_sum[:, :, y : y + crop_h, x : x + crop_w] += resized * weight
            weight_sum[:, :, y : y + crop_h, x : x + crop_w] += weight

    return logits_sum / weight_sum.clamp_min(1e-6)


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

    model = build_model_from_checkpoint(checkpoint, ckpt_path, dataset, device)
    model.eval()

    transform = lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std)
    dataset_obj = SegmentationDataset(data_root, dataset, split=args.split, transform=transform)
    loader = DataLoader(dataset_obj, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    meter = SegmentationMeter(num_classes=num_classes)

    for idx, batch in enumerate(loader, start=1):
        image = batch["image"][0].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        logits = multicrop_logits(
            model=model,
            image=image,
            num_classes=num_classes,
            input_size=args.input_size,
            crop_size=args.crop_size,
            stride=args.stride,
            crop_batch_size=args.crop_batch_size,
            amp=args.amp,
            prediction_head=args.prediction_head,
            merge_mode=args.merge_mode,
            include_full=args.include_full,
            full_weight=args.full_weight,
        )
        pred = logits.argmax(dim=1)
        meter.update(pred, mask)
        if idx == 1 or idx % args.log_every == 0 or idx == len(loader):
            print(f"multicrop eval image={idx}/{len(loader)}", flush=True)

    return meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WSSS checkpoint with sliding-window multi-crop logits.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--input-size", type=int, default=224, help="Model input size for each crop.")
    parser.add_argument("--crop-size", type=int, default=160, help="Window size in original image pixels.")
    parser.add_argument("--stride", type=int, default=80, help="Sliding-window stride in original image pixels.")
    parser.add_argument("--crop-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prediction-head", default="coarse", choices=["auto", "coarse", "refined"])
    parser.add_argument("--merge-mode", default="hann", choices=["uniform", "hann"])
    parser.add_argument("--include-full", action="store_true", help="Also merge the original full-image prediction.")
    parser.add_argument("--full-weight", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metrics = evaluate(args)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved_args = ckpt.get("args", {})
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    class_names = CLASS_NAMES[dataset]

    print(f"split: {args.split}")
    print(f"mIoU: {metrics['miou']:.4f}")
    print(f"mDice: {metrics['mdice']:.4f}")
    print(f"mRecall: {metrics['mrecall']:.4f}")
    print(f"mPrecision: {metrics['mprecision']:.4f}")
    print(f"FwIoU: {metrics['fwiou']:.4f}")
    for class_idx, name in enumerate(class_names):
        print(
            f"{class_idx}:{name} "
            f"IoU={metrics['iou'][class_idx]:.4f} "
            f"Dice={metrics['dice'][class_idx]:.4f} "
            f"Recall={metrics['recall'][class_idx]:.4f} "
            f"Precision={metrics['precision'][class_idx]:.4f}"
        )

    if args.output is not None:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {key: value for key, value in metrics.items() if key != "confusion"}
        serializable["split"] = args.split
        serializable["checkpoint"] = args.checkpoint
        serializable["prediction_head"] = args.prediction_head
        serializable["class_names"] = class_names
        serializable["multicrop_params"] = {
            "input_size": args.input_size,
            "crop_size": args.crop_size,
            "stride": args.stride,
            "crop_batch_size": args.crop_batch_size,
            "merge_mode": args.merge_mode,
            "include_full": args.include_full,
            "full_weight": args.full_weight,
        }
        output.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
