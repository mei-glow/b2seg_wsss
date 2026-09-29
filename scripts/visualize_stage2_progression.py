from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b2seg_wsss.datasets import (  # noqa: E402
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentationDataset,
    pil_to_normalized_tensor,
)
from scripts.evaluate_crf import (  # noqa: E402
    build_model_from_checkpoint,
    dense_crf_predict,
    denormalize_image,
    logits_from_outputs,
    parse_tta,
    transform_spatial,
)
try:  # Older Colab checkouts may not include the optional guided filter.
    from scripts.evaluate_crf import guided_refine_probabilities  # noqa: E402
except ImportError:
    guided_refine_probabilities = None
from scripts.train_online_pseudo import make_pseudo, parse_thresholds  # noqa: E402
from scripts.patch_embed_adapt import adapt_vit_patch_embed  # noqa: E402


IGNORE_INDEX = 255
PALETTE = np.asarray(
    [
        (220, 45, 45),    # tumor
        (62, 177, 83),    # stroma
        (55, 105, 220),   # lymphocyte
        (153, 74, 204),   # necrosis
        (240, 175, 45),
        (45, 185, 185),
    ],
    dtype=np.uint8,
)
IGNORE_COLOR = np.asarray((105, 105, 105), dtype=np.uint8)
PANEL_TITLES = (
    "Image",
    "Stage-1 teacher (coarse)",
    "Pseudo-label Y-hat (gray = ignore)",
    "Student semantic branch",
    "Student spatial branch",
    "Final segmentation (ours)",
    "Ground truth",
)


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def checkpoint_stats(
    checkpoint: dict[str, object],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    saved_args = checkpoint.get("args", {})
    mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(f"Invalid checkpoint normalization: mean={mean}, std={std}")
    return mean, std


def build_final_model(
    checkpoint: dict[str, object],
    checkpoint_path: Path,
    dataset: str,
    device: torch.device,
) -> torch.nn.Module:
    """Build from architecture metadata without reloading the old pretrain path.

    A final Stage-1/Stage-2 checkpoint contains the complete model state.  Its
    saved ``args.checkpoint`` may point to a pretrained DeiT file on the machine
    where training happened, which need not exist on Colab.  Initializing the
    architecture without that file and then strictly loading ``model`` is both
    sufficient and exactly reproduces the saved weights.
    """
    checkpoint_for_build = dict(checkpoint)
    saved_args = dict(checkpoint.get("args", {}))
    saved_args["checkpoint"] = None
    checkpoint_for_build["args"] = saved_args
    return build_model_from_checkpoint(
        checkpoint_for_build, checkpoint_path, dataset, device
    )


def renormalize(
    images: torch.Tensor,
    source_mean: tuple[float, float, float],
    source_std: tuple[float, float, float],
    target_mean: tuple[float, float, float],
    target_std: tuple[float, float, float],
) -> torch.Tensor:
    source_mean_t = images.new_tensor(source_mean).view(1, 3, 1, 1)
    source_std_t = images.new_tensor(source_std).view(1, 3, 1, 1)
    target_mean_t = images.new_tensor(target_mean).view(1, 3, 1, 1)
    target_std_t = images.new_tensor(target_std).view(1, 3, 1, 1)
    rgb = images * source_std_t + source_mean_t
    return (rgb - target_mean_t) / target_std_t


def weak_labels_from_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = torch.zeros(mask.shape[0], num_classes, device=mask.device)
    for class_idx in range(num_classes):
        labels[:, class_idx] = (mask == class_idx).flatten(1).any(dim=1)
    return labels


def token_logits_to_map(
    patch_logits: torch.Tensor,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    batch, num_tokens, num_classes = patch_logits.shape
    grid = int(round(math.sqrt(num_tokens)))
    if grid * grid != num_tokens:
        raise ValueError(f"Expected square token grid, got {num_tokens} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    return F.interpolate(logits, size=size, mode=mode)


@torch.no_grad()
def predict_tta_branch_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    tta_modes: list[str],
    amp: bool,
    output_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    logits_sums: dict[str, torch.Tensor] = {}
    for mode in tta_modes:
        augmented = transform_spatial(images, mode)
        with torch.autocast(
            device_type=images.device.type,
            enabled=amp and images.device.type == "cuda",
        ):
            outputs = model(augmented)
            for output_key in output_keys:
                if output_key not in outputs:
                    raise KeyError(
                        f"Checkpoint model has no '{output_key}'. Available outputs: "
                        f"{sorted(outputs)}"
                    )
                logits_aug = token_logits_to_map(
                    outputs[output_key], augmented.shape[-2:], mode="bilinear"
                )
                logits = transform_spatial(logits_aug, mode, inverse=True)
                logits_sums[output_key] = (
                    logits
                    if output_key not in logits_sums
                    else logits_sums[output_key] + logits
                )
    return {
        key: value / float(len(tta_modes)) for key, value in logits_sums.items()
    }


@torch.no_grad()
def predict_tta_logits(
    model: torch.nn.Module,
    images: torch.Tensor,
    tta_modes: list[str],
    amp: bool,
    output_key: str = "patch_logits",
) -> torch.Tensor:
    return predict_tta_branch_logits(
        model, images, tta_modes, amp, (output_key,)
    )[output_key]


def final_predictions(
    logits: torch.Tensor,
    images: torch.Tensor,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
    use_crf: bool,
    crf_iters: int,
    use_guided: bool = False,
    guided_radius: int = 4,
    guided_eps: float = 1e-3,
) -> torch.Tensor:
    probs_tensor = logits.float().softmax(dim=1)
    if use_guided:
        if guided_refine_probabilities is None:
            raise RuntimeError(
                "--guided-upsample was requested, but this checkout's "
                "scripts/evaluate_crf.py has no guided_refine_probabilities(). "
                "Update evaluate_crf.py or run without --guided-upsample."
            )
        probs_tensor = guided_refine_probabilities(
            probs_tensor,
            images,
            mean,
            std,
            radius=guided_radius,
            eps=guided_eps,
        )
    if not use_crf:
        return probs_tensor.argmax(dim=1)
    probs = probs_tensor.detach().cpu().numpy()
    predictions: list[torch.Tensor] = []
    for index in range(images.shape[0]):
        image_rgb = denormalize_image(images[index], mean, std)
        prediction = dense_crf_predict(
            image_rgb=image_rgb,
            probs=probs[index],
            num_classes=logits.shape[1],
            iterations=crf_iters,
            sxy_gaussian=3,
            compat_gaussian=3,
            sxy_bilateral=40,
            srgb_bilateral=8,
            compat_bilateral=5,
        )
        predictions.append(torch.from_numpy(prediction))
    return torch.stack(predictions).to(logits.device)


def sample_ious(
    prediction: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> list[float]:
    valid = target != IGNORE_INDEX
    values: list[float] = []
    for class_idx in range(num_classes):
        pred_class = (prediction == class_idx) & valid
        target_class = (target == class_idx) & valid
        union = (pred_class | target_class).sum().item()
        intersection = (pred_class & target_class).sum().item()
        values.append(float(intersection / union) if union > 0 else math.nan)
    return values


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else 0.0


def harmonic_mean(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b) or a <= 0.0 or b <= 0.0:
        return 0.0
    return float(2.0 * a * b / (a + b))


def colorize(mask: np.ndarray, num_classes: int) -> Image.Image:
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[...] = IGNORE_COLOR
    for class_idx in range(num_classes):
        rgb[mask == class_idx] = PALETTE[class_idx]
    return Image.fromarray(rgb, mode="RGB")


def resize_export(image: Image.Image, size: int, mask: bool = False) -> Image.Image:
    resampling = Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS
    return image.resize((size, size), resampling)


def safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in Path(name).stem)


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_grid(
    rendered: list[dict[str, object]],
    output_path: Path,
    class_names: tuple[str, ...],
    cell_size: int,
) -> None:
    title_height = 72
    row_label_height = 54
    legend_height = 80
    gap = 12
    width = len(PANEL_TITLES) * cell_size + (len(PANEL_TITLES) - 1) * gap
    height = title_height + len(rendered) * (row_label_height + cell_size + gap) + legend_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    row_font = load_font(21)
    legend_font = load_font(20)

    for column, title in enumerate(PANEL_TITLES):
        x = column * (cell_size + gap)
        box = draw.textbbox((0, 0), title, font=title_font)
        text_width = box[2] - box[0]
        draw.text(
            (x + (cell_size - text_width) / 2, 20),
            title,
            fill="black",
            font=title_font,
        )

    y = title_height
    for rank, sample in enumerate(rendered, start=1):
        label = (
            f"{rank}. {sample['group']} | {sample['name']} | "
            f"mIoU={sample['overall_miou']:.3f}, "
            f"lymph={sample['lymphocyte_iou']:.3f}, "
            f"stroma={sample['stroma_iou']:.3f}"
        )
        draw.text((4, y + 12), label, fill="black", font=row_font)
        y += row_label_height
        for column, image in enumerate(sample["panels"]):
            panel = resize_export(image, cell_size, mask=column > 0)
            x = column * (cell_size + gap)
            canvas.paste(panel, (x, y))
        y += cell_size + gap

    legend_y = height - legend_height + 20
    x = 4
    for class_idx, class_name in enumerate(class_names):
        draw.rectangle((x, legend_y, x + 28, legend_y + 28), fill=tuple(PALETTE[class_idx]))
        draw.text((x + 36, legend_y + 2), class_name, fill="black", font=legend_font)
        x += 36 + draw.textlength(class_name, font=legend_font) + 34
    draw.rectangle((x, legend_y, x + 28, legend_y + 28), fill=tuple(IGNORE_COLOR))
    draw.text((x + 36, legend_y + 2), "ignore", fill="black", font=legend_font)
    canvas.save(output_path, format="PNG", optimize=True)


def build_sample_panel(
    panels: list[Image.Image],
    output_path: Path,
    cell_size: int,
    class_names: tuple[str, ...],
) -> None:
    """Save one publication-ready progression row with a title per column."""
    title_height = 78
    gap = 14
    width = len(PANEL_TITLES) * cell_size + (len(PANEL_TITLES) - 1) * gap
    legend_height = 68
    height = title_height + cell_size + legend_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(max(20, int(round(cell_size * 0.043))))
    for column, (title, panel) in enumerate(zip(PANEL_TITLES, panels)):
        x = column * (cell_size + gap)
        box = draw.textbbox((0, 0), title, font=title_font)
        text_width = box[2] - box[0]
        draw.text(
            (x + max(0.0, (cell_size - text_width) / 2), 22),
            title,
            fill="black",
            font=title_font,
        )
        canvas.paste(
            resize_export(panel, cell_size, mask=column > 0),
            (x, title_height),
        )

    legend_font = load_font(max(18, int(round(cell_size * 0.032))))
    legend_y = title_height + cell_size + 18
    x = 8
    for class_idx, class_name in enumerate(class_names):
        swatch = max(22, int(round(cell_size * 0.043)))
        draw.rectangle(
            (x, legend_y, x + swatch, legend_y + swatch),
            fill=tuple(PALETTE[class_idx]),
        )
        draw.text(
            (x + swatch + 9, legend_y), class_name, fill="black", font=legend_font
        )
        x += swatch + 9 + draw.textlength(class_name, font=legend_font) + 28
    swatch = max(22, int(round(cell_size * 0.043)))
    draw.rectangle(
        (x, legend_y, x + swatch, legend_y + swatch), fill=tuple(IGNORE_COLOR)
    )
    draw.text((x + swatch + 9, legend_y), "ignore", fill="black", font=legend_font)
    canvas.save(output_path, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select three best overall test samples and three distinct samples "
            "that best separate lymphocyte from stroma, then render the five-stage "
            "qualitative progression including both Stage-2 student branches."
        )
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad", "gcss"])
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument(
        "--selection-csv",
        default=None,
        help=(
            "Optional CSV containing a 'name' column. When provided, render exactly "
            "these samples in CSV order instead of selecting a new top set."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tta", default="flip")
    parser.add_argument("--final-crf", action="store_true")
    parser.add_argument("--crf-iters", type=int, default=5)
    parser.add_argument("--guided-upsample", action="store_true")
    parser.add_argument("--guided-radius", type=int, default=4)
    parser.add_argument("--guided-eps", type=float, default=1e-3)
    parser.add_argument("--inference-patch-stride", type=int, default=None)
    parser.add_argument("--inference-patch-padding", type=int, default=0)
    parser.add_argument("--inference-image-size", type=int, default=224)
    parser.add_argument("--overall-count", type=int, default=3)
    parser.add_argument("--minority-count", type=int, default=3)
    parser.add_argument("--overall-min-classes", type=int, default=2)
    parser.add_argument("--overall-min-class-area", type=float, default=0.01)
    parser.add_argument("--minority-min-area", type=float, default=0.01)
    parser.add_argument("--export-size", type=int, default=1024)
    parser.add_argument("--panel-cell-size", type=int, default=640)
    parser.add_argument("--grid-cell-size", type=int, default=448)
    args = parser.parse_args()

    if args.dataset != "bcss":
        raise ValueError(
            "The lymphocyte/stroma selection rule is currently defined for BCSS only."
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    stage1_path = Path(args.stage1_checkpoint)
    stage2_path = Path(args.stage2_checkpoint)
    for path in (stage1_path, stage2_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    stage1_checkpoint = load_checkpoint(stage1_path)
    stage2_checkpoint = load_checkpoint(stage2_path)
    stage2_model = build_final_model(
        stage2_checkpoint, stage2_path, args.dataset, device
    ).eval()
    inference_patch_info: dict[str, object] = {"patch_adapted": False}
    if args.inference_patch_stride is not None:
        inference_patch_info = adapt_vit_patch_embed(
            stage2_model,
            patch_kernel=None,
            patch_stride=args.inference_patch_stride,
            patch_padding=args.inference_patch_padding,
            image_size=args.inference_image_size,
            resample_scale="none",
        )
        print(f"inference_patch_adapt={inference_patch_info}", flush=True)
    stage1_mean, stage1_std = checkpoint_stats(stage1_checkpoint)
    stage2_mean, stage2_std = checkpoint_stats(stage2_checkpoint)
    stage2_args = stage2_checkpoint.get("args", {})
    class_names = CLASS_NAMES[args.dataset]
    num_classes = len(class_names)
    lymphocyte_idx = class_names.index("lymphocyte")
    stroma_idx = class_names.index("stroma")
    tta_modes = parse_tta(args.tta)

    dataset = SegmentationDataset(
        args.data_root,
        args.dataset,
        split="test",
        transform=lambda image: pil_to_normalized_tensor(
            image, mean=stage2_mean, std=stage2_std
        ),
    )
    fixed_selection_rows: list[dict[str, str]] | None = None
    evaluation_indices = list(range(len(dataset)))
    evaluation_dataset: torch.utils.data.Dataset = dataset
    if args.selection_csv is not None:
        selection_path = Path(args.selection_csv)
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        with selection_path.open(newline="", encoding="utf-8") as handle:
            fixed_selection_rows = list(csv.DictReader(handle))
        selected_names = [
            str(row.get("name", "")).strip() for row in fixed_selection_rows
        ]
        if not selected_names or any(not name for name in selected_names):
            raise ValueError(
                f"Every row in --selection-csv must contain a non-empty 'name': "
                f"{selection_path}"
            )
        if len(selected_names) != len(set(selected_names)):
            raise ValueError(f"Duplicate names in --selection-csv: {selection_path}")
        index_by_name = {path.name: index for index, path in enumerate(dataset.images)}
        missing = [name for name in selected_names if name not in index_by_name]
        if missing:
            raise FileNotFoundError(
                f"Selected samples are not present in the test split: {missing}"
            )
        evaluation_indices = [index_by_name[name] for name in selected_names]
        evaluation_dataset = Subset(dataset, evaluation_indices)
    loader = DataLoader(
        evaluation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    records: list[dict[str, object]] = []
    offset = 0
    print(
        f"ranking_test_samples={len(evaluation_dataset)} tta={tta_modes} "
        f"final_crf={args.final_crf}",
        flush=True,
    )
    for batch_index, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if args.inference_patch_stride is not None and tuple(images.shape[-2:]) != (
            args.inference_image_size,
            args.inference_image_size,
        ):
            raise ValueError(
                "Inference patch adaptation expects input size "
                f"{args.inference_image_size}x{args.inference_image_size}, got "
                f"{tuple(images.shape[-2:])}."
            )
        logits = predict_tta_logits(stage2_model, images, tta_modes, args.amp)
        predictions = final_predictions(
            logits,
            images,
            stage2_mean,
            stage2_std,
            args.final_crf,
            args.crf_iters,
            args.guided_upsample,
            args.guided_radius,
            args.guided_eps,
        )
        for local_index, name in enumerate(batch["name"]):
            ious = sample_ious(
                predictions[local_index], masks[local_index], num_classes
            )
            valid = masks[local_index] != IGNORE_INDEX
            valid_count = max(1, int(valid.sum().item()))
            lymph_area = float(
                ((masks[local_index] == lymphocyte_idx) & valid).sum().item()
                / valid_count
            )
            stroma_area = float(
                ((masks[local_index] == stroma_idx) & valid).sum().item()
                / valid_count
            )
            class_areas = [
                float(
                    ((masks[local_index] == class_idx) & valid).sum().item()
                    / valid_count
                )
                for class_idx in range(num_classes)
            ]
            present_class_count = sum(
                area >= args.overall_min_class_area for area in class_areas
            )
            records.append(
                {
                    "index": evaluation_indices[offset + local_index],
                    "name": str(name),
                    "overall_miou": finite_mean(ious),
                    "class_ious": ious,
                    "lymphocyte_iou": ious[lymphocyte_idx],
                    "stroma_iou": ious[stroma_idx],
                    "lymphocyte_area": lymph_area,
                    "stroma_area": stroma_area,
                    "class_areas": class_areas,
                    "present_class_count": present_class_count,
                    "minority_score": harmonic_mean(
                        ious[lymphocyte_idx], ious[stroma_idx]
                    ),
                }
            )
        offset += len(batch["name"])
        print(f"rank batch={batch_index}/{len(loader)}", flush=True)

    selected: list[dict[str, object]] = []
    if args.selection_csv is not None:
        selection_path = Path(args.selection_csv)
        assert fixed_selection_rows is not None
        selection_rows = fixed_selection_rows
        selected_names = [str(row.get("name", "")).strip() for row in selection_rows]
        if not selected_names or any(not name for name in selected_names):
            raise ValueError(
                f"Every row in --selection-csv must contain a non-empty 'name': "
                f"{selection_path}"
            )
        if len(selected_names) != len(set(selected_names)):
            raise ValueError(f"Duplicate names in --selection-csv: {selection_path}")
        records_by_name = {str(row["name"]): row for row in records}
        missing = [name for name in selected_names if name not in records_by_name]
        if missing:
            raise FileNotFoundError(
                f"Selected samples are not present in the test split: {missing}"
            )
        for selection_row, name in zip(selection_rows, selected_names):
            group = str(selection_row.get("group", "fixed-gated-selection")).strip()
            selected.append(
                {**records_by_name[name], "group": group or "fixed-gated-selection"}
            )
        print(
            f"fixed_selection={selection_path} samples={len(selected)}",
            flush=True,
        )
    else:
        overall_candidates = [
            row
            for row in records
            if int(row["present_class_count"]) >= args.overall_min_classes
        ]
        overall = sorted(
            overall_candidates, key=lambda row: row["overall_miou"], reverse=True
        )[: args.overall_count]
        if len(overall) < args.overall_count:
            raise RuntimeError(
                f"Only found {len(overall)} samples with at least "
                f"{args.overall_min_classes} GT classes having area >= "
                f"{args.overall_min_class_area:.4f}. Lower the selection constraints."
            )
        selected_indices = {int(row["index"]) for row in overall}
        minority_candidates = [
            row
            for row in records
            if int(row["index"]) not in selected_indices
            and float(row["lymphocyte_area"]) >= args.minority_min_area
            and float(row["stroma_area"]) >= args.minority_min_area
        ]
        minority = sorted(
            minority_candidates,
            key=lambda row: row["minority_score"],
            reverse=True,
        )[: args.minority_count]
        if len(minority) < args.minority_count:
            raise RuntimeError(
                f"Only found {len(minority)} distinct samples with both lymphocyte and "
                f"stroma area >= {args.minority_min_area:.4f}; lower "
                "--minority-min-area if needed."
            )
        for group, group_records in (
            ("best-overall", overall),
            ("best-lymphocyte-vs-stroma", minority),
        ):
            for row in group_records:
                selected.append({**row, "group": group})

    # Stage 1 is not needed during the full-test ranking pass. Build it only
    # after selecting six samples to keep Colab GPU memory comfortably bounded.
    stage1_model = build_final_model(
        stage1_checkpoint, stage1_path, args.dataset, device
    ).eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_thresholds = stage2_args.get("pseudo_thresholds", "0.85")
    threshold_text = (
        ",".join(str(value) for value in saved_thresholds)
        if isinstance(saved_thresholds, (list, tuple))
        else str(saved_thresholds)
    )
    thresholds = parse_thresholds(
        threshold_text, num_classes
    ).to(device)
    render_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for rank, record in enumerate(selected, start=1):
        sample = dataset[int(record["index"])]
        image = sample["image"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)
        weak_labels = weak_labels_from_mask(mask, num_classes)
        stage1_images = renormalize(
            image, stage2_mean, stage2_std, stage1_mean, stage1_std
        )

        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            enabled=args.amp and device.type == "cuda",
        ):
            stage1_outputs = stage1_model(stage1_images)
            stage2_outputs = stage2_model(image)

        required_branch_outputs = {
            "semantic_patch_logits",
            "spatial_patch_logits",
        }
        missing_branch_outputs = required_branch_outputs.difference(stage2_outputs)
        if missing_branch_outputs:
            raise RuntimeError(
                "The Stage-2 checkpoint is not a dual-route student; missing outputs: "
                f"{sorted(missing_branch_outputs)}"
            )

        stage1_patch_logits = stage1_outputs["patch_logits"].float()
        absent = weak_labels <= 0
        stage1_patch_logits = stage1_patch_logits.masked_fill(
            absent[:, None, :], -1e4
        )
        stage1_map = token_logits_to_map(
            stage1_patch_logits, tuple(mask.shape[-2:]), mode="nearest"
        ).argmax(dim=1)

        pseudo, _, pseudo_stats = make_pseudo(
            stage2_outputs["patch_logits"].float(),
            weak_labels,
            thresholds,
            str(stage2_args.get("pseudo_score", "softmax")),
            bool(stage2_args.get("restrict_present", True)),
            int(stage2_args.get("ignore_index", IGNORE_INDEX)),
            expand_mode=str(stage2_args.get("pseudo_expand_mode", "fixed")),
            expand_min_frac=float(stage2_args.get("pseudo_expand_min_frac", 0.03)),
            expand_min_score=float(stage2_args.get("pseudo_expand_min_score", 0.0)),
            expand_max_frac=float(stage2_args.get("pseudo_expand_max_frac", 0.06)),
            expand_margin_min=float(
                stage2_args.get("pseudo_expand_margin_min", 0.05)
            ),
            expand_under_strength=float(
                stage2_args.get("pseudo_expand_under_strength", 1.0)
            ),
        )
        pseudo_map = token_logits_to_map(
            F.one_hot(
                pseudo.clamp_max(num_classes - 1), num_classes=num_classes
            ).float(),
            tuple(mask.shape[-2:]),
            mode="nearest",
        ).argmax(dim=1)
        pseudo_valid = token_logits_to_map(
            (pseudo != IGNORE_INDEX).unsqueeze(-1).float(),
            tuple(mask.shape[-2:]),
            mode="nearest",
        ).squeeze(1) > 0.5
        pseudo_map = pseudo_map.masked_fill(~pseudo_valid, IGNORE_INDEX)

        branch_logits = predict_tta_branch_logits(
            stage2_model,
            image,
            tta_modes,
            args.amp,
            (
                "semantic_patch_logits",
                "spatial_patch_logits",
                "patch_logits",
            ),
        )
        semantic_logits = branch_logits["semantic_patch_logits"]
        spatial_logits = branch_logits["spatial_patch_logits"]
        semantic_map = semantic_logits.argmax(dim=1)
        spatial_map = spatial_logits.argmax(dim=1)

        final_logits = branch_logits["patch_logits"]
        final_map = final_predictions(
            final_logits,
            image,
            stage2_mean,
            stage2_std,
            args.final_crf,
            args.crf_iters,
            args.guided_upsample,
            args.guided_radius,
            args.guided_eps,
        )

        image_rgb = Image.fromarray(
            denormalize_image(image[0], stage2_mean, stage2_std), mode="RGB"
        )
        panels = [
            image_rgb,
            colorize(stage1_map[0].cpu().numpy(), num_classes),
            colorize(pseudo_map[0].cpu().numpy(), num_classes),
            colorize(semantic_map[0].cpu().numpy(), num_classes),
            colorize(spatial_map[0].cpu().numpy(), num_classes),
            colorize(final_map[0].cpu().numpy(), num_classes),
            colorize(mask[0].cpu().numpy(), num_classes),
        ]
        sample_dir = output_dir / (
            f"{rank:02d}_{record['group']}_{safe_stem(str(record['name']))}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        filenames = (
            "01_image.png",
            "02_stage1_teacher_coarse.png",
            "03_pseudo_label_with_ignore.png",
            "04_student_semantic_branch.png",
            "05_student_spatial_branch.png",
            "06_final_segmentation.png",
            "07_ground_truth.png",
        )
        for column, (panel, filename) in enumerate(zip(panels, filenames)):
            resize_export(panel, args.export_size, mask=column > 0).save(
                sample_dir / filename, format="PNG", optimize=True
            )
        build_sample_panel(
            panels,
            sample_dir / "08_seven_column_panel.png",
            args.panel_cell_size,
            class_names,
        )

        render_record = {**record, "panels": panels}
        render_rows.append(render_record)
        metadata_rows.append(
            {
                "rank": rank,
                "group": record["group"],
                "dataset_index": record["index"],
                "name": record["name"],
                "overall_miou": record["overall_miou"],
                "minority_harmonic_iou": record["minority_score"],
                "lymphocyte_iou": record["lymphocyte_iou"],
                "stroma_iou": record["stroma_iou"],
                "lymphocyte_area": record["lymphocyte_area"],
                "stroma_area": record["stroma_area"],
                "present_class_count": record["present_class_count"],
                "pseudo_kept": pseudo_stats["pseudo_kept"],
                **{
                    f"iou_{name}": record["class_ious"][class_idx]
                    for class_idx, name in enumerate(class_names)
                },
                "sample_dir": str(sample_dir),
            }
        )

    grid_path = output_dir / "selected_examples_grid.png"
    build_grid(render_rows, grid_path, class_names, args.grid_cell_size)
    csv_path = output_dir / "selected_examples_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)
    config = {
        "stage1_checkpoint": str(stage1_path),
        "stage2_checkpoint": str(stage2_path),
        "selection_csv": args.selection_csv,
        "dataset": args.dataset,
        "split": "test",
        "tta": tta_modes,
        "final_crf": args.final_crf,
        "crf_iters": args.crf_iters,
        "guided_upsample": args.guided_upsample,
        "guided_radius": args.guided_radius,
        "guided_eps": args.guided_eps,
        "inference_patch_info": inference_patch_info,
        "selection": (
            {
                "mode": "fixed CSV",
                "path": args.selection_csv,
                "count": len(selected),
                "ordering": "exact CSV row order",
            }
            if args.selection_csv is not None
            else {
                "mode": "ranked",
                "overall": "per-image mean IoU over classes with non-empty union",
                "overall_eligibility": (
                    f"at least {args.overall_min_classes} GT classes, each occupying "
                    f">= {args.overall_min_class_area} of valid pixels"
                ),
                "lymphocyte_vs_stroma": (
                    "harmonic mean of per-image lymphocyte and stroma IoU; both GT "
                    f"areas >= {args.minority_min_area}; disjoint from overall group"
                ),
            }
        ),
        "pseudo_label": (
            "Stage-2 checkpoint patch logits filtered with checkpoint pseudo "
            "threshold, restrict-present weak labels, and configured minimum coverage"
        ),
        "ignore_color_rgb": IGNORE_COLOR.tolist(),
        "class_palette_rgb": {
            name: PALETTE[index].tolist() for index, name in enumerate(class_names)
        },
    }
    (output_dir / "visualization_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"selected_samples={len(render_rows)}", flush=True)
    print(f"grid={grid_path}", flush=True)
    print(f"scores={csv_path}", flush=True)


if __name__ == "__main__":
    main()
