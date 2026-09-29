from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


PALETTE = np.asarray(
    [
        (220, 45, 45),
        (62, 177, 83),
        (55, 105, 220),
        (153, 74, 204),
    ],
    dtype=np.uint8,
)
CLASS_NAMES = ("tumor", "stroma", "lymphocyte", "necrosis")
IGNORE_COLOR = np.asarray((105, 105, 105), dtype=np.uint8)


def torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_state_dict(state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError(f"Expected checkpoint dictionary, got {type(state).__name__}")
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError("Could not find a model state dictionary in checkpoint")
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        normalized[str(key).removeprefix("module.")] = value
    if not normalized:
        raise ValueError("Checkpoint contains no tensor parameters")
    return normalized


def read_selected_names(csv_path: Path) -> list[str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [str(row.get("name", "")).strip() for row in rows]
    names = [name for name in names if name]
    if not names:
        raise ValueError(f"No non-empty 'name' entries in {csv_path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate selected image names in {csv_path}")
    return names


def locate_images(image_dir: Path, names: list[str]) -> list[Path]:
    by_name = {path.name: path for path in image_dir.glob("*.png")}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(
            f"Selected images are missing under {image_dir}: {missing}"
        )
    return [by_name[name] for name in names]


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    # This exactly matches WSSS-Tissue's validation transform: RGB / 255,
    # with the Normalize defaults mean=(0,0,0), std=(1,1,1).
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


def colorize(mask: np.ndarray) -> Image.Image:
    rgb = np.empty((*mask.shape, 3), dtype=np.uint8)
    rgb[...] = IGNORE_COLOR
    for class_idx, color in enumerate(PALETTE):
        rgb[mask == class_idx] = color
    return Image.fromarray(rgb, mode="RGB")


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def stage1_cam_map(
    cams: torch.Tensor,
    probabilities: torch.Tensor,
    output_size: tuple[int, int],
    threshold: float,
) -> np.ndarray:
    """Render the official Net_CAM localization, not an MLPS training pseudo-mask."""
    cams = F.interpolate(cams, size=output_size, mode="bilinear", align_corners=False)
    cams = cams[0].float()
    flat = cams.flatten(1)
    minimum = flat.min(dim=1).values[:, None, None]
    maximum = flat.max(dim=1).values[:, None, None]
    cams = (cams - minimum) / (maximum - minimum).clamp_min(1e-6)
    present = probabilities[0] > threshold
    if not bool(present.any()):
        present[probabilities[0].argmax()] = True
    cams[~present] = -1.0
    return cams.argmax(dim=0).cpu().numpy().astype(np.uint8)


def save_two_stage_panel(
    image: Image.Image,
    stage1: Image.Image,
    stage2: Image.Image,
    target: Image.Image,
    output_path: Path,
    cell_size: int,
) -> None:
    titles = ("Image", "WSSS-Tissue Stage 1 (CAM)", "WSSS-Tissue Stage 2", "Ground truth")
    gap, title_height, legend_height = 14, 76, 66
    width = 4 * cell_size + 3 * gap
    canvas = Image.new("RGB", (width, title_height + cell_size + legend_height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(max(20, int(round(cell_size * 0.038))))
    legend_font = load_font(max(18, int(round(cell_size * 0.029))))
    for column, (title, panel) in enumerate(zip(titles, (image, stage1, stage2, target))):
        x = column * (cell_size + gap)
        text_width = draw.textlength(title, font=title_font)
        draw.text((x + max(0, (cell_size - text_width) / 2), 22), title, fill="black", font=title_font)
        resample = Image.Resampling.LANCZOS if column == 0 else Image.Resampling.NEAREST
        canvas.paste(panel.resize((cell_size, cell_size), resample), (x, title_height))
    x, y, swatch = 8, title_height + cell_size + 17, 26
    for class_idx, class_name in enumerate(CLASS_NAMES):
        draw.rectangle((x, y, x + swatch, y + swatch), fill=tuple(PALETTE[class_idx]))
        draw.text((x + swatch + 8, y), class_name, fill="black", font=legend_font)
        x += swatch + 8 + draw.textlength(class_name, font=legend_font) + 26
    draw.rectangle((x, y, x + swatch, y + swatch), fill=tuple(IGNORE_COLOR))
    draw.text((x + swatch + 8, y), "ignore", fill="black", font=legend_font)
    canvas.save(output_path, format="PNG", optimize=True)


def per_class_iou(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    valid = target != 4
    values: list[float] = []
    for class_idx in range(4):
        pred_class = (prediction == class_idx) & valid
        target_class = (target == class_idx) & valid
        union = np.logical_or(pred_class, target_class).sum()
        intersection = np.logical_and(pred_class, target_class).sum()
        values.append(float(intersection / union) if union > 0 else math.nan)
    return values


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official WSSS-Tissue/MLPS checkpoints on selected BCSS images."
    )
    parser.add_argument("--baseline-repo", required=True)
    parser.add_argument("--data-root", required=True, help="Path to BCSS-WSSS")
    parser.add_argument("--selection-csv", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument("--use-gate", action="store_true")
    parser.add_argument("--gate-threshold", type=float, default=0.1)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--export-size", type=int, default=1024)
    args = parser.parse_args()

    baseline_repo = Path(args.baseline_repo).resolve()
    data_root = Path(args.data_root).resolve()
    selection_csv = Path(args.selection_csv).resolve()
    stage2_path = Path(args.stage2_checkpoint).resolve()
    stage1_path = (
        None if args.stage1_checkpoint is None else Path(args.stage1_checkpoint).resolve()
    )
    for path in (baseline_repo, data_root, selection_csv, stage2_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.use_gate and (stage1_path is None or not stage1_path.is_file()):
        raise ValueError("--use-gate requires a valid --stage1-checkpoint")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)

    sys.path.insert(0, str(baseline_repo))
    DeepLab = importlib.import_module("network.deeplab").DeepLab
    model = DeepLab(
        num_classes=4,
        backbone="resnet",
        output_stride=16,
        sync_bn=False,
        freeze_bn=False,
    )
    stage2_state = normalize_state_dict(torch_load(stage2_path))
    missing, unexpected = model.load_state_dict(stage2_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Stage-2 checkpoint does not exactly match the official DeepLab model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.to(device).eval()

    gate_model = None
    if args.use_gate:
        NetCAM = importlib.import_module("network.resnet38_cls").Net_CAM
        gate_model = NetCAM(n_class=4)
        gate_state = normalize_state_dict(torch_load(stage1_path))
        missing, unexpected = gate_model.load_state_dict(gate_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "Stage-1 checkpoint does not exactly match Net_CAM: "
                f"missing={missing}, unexpected={unexpected}"
            )
        gate_model.to(device).eval()

    names = read_selected_names(selection_csv)
    image_dir = data_root / "test" / "img"
    mask_dir = data_root / "test" / "mask"
    image_paths = locate_images(image_dir, names)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    print(
        f"selected_images={len(image_paths)} use_gate={args.use_gate} "
        f"stage2={stage2_path}",
        flush=True,
    )
    for rank, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        tensor = image_tensor(image_path, device)
        with torch.no_grad():
            logits = model(tensor)
            gate_probabilities = None
            stage1_cams = None
            if gate_model is not None:
                stage1_cams, gate_probabilities = gate_model.forward_cam(tensor)
                present = gate_probabilities > float(args.gate_threshold)
                logits = logits * present[:, :, None, None].to(logits.dtype)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        target = np.asarray(Image.open(mask_path), dtype=np.uint8)
        ious = per_class_iou(prediction, target)
        stem = Path(image_path.name).stem
        raw_path = output_dir / f"{rank:02d}_{stem}_mlps_mask.png"
        color_path = output_dir / f"{rank:02d}_{stem}_mlps_color.png"
        stage1_path_out = output_dir / f"{rank:02d}_{stem}_mlps_stage1_cam.png"
        panel_path = output_dir / f"{rank:02d}_{stem}_mlps_stage1_stage2_panel.png"
        Image.fromarray(prediction, mode="L").save(raw_path, optimize=True)
        stage2_color = colorize(prediction)
        stage2_color.resize(
            (args.export_size, args.export_size), Image.Resampling.NEAREST
        ).save(color_path, optimize=True)
        if stage1_cams is not None and gate_probabilities is not None:
            stage1_prediction = stage1_cam_map(
                stage1_cams,
                gate_probabilities,
                output_size=target.shape,
                threshold=float(args.gate_threshold),
            )
            stage1_color = colorize(stage1_prediction)
            stage1_color.resize(
                (args.export_size, args.export_size), Image.Resampling.NEAREST
            ).save(stage1_path_out, optimize=True)
            save_two_stage_panel(
                Image.open(image_path).convert("RGB"),
                stage1_color,
                stage2_color,
                colorize(target),
                panel_path,
                args.export_size,
            )
        row: dict[str, object] = {
            "rank": rank,
            "name": image_path.name,
            "miou_present_union": finite_mean(ious),
            **{f"iou_{name}": ious[idx] for idx, name in enumerate(CLASS_NAMES)},
            "use_gate": args.use_gate,
            "gate_threshold": args.gate_threshold if args.use_gate else "",
            "raw_mask": str(raw_path),
            "color_mask": str(color_path),
            "stage1_cam": str(stage1_path_out) if stage1_cams is not None else "",
            "two_stage_panel": str(panel_path) if stage1_cams is not None else "",
        }
        if gate_probabilities is not None:
            for idx, name in enumerate(CLASS_NAMES):
                row[f"gate_probability_{name}"] = float(gate_probabilities[0, idx].cpu())
        rows.append(row)
        print(
            f"[{rank}/{len(image_paths)}] {image_path.name} "
            f"mIoU={row['miou_present_union']:.4f}",
            flush=True,
        )

    manifest = output_dir / "mlps_selected_predictions.csv"
    fieldnames = list(rows[0])
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    config = {
        "method": "WSSS-Tissue / MLPS",
        "official_repository": "https://github.com/ChuHan89/WSSS-Tissue",
        "stage2_checkpoint": str(stage2_path),
        "stage1_checkpoint": None if stage1_path is None else str(stage1_path),
        "use_gate": args.use_gate,
        "gate_threshold": args.gate_threshold,
        "selection_csv": str(selection_csv),
        "preprocessing": "RGB float32 divided by 255; no mean/std shift",
        "class_order": list(CLASS_NAMES),
        "palette": {name: PALETTE[idx].tolist() for idx, name in enumerate(CLASS_NAMES)},
        "ignore_color_rgb": IGNORE_COLOR.tolist(),
        "stage1_visualization": "Net_CAM class localization; not a training pseudo-mask",
    }
    (output_dir / "mlps_inference_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(f"manifest={manifest}", flush=True)


if __name__ == "__main__":
    main()
