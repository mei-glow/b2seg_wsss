from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = ("tumor", "stroma", "lymphocyte", "necrosis")
PALETTE = np.asarray(
    [(220, 45, 45), (62, 177, 83), (55, 105, 220), (153, 74, 204)],
    dtype=np.uint8,
)
IGNORE_COLOR = (105, 105, 105)
TITLES = ("Image", "WSSS-Tissue (MLPS)", "Ours", "Ground truth")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def load_font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def open_rgb(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def render_panel(
    panels: tuple[Image.Image, ...],
    output: Path,
    cell_size: int,
    subtitle: str,
) -> None:
    gap, title_height, subtitle_height, legend_height = 14, 72, 46, 64
    width = len(TITLES) * cell_size + (len(TITLES) - 1) * gap
    height = title_height + subtitle_height + cell_size + legend_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(max(20, int(cell_size * 0.040)))
    subtitle_font = load_font(max(18, int(cell_size * 0.030)))
    legend_font = load_font(max(17, int(cell_size * 0.027)))
    for column, (title, panel) in enumerate(zip(TITLES, panels)):
        x = column * (cell_size + gap)
        text_width = draw.textlength(title, font=title_font)
        draw.text((x + max(0, (cell_size - text_width) / 2), 20), title, fill="black", font=title_font)
        resampling = Image.Resampling.LANCZOS if column == 0 else Image.Resampling.NEAREST
        canvas.paste(
            panel.resize((cell_size, cell_size), resampling),
            (x, title_height + subtitle_height),
        )
    subtitle_width = draw.textlength(subtitle, font=subtitle_font)
    draw.text(((width - subtitle_width) / 2, title_height + 8), subtitle, fill="black", font=subtitle_font)
    x, y, swatch = 8, title_height + subtitle_height + cell_size + 17, 25
    for class_idx, class_name in enumerate(CLASS_NAMES):
        draw.rectangle((x, y, x + swatch, y + swatch), fill=tuple(PALETTE[class_idx]))
        draw.text((x + swatch + 8, y), class_name, fill="black", font=legend_font)
        x += swatch + 8 + draw.textlength(class_name, font=legend_font) + 25
    draw.rectangle((x, y, x + swatch, y + swatch), fill=IGNORE_COLOR)
    draw.text((x + swatch + 8, y), "ignore", fill="black", font=legend_font)
    canvas.save(output, format="PNG", optimize=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select fixed-rule multi-class examples where Ours outperforms official MLPS."
    )
    parser.add_argument("--ours-csv", required=True)
    parser.add_argument("--mlps-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--min-classes", type=int, default=3)
    parser.add_argument("--min-ours-miou", type=float, default=0.70)
    parser.add_argument("--min-gap", type=float, default=0.03)
    parser.add_argument("--cell-size", type=int, default=640)
    args = parser.parse_args()

    ours_path, mlps_path = Path(args.ours_csv), Path(args.mlps_csv)
    ours_rows = read_csv(ours_path)
    mlps_by_name = {row["name"]: row for row in read_csv(mlps_path)}
    candidates: list[dict[str, object]] = []
    for ours in ours_rows:
        name = ours["name"]
        if name not in mlps_by_name:
            raise ValueError(f"MLPS CSV is missing selected image: {name}")
        mlps = mlps_by_name[name]
        ours_miou = float(ours["overall_miou"])
        mlps_miou = float(mlps["miou_present_union"])
        present_classes = int(ours["present_class_count"])
        gap = ours_miou - mlps_miou
        row = {
            "name": name,
            "present_class_count": present_classes,
            "ours_miou": ours_miou,
            "mlps_miou": mlps_miou,
            "miou_gap": gap,
            "ours_sample_dir": ours["sample_dir"],
            "mlps_color_mask": mlps["color_mask"],
        }
        for class_name in CLASS_NAMES:
            row[f"ours_iou_{class_name}"] = float(ours[f"iou_{class_name}"])
            row[f"mlps_iou_{class_name}"] = float(mlps[f"iou_{class_name}"])
        if (
            present_classes >= args.min_classes
            and ours_miou >= args.min_ours_miou
            and gap >= args.min_gap
        ):
            candidates.append(row)

    candidates.sort(key=lambda row: (float(row["miou_gap"]), float(row["ours_miou"])), reverse=True)
    if len(candidates) < args.count:
        all_gaps = sorted(
            (
                (row["name"], float(row["overall_miou"]) - float(mlps_by_name[row["name"]]["miou_present_union"]))
                for row in ours_rows
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        raise RuntimeError(
            f"Only {len(candidates)} examples satisfy the fixed rule, requested {args.count}. "
            f"Do not silently relax it. Available gaps: {all_gaps}"
        )
    selected = candidates[: args.count]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_paths: list[Path] = []
    for rank, row in enumerate(selected, start=1):
        sample_dir = Path(str(row["ours_sample_dir"]))
        panels = (
            open_rgb(sample_dir / "01_image.png"),
            open_rgb(Path(str(row["mlps_color_mask"]))),
            open_rgb(sample_dir / "04_final_segmentation.png"),
            open_rgb(sample_dir / "05_ground_truth.png"),
        )
        subtitle = (
            f"{row['name']} | mIoU: MLPS={float(row['mlps_miou']):.3f}, "
            f"Ours={float(row['ours_miou']):.3f}, gap={float(row['miou_gap']):+.3f}"
        )
        panel_path = output_dir / f"{rank:02d}_{Path(str(row['name'])).stem}_comparison.png"
        render_panel(
            panels,
            panel_path,
            args.cell_size,
            subtitle,
        )
        rendered_paths.append(panel_path)

    rendered = [open_rgb(path) for path in rendered_paths]
    grid_gap = 18
    grid_width = max(image.width for image in rendered)
    grid_height = sum(image.height for image in rendered) + grid_gap * (len(rendered) - 1)
    grid = Image.new("RGB", (grid_width, grid_height), "white")
    y = 0
    for image in rendered:
        grid.paste(image, ((grid_width - image.width) // 2, y))
        y += image.height + grid_gap
    grid_path = output_dir / "selected_comparison_grid.png"
    grid.save(grid_path, format="PNG", optimize=True)

    csv_output = output_dir / "selected_comparison_scores.csv"
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    config = {
        "selection_pool": "the exact images already selected by the Ours qualitative script",
        "selection_rule": {
            "min_classes": args.min_classes,
            "min_ours_miou": args.min_ours_miou,
            "min_ours_minus_mlps_miou": args.min_gap,
            "ranking": "descending Ours-minus-MLPS per-image mIoU",
        },
        "ours_csv": str(ours_path),
        "mlps_csv": str(mlps_path),
        "palette_rgb": {name: PALETTE[idx].tolist() for idx, name in enumerate(CLASS_NAMES)},
        "ignore_rgb": list(IGNORE_COLOR),
    }
    (output_dir / "comparison_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"selected={len(selected)}", flush=True)
    print(f"grid={grid_path}", flush=True)
    print(f"scores={csv_output}", flush=True)


if __name__ == "__main__":
    main()
