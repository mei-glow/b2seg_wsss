from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = {
    "bcss": ("tumor", "stroma", "lymphocyte", "necrosis"),
    "luad": ("tumor_epithelial", "necrosis", "lymphocyte", "tumor_stroma"),
}

BCSS_PALETTE = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 0, 255),
    3: (153, 0, 255),
    4: (255, 255, 255),
}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    val_img_dir: Path
    val_mask_dir: Path
    test_img_dir: Path
    test_mask_dir: Path


def resolve_dataset_paths(data_root: str | Path, dataset: str) -> DatasetPaths:
    root = Path(data_root) / {"bcss": "BCSS-WSSS", "luad": "LUAD-HistoSeg"}[dataset]
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    paths = DatasetPaths(
        root=root,
        val_img_dir=root / "val" / "img",
        val_mask_dir=root / "val" / "mask",
        test_img_dir=root / "test" / "img",
        test_mask_dir=root / "test" / "mask",
    )
    for path in (paths.val_img_dir, paths.val_mask_dir, paths.test_img_dir, paths.test_mask_dir):
        if not path.exists():
            raise FileNotFoundError(f"Expected folder not found: {path}")
    return paths


HE_DAB_STAIN_MATRIX = np.array(
    [
        [0.650, 0.072, 0.268],
        [0.704, 0.990, 0.570],
        [0.286, 0.105, 0.776],
    ],
    dtype=np.float32,
)


def hematoxylin_density(image: Image.Image, percentile: float = 99.0) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    od = -np.log((rgb + 1.0) / 255.0)
    concentrations = od.reshape(-1, 3) @ np.linalg.inv(HE_DAB_STAIN_MATRIX).T
    h = concentrations[:, 0].reshape(rgb.shape[:2])
    h = np.clip(h, 0.0, None)
    scale = float(np.percentile(h, percentile))
    if scale <= 1e-6:
        return np.zeros_like(h, dtype=np.float32)
    return np.clip(h / scale, 0.0, 1.0).astype(np.float32)


def auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float | None:
    pos = float(pos_hist.sum())
    neg = float(neg_hist.sum())
    if pos == 0.0 or neg == 0.0:
        return None
    neg_less = np.cumsum(neg_hist) - neg_hist
    wins = (pos_hist * (neg_less + 0.5 * neg_hist)).sum()
    return float(wins / (pos * neg))


def add_density_hist(hist: np.ndarray, values: np.ndarray, bins: int) -> None:
    idx = np.clip((values * bins).astype(np.int64), 0, bins - 1)
    hist += np.bincount(idx, minlength=bins)


def colorize_density(density: np.ndarray) -> Image.Image:
    x = np.clip(density, 0.0, 1.0)
    red = (255.0 * x).astype(np.uint8)
    green = (180.0 * np.sqrt(x)).astype(np.uint8)
    blue = (255.0 * (1.0 - x)).astype(np.uint8)
    return Image.fromarray(np.stack([red, green, blue], axis=-1), mode="RGB")


def colorize_mask(mask: np.ndarray) -> Image.Image:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for idx, color in BCSS_PALETTE.items():
        out[mask == idx] = color
    out[mask == 255] = (255, 255, 255)
    return Image.fromarray(out, mode="RGB")


def save_visual(image: Image.Image, mask: np.ndarray, density: np.ndarray, path: Path) -> None:
    rgb = image.convert("RGB")
    heat = colorize_density(density)
    overlay = Image.blend(rgb, heat, alpha=0.42)
    mask_img = colorize_mask(mask)
    canvas = Image.new("RGB", (rgb.width * 4, rgb.height), "white")
    canvas.paste(rgb, (0, 0))
    canvas.paste(heat, (rgb.width, 0))
    canvas.paste(overlay, (rgb.width * 2, 0))
    canvas.paste(mask_img, (rgb.width * 3, 0))
    canvas.save(path)


def summarize_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether hematoxylin density is a useful class prior."
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default="runs/density_prior_bcss_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--top-percents", default="1,5,10")
    parser.add_argument("--visual-count", type=int, default=12)
    parser.add_argument("--normalize-percentile", type=float, default=99.0)
    args = parser.parse_args()

    if args.dataset != "bcss":
        raise ValueError("This diagnostic currently expects BCSS palette masks.")

    paths = resolve_dataset_paths(args.data_root, args.dataset)
    image_dir = paths.val_img_dir if args.split == "val" else paths.test_img_dir
    mask_dir = paths.val_mask_dir if args.split == "val" else paths.test_mask_dir
    image_paths = sorted(image_dir.glob("*.png"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    output_dir = Path(args.output_dir)
    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)

    class_names = CLASS_NAMES[args.dataset]
    num_classes = len(class_names)
    bins = int(args.bins)
    top_percents = [float(x) for x in args.top_percents.replace(";", ",").split(",") if x.strip()]

    pixel_count = np.zeros(num_classes, dtype=np.int64)
    density_sum = np.zeros(num_classes, dtype=np.float64)
    density_sq_sum = np.zeros(num_classes, dtype=np.float64)
    density_hists = np.zeros((num_classes, bins), dtype=np.int64)
    top_counts = {str(p): np.zeros(num_classes, dtype=np.int64) for p in top_percents}
    image_level_rows: list[dict[str, object]] = []
    class_image_means: list[list[float]] = [[] for _ in range(num_classes)]

    for idx, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask missing for {image_path.name}: {mask_path}")
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()
        density = hematoxylin_density(image, percentile=args.normalize_percentile)

        row: dict[str, object] = {"name": image_path.name}
        valid = (mask >= 0) & (mask < num_classes)
        for class_idx, class_name in enumerate(class_names):
            cls = valid & (mask == class_idx)
            count = int(cls.sum())
            row[f"{class_name}_pixels"] = count
            if count:
                vals = density[cls]
                pixel_count[class_idx] += count
                density_sum[class_idx] += float(vals.sum())
                density_sq_sum[class_idx] += float((vals.astype(np.float64) ** 2).sum())
                add_density_hist(density_hists[class_idx], vals, bins)
                mean_val = float(vals.mean())
                class_image_means[class_idx].append(mean_val)
                row[f"{class_name}_density_mean"] = mean_val
                row[f"{class_name}_density_p90"] = float(np.percentile(vals, 90))
            else:
                row[f"{class_name}_density_mean"] = None
                row[f"{class_name}_density_p90"] = None

        flat_density = density[valid]
        flat_mask = mask[valid]
        for percent in top_percents:
            if flat_density.size == 0:
                continue
            kth = max(1, int(round(flat_density.size * percent / 100.0)))
            threshold = np.partition(flat_density, flat_density.size - kth)[flat_density.size - kth]
            selected = flat_density >= threshold
            selected_mask = flat_mask[selected]
            for class_idx in range(num_classes):
                top_counts[str(percent)][class_idx] += int((selected_mask == class_idx).sum())

        image_level_rows.append(row)
        if idx <= args.visual_count:
            save_visual(image, mask, density, visual_dir / f"{idx:03d}_{image_path.stem}.png")
        if idx == 1 or idx % 100 == 0 or idx == len(image_paths):
            print(f"processed {idx}/{len(image_paths)}", flush=True)

    class_stats = []
    total_pixels = int(pixel_count.sum())
    for class_idx, class_name in enumerate(class_names):
        count = int(pixel_count[class_idx])
        mean = float(density_sum[class_idx] / max(count, 1))
        var = float(density_sq_sum[class_idx] / max(count, 1) - mean * mean)
        pos_hist = density_hists[class_idx]
        neg_hist = density_hists.sum(axis=0) - pos_hist
        class_stats.append(
            {
                "class": class_name,
                "pixels": count,
                "pixel_fraction": float(count / max(total_pixels, 1)),
                "density_mean": mean,
                "density_std": float(np.sqrt(max(var, 0.0))),
                "one_vs_rest_auc": auc_from_hist(pos_hist, neg_hist),
                "image_mean_stats": summarize_stats(class_image_means[class_idx]),
            }
        )

    top_summary = {}
    for percent, counts in top_counts.items():
        total_top = int(counts.sum())
        top_summary[percent] = {
            class_names[class_idx]: {
                "top_pixels": int(counts[class_idx]),
                "top_fraction": float(counts[class_idx] / max(total_top, 1)),
                "class_recall_in_top": float(counts[class_idx] / max(pixel_count[class_idx], 1)),
            }
            for class_idx in range(num_classes)
        }

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "num_images": len(image_paths),
        "normalize_percentile": args.normalize_percentile,
        "class_stats": class_stats,
        "top_density_summary": top_summary,
        "visual_dir": str(visual_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "density_prior_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "image_level_density.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_level_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_level_rows)

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
