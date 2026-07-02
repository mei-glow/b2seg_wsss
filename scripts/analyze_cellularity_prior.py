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

HE_DAB_STAIN_MATRIX = np.array(
    [
        [0.650, 0.072, 0.268],
        [0.704, 0.990, 0.570],
        [0.286, 0.105, 0.776],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    val_img_dir: Path
    val_mask_dir: Path
    test_img_dir: Path
    test_mask_dir: Path


@dataclass
class FeatureAccumulator:
    name: str
    num_classes: int
    bins: int
    top_percents: list[float]
    pixel_count: np.ndarray
    score_sum: np.ndarray
    score_sq_sum: np.ndarray
    hists: np.ndarray
    top_counts: dict[str, np.ndarray]
    image_means: list[list[float]]

    @classmethod
    def create(cls, name: str, num_classes: int, bins: int, top_percents: list[float]) -> "FeatureAccumulator":
        return cls(
            name=name,
            num_classes=num_classes,
            bins=bins,
            top_percents=top_percents,
            pixel_count=np.zeros(num_classes, dtype=np.int64),
            score_sum=np.zeros(num_classes, dtype=np.float64),
            score_sq_sum=np.zeros(num_classes, dtype=np.float64),
            hists=np.zeros((num_classes, bins), dtype=np.int64),
            top_counts={str(p): np.zeros(num_classes, dtype=np.int64) for p in top_percents},
            image_means=[[] for _ in range(num_classes)],
        )

    def update(self, score: np.ndarray, mask: np.ndarray) -> None:
        valid = (mask >= 0) & (mask < self.num_classes)
        flat_score = score[valid]
        flat_mask = mask[valid]
        for class_idx in range(self.num_classes):
            cls = flat_mask == class_idx
            count = int(cls.sum())
            if count == 0:
                continue
            vals = flat_score[cls]
            self.pixel_count[class_idx] += count
            self.score_sum[class_idx] += float(vals.sum())
            self.score_sq_sum[class_idx] += float((vals.astype(np.float64) ** 2).sum())
            self.image_means[class_idx].append(float(vals.mean()))
            add_score_hist(self.hists[class_idx], vals, self.bins)

        for percent in self.top_percents:
            if flat_score.size == 0:
                continue
            kth = max(1, int(round(flat_score.size * percent / 100.0)))
            threshold = np.partition(flat_score, flat_score.size - kth)[flat_score.size - kth]
            selected_mask = flat_mask[flat_score >= threshold]
            for class_idx in range(self.num_classes):
                self.top_counts[str(percent)][class_idx] += int((selected_mask == class_idx).sum())

    def report(self, class_names: tuple[str, ...]) -> dict[str, object]:
        total_pixels = int(self.pixel_count.sum())
        total_hist = self.hists.sum(axis=0)
        class_stats = []
        for class_idx, class_name in enumerate(class_names):
            count = int(self.pixel_count[class_idx])
            mean = float(self.score_sum[class_idx] / max(count, 1))
            var = float(self.score_sq_sum[class_idx] / max(count, 1) - mean * mean)
            pos_hist = self.hists[class_idx]
            neg_hist = total_hist - pos_hist
            class_stats.append(
                {
                    "class": class_name,
                    "pixels": count,
                    "pixel_fraction": float(count / max(total_pixels, 1)),
                    "score_mean": mean,
                    "score_std": float(np.sqrt(max(var, 0.0))),
                    "one_vs_rest_auc": auc_from_hist(pos_hist, neg_hist),
                    "image_mean_stats": summarize_stats(self.image_means[class_idx]),
                }
            )

        pairwise_auc: dict[str, float | None] = {}
        for i, name_i in enumerate(class_names):
            for j, name_j in enumerate(class_names):
                if i == j:
                    continue
                pairwise_auc[f"{name_i}_vs_{name_j}"] = auc_from_hist(self.hists[i], self.hists[j])

        top_summary = {}
        for percent, counts in self.top_counts.items():
            total_top = int(counts.sum())
            top_summary[percent] = {
                class_names[class_idx]: {
                    "top_pixels": int(counts[class_idx]),
                    "top_fraction": float(counts[class_idx] / max(total_top, 1)),
                    "class_recall_in_top": float(counts[class_idx] / max(self.pixel_count[class_idx], 1)),
                }
                for class_idx in range(self.num_classes)
            }

        lymph_idx = class_names.index("lymphocyte") if "lymphocyte" in class_names else None
        tumor_idx = class_names.index("tumor") if "tumor" in class_names else 0
        summary_score = None
        if lymph_idx is not None:
            lymph_vs_tumor = auc_from_hist(self.hists[lymph_idx], self.hists[tumor_idx])
            top_1 = top_summary.get("1.0") or top_summary.get("1")
            top_1_lym = None
            if top_1 is not None:
                top_1_lym = top_1["lymphocyte"]["top_fraction"]
            summary_score = {
                "lymphocyte_vs_tumor_auc": lymph_vs_tumor,
                "lymphocyte_vs_rest_auc": class_stats[lymph_idx]["one_vs_rest_auc"],
                "top1_lymphocyte_fraction": top_1_lym,
            }

        return {
            "name": self.name,
            "class_stats": class_stats,
            "pairwise_auc": pairwise_auc,
            "top_summary": top_summary,
            "selection_summary": summary_score,
        }


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


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 1e-3)
    half = max(1, int(round(3.0 * sigma)))
    x = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_axis_reflect(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = len(kernel) // 2
    if axis == 0:
        padded = np.pad(array, ((pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(array, dtype=np.float32)
        for idx, weight in enumerate(kernel):
            out += float(weight) * padded[idx : idx + array.shape[0], :]
        return out
    if axis == 1:
        padded = np.pad(array, ((0, 0), (pad, pad)), mode="reflect")
        out = np.zeros_like(array, dtype=np.float32)
        for idx, weight in enumerate(kernel):
            out += float(weight) * padded[:, idx : idx + array.shape[1]]
        return out
    raise ValueError(f"Unsupported axis: {axis}")


def gaussian_blur(array: np.ndarray, radius: float) -> np.ndarray:
    kernel = gaussian_kernel1d(radius)
    src = array.astype(np.float32, copy=False)
    return convolve_axis_reflect(convolve_axis_reflect(src, kernel, axis=1), kernel, axis=0)


def normalize_positive(array: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    score = np.clip(array, 0.0, None).astype(np.float32)
    scale = float(np.percentile(score, percentile))
    if scale <= 1e-6:
        return np.zeros_like(score, dtype=np.float32)
    return np.clip(score / scale, 0.0, 1.0).astype(np.float32)


def local_maxima(score: np.ndarray) -> np.ndarray:
    padded = np.pad(score, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    is_max = np.ones_like(center, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            is_max &= center >= padded[1 + dy : 1 + dy + score.shape[0], 1 + dx : 1 + dx + score.shape[1]]
    return is_max


def nuclei_count_map(blob_score: np.ndarray, percentile: float, count_radius: float) -> np.ndarray:
    threshold = float(np.percentile(blob_score, percentile))
    peaks = (blob_score >= threshold) & local_maxima(blob_score)
    count = gaussian_blur(peaks.astype(np.float32), radius=count_radius)
    return normalize_positive(count, percentile=99.0)


def add_score_hist(hist: np.ndarray, values: np.ndarray, bins: int) -> None:
    idx = np.clip((values * bins).astype(np.int64), 0, bins - 1)
    hist += np.bincount(idx, minlength=bins)


def auc_from_hist(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float | None:
    pos = float(pos_hist.sum())
    neg = float(neg_hist.sum())
    if pos == 0.0 or neg == 0.0:
        return None
    neg_less = np.cumsum(neg_hist) - neg_hist
    wins = (pos_hist * (neg_less + 0.5 * neg_hist)).sum()
    return float(wins / (pos * neg))


def summarize_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "p90": float(np.percentile(arr, 90))}


def colorize_score(score: np.ndarray) -> Image.Image:
    x = np.clip(score, 0.0, 1.0)
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


def parse_scale_pairs(text: str) -> list[tuple[float, float]]:
    pairs = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        small, medium = item.split(":")
        pairs.append((float(small), float(medium)))
    return pairs


def build_features(
    h: np.ndarray,
    dog_scales: list[tuple[float, float]],
    local_radius: float,
    peak_percentile: float,
    count_radius: float,
) -> dict[str, np.ndarray]:
    features: dict[str, np.ndarray] = {
        "h_density": h,
        f"local_contrast_r{local_radius:g}": normalize_positive(h - gaussian_blur(h, local_radius)),
    }
    for small, medium in dog_scales:
        dog = normalize_positive(gaussian_blur(h, small) - gaussian_blur(h, medium))
        features[f"dog_s{small:g}_m{medium:g}"] = dog
        features[f"count_s{small:g}_m{medium:g}_p{peak_percentile:g}_r{count_radius:g}"] = nuclei_count_map(
            dog, peak_percentile, count_radius
        )
    return features


def save_visuals(
    image: Image.Image,
    mask: np.ndarray,
    features: dict[str, np.ndarray],
    feature_names: list[str],
    path: Path,
) -> None:
    rgb = image.convert("RGB")
    panels = [rgb]
    for name in feature_names:
        score = features[name]
        panels.append(colorize_score(score))
        panels.append(Image.blend(rgb, colorize_score(score), alpha=0.42))
    panels.append(colorize_mask(mask))
    canvas = Image.new("RGB", (rgb.width * len(panels), rgb.height), "white")
    for idx, panel in enumerate(panels):
        canvas.paste(panel, (rgb.width * idx, 0))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze morphology-aware cellularity priors for WSSS.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output-dir", default="runs/cellularity_prior_bcss_test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--bins", type=int, default=256)
    parser.add_argument("--top-percents", default="1,5,10")
    parser.add_argument("--visual-count", type=int, default=12)
    parser.add_argument("--visual-features", default="h_density,dog_s1.5_m3,count_s1.5_m3_p94_r6")
    parser.add_argument("--normalize-percentile", type=float, default=99.0)
    parser.add_argument("--dog-scales", default="1:2,1.5:3,2:4,3:6")
    parser.add_argument("--local-radius", type=float, default=12.0)
    parser.add_argument("--peak-percentile", type=float, default=94.0)
    parser.add_argument("--count-radius", type=float, default=6.0)
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
    top_percents = [float(x) for x in args.top_percents.replace(";", ",").split(",") if x.strip()]
    dog_scales = parse_scale_pairs(args.dog_scales)
    visual_feature_names = [x.strip() for x in args.visual_features.replace(";", ",").split(",") if x.strip()]

    accumulators: dict[str, FeatureAccumulator] = {}
    image_rows: list[dict[str, object]] = []

    for idx, image_path in enumerate(image_paths, start=1):
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask missing for {image_path.name}: {mask_path}")
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(Image.open(mask_path), dtype=np.uint8).copy()
        h = hematoxylin_density(image, percentile=args.normalize_percentile)
        features = build_features(h, dog_scales, args.local_radius, args.peak_percentile, args.count_radius)

        if not accumulators:
            for name in features:
                accumulators[name] = FeatureAccumulator.create(name, num_classes, args.bins, top_percents)

        row: dict[str, object] = {"name": image_path.name}
        for name, score in features.items():
            accumulators[name].update(score, mask)
            valid = (mask >= 0) & (mask < num_classes)
            for class_idx, class_name in enumerate(class_names):
                cls = valid & (mask == class_idx)
                row[f"{name}_{class_name}_mean"] = float(score[cls].mean()) if int(cls.sum()) else None
        image_rows.append(row)

        if idx <= args.visual_count:
            available_visuals = [name for name in visual_feature_names if name in features]
            save_visuals(image, mask, features, available_visuals, visual_dir / f"{idx:03d}_{image_path.stem}.png")
        if idx == 1 or idx % 100 == 0 or idx == len(image_paths):
            print(f"processed {idx}/{len(image_paths)}", flush=True)

    feature_reports = {name: acc.report(class_names) for name, acc in accumulators.items()}
    ranked = sorted(
        [
            {
                "feature": name,
                **(report["selection_summary"] or {}),
            }
            for name, report in feature_reports.items()
        ],
        key=lambda item: (
            item.get("lymphocyte_vs_tumor_auc") is not None,
            item.get("lymphocyte_vs_tumor_auc") or -1.0,
            item.get("top1_lymphocyte_fraction") or -1.0,
        ),
        reverse=True,
    )

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "num_images": len(image_paths),
        "normalize_percentile": args.normalize_percentile,
        "dog_scales": dog_scales,
        "local_radius": args.local_radius,
        "peak_percentile": args.peak_percentile,
        "count_radius": args.count_radius,
        "ranked_features": ranked,
        "features": feature_reports,
        "visual_dir": str(visual_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cellularity_prior_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "image_level_cellularity.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
