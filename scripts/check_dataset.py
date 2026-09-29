from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b2seg_wsss.datasets import (
    CLASS_NAMES,
    ImageLevelDataset,
    SegmentationDataset,
    parse_image_level_label,
    resolve_dataset_paths,
)


def count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png")))


def label_distribution(train_images: list[Path], dataset: str) -> tuple[Counter, Counter]:
    per_class = Counter()
    patterns = Counter()
    for path in train_images:
        label = tuple(int(x) for x in parse_image_level_label(path.name, dataset).tolist())
        patterns[label] += 1
        for idx, value in enumerate(label):
            if value:
                per_class[idx] += 1
    return per_class, patterns


def mask_ids(mask_dir: Path, limit: int) -> Counter:
    counts = Counter()
    for path in sorted(mask_dir.glob("*.png"))[:limit]:
        ids = np.unique(np.asarray(Image.open(path), dtype=np.uint8))
        for value in ids.tolist():
            counts[int(value)] += 1
    return counts


def check_pairs(image_dir: Path, mask_dir: Path) -> tuple[int, list[str]]:
    image_names = {path.name for path in image_dir.glob("*.png")}
    mask_names = {path.name for path in mask_dir.glob("*.png")}
    missing = sorted(image_names - mask_names)
    extra = sorted(mask_names - image_names)
    return len(image_names), missing[:5] + extra[:5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity check WSSS dataset layout and labels.")
    parser.add_argument("--data-root", default="data", help="Folder containing BCSS-WSSS, LUAD-HistoSeg, or GCSS.")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad", "gcss"])
    parser.add_argument("--mask-scan-limit", type=int, default=256)
    args = parser.parse_args()

    paths = resolve_dataset_paths(args.data_root, args.dataset)
    train_images = sorted(paths.train_dir.glob("*.png"))
    per_class, patterns = label_distribution(train_images, args.dataset)

    print(f"dataset: {args.dataset}")
    print(f"root: {paths.root}")
    print(f"train_dir: {paths.train_dir}")
    print(f"train_count: {count_pngs(paths.train_dir)}")
    print(f"val_img_count: {count_pngs(paths.val_img_dir)}")
    print(f"val_mask_count: {count_pngs(paths.val_mask_dir)}")
    print(f"test_img_count: {count_pngs(paths.test_img_dir)}")
    print(f"test_mask_count: {count_pngs(paths.test_mask_dir)}")

    print("class_positive_counts:")
    for idx, name in enumerate(CLASS_NAMES[args.dataset]):
        print(f"  {idx}:{name}: {per_class[idx]}")

    print("top_label_patterns:")
    for pattern, count in patterns.most_common(10):
        print(f"  {pattern}: {count}")

    val_count, val_pair_issues = check_pairs(paths.val_img_dir, paths.val_mask_dir)
    test_count, test_pair_issues = check_pairs(paths.test_img_dir, paths.test_mask_dir)
    print(f"val_pairs_checked: {val_count}, issues_sample: {val_pair_issues}")
    print(f"test_pairs_checked: {test_count}, issues_sample: {test_pair_issues}")
    print(f"val_mask_ids_first_{args.mask_scan_limit}: {dict(sorted(mask_ids(paths.val_mask_dir, args.mask_scan_limit).items()))}")
    print(f"test_mask_ids_first_{args.mask_scan_limit}: {dict(sorted(mask_ids(paths.test_mask_dir, args.mask_scan_limit).items()))}")

    train_dataset = ImageLevelDataset(args.data_root, args.dataset)
    val_dataset = SegmentationDataset(args.data_root, args.dataset, "val")
    train_item = train_dataset[0]
    val_item = val_dataset[0]
    print("loader_sample:")
    print(f"  train_image_shape: {tuple(train_item['image'].shape)}, label: {train_item['label'].tolist()}, name: {train_item['name']}")
    print(f"  val_image_shape: {tuple(val_item['image'].shape)}, mask_shape: {tuple(val_item['mask'].shape)}, name: {val_item['name']}")


if __name__ == "__main__":
    main()
