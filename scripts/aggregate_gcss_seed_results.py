import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SCALAR_METRICS = ("miou", "mdice", "pixel_accuracy", "fwiou")
CLASS_METRICS = ("iou", "dice")
CANONICAL_CLASSES = (
    "background",
    "tumor",
    "lymphoid_stroma",
    "desmoplastic_stroma",
    "smooth_muscle",
    "necrosis",
)
CLASS_ALIASES = {
    "background": "background",
    "back": "background",
    "tum": "tumor",
    "tumor": "tumor",
    "lym_str": "lymphoid_stroma",
    "lymphoid_stroma": "lymphoid_stroma",
    "des_str": "desmoplastic_stroma",
    "desmoplastic_stroma": "desmoplastic_stroma",
    "smooth_muscle": "smooth_muscle",
    "nec": "necrosis",
    "necrosis": "necrosis",
}


def canonical_class_order(result, path):
    names = result.get("class_names")
    if not isinstance(names, list) or len(names) != len(CANONICAL_CLASSES):
        raise ValueError(f"Expected six class_names in {path}, got {names}")
    normalized = []
    for name in names:
        key = str(name).strip().lower().replace(" ", "_")
        if key not in CLASS_ALIASES:
            raise ValueError(f"Unknown GCSS class name {name!r} in {path}")
        normalized.append(CLASS_ALIASES[key])
    if set(normalized) != set(CANONICAL_CLASSES):
        raise ValueError(f"Class names are incomplete or duplicated in {path}: {normalized}")
    return [normalized.index(name) for name in CANONICAL_CLASSES]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--all-seeds-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    rows = []
    grouped = defaultdict(lambda: defaultdict(list))
    expected_dataset = None
    expected_split = None
    for pattern in args.inputs:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        for path in matches:
            result = json.loads(path.read_text(encoding="utf-8"))
            dataset = result.get("dataset", "")
            split = result.get("split", "")
            if expected_dataset is None:
                expected_dataset, expected_split = dataset, split
            if dataset != expected_dataset or split != expected_split:
                raise ValueError(
                    f"Cannot aggregate mixed dataset/split results: expected "
                    f"{expected_dataset}/{expected_split}, got {dataset}/{split} in {path}"
                )
            row = {
                "model": result["model"],
                "dataset": dataset,
                "split": split,
                "seed": int(result["seed"]),
                "checkpoint": result.get("checkpoint", ""),
                "result_file": str(path),
            }
            for metric in SCALAR_METRICS:
                row[metric] = float(result["metrics"][metric])
                grouped[result["model"]][metric].append(row[metric])
            canonical_indices = canonical_class_order(result, path)
            for metric in CLASS_METRICS:
                values = result["metrics"].get(metric)
                if not isinstance(values, list) or len(values) != len(CANONICAL_CLASSES):
                    raise ValueError(f"Expected six per-class {metric} values in {path}")
                for class_name, source_index in zip(CANONICAL_CLASSES, canonical_indices):
                    key = f"{metric}_{class_name}"
                    row[key] = float(values[source_index])
                    grouped[result["model"]][key].append(row[key])
            rows.append(row)
    if not rows:
        raise FileNotFoundError("No result JSON files matched --inputs")
    rows.sort(key=lambda row: (row["model"], row["seed"]))

    all_output = Path(args.all_seeds_output)
    all_output.parent.mkdir(parents=True, exist_ok=True)
    with all_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for model in sorted(grouped):
        seed_count = len(grouped[model][SCALAR_METRICS[0]])
        row = {"model": model, "num_seeds": seed_count}
        for metric in SCALAR_METRICS:
            values = np.asarray(grouped[model][metric], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_mean_std"] = f"{values.mean():.6f} +/- {(values.std(ddof=1) if len(values) > 1 else 0.0):.6f}"
        for metric in CLASS_METRICS:
            for class_name in CANONICAL_CLASSES:
                key = f"{metric}_{class_name}"
                values = np.asarray(grouped[model][key], dtype=np.float64)
                mean = float(np.nanmean(values))
                valid = values[~np.isnan(values)]
                std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
                row[f"{key}_mean"] = mean
                row[f"{key}_std"] = std
                row[f"{key}_mean_std"] = f"{mean:.6f} +/- {std:.6f}"
        summary_rows.append(row)

    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"all_seeds={all_output}")
    print(f"summary={summary_output}")


if __name__ == "__main__":
    main()
