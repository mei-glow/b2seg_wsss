#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path


CLASS_NAMES = ("tumor", "stroma", "lymphocyte", "necrosis")
LABEL_RE = re.compile(r"\[([01]{4})\](?=\.[^.]+$)")


def default_training_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "BCSS-WSSS" / "training"


def parse_label(path: Path) -> tuple[int, ...]:
    match = LABEL_RE.search(path.name)
    if match is None:
        raise ValueError(f"No 4-bit [label] block found in filename: {path.name}")
    return tuple(int(ch) for ch in match.group(1))


def pct(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def make_report(data_dir: Path, pattern: str) -> str:
    image_paths = sorted(data_dir.glob(pattern))
    if not image_paths:
        raise FileNotFoundError(f"No files matched {pattern!r} in {data_dir}")

    per_class = Counter({name: 0 for name in CLASS_NAMES})
    combinations: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    invalid_files: list[str] = []

    for path in image_paths:
        try:
            label = parse_label(path)
        except ValueError as exc:
            invalid_files.append(str(exc))
            continue

        label_text = "".join(str(bit) for bit in label)
        combinations[label_text] += 1
        cardinality[sum(label)] += 1
        for class_name, bit in zip(CLASS_NAMES, label):
            if bit:
                per_class[class_name] += 1

    valid_count = sum(combinations.values())

    lines = [
        "# BCSS-WSSS Training Label Counts",
        "",
        f"- Data dir: `{data_dir}`",
        f"- File pattern: `{pattern}`",
        f"- Files scanned: `{len(image_paths)}`",
        f"- Files with valid labels: `{valid_count}`",
        f"- Files without valid labels: `{len(invalid_files)}`",
        "",
        "## Per-Class Positive Labels",
        "",
        "| index | bit | class | positive files | percent of valid files |",
        "|---:|---:|---|---:|---:|",
    ]

    for index, class_name in enumerate(CLASS_NAMES):
        count = per_class[class_name]
        lines.append(f"| {index} | {index + 1} | {class_name} | {count} | {pct(count, valid_count)} |")

    lines.extend(
        [
            "",
            "## Label Combinations",
            "",
            "| label | tumor | stroma | lymphocyte | necrosis | files | percent of valid files |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label_text in sorted(combinations):
        bits = [int(ch) for ch in label_text]
        lines.append(
            f"| `{label_text}` | {bits[0]} | {bits[1]} | {bits[2]} | {bits[3]} | "
            f"{combinations[label_text]} | {pct(combinations[label_text], valid_count)} |"
        )

    lines.extend(
        [
            "",
            "## Labels Per File",
            "",
            "| positive labels in file | files | percent of valid files |",
            "|---:|---:|---:|",
        ]
    )
    for label_count in sorted(cardinality):
        count = cardinality[label_count]
        lines.append(f"| {label_count} | {count} | {pct(count, valid_count)} |")

    if invalid_files:
        lines.extend(["", "## Invalid Files", ""])
        lines.extend(f"- {message}" for message in invalid_files[:50])
        if len(invalid_files) > 50:
            lines.append(f"- ... {len(invalid_files) - 50} more")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count BCSS-WSSS image-level training labels encoded as [abcd] in filenames."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_training_dir(),
        help="Training folder containing BCSS-WSSS PNG files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Glob pattern used inside --data-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional Markdown file to write. Prints to stdout when omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    if not data_dir.exists():
        print(f"Data dir does not exist: {data_dir}", file=sys.stderr)
        return 1
    if not data_dir.is_dir():
        print(f"Data dir is not a directory: {data_dir}", file=sys.stderr)
        return 1

    try:
        report = make_report(data_dir, args.pattern)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.output:
        output_path = args.output.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
