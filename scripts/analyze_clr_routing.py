from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must contain unique integers")
    return seeds


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a full checkpoint containing 'model': {path}")
    return checkpoint


def find_route_logits(state: dict[str, torch.Tensor]) -> tuple[str, torch.Tensor]:
    matches = [
        (key, value)
        for key, value in state.items()
        if key.removeprefix("module.").endswith("semantic_route_logits")
        and torch.is_tensor(value)
    ]
    if len(matches) != 1:
        raise KeyError(
            f"Expected exactly one semantic_route_logits tensor, found {[key for key, _ in matches]}"
        )
    return matches[0]


def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return values.mean(axis=0), values.std(axis=0, ddof=1) if len(values) > 1 else np.full(values.shape[1:], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and average class-wise CLR routing weights.")
    parser.add_argument("--checkpoint-template", required=True, help="Path containing {seed}")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0,1,2"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Class-wise layer routing")
    args = parser.parse_args()
    if "{seed}" not in args.checkpoint_template:
        parser.error("--checkpoint-template must contain {seed}")

    weights_by_seed: list[np.ndarray] = []
    class_names: list[str] | None = None
    checkpoint_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        path = Path(args.checkpoint_template.format(seed=seed))
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = load_checkpoint(path)
        saved_args = dict(checkpoint.get("args", {}))
        names = [str(item) for item in saved_args.get("class_names", [])]
        key, logits = find_route_logits(checkpoint["model"])
        weights = logits.float().softmax(dim=-1).cpu().numpy()
        if not names:
            names = [f"class_{index}" for index in range(weights.shape[0])]
        if len(names) != weights.shape[0]:
            raise ValueError(f"Class names do not match route tensor in {path}: {names}, {weights.shape}")
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError(f"Class order differs across checkpoints: {names} vs {class_names}")
        if weights_by_seed and weights.shape != weights_by_seed[0].shape:
            raise ValueError("Route tensor shapes differ across seeds")
        weights_by_seed.append(weights)
        checkpoint_rows.append({"seed": seed, "checkpoint": str(path), "state_key": key})
        print(f"loaded seed={seed} checkpoint={path} shape={weights.shape}", flush=True)

    assert class_names is not None
    stacked = np.stack(weights_by_seed, axis=0)
    mean, std = mean_std(stacked)
    layer_numbers = np.arange(1, mean.shape[1] + 1, dtype=np.float64)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_csv = output_dir / "clr_routing_all_seeds.csv"
    seed_rows: list[dict[str, object]] = []
    for seed_index, seed in enumerate(args.seeds):
        for class_index, class_name in enumerate(class_names):
            for layer_index in range(mean.shape[1]):
                seed_rows.append({
                    "seed": seed,
                    "class": class_name,
                    "layer": layer_index + 1,
                    "weight": float(stacked[seed_index, class_index, layer_index]),
                })
    with seed_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)

    summary_csv = output_dir / "clr_routing_mean_std.csv"
    summary_rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(class_names):
        for layer_index in range(mean.shape[1]):
            summary_rows.append({
                "class": class_name,
                "layer": layer_index + 1,
                "weight_mean": float(mean[class_index, layer_index]),
                "weight_std": float(std[class_index, layer_index]),
            })
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    class_summary = []
    for class_index, class_name in enumerate(class_names):
        expected_by_seed = (stacked[:, class_index, :] * layer_numbers).sum(axis=1)
        entropy_by_seed = -(stacked[:, class_index, :] * np.log(stacked[:, class_index, :] + 1e-12)).sum(axis=1)
        class_summary.append({
            "class": class_name,
            "expected_layer_mean": float(expected_by_seed.mean()),
            "expected_layer_std": float(expected_by_seed.std(ddof=1)) if len(expected_by_seed) > 1 else math.nan,
            "entropy_mean": float(entropy_by_seed.mean()),
            "peak_layer": int(mean[class_index].argmax() + 1),
            "peak_weight": float(mean[class_index].max()),
        })
    with (output_dir / "clr_routing_class_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_summary[0]))
        writer.writeheader()
        writer.writerows(class_summary)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the heatmap") from exc
    figure_width = max(8.0, mean.shape[1] * 0.65)
    fig, axis = plt.subplots(figsize=(figure_width, max(3.0, len(class_names) * 0.7)))
    image = axis.imshow(mean, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.25, float(mean.max())))
    axis.set_xticks(np.arange(mean.shape[1]), labels=[str(index + 1) for index in range(mean.shape[1])])
    axis.set_yticks(np.arange(len(class_names)), labels=class_names)
    axis.set_xlabel("Transformer layer")
    axis.set_ylabel("Tissue class")
    axis.set_title(args.title)
    for row in range(mean.shape[0]):
        for column in range(mean.shape[1]):
            axis.text(column, row, f"{mean[row, column]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if mean[row, column] > mean.max() * 0.55 else "black")
    fig.colorbar(image, ax=axis, label="Mean routing weight")
    fig.tight_layout()
    figure_path = output_dir / "clr_routing_heatmap.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "definition": "softmax(semantic_route_logits, dim=layer), averaged over seeds",
        "seeds": args.seeds,
        "checkpoints": checkpoint_rows,
        "class_names": class_names,
        "num_layers": int(mean.shape[1]),
    }
    (output_dir / "clr_routing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"heatmap={figure_path}", flush=True)
    print(f"all_seeds={seed_csv}", flush=True)
    print(f"summary={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
