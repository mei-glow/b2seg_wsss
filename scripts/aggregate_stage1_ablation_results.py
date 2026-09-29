from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


SCALAR_METRICS = ("miou", "mdice", "mrecall", "mprecision", "fwiou")
CLASS_METRICS = ("iou", "dice", "recall", "precision")
STAGE2_FLIP_TTA_ABLATIONS = {
    "S2_A4_full_n2",
    "S2_A4_full_n2_2",
    "S2_A4_full_n2_best",
}


@dataclass(frozen=True)
class Ablation:
    key: str
    label: str
    directory: str
    class_balance: str
    pl: str = ""
    dual_fusion: str = ""
    adaptive_ema: str = ""
    crf_teacher: str = ""
    consistency: str = ""


def stage1_ablations(dataset: str) -> tuple[Ablation, ...]:
    return (
        Ablation(
            "A0_no_balance",
            "A0 DeiT baseline (no class balance)",
            f"{dataset}_s1_ab0_deit_nobalance_5ep_seed{{seed}}",
            "no",
        ),
        Ablation(
            "A0_balanced",
            "A0 DeiT baseline (class balanced)",
            f"{dataset}_s1_ab0_deit_5ep_seed{{seed}}",
            "yes",
        ),
        Ablation(
            "A1_routing",
            "A1 + class-wise layer routing",
            f"{dataset}_s1_ab1_routing_5ep_seed{{seed}}",
            "yes",
        ),
        Ablation(
            "A2_lse",
            "A2 + LSE-MIL",
            f"{dataset}_s1_ab2_lse_5ep_seed{{seed}}",
            "yes",
        ),
        Ablation(
            "A3_abs",
            "A3 + absent-token loss",
            f"{dataset}_s1_ab3_abs_5ep_seed{{seed}}",
            "yes",
        ),
        Ablation(
            "A4_equiv",
            "A4 + equivariance loss",
            f"{dataset}_s1_ab4_equiv_5ep_seed{{seed}}",
            "yes",
        ),
        Ablation(
            "A5_full",
            "A5 full Stage 1",
            f"{dataset}_s1_F_seed{{seed}}",
            "yes",
        ),
        Ablation(
            "A5_full_noeq",
            "A5 full Stage 1 without equivariance loss",
            f"{dataset}_s1_full_noeq_seed{{seed}}",
            "yes",
        ),
    )


def stage2_ablations(dataset: str) -> tuple[Ablation, ...]:
    return (
        Ablation(
            "S2_A0_fixed_pl",
            "S2-A0 fixed teacher + fixed-threshold L_PL (single semantic branch)",
            f"{dataset}_s2_ab0_fixed_pl_seed{{seed}}",
            "",
            "yes",
        ),
        Ablation(
            "S2_A1_dual_fusion",
            "S2-A1 + dual routing and class-wise learned fusion",
            f"{dataset}_s2_ab1_dual_fusion_seed{{seed}}",
            "",
            "yes",
            "yes",
        ),
        Ablation(
            "S2_A2_adaptive_ema",
            "S2-A2 + adaptive EMA pseudo-label teacher",
            f"{dataset}_s2_ab2_adaptive_ema_seed{{seed}}",
            "",
            "yes",
            "yes",
            "yes",
        ),
        Ablation(
            "S2_A2C_ema_cons",
            "S2-A2C adaptive EMA + strong-view consistency (no CRF teacher)",
            f"{dataset}_s2_ab2c_ema_cons_seed{{seed}}",
            "",
            "yes",
            "yes",
            "yes",
            "",
            "yes",
        ),
        Ablation(
            "S2_A3_crf_teacher",
            "S2-A3 + training-time CRF teacher (AAD)",
            f"{dataset}_s2_ab3_crf_teacher_seed{{seed}}",
            "",
            "yes",
            "yes",
            "yes",
            "yes",
        ),
        Ablation(
            "S2_A4_full_n2",
            "S2-A4 full N2 + strong-view consistency",
            f"{dataset}_s2_ab4_full_n2_seed{{seed}}",
            "",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ),
        Ablation(
            "S2_A4_full_n2_2",
            "S2-A4 full N2 rerun 2",
            f"{dataset}_s2_ab4_full_n2_2_seed{{seed}}",
            "",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ),
        Ablation(
            "S2_A4_full_n2_best",
            "S2-A4 full N2 best variant",
            f"{dataset}_s2_ab4_full_n2_best_seed{{seed}}",
            "",
            "yes",
            "yes",
            "yes",
            "yes",
            "yes",
        ),
    )


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"Expected unique comma-separated seeds, got {value!r}")
    return seeds


def metric_columns(class_names: list[str], metric_view: str) -> list[str]:
    base_columns = list(SCALAR_METRICS)
    for metric in CLASS_METRICS:
        base_columns.extend(f"{metric}_{name}" for name in class_names)
    return base_columns


def read_result(
    path: Path,
    expected_split: str,
    metric_view: str,
    expected_tta: tuple[str, ...] | None = None,
) -> tuple[dict[str, float], list[str], str]:
    result = json.loads(path.read_text(encoding="utf-8"))
    split = str(result.get("split", ""))
    if split != expected_split:
        raise ValueError(f"Expected split={expected_split!r}, got {split!r} in {path}")
    if expected_tta is not None:
        actual_tta = result.get("tta")
        if not isinstance(actual_tta, list):
            raise ValueError(f"Missing TTA metadata in {path}")
        actual_tta = tuple(str(mode) for mode in actual_tta)
        if actual_tta != expected_tta:
            raise ValueError(
                f"TTA mismatch in {path}: expected {expected_tta}, got {actual_tta}"
            )

    class_names = result.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError(f"Missing class_names in {path}")
    class_names = [str(name) for name in class_names]

    selected_view = metric_view
    if metric_view == "auto":
        selected_view = "crf" if isinstance(result.get("crf"), dict) else "top_level"
    if selected_view == "both":
        raise ValueError(
            "read_result expects one metric view at a time; split 'both' in the caller"
        )
    selected_views = (selected_view,)
    values: dict[str, float] = {}
    for view in selected_views:
        if view == "top_level":
            metrics_source = result
        else:
            metrics_source = result.get(view)
            if not isinstance(metrics_source, dict):
                raise ValueError(
                    f"Requested metric view {view!r}, but it is missing in {path}"
                )
        for metric in SCALAR_METRICS:
            if metric not in metrics_source:
                raise ValueError(f"Missing scalar metric {metric!r} in {path}")
            value = float(metrics_source[metric])
            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite scalar metric {metric!r}={value} in {path}"
                )
            values[metric] = value
        for metric in CLASS_METRICS:
            per_class = metrics_source.get(metric)
            if not isinstance(per_class, list) or len(per_class) != len(class_names):
                raise ValueError(
                    f"Expected {len(class_names)} per-class {metric} values in {path}, got {per_class}"
                )
            for class_name, value in zip(class_names, per_class):
                numeric_value = float(value)
                if not math.isfinite(numeric_value):
                    raise ValueError(
                        f"Non-finite {metric}_{class_name}={numeric_value} in {path}"
                    )
                values[f"{metric}_{class_name}"] = numeric_value
        for scalar, per_class_metric in (
            ("miou", "iou"),
            ("mdice", "dice"),
            ("mrecall", "recall"),
            ("mprecision", "precision"),
        ):
            recomputed = statistics.fmean(
                values[f"{per_class_metric}_{name}"] for name in class_names
            )
            if not math.isclose(values[scalar], recomputed, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(
                    f"Inconsistent {scalar} in {path}: stored={values[scalar]}, "
                    f"mean({per_class_metric})={recomputed}"
                )
    return values, class_names, selected_view


def finite_mean_std(values: list[float]) -> tuple[float, float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return math.nan, math.nan
    mean = statistics.fmean(finite)
    std = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return mean, std


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate cumulative Stage-1 or Stage-2 ablation results."
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage", default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad", "gcss"])
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--result-name",
        default=None,
        help=(
            "Result filename inside each run. Defaults to test_metrics.json for "
            "Stage 1 and test_metrics_crf.json for Stage 2."
        ),
    )
    parser.add_argument(
        "--metric-view",
        default="auto",
        choices=["auto", "top_level", "raw", "crf", "both"],
        help=(
            "Metric block to aggregate. auto selects CRF for evaluate_crf.py JSON "
            "and top-level metrics for evaluate.py JSON; both emits separate raw "
            "and CRF rows with identical metric columns."
        ),
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail on a missing result instead of warning and skipping it.",
    )
    parser.add_argument("--all-seeds-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    seeds = parse_seeds(args.seeds)
    specs = (
        stage1_ablations(args.dataset)
        if args.stage == "stage1"
        else stage2_ablations(args.dataset)
    )
    result_name = args.result_name or (
        "test_metrics.json" if args.stage == "stage1" else "test_metrics_crf.json"
    )
    metric_view = (
        "both"
        if args.stage == "stage2" and args.metric_view == "auto"
        else args.metric_view
    )
    rows: list[dict[str, object]] = []
    expected_class_names: list[str] | None = None
    missing: list[Path] = []

    for order, spec in enumerate(specs):
        for seed in seeds:
            path = run_root / spec.directory.format(seed=seed) / result_name
            if not path.is_file():
                if args.strict_missing:
                    raise FileNotFoundError(
                        f"Missing result for {spec.key}, seed {seed}: {path}"
                    )
                missing.append(path)
                print(f"SKIP missing {spec.key} seed={seed}: {path}")
                continue
            views = ("raw", "crf") if metric_view == "both" else (metric_view,)
            for view in views:
                values, class_names, selected_view = read_result(
                    path,
                    args.split,
                    view,
                    expected_tta=("id",) if args.stage == "stage2" else None,
                )
                if expected_class_names is None:
                    expected_class_names = class_names
                elif class_names != expected_class_names:
                    raise ValueError(
                        f"Class order mismatch in {path}: expected {expected_class_names}, got {class_names}"
                    )
                rows.append(
                    {
                        "order": order,
                        "stage": args.stage,
                        "ablation": spec.key,
                        "description": spec.label,
                        "class_balance": spec.class_balance,
                        "baseline_pl": spec.pl,
                        "dual_routing_fusion": spec.dual_fusion,
                        "adaptive_ema": spec.adaptive_ema,
                        "crf_teacher": spec.crf_teacher,
                        "consistency_loss": spec.consistency,
                        "dataset": args.dataset,
                        "split": args.split,
                        "metric_view": selected_view,
                        "seed": seed,
                        **values,
                        "checkpoint": str(
                            run_root / spec.directory.format(seed=seed) / "best.pt"
                        ),
                        "result_file": str(path),
                    }
                )
            if (
                args.stage == "stage2"
                and spec.key in STAGE2_FLIP_TTA_ABLATIONS
                and metric_view == "both"
            ):
                tta_path = (
                    run_root
                    / spec.directory.format(seed=seed)
                    / "test_metrics_crf_tta_flip.json"
                )
                if not tta_path.is_file():
                    if args.strict_missing:
                        raise FileNotFoundError(
                            f"Missing CRF flip-TTA result for {spec.key}, "
                            f"seed {seed}: {tta_path}"
                        )
                    missing.append(tta_path)
                    print(
                        f"SKIP missing {spec.key} metric_view=crf_tta_flip "
                        f"seed={seed}: {tta_path}"
                    )
                else:
                    values, class_names, _ = read_result(
                        tta_path,
                        args.split,
                        "crf",
                        expected_tta=("id", "hflip", "vflip", "hvflip"),
                    )
                    if class_names != expected_class_names:
                        raise ValueError(
                            f"Class order mismatch in {tta_path}: expected "
                            f"{expected_class_names}, got {class_names}"
                        )
                    rows.append(
                        {
                            "order": order,
                            "stage": args.stage,
                            "ablation": spec.key,
                            "description": spec.label,
                            "class_balance": spec.class_balance,
                            "baseline_pl": spec.pl,
                            "dual_routing_fusion": spec.dual_fusion,
                            "adaptive_ema": spec.adaptive_ema,
                            "crf_teacher": spec.crf_teacher,
                            "consistency_loss": spec.consistency,
                            "dataset": args.dataset,
                            "split": args.split,
                            "metric_view": "crf_tta_flip",
                            "seed": seed,
                            **values,
                            "checkpoint": str(
                                run_root
                                / spec.directory.format(seed=seed)
                                / "best.pt"
                            ),
                            "result_file": str(tta_path),
                        }
                    )

    if expected_class_names is None:
        raise FileNotFoundError(
            f"No {result_name} files were found under {run_root}"
        )
    row_keys = [
        (str(row["ablation"]), int(row["seed"]), str(row["metric_view"]))
        for row in rows
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError("Duplicate (ablation, seed, metric_view) rows detected")
    metrics = metric_columns(expected_class_names, metric_view)
    full_fields = [
        "order",
        "stage",
        "ablation",
        "description",
        "class_balance",
        "baseline_pl",
        "dual_routing_fusion",
        "adaptive_ema",
        "crf_teacher",
        "consistency_loss",
        "dataset",
        "split",
        "metric_view",
        "seed",
        *metrics,
        "checkpoint",
        "result_file",
    ]
    write_csv(Path(args.all_seeds_output), rows, full_fields)

    summary_rows: list[dict[str, object]] = []
    for order, spec in enumerate(specs):
        ablation_rows = [row for row in rows if row["ablation"] == spec.key]
        if not ablation_rows:
            print(f"SKIP summary {spec.key}: no completed seeds")
            continue
        if metric_view == "both":
            view_order = (
                ("raw", "crf", "crf_tta_flip")
                if spec.key in STAGE2_FLIP_TTA_ABLATIONS
                else ("raw", "crf")
            )
        else:
            view_order = tuple(
                dict.fromkeys(str(row["metric_view"]) for row in ablation_rows)
            )
        for view in view_order:
            selected = [
                row for row in ablation_rows if row["metric_view"] == view
            ]
            if not selected:
                continue
            summary: dict[str, object] = {
                "order": order,
                "stage": args.stage,
                "ablation": spec.key,
                "description": spec.label,
                "class_balance": spec.class_balance,
                "baseline_pl": spec.pl,
                "dual_routing_fusion": spec.dual_fusion,
                "adaptive_ema": spec.adaptive_ema,
                "crf_teacher": spec.crf_teacher,
                "consistency_loss": spec.consistency,
                "dataset": args.dataset,
                "split": args.split,
                "metric_view": view,
                "num_seeds": len(selected),
            }
            for metric in metrics:
                mean, std = finite_mean_std(
                    [float(row[metric]) for row in selected]
                )
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
                summary[f"{metric}_mean_std"] = f"{mean:.6f} +/- {std:.6f}"
            summary_rows.append(summary)

    summary_fields = [
        "order",
        "stage",
        "ablation",
        "description",
        "class_balance",
        "baseline_pl",
        "dual_routing_fusion",
        "adaptive_ema",
        "crf_teacher",
        "consistency_loss",
        "dataset",
        "split",
        "metric_view",
        "num_seeds",
    ]
    for metric in metrics:
        summary_fields.extend(
            (f"{metric}_mean", f"{metric}_std", f"{metric}_mean_std")
        )
    write_csv(Path(args.summary_output), summary_rows, summary_fields)

    completed_configs = len({str(row["ablation"]) for row in rows})
    print(
        f"configurations={completed_configs}/{len(specs)} "
        f"summary_rows={len(summary_rows)} requested_seeds={len(seeds)} "
        f"seed_view_rows={len(rows)} missing={len(missing)}"
    )
    print(f"all_seeds={args.all_seeds_output}")
    print(f"summary={args.summary_output}")


if __name__ == "__main__":
    main()
