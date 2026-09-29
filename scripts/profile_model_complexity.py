from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from b2seg_wsss.datasets import CLASS_NAMES  # noqa: E402
from b2seg_wsss.models import DualRouteLinearWSSSModel, LinearWSSSModel  # noqa: E402
from scripts.evaluate_crf import build_model_from_checkpoint  # noqa: E402


class TensorOutput(nn.Module):
    """Expose one tensor so fvcore traces models whose normal output is a dict."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)["patch_logits"]


def load_checkpoint(path: Path) -> dict[str, object]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Expected a training checkpoint containing 'model': {path}")
    return checkpoint


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def profile_flops(model: nn.Module, image_size: int) -> tuple[int, dict[str, int]]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError("fvcore is required: pip install fvcore") from exc

    model = TensorOutput(model.eval()).cpu()
    example = torch.zeros(1, 3, image_size, image_size)
    analysis = FlopCountAnalysis(model, example)
    analysis.unsupported_ops_warnings(False)
    analysis.uncalled_modules_warnings(False)
    total = int(analysis.total())
    unsupported = {str(name): int(count) for name, count in analysis.unsupported_ops().items()}
    return total, unsupported


def model_row(name: str, model: nn.Module, image_size: int) -> dict[str, object]:
    total, trainable = count_parameters(model)
    flops, unsupported = profile_flops(model, image_size)
    return {
        "configuration": name,
        "params": total,
        "params_m": total / 1e6,
        "trainable_params": trainable,
        "trainable_params_m": trainable / 1e6,
        "flops": flops,
        "gflops": flops / 1e9,
        "unsupported_ops": unsupported,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile reproducible inference complexity from one final checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=sorted(CLASS_NAMES), required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--csv-output", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    saved = dict(checkpoint.get("args", {}))
    num_classes = len(CLASS_NAMES[args.dataset])
    model_name = str(saved.get("model", "deit_base_patch16_224"))
    route_layers = str(saved.get("route_layers", "all"))
    semantic_pooling = str(saved.get("semantic_pooling", "lse"))
    lse_tau = float(saved.get("lse_tau", 1.0))
    topk_frac = float(saved.get("topk_frac", 0.05))

    # The checkpoint has all learned weights. Do not reopen the machine-specific
    # pretraining path embedded in its saved arguments.
    build_checkpoint = dict(checkpoint)
    build_args = dict(saved)
    build_args["checkpoint"] = None
    build_checkpoint["args"] = build_args
    final_model = build_model_from_checkpoint(
        build_checkpoint, checkpoint_path, args.dataset, torch.device("cpu")
    ).eval()

    baseline = LinearWSSSModel(
        model_name=model_name,
        checkpoint_path=None,
        num_classes=num_classes,
        grad_checkpointing=False,
        pooling="max",
        topk_frac=topk_frac,
    ).eval()
    clr = DualRouteLinearWSSSModel(
        model_name=model_name,
        checkpoint_path=None,
        num_classes=num_classes,
        route_layers=route_layers,
        variant="single_semantic",
        topk_frac=topk_frac,
        semantic_pooling=semantic_pooling,
        lse_tau=lse_tau,
        semantic_init="final",
        grad_checkpointing=False,
    ).eval()

    baseline_row = model_row("DeiT-B/16", baseline, args.image_size)
    clr_row = model_row("+ CLR", clr, args.image_size)
    dual_row = model_row("+ Dual routing / fusion", final_model, args.image_size)

    def inherited_row(name: str, source: dict[str, object], status: str, note: str) -> dict[str, object]:
        row = dict(source)
        row["configuration"] = name
        row["inference_status"] = status
        row["note"] = note
        return row

    baseline_row["inference_status"] = "raw inference"
    baseline_row["note"] = "single DeiT encoder with a linear segmentation head"
    clr_row["inference_status"] = "raw inference"
    clr_row["note"] = "class-wise layer routing semantic head"
    dual_row["inference_status"] = "raw inference"
    dual_row["note"] = "final student architecture before training-only components"
    rows = [
        baseline_row,
        clr_row,
        dual_row,
        inherited_row(
            "+ EMA teacher",
            dual_row,
            "training only",
            "no test-time parameters or FLOPs; student inference is unchanged",
        ),
        inherited_row(
            "+ AAD",
            dual_row,
            "training only",
            "parameter-free training refinement; student inference is unchanged",
        ),
        inherited_row(
            "Full raw",
            dual_row,
            "raw inference",
            "EMA/AAD/loss modules are discarded; same raw student architecture",
        ),
    ]
    previous_params = int(rows[0]["params"])
    previous_flops = int(rows[0]["flops"])
    for index, row in enumerate(rows):
        current_params = int(row["params"])
        current_flops = int(row["flops"])
        row["delta_params"] = 0 if index == 0 else current_params - previous_params
        row["delta_params_m"] = row["delta_params"] / 1e6
        row["delta_gflops"] = 0.0 if index == 0 else (current_flops - previous_flops) / 1e9
        previous_params = current_params
        previous_flops = current_flops

    output = {
        "checkpoint": str(checkpoint_path),
        "dataset": args.dataset,
        "input_shape": [1, 3, args.image_size, args.image_size],
        "flop_convention": "fvcore: one fused multiply-add is counted as one FLOP",
        "configurations": rows,
        "parameter_free_components": [
            {"name": "L_abs / L_eq / L_pseudo / L_con", "test_overhead": "none"},
            {"name": "EMA teacher", "test_overhead": "none; training-only copy"},
            {"name": "AAD teacher", "test_overhead": "none; training-only"},
            {"name": "CRF", "test_overhead": "post-processing; excluded from raw GFLOPs"},
            {"name": "flip TTA", "test_overhead": "multiple forwards; excluded from raw GFLOPs"},
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    csv_path = Path(args.csv_output) if args.csv_output else output_path.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "configuration", "params_m", "delta_params_m", "trainable_params_m",
        "gflops", "delta_gflops", "inference_status", "note",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

    print(f"input_shape={output['input_shape']}", flush=True)
    for row in rows:
        print(
            f"{row['configuration']}: params={row['params_m']:.4f}M "
            f"(delta={row['delta_params_m']:+.4f}M), "
            f"GFLOPs={row['gflops']:.4f} (delta={row['delta_gflops']:+.4f})",
            flush=True,
        )
        if row["unsupported_ops"]:
            print(f"  unsupported_ops={row['unsupported_ops']}", flush=True)
    print(f"json={output_path}", flush=True)
    print(f"csv={csv_path}", flush=True)


if __name__ == "__main__":
    main()
