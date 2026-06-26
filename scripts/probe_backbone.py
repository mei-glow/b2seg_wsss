from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.backbones import create_vit_backbone, load_local_checkpoint


def default_checkpoint(model_name: str) -> Path:
    mapping = {
        "deit_small_patch16_224": "deit_small_patch16_224-cd65a155.pth",
        "deit_base_patch16_224": "deit_base_patch16_224-b5f2ef4d.pth",
    }
    if model_name not in mapping:
        raise ValueError(f"No default checkpoint known for {model_name}")
    return Path("pretrained") / mapping[model_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DeiT backbone smoke test.")
    parser.add_argument("--model", default="deit_small_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model)
    device = torch.device(args.device)

    model = create_vit_backbone(args.model, grad_checkpointing=args.grad_checkpointing)
    report = load_local_checkpoint(model, checkpoint)
    model.to(device).eval()

    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_steps = args.warmup + args.steps
    start = None
    with torch.no_grad():
        for step in range(total_steps):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step == args.warmup:
                start = time.perf_counter()
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                y = model.forward_features(x)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start if start is not None else 0.0
    y_shape = tuple(y.shape) if hasattr(y, "shape") else type(y).__name__

    print(f"model: {args.model}")
    print(f"checkpoint: {checkpoint}")
    print(f"checkpoint_loaded_tensors: {report['loaded']}")
    print(f"batch_size: {args.batch_size}")
    print(f"amp: {args.amp}")
    print(f"grad_checkpointing: {args.grad_checkpointing}")
    print(f"forward_features_shape: {y_shape}")
    print(f"sec_per_batch: {elapsed / max(args.steps, 1):.4f}")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"peak_allocated_gb: {peak_gb:.3f}")


if __name__ == "__main__":
    main()

