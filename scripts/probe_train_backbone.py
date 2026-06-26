from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.models import PrototypeWSSSModel


def default_checkpoint(model_name: str) -> Path:
    mapping = {
        "deit_small_patch16_224": "deit_small_patch16_224-cd65a155.pth",
        "deit_base_patch16_224": "deit_base_patch16_224-b5f2ef4d.pth",
    }
    if model_name not in mapping:
        raise ValueError(f"No default checkpoint known for {model_name}")
    return Path("pretrained") / mapping[model_name]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-mode VRAM probe for prototype WSSS baseline.")
    parser.add_argument("--model", default="deit_small_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--div-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model)
    device = torch.device(args.device)
    model = PrototypeWSSSModel(
        model_name=args.model,
        checkpoint_path=str(checkpoint),
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    x = torch.randn(args.batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (args.batch_size, 4), device=device).float()
    labels[labels.sum(dim=1) == 0, 0] = 1.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_steps = args.warmup + args.steps
    start = None
    last_loss = None
    for step in range(total_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if step == args.warmup:
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            outputs = model(x)
            loss = F.multilabel_soft_margin_loss(outputs["image_logits"], labels)
            loss = loss + args.div_weight * model.diversity_loss()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.detach().cpu())
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start if start is not None else 0.0
    print(f"model: {args.model}")
    print(f"checkpoint: {checkpoint}")
    print(f"batch_size: {args.batch_size}")
    print(f"amp: {args.amp}")
    print(f"grad_checkpointing: {args.grad_checkpointing}")
    print(f"loss: {last_loss:.4f}")
    print(f"sec_per_train_step: {elapsed / max(args.steps, 1):.4f}")
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        print(f"peak_allocated_gb: {peak_gb:.3f}")


if __name__ == "__main__":
    main()

