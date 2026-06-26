from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import CLASS_NAMES, ImageLevelDataset, SegmentationDataset, pil_to_normalized_tensor
from jepa_wsss.losses import compute_pos_weight, multilabel_loss
from jepa_wsss.metrics import SegmentationMeter
from jepa_wsss.models import PrototypeWSSSModel


def find_checkpoint(model_name: str, checkpoint: str | None) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    names = {
        "deit_small_patch16_224": "deit_small_patch16_224-cd65a155.pth",
        "deit_base_patch16_224": "deit_base_patch16_224-b5f2ef4d.pth",
    }
    filename = names[model_name]
    candidates = [Path("pretrained") / filename, Path("jepa_wsss") / "pretrained" / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename}; checked: {', '.join(str(p) for p in candidates)}")


class TrainTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return pil_to_normalized_tensor(image)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def segmentation_from_patch_logits(patch_logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    batch, num_patches, num_classes = patch_logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    logits = patch_logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)
    logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)


@torch.no_grad()
def evaluate(model: PrototypeWSSSModel, loader: DataLoader, device: torch.device, amp: bool, num_classes: int) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes)
    cls_loss_sum = 0.0
    num_batches = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
        pred = segmentation_from_patch_logits(outputs["patch_logits"], masks.shape[-2:])
        meter.update(pred, masks)
        num_batches += 1
    metrics = meter.compute()
    metrics["loss"] = cls_loss_sum / max(num_batches, 1)
    return metrics


def write_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for key, value in row.items():
        flat[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def parse_prototype_counts(value: str | None, num_classes: int) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    counts = [int(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]
    if len(counts) != num_classes:
        raise ValueError(f"--prototype-counts expects {num_classes} comma-separated integers, got {counts}")
    if min(counts) < 1:
        raise ValueError("--prototype-counts values must be positive.")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Train prototype WSSS image-level baseline.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--dataset", default="bcss", choices=["bcss", "luad"])
    parser.add_argument("--model", default="deit_base_patch16_224", choices=["deit_small_patch16_224", "deit_base_patch16_224"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default="runs/baseline_deit_base")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--div-weight", type=float, default=0.01)
    parser.add_argument("--div-type", default="vector", choices=["vector", "spatial", "both"])
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--max-pos-weight", type=float, default=5.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--prototypes-per-class", type=int, default=4)
    parser.add_argument("--prototype-counts", default=None, help="Optional per-class prototype counts, e.g. 10,10,16,16 for BCSS.")
    parser.add_argument("--prototype-gating", action="store_true", help="Learn soft gates over an overcomplete prototype bank.")
    parser.add_argument("--gate-init", type=float, default=2.0, help="Initial prototype gate logit when --prototype-gating is enabled.")
    parser.add_argument("--gate-weight", type=float, default=0.0, help="Sparsity weight for learned prototype gates.")
    parser.add_argument("--prototype-usage-weight", type=float, default=0.0, help="Encourage present-class prototype usage diversity without changing max pooling.")
    parser.add_argument("--prototype-aggregation", default="max", choices=["max", "logmeanexp"])
    parser.add_argument("--prototype-lse-tau", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_checkpoint(args.model, args.checkpoint)
    device = torch.device(args.device)
    num_classes = len(CLASS_NAMES[args.dataset])
    prototype_counts = parse_prototype_counts(args.prototype_counts, num_classes)

    train_set = ImageLevelDataset(args.data_root, args.dataset, transform=TrainTransform())
    val_set = SegmentationDataset(args.data_root, args.dataset, split="val")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = PrototypeWSSSModel(
        model_name=args.model,
        checkpoint_path=str(checkpoint),
        num_classes=num_classes,
        prototypes_per_class=args.prototypes_per_class,
        prototype_counts=prototype_counts,
        prototype_gating=args.prototype_gating,
        gate_init=args.gate_init,
        prototype_aggregation=args.prototype_aggregation,
        lse_tau=args.prototype_lse_tau,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    pos_weight = compute_pos_weight(args.data_root, args.dataset, args.max_pos_weight, device) if args.class_balance else None
    if pos_weight is not None:
        print(f"pos_weight={pos_weight.detach().cpu().tolist()}", flush=True)

    config = vars(args).copy()
    config["checkpoint"] = str(checkpoint)
    config["class_names"] = CLASS_NAMES[args.dataset]
    config["resolved_prototype_counts"] = prototype_counts or [args.prototypes_per_class] * num_classes
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_miou = -1.0
    global_step = 0
    log_path = output_dir / "log.csv"
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = 0.0
        cls_sum = 0.0
        div_sum = 0.0
        proto_gate_sum = 0.0
        usage_sum = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                cls_loss = multilabel_loss(outputs["image_logits"], labels, pos_weight)
                vector_div_loss = model.diversity_loss()
                spatial_div_loss = model.spatial_diversity_loss(outputs["prototype_sims"], labels)
                if args.div_type == "spatial":
                    div_loss = spatial_div_loss
                elif args.div_type == "both":
                    div_loss = vector_div_loss + spatial_div_loss
                else:
                    div_loss = vector_div_loss
                proto_gate_loss = model.gate_loss()
                usage_loss = model.usage_loss(outputs["prototype_sims"], labels)
                loss = (
                    cls_loss
                    + args.div_weight * div_loss
                    + args.gate_weight * proto_gate_loss
                    + args.prototype_usage_weight * usage_loss
                ) / args.grad_accum_steps
            scaler.scale(loss).backward()
            if step % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            loss_sum += float(loss.detach().cpu()) * args.grad_accum_steps
            cls_sum += float(cls_loss.detach().cpu())
            div_sum += float(div_loss.detach().cpu())
            proto_gate_sum += float(proto_gate_loss.detach().cpu())
            usage_sum += float(usage_loss.detach().cpu())
            if args.log_every > 0 and (step == 1 or step % args.log_every == 0 or step == len(train_loader)):
                elapsed = time.perf_counter() - start
                avg_loss = loss_sum / step
                lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} cls={cls_sum / step:.4f} div={div_sum / step:.4f} "
                    f"proto_gate={proto_gate_sum / step:.4f} usage={usage_sum / step:.4f} "
                    f"lr={lr:.2e} elapsed={elapsed:.1f}s"
                )
                if device.type == "cuda":
                    mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                    msg += f" peak_mem={mem_gb:.2f}GB"
                print(msg, flush=True)

        train_loss = loss_sum / max(len(train_loader), 1)
        train_cls = cls_sum / max(len(train_loader), 1)
        train_div = div_sum / max(len(train_loader), 1)
        row: dict[str, object] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "train_cls_loss": train_cls,
            "train_div_loss": train_div,
            "train_proto_gate_loss": proto_gate_sum / max(len(train_loader), 1),
            "train_usage_loss": usage_sum / max(len(train_loader), 1),
            "active_prototypes": model.effective_prototypes().detach().cpu().tolist(),
            "sec_epoch": time.perf_counter() - start,
        }

        if epoch % args.eval_every == 0:
            val_metrics = evaluate(model, val_loader, device, args.amp, num_classes)
            row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "confusion"})
            print(
                f"epoch={epoch} train_loss={train_loss:.4f} "
                f"val_miou={val_metrics['miou']:.4f} val_mrecall={val_metrics['mrecall']:.4f} "
                f"val_fwiou={val_metrics['fwiou']:.4f}"
            )
            if float(val_metrics["miou"]) > best_miou:
                best_miou = float(val_metrics["miou"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_miou": best_miou,
                        "args": config,
                        "val_metrics": val_metrics,
                    },
                    output_dir / "best.pt",
                )
        else:
            print(f"epoch={epoch} train_loss={train_loss:.4f}")

        torch.save({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "args": config}, output_dir / "last.pt")
        write_log(log_path, row)


if __name__ == "__main__":
    main()
