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
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from jepa_wsss.datasets import (
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SegmentationDataset,
    parse_image_level_label,
    pil_to_normalized_tensor,
    resolve_dataset_paths,
)
from jepa_wsss.losses import multilabel_loss
from jepa_wsss.metrics import SegmentationMeter
from scripts.evaluate_crf import build_model_from_checkpoint, logits_from_outputs


class PseudoSegDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        dataset: str,
        pseudo_dir: str | Path,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        train: bool = True,
        ignore_index: int = 255,
    ) -> None:
        self.dataset = dataset
        self.paths = resolve_dataset_paths(data_root, dataset)
        self.images = sorted(self.paths.train_dir.glob("*.png"))
        if not self.images:
            raise FileNotFoundError(f"No training PNG files found in {self.paths.train_dir}")
        self.pseudo_dir = Path(pseudo_dir)
        self.mean = mean
        self.std = std
        self.train = train
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, object]:
        path = self.images[index]
        pseudo_path = self.pseudo_dir / path.name
        if not pseudo_path.exists():
            raise FileNotFoundError(f"Pseudo mask missing for {path.name}: {pseudo_path}")
        image = Image.open(path).convert("RGB")
        pseudo = Image.open(pseudo_path)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            pseudo = pseudo.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            pseudo = pseudo.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        pseudo_array = np.asarray(pseudo, dtype=np.uint8).copy()
        return {
            "image": pil_to_normalized_tensor(image, mean=self.mean, std=self.std),
            "pseudo": torch.from_numpy(pseudo_array).long(),
            "label": parse_image_level_label(path.name, self.dataset),
            "name": path.name,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_log(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(flat)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * float(cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    num_classes: int,
    prediction_head: str,
) -> dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(images)
            logits = logits_from_outputs(outputs, masks.shape[-2:], prediction_head)
        meter.update(logits.argmax(dim=1), masks)
    return meter.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune WSSS model with partial pseudo segmentation masks.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pseudo-dir", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset", default=None, choices=["bcss", "luad"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--seg-weight", type=float, default=1.0)
    parser.add_argument("--cls-weight", type=float, default=0.2)
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone and fine-tune only WSSS head/decoder on pseudo masks.")
    parser.add_argument("--ignore-index", type=int, default=255)
    parser.add_argument("--prediction-head", default="coarse", choices=["auto", "coarse", "refined"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    seed_everything(args.seed)
    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    data_root = args.data_root or saved_args.get("data_root", "data")
    dataset = args.dataset or saved_args.get("dataset", "bcss")
    image_mean = tuple(float(x) for x in saved_args.get("image_mean", IMAGENET_MEAN))
    image_std = tuple(float(x) for x in saved_args.get("image_std", IMAGENET_STD))
    num_classes = len(CLASS_NAMES[dataset])
    device = torch.device(args.device)

    model = build_model_from_checkpoint(checkpoint, ckpt_path, dataset, device)
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad_(False)
        print("freeze_backbone=true", flush=True)
    train_set = PseudoSegDataset(
        data_root=data_root,
        dataset=dataset,
        pseudo_dir=args.pseudo_dir,
        mean=image_mean,
        std=image_std,
        train=True,
        ignore_index=args.ignore_index,
    )
    val_set = SegmentationDataset(
        data_root,
        dataset,
        split="val",
        transform=lambda image: pil_to_normalized_tensor(image, mean=image_mean, std=image_std),
    )
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

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters selected.")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer,
        total_steps=len(train_loader) * args.epochs,
        warmup_steps=int(round(len(train_loader) * args.warmup_epochs)),
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = saved_args.copy()
    config["pseudo_source_checkpoint"] = str(ckpt_path)
    config["pseudo_dir"] = args.pseudo_dir
    config["pseudo_epochs"] = args.epochs
    config["pseudo_lr"] = args.lr
    config["pseudo_weight_decay"] = args.weight_decay
    config["pseudo_seg_weight"] = args.seg_weight
    config["pseudo_cls_weight"] = args.cls_weight
    config["pseudo_freeze_backbone"] = args.freeze_backbone
    config["pseudo_ignore_index"] = args.ignore_index
    config["data_root"] = data_root
    config["dataset"] = dataset
    config["image_mean"] = image_mean
    config["image_std"] = image_std
    config["class_names"] = CLASS_NAMES[dataset]
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_miou = -1.0
    global_step = 0
    log_path = output_dir / "log.csv"
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        loss_sum = seg_sum = cls_sum = valid_sum = 0.0
        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            pseudo = batch["pseudo"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                outputs = model(images)
                logits = logits_from_outputs(outputs, pseudo.shape[-2:], args.prediction_head)
                seg_loss = F.cross_entropy(logits, pseudo, ignore_index=args.ignore_index)
                cls_loss = multilabel_loss(outputs["image_logits"], labels)
                loss = args.seg_weight * seg_loss + args.cls_weight * cls_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            valid_fraction = (pseudo != args.ignore_index).float().mean()
            loss_sum += float(loss.detach().cpu())
            seg_sum += float(seg_loss.detach().cpu())
            cls_sum += float(cls_loss.detach().cpu())
            valid_sum += float(valid_fraction.detach().cpu())
            if step == 1 or step % args.log_every == 0 or step == len(train_loader):
                peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                print(
                    f"epoch={epoch}/{args.epochs} step={step}/{len(train_loader)} "
                    f"loss={loss_sum / step:.4f} seg={seg_sum / step:.4f} cls={cls_sum / step:.4f} "
                    f"valid={valid_sum / step:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
                    f"elapsed={time.perf_counter() - start:.1f}s peak_mem={peak:.2f}GB",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device, args.amp, num_classes, args.prediction_head)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": loss_sum / max(len(train_loader), 1),
            "train_seg_loss": seg_sum / max(len(train_loader), 1),
            "train_cls_loss": cls_sum / max(len(train_loader), 1),
            "train_valid_fraction": valid_sum / max(len(train_loader), 1),
            "sec_epoch": time.perf_counter() - start,
            **{f"val_{key}": value for key, value in val_metrics.items() if key != "confusion"},
        }
        write_log(log_path, row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} "
            f"val_miou={val_metrics['miou']:.4f} val_mrecall={val_metrics['mrecall']:.4f} "
            f"val_fwiou={val_metrics['fwiou']:.4f}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": config,
            "val_metrics": val_metrics,
        }
        torch.save(state, output_dir / "last.pt")
        if float(val_metrics["miou"]) > best_miou:
            best_miou = float(val_metrics["miou"])
            state["best_miou"] = best_miou
            torch.save(state, output_dir / "best.pt")


if __name__ == "__main__":
    main()
