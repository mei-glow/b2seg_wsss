from __future__ import annotations

import numpy as np
import torch


class SegmentationMeter:
    def __init__(self, num_classes: int = 4, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.float64)

    def reset(self) -> None:
        self.confusion[...] = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        target_np = target.detach().cpu().numpy().astype(np.int64)
        mask = (target_np != self.ignore_index) & (target_np >= 0) & (target_np < self.num_classes)
        encoded = self.num_classes * target_np[mask] + pred_np[mask]
        counts = np.bincount(encoded, minlength=self.num_classes**2)
        self.confusion += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        tp = np.diag(self.confusion)
        gt = self.confusion.sum(axis=1)
        pred = self.confusion.sum(axis=0)
        union = gt + pred - tp
        total = self.confusion.sum()

        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
        dice = np.divide(2 * tp, gt + pred, out=np.zeros_like(tp), where=(gt + pred) > 0)
        recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt > 0)
        precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
        freq = np.divide(gt, total, out=np.zeros_like(gt), where=total > 0)

        return {
            "miou": float(iou.mean()),
            "mdice": float(dice.mean()),
            "mrecall": float(recall.mean()),
            "mprecision": float(precision.mean()),
            "fwiou": float((freq * iou).sum()),
            "iou": iou.tolist(),
            "dice": dice.tolist(),
            "recall": recall.tolist(),
            "precision": precision.tolist(),
            "confusion": self.confusion.copy(),
        }

