from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from b2seg_wsss.datasets import ImageLevelDataset


def compute_pos_weight(
    data_root: str | Path,
    dataset: str,
    max_weight: float = 5.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    ds = ImageLevelDataset(data_root, dataset)
    labels = torch.stack([ds[idx]["label"] for idx in range(len(ds))]).float()
    pos = labels.sum(dim=0).clamp_min(1.0)
    neg = labels.shape[0] - pos
    weights = neg / pos
    weights = weights.clamp(min=1.0, max=max_weight)
    return weights.to(device) if device is not None else weights


def compute_frequency_weight(
    data_root: str | Path,
    dataset: str,
    mode: str = "sqrt_inv",
    max_weight: float = 5.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    ds = ImageLevelDataset(data_root, dataset)
    labels = torch.stack([ds[idx]["label"] for idx in range(len(ds))]).float()
    freq = labels.mean(dim=0).clamp_min(1e-6)
    mean_freq = freq.mean()
    if mode == "sqrt_inv":
        weights = torch.sqrt(mean_freq / freq)
    elif mode == "inv":
        weights = mean_freq / freq
    elif mode == "none":
        weights = torch.ones_like(freq)
    else:
        raise ValueError(f"Unknown frequency weight mode: {mode}")
    weights = weights / weights.mean().clamp_min(1e-6)
    weights = weights.clamp(min=1.0 / max_weight, max=max_weight)
    return weights.to(device) if device is not None else weights


def multilabel_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
    )


class AuxiliaryUncertaintyWeighting(nn.Module):
    """Kendall-style uncertainty weighting for auxiliary objectives.

    Main classification supervision should remain outside this module.
    """

    def __init__(self, names: tuple[str, ...] = ("div", "jepa", "propagate"), init_log_vars: dict[str, float] | None = None) -> None:
        super().__init__()
        self.names = names
        init_log_vars = init_log_vars or {}
        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(float(init_log_vars.get(name, 0.0))))
                for name in names
            }
        )

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = None
        weights = {}
        for name, loss in losses.items():
            if name not in self.log_vars:
                raise KeyError(f"Unknown auxiliary loss for uncertainty weighting: {name}")
            log_var = self.log_vars[name].clamp(-5.0, 5.0)
            weight = torch.exp(-log_var)
            term = weight * loss + log_var
            total = term if total is None else total + term
            weights[f"{name}_uw"] = float(weight.detach().cpu())
        if total is None:
            raise ValueError("No auxiliary losses passed to uncertainty weighting.")
        return total, weights
