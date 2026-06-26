from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import timm


def load_local_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()

    compatible = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "loaded": len(compatible),
        "skipped": skipped,
        "missing": missing,
        "unexpected": unexpected,
    }


def parse_layer_indices(layer_spec: str | None, num_blocks: int) -> list[int] | None:
    if layer_spec is None or str(layer_spec).strip() == "":
        return None
    indices = []
    for item in str(layer_spec).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        layer = int(item)
        if layer < 1 or layer > num_blocks:
            raise ValueError(f"Fusion layer {layer} out of range 1..{num_blocks}")
        indices.append(layer - 1)
    if not indices:
        return None
    return indices


class MultiLayerViTBackbone(nn.Module):
    """Fuse intermediate ViT token layers while preserving timm ViT output shape."""

    def __init__(
        self,
        vit: nn.Module,
        layer_indices: list[int],
        fusion_mode: str = "weighted_sum",
        fusion_init: str = "average",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"weighted_sum", "concat_proj"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        if fusion_init not in {"average", "final"}:
            raise ValueError(f"Unknown fusion init: {fusion_init}")
        self.vit = vit
        self.layer_indices = sorted(set(layer_indices))
        self.fusion_mode = fusion_mode
        self.fusion_init = fusion_init
        self.num_features = vit.num_features
        self.fusion_norms = nn.ModuleList([nn.LayerNorm(self.num_features) for _ in self.layer_indices])
        if self.fusion_mode == "weighted_sum":
            logits = torch.zeros(len(self.layer_indices))
            if self.fusion_init == "final":
                logits.fill_(-4.0)
                logits[-1] = 4.0
            self.fusion_logits = nn.Parameter(logits)
            self.fusion_proj = None
        else:
            self.register_parameter("fusion_logits", None)
            self.fusion_proj = nn.Linear(self.num_features * len(self.layer_indices), self.num_features)
            if self.fusion_init == "final":
                nn.init.zeros_(self.fusion_proj.weight)
                nn.init.zeros_(self.fusion_proj.bias)
                start = self.num_features * (len(self.layer_indices) - 1)
                end = start + self.num_features
                with torch.no_grad():
                    self.fusion_proj.weight[:, start:end].copy_(torch.eye(self.num_features))
        self.fusion_out_norm = nn.LayerNorm(self.num_features)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enable)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed(x)
        x = vit._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)

        features = []
        norm_idx = 0
        use_checkpoint = bool(getattr(vit, "grad_checkpointing", False)) and self.training
        for block_idx, block in enumerate(vit.blocks):
            if use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if block_idx in self.layer_indices:
                features.append(self.fusion_norms[norm_idx](x))
                norm_idx += 1

        if not features:
            raise RuntimeError("No fusion features were collected.")
        if self.fusion_mode == "weighted_sum":
            weights = self.fusion_logits.softmax(dim=0).to(features[0].dtype)
            fused = sum(weight * feature for weight, feature in zip(weights, features))
        else:
            fused = self.fusion_proj(torch.cat(features, dim=-1))
        return self.fusion_out_norm(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def create_vit_backbone(
    model_name: str,
    checkpoint_path: str | Path | None = None,
    grad_checkpointing: bool = False,
    fusion_layers: str | None = None,
    fusion_mode: str = "weighted_sum",
    fusion_init: str = "average",
) -> torch.nn.Module:
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if checkpoint_path is not None:
        load_local_checkpoint(model, checkpoint_path)
    layer_indices = parse_layer_indices(fusion_layers, len(model.blocks))
    if layer_indices is not None:
        model = MultiLayerViTBackbone(model, layer_indices, fusion_mode=fusion_mode, fusion_init=fusion_init)
    if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model
