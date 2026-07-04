from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _patch_grid_size(img_size: int | tuple[int, int], patch_size: int | tuple[int, int], stride: int, padding: int) -> tuple[int, int]:
    if isinstance(img_size, tuple):
        img_h, img_w = img_size
    else:
        img_h = img_w = int(img_size)
    if isinstance(patch_size, tuple):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = int(patch_size)
    grid_h = (img_h + 2 * padding - patch_h) // stride + 1
    grid_w = (img_w + 2 * padding - patch_w) // stride + 1
    return int(grid_h), int(grid_w)


def configure_patch_stride(vit: nn.Module, patch_stride: int | None = None, patch_padding: int = 0) -> None:
    if patch_stride is None:
        return
    patch_embed = vit.patch_embed
    if not hasattr(patch_embed, "proj"):
        raise ValueError("Expected timm ViT patch_embed to have a proj convolution.")
    patch_embed.proj.stride = (int(patch_stride), int(patch_stride))
    patch_embed.proj.padding = (int(patch_padding), int(patch_padding))
    img_size = getattr(patch_embed, "img_size", (224, 224))
    patch_size = getattr(patch_embed, "patch_size", patch_embed.proj.kernel_size)
    grid_size = _patch_grid_size(img_size, patch_size, int(patch_stride), int(patch_padding))
    patch_embed.grid_size = grid_size
    patch_embed.num_patches = grid_size[0] * grid_size[1]
    vit.patch_grid_size = grid_size


def _custom_pos_embed(vit: nn.Module, x: torch.Tensor) -> torch.Tensor:
    cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
    if getattr(vit, "no_embed_class", False):
        x = x + _resized_patch_pos_embed(vit, x.shape[1], x.dtype)
        return torch.cat((cls_token, x), dim=1)
    x = torch.cat((cls_token, x), dim=1)
    pos_embed = vit.pos_embed
    num_prefix_tokens = x.shape[1] - (x.shape[1] - 1)
    prefix_pos = pos_embed[:, :num_prefix_tokens]
    patch_pos = pos_embed[:, num_prefix_tokens:]
    resized_patch_pos = _resize_patch_pos(patch_pos, x.shape[1] - num_prefix_tokens, x.dtype)
    return vit.pos_drop(x + torch.cat((prefix_pos.to(x.dtype), resized_patch_pos), dim=1))


def _resized_patch_pos_embed(vit: nn.Module, num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
    pos_embed = vit.pos_embed
    return _resize_patch_pos(pos_embed, num_tokens, dtype)


def _resize_patch_pos(patch_pos: torch.Tensor, num_tokens: int, dtype: torch.dtype) -> torch.Tensor:
    if patch_pos.shape[1] == num_tokens:
        return patch_pos.to(dtype)
    old_grid = int(patch_pos.shape[1] ** 0.5)
    new_grid = int(num_tokens ** 0.5)
    if old_grid * old_grid != patch_pos.shape[1] or new_grid * new_grid != num_tokens:
        raise ValueError(f"Expected square position grids, got old={patch_pos.shape[1]} new={num_tokens}")
    patch_pos_map = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    patch_pos_map = F.interpolate(patch_pos_map, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    return patch_pos_map.permute(0, 2, 3, 1).reshape(1, num_tokens, -1).to(dtype)


class ViTForwardMixin:
    vit: nn.Module
    use_custom_pos_embed: bool

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_custom_pos_embed:
            return _custom_pos_embed(self.vit, x)
        return self.vit._pos_embed(x)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing(enable)


class OverlapViTBackbone(ViTForwardMixin, nn.Module):
    """ViT wrapper with optional overlapping patch embedding while preserving timm output shape."""

    def __init__(self, vit: nn.Module, use_custom_pos_embed: bool = False) -> None:
        super().__init__()
        self.vit = vit
        self.use_custom_pos_embed = use_custom_pos_embed
        self.num_features = vit.num_features
        self.num_patches = vit.patch_embed.num_patches
        self.patch_grid_size = vit.patch_embed.grid_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed(x)
        x = self._pos_embed(x)
        x = vit.patch_drop(x)
        x = vit.norm_pre(x)
        use_checkpoint = bool(getattr(vit, "grad_checkpointing", False)) and self.training
        for block in vit.blocks:
            if use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = vit.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


class MultiLayerViTBackbone(ViTForwardMixin, nn.Module):
    """Fuse intermediate ViT token layers while preserving timm ViT output shape."""

    def __init__(
        self,
        vit: nn.Module,
        layer_indices: list[int],
        fusion_mode: str = "weighted_sum",
        fusion_init: str = "average",
        use_custom_pos_embed: bool = False,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"weighted_sum", "concat_proj"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        if fusion_init not in {"average", "final"}:
            raise ValueError(f"Unknown fusion init: {fusion_init}")
        self.vit = vit
        self.use_custom_pos_embed = use_custom_pos_embed
        self.layer_indices = sorted(set(layer_indices))
        self.fusion_mode = fusion_mode
        self.fusion_init = fusion_init
        self.num_features = vit.num_features
        self.num_patches = vit.patch_embed.num_patches
        self.patch_grid_size = vit.patch_embed.grid_size
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        vit = self.vit
        x = vit.patch_embed(x)
        x = self._pos_embed(x)
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


class HFDinoV2Backbone(nn.Module):
    """Local HuggingFace DINOv2/Hibou adapter with a timm-like ViT token API."""

    def __init__(
        self,
        model_dir: str | Path,
        layer_indices: list[int] | None = None,
        fusion_mode: str = "weighted_sum",
        fusion_init: str = "average",
    ) -> None:
        super().__init__()
        if fusion_mode not in {"weighted_sum", "concat_proj"}:
            raise ValueError(f"Unknown fusion mode: {fusion_mode}")
        if fusion_init not in {"average", "final"}:
            raise ValueError(f"Unknown fusion init: {fusion_init}")
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Hibou/Phikon local folders require transformers. Install with: pip install transformers safetensors"
            ) from exc

        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"HF backbone folder not found: {model_dir}")
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        self.vit = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.num_features = int(config.hidden_size)
        image_size = int(getattr(config, "image_size", 224))
        patch_size = int(getattr(config, "patch_size", 14))
        grid = image_size // patch_size
        self.patch_grid_size = (grid, grid)
        self.num_patches = grid * grid
        self.num_register_tokens = int(getattr(config, "num_register_tokens", 0))
        self.layer_indices = layer_indices
        self.fusion_mode = fusion_mode
        self.fusion_init = fusion_init

        if self.layer_indices is not None:
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

    def _strip_register_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        cls = tokens[:, :1, :]
        if tokens.shape[1] == 1 + self.num_register_tokens + self.num_patches:
            patches = tokens[:, 1 + self.num_register_tokens :, :]
        elif tokens.shape[1] == 1 + self.num_patches:
            patches = tokens[:, 1:, :]
        else:
            raise ValueError(
                f"Unexpected HF DINOv2 token count: got {tokens.shape[1]}, "
                f"expected {1 + self.num_patches} or {1 + self.num_register_tokens + self.num_patches}"
            )
        return torch.cat([cls, patches], dim=1)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        if hasattr(self.vit, "gradient_checkpointing_enable") and enable:
            self.vit.gradient_checkpointing_enable()
        elif hasattr(self.vit, "gradient_checkpointing_disable") and not enable:
            self.vit.gradient_checkpointing_disable()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_indices is None:
            outputs = self.vit(pixel_values=x, return_dict=True)
            return self._strip_register_tokens(outputs.last_hidden_state)

        outputs = self.vit(pixel_values=x, output_hidden_states=True, return_dict=True)
        features = []
        for norm, layer_idx in zip(self.fusion_norms, self.layer_indices):
            hidden = outputs.hidden_states[layer_idx + 1]
            features.append(norm(self._strip_register_tokens(hidden)))
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
    patch_stride: int | None = None,
    patch_padding: int = 0,
) -> torch.nn.Module:
    if model_name in {"hibou_b", "hf_dinov2"}:
        if checkpoint_path is None:
            raise ValueError(f"{model_name} requires --checkpoint pointing to a local HF model folder.")
        if patch_stride is not None:
            raise ValueError("--patch-stride is only supported for timm ViT backbones, not HF DINOv2/Hibou.")
        layer_indices = parse_layer_indices(fusion_layers, 12)
        model = HFDinoV2Backbone(
            checkpoint_path,
            layer_indices=layer_indices,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
        )
        if grad_checkpointing:
            model.set_grad_checkpointing(True)
        return model

    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if checkpoint_path is not None:
        load_local_checkpoint(model, checkpoint_path)
    configure_patch_stride(model, patch_stride=patch_stride, patch_padding=patch_padding)
    use_custom_pos_embed = patch_stride is not None
    layer_indices = parse_layer_indices(fusion_layers, len(model.blocks))
    if layer_indices is not None:
        model = MultiLayerViTBackbone(
            model,
            layer_indices,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
            use_custom_pos_embed=use_custom_pos_embed,
        )
    elif use_custom_pos_embed:
        model = OverlapViTBackbone(model, use_custom_pos_embed=use_custom_pos_embed)
    if grad_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model
