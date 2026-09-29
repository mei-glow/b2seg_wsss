from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


def _iter_modules(root: torch.nn.Module) -> Iterable[torch.nn.Module]:
    yield root
    for module in root.modules():
        if module is not root:
            yield module


def _find_vit_with_patch_embed(model: torch.nn.Module) -> torch.nn.Module:
    for candidate_name in ("vit", "backbone", "encoder"):
        candidate = getattr(model, candidate_name, None)
        if isinstance(candidate, torch.nn.Module) and hasattr(candidate, "patch_embed"):
            return candidate
        nested = getattr(candidate, "vit", None)
        if isinstance(nested, torch.nn.Module) and hasattr(nested, "patch_embed"):
            return nested
    for module in _iter_modules(model):
        if hasattr(module, "patch_embed") and hasattr(getattr(module, "patch_embed"), "proj"):
            return module
    raise AttributeError("Could not find a ViT module with patch_embed.proj.")


def _as_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _resample_conv_weight(weight: torch.Tensor, kernel_size: tuple[int, int], scale_mode: str) -> torch.Tensor:
    old_h, old_w = int(weight.shape[-2]), int(weight.shape[-1])
    new_h, new_w = kernel_size
    if (old_h, old_w) == (new_h, new_w):
        return weight.detach().clone()
    out_channels, in_channels = int(weight.shape[0]), int(weight.shape[1])
    resized = F.interpolate(
        weight.detach().float().reshape(out_channels * in_channels, 1, old_h, old_w),
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    ).reshape(out_channels, in_channels, new_h, new_w)
    if scale_mode == "area":
        resized = resized * float(old_h * old_w) / float(new_h * new_w)
    elif scale_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown patch resample scale mode: {scale_mode}")
    return resized.to(dtype=weight.dtype, device=weight.device)


def _resize_pos_embed(vit: torch.nn.Module, old_num_patches: int, new_grid: tuple[int, int]) -> None:
    pos_embed = getattr(vit, "pos_embed", None)
    if pos_embed is None:
        return
    if pos_embed.ndim != 3:
        raise ValueError(f"Expected pos_embed with shape [1, tokens, dim], got {tuple(pos_embed.shape)}")
    total_tokens = int(pos_embed.shape[1])
    prefix_tokens = total_tokens - int(old_num_patches)
    if prefix_tokens < 0:
        raise ValueError(
            f"pos_embed has {total_tokens} tokens but old_num_patches={old_num_patches}; cannot infer prefix tokens."
        )
    old_grid_size = int(round(old_num_patches**0.5))
    if old_grid_size * old_grid_size != old_num_patches:
        raise ValueError(f"Expected square old patch grid, got {old_num_patches} patches.")
    prefix = pos_embed[:, :prefix_tokens]
    grid = pos_embed[:, prefix_tokens:]
    if grid.shape[1] != old_num_patches:
        raise ValueError(f"pos_embed grid token mismatch: {grid.shape[1]} vs {old_num_patches}")
    grid = grid.reshape(1, old_grid_size, old_grid_size, pos_embed.shape[-1]).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=new_grid, mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid[0] * new_grid[1], pos_embed.shape[-1])
    resized = torch.cat([prefix.float(), grid], dim=1).to(dtype=pos_embed.dtype, device=pos_embed.device)
    vit.pos_embed = torch.nn.Parameter(resized, requires_grad=pos_embed.requires_grad)


def adapt_vit_patch_embed(
    model: torch.nn.Module,
    patch_kernel: int | None,
    patch_stride: int | None,
    patch_padding: int = 0,
    image_size: int = 224,
    resample_scale: str = "area",
) -> dict[str, object]:
    """Adapt a timm ViT patch embed in-place and resize positional embeddings.

    This is intentionally script-side and duck-typed so checkpoints can be
    reconstructed even when the model package does not expose patch-size args.
    """
    if patch_kernel is None and patch_stride is None and int(patch_padding) == 0:
        return {"patch_adapted": False}

    vit = _find_vit_with_patch_embed(model)
    patch_embed = getattr(vit, "patch_embed")
    proj = getattr(patch_embed, "proj", None)
    if not isinstance(proj, torch.nn.Conv2d):
        raise TypeError("Expected vit.patch_embed.proj to be torch.nn.Conv2d.")

    old_num_patches = int(getattr(patch_embed, "num_patches", 0) or 0)
    if old_num_patches <= 0:
        old_grid = getattr(patch_embed, "grid_size", None)
        if old_grid is not None:
            old_grid_pair = _as_pair(old_grid)
            old_num_patches = old_grid_pair[0] * old_grid_pair[1]
        else:
            old_num_patches = int(round((image_size / max(1, proj.stride[0])) ** 2))

    kernel = _as_pair(patch_kernel if patch_kernel is not None else proj.kernel_size)
    stride = _as_pair(patch_stride if patch_stride is not None else proj.stride)
    padding = _as_pair(int(patch_padding))
    new_grid = (
        (int(image_size) + 2 * padding[0] - kernel[0]) // stride[0] + 1,
        (int(image_size) + 2 * padding[1] - kernel[1]) // stride[1] + 1,
    )
    if new_grid[0] <= 0 or new_grid[1] <= 0:
        raise ValueError(f"Invalid patch grid {new_grid}; check kernel/stride/padding.")

    new_proj = torch.nn.Conv2d(
        proj.in_channels,
        proj.out_channels,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=proj.bias is not None,
    ).to(device=proj.weight.device, dtype=proj.weight.dtype)
    with torch.no_grad():
        new_proj.weight.copy_(_resample_conv_weight(proj.weight, kernel, resample_scale))
        if proj.bias is not None and new_proj.bias is not None:
            new_proj.bias.copy_(proj.bias.detach())
    patch_embed.proj = new_proj

    _resize_pos_embed(vit, old_num_patches=old_num_patches, new_grid=new_grid)

    patch_embed.img_size = (int(image_size), int(image_size))
    patch_embed.patch_size = kernel
    patch_embed.grid_size = new_grid
    patch_embed.num_patches = int(new_grid[0] * new_grid[1])
    for target in (model, getattr(model, "backbone", None), vit):
        if target is not None and hasattr(target, "num_patches"):
            setattr(target, "num_patches", patch_embed.num_patches)

    return {
        "patch_adapted": True,
        "patch_kernel": list(kernel),
        "patch_stride": list(stride),
        "patch_padding": list(padding),
        "patch_grid": list(new_grid),
        "num_patches": int(patch_embed.num_patches),
        "old_num_patches": int(old_num_patches),
        "pos_prefix_tokens": int(getattr(vit, "pos_embed").shape[1] - patch_embed.num_patches)
        if hasattr(vit, "pos_embed")
        else None,
    }
