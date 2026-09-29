from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from b2seg_wsss.backbones import create_vit_backbone


def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    batch, num_patches, dim = tokens.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return tokens.transpose(1, 2).reshape(batch, dim, grid, grid)


def _map_to_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    return feature_map.flatten(2).transpose(1, 2).contiguous()


def _logits_to_map(logits: torch.Tensor) -> torch.Tensor:
    batch, num_patches, num_classes = logits.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return logits.transpose(1, 2).reshape(batch, num_classes, grid, grid)


class RefinementDecoder(nn.Module):
    """Lightweight trainable decoder from ViT patch tokens and coarse prototype logits."""

    def __init__(self, dim: int, num_classes: int, hidden_dim: int = 256, upsample_scale: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.upsample_scale = upsample_scale
        self.token_proj = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.logit_proj = nn.Conv2d(num_classes, hidden_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, patch_tokens: torch.Tensor, patch_logits: torch.Tensor) -> torch.Tensor:
        token_map = self.token_proj(_tokens_to_map(patch_tokens))
        logit_map = self.logit_proj(_logits_to_map(patch_logits))
        fused = self.fuse(torch.cat([token_map, logit_map], dim=1))
        if self.upsample_scale > 1:
            fused = F.interpolate(fused, scale_factor=self.upsample_scale, mode="bilinear", align_corners=False)
        return self.refine(fused)


class MultiScalePrototypeEvidence(nn.Module):
    """Create multiple token evidence scales before prototype matching."""

    def __init__(
        self,
        dim: int,
        num_classes: int,
        branches: str = "identity,local,coarse",
        init: str = "identity",
        residual_init: float = 0.05,
        residual_logit_init: float = -4.0,
    ) -> None:
        super().__init__()
        names = [item.strip() for item in branches.replace(";", ",").split(",") if item.strip()]
        if not names:
            raise ValueError("--prototype-scale-branches must contain at least one branch.")
        allowed = {"identity", "local", "coarse"}
        unknown = sorted(set(names) - allowed)
        if unknown:
            raise ValueError(f"Unknown prototype scale branches: {unknown}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate prototype scale branches: {names}")
        if init not in {"identity", "soft_identity", "uniform"}:
            raise ValueError(f"Unknown prototype scale init: {init}")

        self.names = names
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
        )
        self.coarse = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.norms = nn.ModuleDict({name: nn.LayerNorm(dim) for name in names})
        self.residual_gains = nn.ParameterDict()
        for name in names:
            if name != "identity":
                self.residual_gains[name] = nn.Parameter(torch.tensor(float(residual_init)))
        correction_names = [name for name in names if name != "identity"]
        self.correction_names = correction_names
        self.residual_logits = nn.ParameterDict(
            {
                name: nn.Parameter(torch.full((num_classes,), float(residual_logit_init)))
                for name in correction_names
            }
        )
        logits = torch.zeros(num_classes, len(names))
        if init == "identity" and "identity" in names:
            logits.fill_(-4.0)
            logits[:, names.index("identity")] = 4.0
        elif init == "soft_identity" and "identity" in names:
            logits[:, names.index("identity")] = 1.0
        self.scale_logits = nn.Parameter(logits)

    def forward(self, patch_tokens: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        token_map = _tokens_to_map(patch_tokens)
        branch_tokens = []
        for name in self.names:
            if name == "identity":
                tokens = patch_tokens
            elif name == "local":
                tokens = patch_tokens + self.residual_gains[name].to(patch_tokens.dtype) * _map_to_tokens(self.local(token_map))
            elif name == "coarse":
                tokens = patch_tokens + self.residual_gains[name].to(patch_tokens.dtype) * _map_to_tokens(self.coarse(token_map))
            else:
                raise RuntimeError(f"Unhandled branch: {name}")
            branch_tokens.append(self.norms[name](tokens))
        return branch_tokens, self.scale_logits.softmax(dim=-1)

    def residual_alphas(self) -> dict[str, torch.Tensor]:
        return {name: logits.sigmoid() for name, logits in self.residual_logits.items()}


def image_logits_from_map(
    logit_map: torch.Tensor,
    pooling: str = "max",
    topk_frac: float = 0.05,
    mix_alpha: float = 0.5,
) -> torch.Tensor:
    flat = logit_map.flatten(2)
    max_logits = flat.amax(dim=-1)
    if pooling == "max":
        return max_logits
    if pooling in {"topk", "mix_max_topk"}:
        k = max(1, int(round(int(flat.shape[-1]) * topk_frac)))
        topk_logits = flat.topk(k, dim=-1).values.mean(dim=-1)
        if pooling == "topk":
            return topk_logits
        alpha = max(0.0, min(1.0, float(mix_alpha)))
        return alpha * max_logits + (1.0 - alpha) * topk_logits
    raise ValueError(f"Unknown map pooling: {pooling}")


class PrototypeHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int = 4,
        prototypes_per_class: int = 4,
        prototype_counts: list[int] | None = None,
        prototype_gating: bool = False,
        gate_init: float = 2.0,
        prototype_dropout: float = 0.0,
        prototype_aggregation: str = "max",
        lse_tau: float = 1.0,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if prototype_counts is None:
            prototype_counts = [prototypes_per_class] * num_classes
        if len(prototype_counts) != num_classes:
            raise ValueError(f"Expected {num_classes} prototype counts, got {len(prototype_counts)}")
        if min(prototype_counts) < 1:
            raise ValueError("Each class must have at least one prototype.")
        self.prototype_counts = [int(count) for count in prototype_counts]
        self.prototypes_per_class = max(self.prototype_counts)
        self.prototype_gating = prototype_gating
        self.prototype_dropout = float(prototype_dropout)
        self.prototype_aggregation = prototype_aggregation
        self.lse_tau = lse_tau
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.prototypes_per_class, dim) * 0.02)
        if self.prototype_gating:
            self.gate_logits = nn.Parameter(torch.full((num_classes, self.prototypes_per_class), float(gate_init)))
        else:
            self.register_parameter("gate_logits", None)
        valid = torch.zeros(num_classes, self.prototypes_per_class, dtype=torch.bool)
        for class_idx, count in enumerate(self.prototype_counts):
            valid[class_idx, :count] = True
        self.register_buffer("prototype_valid", valid, persistent=False)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        sims = self.prototype_similarity(patch_tokens)
        return self.aggregate_prototypes(sims)

    def prototype_similarity(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        tokens = F.normalize(patch_tokens, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        sims = torch.einsum("bnd,kmd->bnkm", tokens, prototypes) / self.temperature
        return sims

    def mask_invalid_prototypes(self, sims: torch.Tensor) -> torch.Tensor:
        valid = self.prototype_valid.view(1, 1, self.num_classes, self.prototypes_per_class)
        sims = sims.masked_fill(~valid, -torch.inf)
        if self.prototype_gating:
            gate_probs = self.prototype_gate_probs()
            if self.training and self.prototype_dropout > 0.0:
                keep_prob = max(1e-6, 1.0 - self.prototype_dropout)
                keep = torch.rand_like(gate_probs) < keep_prob
                keep = keep | (~self.prototype_valid)
                gate_probs = gate_probs * keep.to(gate_probs.dtype)
            gate = gate_probs.clamp_min(1e-6).log().view(1, 1, self.num_classes, self.prototypes_per_class)
            sims = sims + gate
        return sims

    def aggregate_prototypes(self, sims: torch.Tensor) -> torch.Tensor:
        sims = self.mask_invalid_prototypes(sims)
        if self.prototype_aggregation == "max":
            return sims.max(dim=-1).values
        if self.prototype_aggregation == "logmeanexp":
            tau = max(float(self.lse_tau), 1e-6)
            valid_counts = self.prototype_valid.sum(dim=1).clamp_min(1).to(sims.dtype)
            normalizer = valid_counts.clamp_min(1.0).log().view(1, 1, self.num_classes)
            return tau * torch.logsumexp(sims / tau, dim=-1) - tau * normalizer
        raise ValueError(f"Unknown prototype aggregation: {self.prototype_aggregation}")

    def prototype_gate_probs(self) -> torch.Tensor:
        if not self.prototype_gating:
            return self.prototype_valid.to(self.prototypes.dtype)
        return self.gate_logits.sigmoid() * self.prototype_valid.to(self.gate_logits.dtype)

    def gate_loss(self) -> torch.Tensor:
        if not self.prototype_gating:
            return self.prototypes.sum() * 0.0
        valid = self.prototype_valid.to(self.gate_logits.dtype)
        return (self.gate_logits.sigmoid() * valid).sum() / valid.sum().clamp_min(1.0)

    def effective_prototypes(self, threshold: float = 0.5) -> torch.Tensor:
        probs = self.prototype_gate_probs()
        return (probs >= threshold).sum(dim=1)

    def diversity_loss(self) -> torch.Tensor:
        prototypes = F.normalize(self.prototypes, dim=-1)
        gram = torch.einsum("kmd,knd->kmn", prototypes, prototypes)
        valid = self.prototype_valid.to(gram.device)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(self.prototypes_per_class, device=gram.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = pair_valid & (~eye)
        if not pair_valid.any():
            return gram.sum() * 0.0
        if not self.prototype_gating:
            return gram[pair_valid].pow(2).mean()
        gates = self.prototype_gate_probs().to(gram.device)
        pair_weight = gates.unsqueeze(2) * gates.unsqueeze(1) * pair_valid.to(gates.dtype)
        denom = pair_weight.sum().clamp_min(1.0)
        return (gram.pow(2) * pair_weight).sum() / denom

    def spatial_diversity_loss(self, prototype_sims: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        activations = prototype_sims.softmax(dim=1)
        overlap = torch.einsum("bnkm,bnkp->bkmp", activations, activations)
        valid = self.prototype_valid.to(overlap.device)
        pair_valid = valid.unsqueeze(2) & valid.unsqueeze(1)
        eye = torch.eye(self.prototypes_per_class, device=overlap.device, dtype=torch.bool).unsqueeze(0)
        pair_valid = pair_valid & (~eye)
        if self.prototype_gating:
            gates = self.prototype_gate_probs().to(overlap.device)
            pair_weight = gates.unsqueeze(2) * gates.unsqueeze(1) * pair_valid.to(gates.dtype)
        else:
            pair_weight = pair_valid.to(overlap.dtype)
        masked_overlap = overlap * pair_weight.unsqueeze(0).to(overlap.dtype)
        denom = pair_weight.sum(dim=(1, 2)).clamp_min(1.0).to(overlap.dtype)
        per_class = masked_overlap.sum(dim=(2, 3)) / denom.unsqueeze(0)
        if labels is not None:
            weights = labels.to(per_class.dtype)
            denom = weights.sum().clamp_min(1.0)
            return (per_class * weights).sum() / denom
        return per_class.mean()

    def usage_loss(self, prototype_sims: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        sims = self.mask_invalid_prototypes(prototype_sims)
        probs = sims.softmax(dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        usage = probs.mean(dim=1).clamp_min(1e-8)
        valid = self.prototype_valid.to(usage.device)
        valid_counts = valid.sum(dim=1).clamp_min(1).to(usage.dtype)
        log_valid_counts = valid_counts.log().clamp_min(1e-6)
        kl_to_uniform = (usage * (usage.log() + valid_counts.log().view(1, self.num_classes, 1))).sum(dim=-1)
        kl_to_uniform = kl_to_uniform / log_valid_counts.view(1, self.num_classes)
        single_proto = valid_counts <= 1
        kl_to_uniform = kl_to_uniform.masked_fill(single_proto.view(1, self.num_classes), 0.0)
        if labels is not None:
            weights = labels.to(kl_to_uniform.dtype)
            denom = weights.sum().clamp_min(1.0)
            return (kl_to_uniform * weights).sum() / denom
        return kl_to_uniform.mean()


class PrototypeWSSSModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        num_classes: int = 4,
        prototypes_per_class: int = 4,
        prototype_counts: list[int] | None = None,
        prototype_gating: bool = False,
        gate_init: float = 2.0,
        prototype_dropout: float = 0.0,
        prototype_aggregation: str = "max",
        lse_tau: float = 1.0,
        refine_head: bool = False,
        refine_dim: int = 256,
        refine_scale: int = 2,
        refine_pooling: str = "topk",
        refine_topk_frac: float = 0.05,
        grad_checkpointing: bool = False,
        fusion_layers: str | None = None,
        fusion_mode: str = "weighted_sum",
        fusion_init: str = "average",
        dense_prototype_scale: int = 1,
        patch_stride: int | None = None,
        patch_padding: int = 0,
        prototype_pooling: str = "max",
        prototype_topk_frac: float = 0.05,
        prototype_mix_alpha: float = 0.5,
        prototype_multiscale: bool = False,
        prototype_scale_branches: str = "identity,local,coarse",
        prototype_scale_init: str = "identity",
        prototype_scale_residual_init: float = 0.05,
        prototype_scale_mode: str = "mixture",
        prototype_scale_alpha_init: float = 0.02,
    ) -> None:
        super().__init__()
        self.backbone = create_vit_backbone(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            grad_checkpointing=grad_checkpointing,
            fusion_layers=fusion_layers,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
        )
        dim = self.backbone.num_features
        self.head = PrototypeHead(
            dim=dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            prototype_counts=prototype_counts,
            prototype_gating=prototype_gating,
            gate_init=gate_init,
            prototype_dropout=prototype_dropout,
            prototype_aggregation=prototype_aggregation,
            lse_tau=lse_tau,
        )
        self.refine_pooling = refine_pooling
        self.refine_topk_frac = refine_topk_frac
        self.prototype_pooling = prototype_pooling
        self.prototype_topk_frac = prototype_topk_frac
        self.prototype_mix_alpha = prototype_mix_alpha
        self.prototype_multiscale = bool(prototype_multiscale)
        if prototype_scale_mode not in {"mixture", "residual"}:
            raise ValueError(f"Unknown prototype scale mode: {prototype_scale_mode}")
        self.prototype_scale_mode = prototype_scale_mode
        self.dense_prototype_scale = int(dense_prototype_scale)
        if self.dense_prototype_scale < 1:
            raise ValueError("--dense-prototype-scale must be >= 1.")
        self.prototype_scale_mixer = (
            MultiScalePrototypeEvidence(
                dim=dim,
                num_classes=num_classes,
                branches=prototype_scale_branches,
                init=prototype_scale_init,
                residual_init=prototype_scale_residual_init,
                residual_logit_init=torch.logit(torch.tensor(float(max(1e-4, min(1.0 - 1e-4, prototype_scale_alpha_init))))).item(),
            )
            if self.prototype_multiscale
            else None
        )
        self.refine_decoder = (
            RefinementDecoder(dim=dim, num_classes=num_classes, hidden_dim=refine_dim, upsample_scale=refine_scale)
            if refine_head
            else None
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.forward_features(images)
        patch_tokens = features[:, 1:, :]
        prototype_sims = self.head.prototype_similarity(patch_tokens)
        backbone_patch_logits = self.head.aggregate_prototypes(prototype_sims)
        if self.dense_prototype_scale > 1:
            dense_tokens = _map_to_tokens(
                F.interpolate(
                    _tokens_to_map(patch_tokens),
                    scale_factor=self.dense_prototype_scale,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            dense_prototype_sims = self.head.prototype_similarity(dense_tokens)
            patch_logits = self.head.aggregate_prototypes(dense_prototype_sims)
        else:
            dense_tokens = patch_tokens
            dense_prototype_sims = prototype_sims
            patch_logits = backbone_patch_logits
        scale_weights = None
        if self.prototype_scale_mixer is not None:
            branch_tokens, scale_weights = self.prototype_scale_mixer(dense_tokens)
            branch_logits = {}
            for name, tokens in zip(self.prototype_scale_mixer.names, branch_tokens):
                branch_sims = self.head.prototype_similarity(tokens)
                branch_logits[name] = self.head.aggregate_prototypes(branch_sims)
            if self.prototype_scale_mode == "residual" and "identity" in branch_logits:
                base_logits = branch_logits["identity"]
                patch_logits = base_logits
                residual_alphas = self.prototype_scale_mixer.residual_alphas()
                for name, alpha in residual_alphas.items():
                    if name in branch_logits:
                        patch_logits = patch_logits + alpha.view(1, 1, self.head.num_classes) * (branch_logits[name] - base_logits)
            else:
                stacked_logits = torch.stack([branch_logits[name] for name in self.prototype_scale_mixer.names], dim=-1)
                patch_logits = (stacked_logits * scale_weights.view(1, 1, self.head.num_classes, -1)).sum(dim=-1)
        prototype_image_logits = image_logits_from_map(
            _logits_to_map(patch_logits),
            pooling=self.prototype_pooling,
            topk_frac=self.prototype_topk_frac,
            mix_alpha=self.prototype_mix_alpha,
        )
        outputs = {
            "patch_tokens": patch_tokens,
            "dense_patch_tokens": dense_tokens,
            "prototype_sims": prototype_sims,
            "dense_prototype_sims": dense_prototype_sims,
            "backbone_patch_logits": backbone_patch_logits,
            "patch_logits": patch_logits,
            "prototype_image_logits": prototype_image_logits,
            "image_logits": prototype_image_logits,
        }
        if scale_weights is not None:
            outputs["prototype_scale_weights"] = scale_weights
            if self.prototype_scale_mode == "residual":
                outputs["prototype_scale_alphas"] = {
                    name: alpha.detach()
                    for name, alpha in self.prototype_scale_mixer.residual_alphas().items()
                }
        if self.refine_decoder is not None:
            refined_logits = self.refine_decoder(patch_tokens, backbone_patch_logits)
            refined_image_logits = image_logits_from_map(refined_logits, self.refine_pooling, self.refine_topk_frac)
            outputs["refined_logits"] = refined_logits
            outputs["refined_image_logits"] = refined_image_logits
            outputs["image_logits"] = refined_image_logits
        return outputs

    def diversity_loss(self) -> torch.Tensor:
        return self.head.diversity_loss()

    def spatial_diversity_loss(self, prototype_sims: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.head.spatial_diversity_loss(prototype_sims, labels)

    def gate_loss(self) -> torch.Tensor:
        return self.head.gate_loss()

    def usage_loss(self, prototype_sims: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.head.usage_loss(prototype_sims, labels)

    def effective_prototypes(self, threshold: float = 0.5) -> torch.Tensor:
        return self.head.effective_prototypes(threshold)


class LinearWSSSModel(nn.Module):
    """No-prototype WSSS ablation: backbone/fusion plus a linear patch classifier."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        num_classes: int = 4,
        grad_checkpointing: bool = False,
        fusion_layers: str | None = None,
        fusion_mode: str = "weighted_sum",
        fusion_init: str = "average",
        patch_stride: int | None = None,
        patch_padding: int = 0,
        pooling: str = "max",
        topk_frac: float = 0.05,
    ) -> None:
        super().__init__()
        self.backbone = create_vit_backbone(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            grad_checkpointing=grad_checkpointing,
            fusion_layers=fusion_layers,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
            patch_stride=patch_stride,
            patch_padding=patch_padding,
        )
        self.classifier = nn.Linear(self.backbone.num_features, num_classes)
        self.pooling = pooling
        self.topk_frac = topk_frac

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.forward_features(images)
        patch_tokens = features[:, 1:, :]
        patch_logits = self.classifier(patch_tokens)
        image_logits = image_logits_from_map(
            _logits_to_map(patch_logits),
            pooling=self.pooling,
            topk_frac=self.topk_frac,
        )
        return {
            "patch_tokens": patch_tokens,
            "patch_logits": patch_logits,
            "image_logits": image_logits,
        }

    def refine_consistency_loss(self, refined_logits: torch.Tensor, patch_logits: torch.Tensor) -> torch.Tensor:
        coarse = _logits_to_map(patch_logits.detach())
        refined_down = F.interpolate(refined_logits, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(refined_down, coarse)


def token_image_logits(
    patch_logits: torch.Tensor,
    pooling: str = "max",
    topk_frac: float = 0.05,
    lse_tau: float = 1.0,
) -> torch.Tensor:
    if pooling == "max":
        return patch_logits.amax(dim=1)
    if pooling == "topk":
        # Materialize the token count as a Python integer. This is identical for
        # normal execution and also lets FLOP/JIT tracers treat k as a static
        # architectural constant instead of passing a Tensor to round().
        num_tokens = int(patch_logits.shape[1])
        k = max(1, int(round(num_tokens * float(topk_frac))))
        return patch_logits.topk(k, dim=1).values.mean(dim=1)
    if pooling == "lse":
        # Smooth max-MIL: every token receives gradient weighted by softmax(logit / tau).
        # tau -> 0 recovers `max`, tau -> inf recovers `mean`.
        tau = max(float(lse_tau), 1e-4)
        num_tokens = patch_logits.shape[1]
        return tau * (torch.logsumexp(patch_logits.float() / tau, dim=1) - math.log(num_tokens)).to(patch_logits.dtype)
    raise ValueError(f"Unknown token pooling: {pooling}")


class DualRouteLinearWSSSModel(nn.Module):
    """Adaptive all-layer semantic/spatial routing from a raw ViT backbone.

    Variants:
    - single_semantic: one parameter router trained with max-MIL.
    - single_spatial: one parameter router trained/evaluated with top-k-MIL.
    - dual_param: separate parameter routers and heads for semantic/spatial routes.
    - dual_quality: parameter semantic router plus image-adaptive spatial quality router.
    - dual_shift: parameter semantic router plus learnable earlier spatial shift.
    """

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        num_classes: int = 4,
        route_layers: str = "all",
        variant: str = "dual_quality",
        topk_frac: float = 0.05,
        semantic_pooling: str = "max",
        lse_tau: float = 1.0,
        spatial_weight: float = 0.5,
        quality_hidden_dim: int = 32,
        semantic_init: str = "final",
        spatial_init: str = "uniform",
        output_mode: str = "spatial",
        output_fuse_alpha: float = 0.5,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if variant not in {"single_semantic", "single_spatial", "dual_param", "dual_quality", "dual_shift"}:
            raise ValueError(f"Unknown dual route variant: {variant}")
        if output_mode not in {"spatial", "semantic", "fuse", "learned_fuse"}:
            raise ValueError(f"Unknown dual route output mode: {output_mode}")
        if semantic_pooling not in {"max", "lse", "topk"}:
            raise ValueError(f"Unknown semantic pooling: {semantic_pooling}")
        if model_name in {"hibou_b", "hf_dinov2"}:
            raise ValueError("DualRouteLinearWSSSModel currently supports timm DeiT/ViT backbones only.")
        import timm
        from b2seg_wsss.backbones import load_local_checkpoint, parse_layer_indices

        vit = timm.create_model(model_name, pretrained=False, num_classes=0)
        if checkpoint_path is not None:
            load_local_checkpoint(vit, checkpoint_path)
        if grad_checkpointing and hasattr(vit, "set_grad_checkpointing"):
            vit.set_grad_checkpointing(True)
        self.vit = vit
        self.num_classes = int(num_classes)
        self.num_features = vit.num_features
        self.num_patches = vit.patch_embed.num_patches
        self.variant = variant
        self.topk_frac = float(topk_frac)
        self.semantic_pooling = semantic_pooling
        self.lse_tau = float(lse_tau)
        self.spatial_weight = float(spatial_weight)
        self.output_mode = output_mode
        self.output_fuse_alpha = float(output_fuse_alpha)
        if output_mode == "learned_fuse":
            init_alpha = max(1e-4, min(1.0 - 1e-4, float(output_fuse_alpha)))
            self.output_fuse_alpha_logits = nn.Parameter(
                torch.full((self.num_classes,), torch.logit(torch.tensor(init_alpha)).item())
            )
        else:
            self.register_parameter("output_fuse_alpha_logits", None)
        self.route_layer_indices = (
            list(range(len(vit.blocks)))
            if route_layers == "all"
            else parse_layer_indices(route_layers, len(vit.blocks))
        )
        if self.route_layer_indices is None or not self.route_layer_indices:
            raise ValueError("--route-layers must resolve to at least one layer.")
        self.route_layers = [idx + 1 for idx in self.route_layer_indices]
        num_layers = len(self.route_layer_indices)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.num_features) for _ in range(num_layers)])

        if variant in {"single_semantic", "single_spatial"}:
            self.shared_classifier = nn.Linear(self.num_features, self.num_classes)
            self.semantic_classifier = None
            self.spatial_classifier = None
        else:
            self.shared_classifier = None
            self.semantic_classifier = nn.Linear(self.num_features, self.num_classes)
            self.spatial_classifier = nn.Linear(self.num_features, self.num_classes)

        self.semantic_route_logits = nn.Parameter(self._init_route_logits(num_classes, num_layers, semantic_init))
        if variant in {"single_semantic", "single_spatial", "dual_param"}:
            self.spatial_route_logits = nn.Parameter(self._init_route_logits(num_classes, num_layers, spatial_init))
        else:
            self.register_parameter("spatial_route_logits", None)
            if variant == "dual_quality":
                self.spatial_quality_bias = nn.Parameter(self._init_route_logits(num_classes, num_layers, spatial_init))
                self.quality_router = nn.Sequential(
                    nn.Linear(4, quality_hidden_dim),
                    nn.GELU(),
                    nn.Linear(quality_hidden_dim, 1),
                )
            elif variant == "dual_shift":
                self.spatial_shift_raw = nn.Parameter(torch.full((num_classes,), 0.54))
                self.spatial_width_raw = nn.Parameter(torch.full((num_classes,), 0.85))

    @staticmethod
    def _init_route_logits(num_classes: int, num_layers: int, init: str) -> torch.Tensor:
        logits = torch.zeros(num_classes, num_layers)
        if init == "uniform":
            return logits
        if init == "final":
            logits.fill_(-2.0)
            logits[:, -1] = 2.0
            return logits
        if init == "middle":
            center = (num_layers - 1) / 2.0
            for idx in range(num_layers):
                logits[:, idx] = -abs(idx - center)
            return logits
        raise ValueError(f"Unknown route init: {init}")

    def _collect_layer_tokens(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vit.patch_embed(images)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)
        wanted = set(self.route_layer_indices)
        tokens = []
        norm_idx = 0
        use_checkpoint = bool(getattr(self.vit, "grad_checkpointing", False)) and self.training
        for block_idx, block in enumerate(self.vit.blocks):
            if use_checkpoint:
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            if block_idx in wanted:
                tokens.append(self.layer_norms[norm_idx](x)[:, 1:, :])
                norm_idx += 1
        if len(tokens) != len(self.route_layer_indices):
            raise RuntimeError("Did not collect all requested route layers.")
        return torch.stack(tokens, dim=1)

    @staticmethod
    def _route_logits(per_layer_logits: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        # per_layer_logits: [B, L, N, C]. route_weights: [C, L] or [B, C, L].
        if route_weights.dim() == 2:
            return torch.einsum("blnc,cl->bnc", per_layer_logits, route_weights)
        return torch.einsum("blnc,bcl->bnc", per_layer_logits, route_weights)

    def _quality_spatial_weights(self, per_layer_logits: torch.Tensor) -> torch.Tensor:
        # Use detached shape/confidence statistics to route; classifier still learns from BCE.
        logits = per_layer_logits.detach()
        probs = logits.sigmoid()
        topk = max(1, int(round(int(logits.shape[2]) * self.topk_frac)))
        topk_mean = logits.topk(topk, dim=2).values.mean(dim=2)
        mean = logits.mean(dim=2)
        std = logits.std(dim=2)
        area = (probs > 0.5).to(logits.dtype).mean(dim=2)
        stats = torch.stack([topk_mean, mean, std, area], dim=-1)  # [B, L, C, 4]
        quality = self.quality_router(stats).squeeze(-1).permute(0, 2, 1)  # [B, C, L]
        return (quality + self.spatial_quality_bias.unsqueeze(0)).softmax(dim=-1)

    def _shift_spatial_weights(self, semantic_weights: torch.Tensor) -> torch.Tensor:
        layers = torch.tensor(self.route_layers, dtype=semantic_weights.dtype, device=semantic_weights.device)
        sem_expected = (semantic_weights * layers.view(1, -1)).sum(dim=-1)
        shift = F.softplus(self.spatial_shift_raw).to(semantic_weights.dtype) + 0.25
        width = F.softplus(self.spatial_width_raw).to(semantic_weights.dtype) + 0.75
        center = (sem_expected - shift).clamp(min=float(layers.min()), max=float(layers.max()))
        logits = -0.5 * ((layers.view(1, -1) - center.view(-1, 1)) / width.view(-1, 1)).pow(2)
        return logits.softmax(dim=-1)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        layer_tokens = self._collect_layer_tokens(images)
        if self.variant in {"single_semantic", "single_spatial"}:
            per_layer_logits = self.shared_classifier(layer_tokens)
            route = self.semantic_route_logits.softmax(dim=-1)
            patch_logits = self._route_logits(per_layer_logits, route)
            pooling = self.semantic_pooling if self.variant == "single_semantic" else "topk"
            image_logits = token_image_logits(
                patch_logits,
                pooling=pooling,
                topk_frac=self.topk_frac,
                lse_tau=self.lse_tau,
            )
            return {
                "layer_tokens": layer_tokens,
                "per_layer_logits": per_layer_logits,
                "patch_logits": patch_logits,
                "image_logits": image_logits,
                "semantic_image_logits": image_logits,
                "spatial_image_logits": image_logits,
                "semantic_route_weights": route,
                "spatial_route_weights": route,
            }

        sem_per_layer = self.semantic_classifier(layer_tokens)
        spa_per_layer = self.spatial_classifier(layer_tokens)
        semantic_weights = self.semantic_route_logits.softmax(dim=-1)
        if self.variant == "dual_param":
            spatial_weights = self.spatial_route_logits.softmax(dim=-1)
        elif self.variant == "dual_quality":
            spatial_weights = self._quality_spatial_weights(spa_per_layer)
        else:
            spatial_weights = self._shift_spatial_weights(semantic_weights)
        semantic_logits = self._route_logits(sem_per_layer, semantic_weights)
        spatial_logits = self._route_logits(spa_per_layer, spatial_weights)
        semantic_image_logits = token_image_logits(
            semantic_logits,
            pooling=self.semantic_pooling,
            topk_frac=self.topk_frac,
            lse_tau=self.lse_tau,
        )
        spatial_image_logits = token_image_logits(spatial_logits, pooling="topk", topk_frac=self.topk_frac)
        if self.output_mode == "semantic":
            patch_logits = semantic_logits
        elif self.output_mode == "fuse":
            alpha = max(0.0, min(1.0, self.output_fuse_alpha))
            patch_logits = (1.0 - alpha) * semantic_logits + alpha * spatial_logits
        elif self.output_mode == "learned_fuse":
            alpha = self.output_fuse_alpha_logits.sigmoid().to(semantic_logits.dtype)
            patch_logits = (1.0 - alpha.view(1, 1, -1)) * semantic_logits + alpha.view(1, 1, -1) * spatial_logits
        else:
            patch_logits = spatial_logits
        outputs = {
            "layer_tokens": layer_tokens,
            "semantic_per_layer_logits": sem_per_layer,
            "spatial_per_layer_logits": spa_per_layer,
            "semantic_patch_logits": semantic_logits,
            "spatial_patch_logits": spatial_logits,
            "patch_logits": patch_logits,
            "image_logits": semantic_image_logits,
            "semantic_image_logits": semantic_image_logits,
            "spatial_image_logits": spatial_image_logits,
            "semantic_route_weights": semantic_weights,
            "spatial_route_weights": spatial_weights,
        }
        if self.output_mode == "learned_fuse":
            outputs["output_fuse_alpha"] = self.output_fuse_alpha_logits.sigmoid()
        return outputs
