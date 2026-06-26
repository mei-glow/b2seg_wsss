from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jepa_wsss.backbones import create_vit_backbone


def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    batch, num_patches, dim = tokens.shape
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square patch grid, got {num_patches} tokens")
    return tokens.transpose(1, 2).reshape(batch, dim, grid, grid)


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


def image_logits_from_map(logit_map: torch.Tensor, pooling: str = "max", topk_frac: float = 0.05) -> torch.Tensor:
    flat = logit_map.flatten(2)
    if pooling == "max":
        return flat.amax(dim=-1)
    if pooling == "topk":
        k = max(1, int(round(flat.shape[-1] * topk_frac)))
        return flat.topk(k, dim=-1).values.mean(dim=-1)
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
            gate = self.prototype_gate_probs().clamp_min(1e-6).log().view(1, 1, self.num_classes, self.prototypes_per_class)
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
    ) -> None:
        super().__init__()
        self.backbone = create_vit_backbone(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            grad_checkpointing=grad_checkpointing,
            fusion_layers=fusion_layers,
            fusion_mode=fusion_mode,
            fusion_init=fusion_init,
        )
        dim = self.backbone.num_features
        self.head = PrototypeHead(
            dim=dim,
            num_classes=num_classes,
            prototypes_per_class=prototypes_per_class,
            prototype_counts=prototype_counts,
            prototype_gating=prototype_gating,
            gate_init=gate_init,
            prototype_aggregation=prototype_aggregation,
            lse_tau=lse_tau,
        )
        self.refine_pooling = refine_pooling
        self.refine_topk_frac = refine_topk_frac
        self.refine_decoder = (
            RefinementDecoder(dim=dim, num_classes=num_classes, hidden_dim=refine_dim, upsample_scale=refine_scale)
            if refine_head
            else None
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone.forward_features(images)
        patch_tokens = features[:, 1:, :]
        prototype_sims = self.head.prototype_similarity(patch_tokens)
        patch_logits = self.head.aggregate_prototypes(prototype_sims)
        prototype_image_logits = patch_logits.amax(dim=1)
        outputs = {
            "patch_tokens": patch_tokens,
            "prototype_sims": prototype_sims,
            "patch_logits": patch_logits,
            "prototype_image_logits": prototype_image_logits,
            "image_logits": prototype_image_logits,
        }
        if self.refine_decoder is not None:
            refined_logits = self.refine_decoder(patch_tokens, patch_logits)
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

    def refine_consistency_loss(self, refined_logits: torch.Tensor, patch_logits: torch.Tensor) -> torch.Tensor:
        coarse = _logits_to_map(patch_logits.detach())
        refined_down = F.interpolate(refined_logits, size=coarse.shape[-2:], mode="bilinear", align_corners=False)
        return F.mse_loss(refined_down, coarse)
