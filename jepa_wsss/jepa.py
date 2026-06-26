from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAPredictor(nn.Module):
    """Context-to-target latent predictor for 14x14 ViT patch tokens."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 384,
        num_patches: int = 196,
        predictor_type: str = "mean_mlp",
        num_heads: int = 6,
        depth: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.predictor_type = predictor_type
        self.context_proj = nn.Linear(dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, hidden_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, dim),
        )
        self.decoder_layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_out = nn.Linear(hidden_dim, dim)
        if predictor_type not in {"mean_mlp", "cross_attn"}:
            raise ValueError(f"Unknown JEPA predictor type: {predictor_type}")

    def forward(self, patch_tokens: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        batch, num_patches, _ = patch_tokens.shape
        hidden = self.context_proj(patch_tokens)
        mask = torch.zeros(batch, num_patches, dtype=torch.bool, device=patch_tokens.device)
        mask.scatter_(1, mask_indices, True)
        if self.predictor_type == "cross_attn":
            hidden = hidden + self.pos_embed[:num_patches].unsqueeze(0).to(hidden.dtype)
            query_pos = self.pos_embed[mask_indices].to(hidden.dtype)
            query = self.mask_token.to(hidden.dtype).expand(batch, mask_indices.shape[1], -1) + query_pos
            for layer in self.decoder_layers:
                query = layer(query, hidden, memory_key_padding_mask=mask)
            return self.cross_out(self.cross_norm(query))

        visible = hidden.masked_fill(mask.unsqueeze(-1), 0.0)
        counts = (~mask).sum(dim=1).clamp_min(1).to(hidden.dtype).unsqueeze(-1)
        context = visible.sum(dim=1) / counts
        pos = self.pos_embed[mask_indices]
        context = context.unsqueeze(1).expand_as(pos)
        return self.mlp(torch.cat([context, pos], dim=-1))


def make_ema_backbone(backbone: nn.Module) -> nn.Module:
    ema = copy.deepcopy(backbone)
    ema.eval()
    for param in ema.parameters():
        param.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema_backbone(ema_backbone: nn.Module, backbone: nn.Module, momentum: float) -> None:
    for ema_param, param in zip(ema_backbone.parameters(), backbone.parameters()):
        ema_param.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)
    for ema_buffer, buffer in zip(ema_backbone.buffers(), backbone.buffers()):
        ema_buffer.copy_(buffer)


def entropy_mask_indices(patch_logits: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    probs = patch_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
    num_patches = patch_logits.shape[1]
    num_mask = max(1, min(num_patches - 1, int(round(num_patches * mask_ratio))))
    return entropy.topk(num_mask, dim=1).indices


def balanced_entropy_mask_indices(
    patch_logits: torch.Tensor,
    image_labels: torch.Tensor,
    mask_ratio: float,
    minority_classes: tuple[int, ...] = (2, 3),
    minority_boost: float = 2.0,
) -> torch.Tensor:
    probs = patch_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
    pred_class = probs.argmax(dim=-1)
    batch, num_patches, num_classes = patch_logits.shape
    num_mask = max(1, min(num_patches - 1, int(round(num_patches * mask_ratio))))
    selections = []

    for bidx in range(batch):
        present = torch.where(image_labels[bidx] > 0)[0].tolist()
        if not present:
            selections.append(entropy[bidx].topk(num_mask).indices)
            continue

        weights = torch.ones(len(present), device=patch_logits.device)
        for pidx, class_id in enumerate(present):
            if class_id in minority_classes:
                weights[pidx] *= minority_boost
        quotas = torch.floor(weights / weights.sum() * num_mask).long()
        quotas = torch.clamp(quotas, min=1)
        while int(quotas.sum()) > num_mask:
            max_pos = int(torch.argmax(quotas).item())
            if quotas[max_pos] > 1:
                quotas[max_pos] -= 1
            else:
                break
        while int(quotas.sum()) < num_mask:
            max_pos = int(torch.argmax(weights).item())
            quotas[max_pos] += 1

        chosen_parts = []
        chosen_mask = torch.zeros(num_patches, dtype=torch.bool, device=patch_logits.device)
        for class_id, quota in zip(present, quotas.tolist()):
            candidates = torch.where(pred_class[bidx] == class_id)[0]
            candidates = candidates[~chosen_mask[candidates]]
            if candidates.numel() == 0:
                continue
            take = min(quota, candidates.numel())
            order = entropy[bidx, candidates].topk(take).indices
            selected = candidates[order]
            chosen_parts.append(selected)
            chosen_mask[selected] = True

        if chosen_parts:
            chosen = torch.cat(chosen_parts)
        else:
            chosen = torch.empty(0, dtype=torch.long, device=patch_logits.device)

        remaining = num_mask - chosen.numel()
        if remaining > 0:
            entropy_fill = entropy[bidx].masked_fill(chosen_mask, -torch.inf)
            fill = entropy_fill.topk(remaining).indices
            chosen = torch.cat([chosen, fill])
        elif chosen.numel() > num_mask:
            order = entropy[bidx, chosen].topk(num_mask).indices
            chosen = chosen[order]

        selections.append(chosen)

    return torch.stack(selections, dim=0)


def class_weighted_entropy_mask_indices(
    patch_logits: torch.Tensor,
    image_labels: torch.Tensor,
    mask_ratio: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    probs = patch_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
    pred_class = probs.argmax(dim=-1)
    batch, num_patches, _ = patch_logits.shape
    num_mask = max(1, min(num_patches - 1, int(round(num_patches * mask_ratio))))
    class_weights = class_weights.to(patch_logits.device)
    selections = []

    for bidx in range(batch):
        present = torch.where(image_labels[bidx] > 0)[0]
        if present.numel() == 0:
            selections.append(entropy[bidx].topk(num_mask).indices)
            continue

        weights = class_weights[present].clamp_min(1e-6)
        quotas = torch.floor(weights / weights.sum() * num_mask).long().clamp_min(1)
        while int(quotas.sum()) > num_mask:
            max_pos = int(torch.argmax(quotas).item())
            if quotas[max_pos] > 1:
                quotas[max_pos] -= 1
            else:
                break
        while int(quotas.sum()) < num_mask:
            max_pos = int(torch.argmax(weights).item())
            quotas[max_pos] += 1

        chosen_parts = []
        chosen_mask = torch.zeros(num_patches, dtype=torch.bool, device=patch_logits.device)
        for class_id, quota in zip(present.tolist(), quotas.tolist()):
            candidates = torch.where(pred_class[bidx] == class_id)[0]
            candidates = candidates[~chosen_mask[candidates]]
            if candidates.numel() == 0:
                continue
            take = min(quota, candidates.numel())
            order = entropy[bidx, candidates].topk(take).indices
            selected = candidates[order]
            chosen_parts.append(selected)
            chosen_mask[selected] = True

        if chosen_parts:
            chosen = torch.cat(chosen_parts)
        else:
            chosen = torch.empty(0, dtype=torch.long, device=patch_logits.device)

        remaining = num_mask - chosen.numel()
        if remaining > 0:
            entropy_fill = entropy[bidx].masked_fill(chosen_mask, -torch.inf)
            fill = entropy_fill.topk(remaining).indices
            chosen = torch.cat([chosen, fill])
        elif chosen.numel() > num_mask:
            order = entropy[bidx, chosen].topk(num_mask).indices
            chosen = chosen[order]

        selections.append(chosen)

    return torch.stack(selections, dim=0)


def frequency_uncertainty_mask_indices(
    patch_logits: torch.Tensor,
    image_labels: torch.Tensor,
    mask_ratio: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    probs = patch_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
    pred_class = probs.argmax(dim=-1)
    batch, num_patches, _ = patch_logits.shape
    num_mask = max(1, min(num_patches - 1, int(round(num_patches * mask_ratio))))

    class_weights = class_weights.to(patch_logits.device)
    pred_weight = class_weights[pred_class]
    present = image_labels.gather(dim=1, index=pred_class).to(patch_logits.dtype)
    score = entropy * pred_weight * present

    empty = score.sum(dim=1) <= 0
    if empty.any():
        score = score.clone()
        score[empty] = entropy[empty] * pred_weight[empty]

    return score.topk(num_mask, dim=1).indices


def soft_frequency_uncertainty_mask_indices(
    patch_logits: torch.Tensor,
    image_labels: torch.Tensor,
    mask_ratio: float,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    probs = patch_logits.softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-6).log()).sum(dim=-1)
    num_patches = patch_logits.shape[1]
    num_mask = max(1, min(num_patches - 1, int(round(num_patches * mask_ratio))))

    class_weights = class_weights.to(patch_logits.device)
    label_mask = image_labels.to(probs.dtype)
    weighted_labels = label_mask * class_weights.unsqueeze(0)
    expected_weight = (probs * weighted_labels.unsqueeze(1)).sum(dim=-1)
    normalizer = (probs * label_mask.unsqueeze(1)).sum(dim=-1).clamp_min(1e-6)
    expected_weight = expected_weight / normalizer
    score = entropy * expected_weight

    empty = label_mask.sum(dim=1) <= 0
    if empty.any():
        fallback_weight = (probs * class_weights.view(1, 1, -1)).sum(dim=-1)
        score = score.clone()
        score[empty] = entropy[empty] * fallback_weight[empty]

    return score.topk(num_mask, dim=1).indices


def gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    expand = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
    return tokens.gather(dim=1, index=expand)


def jepa_smooth_l1_loss(pred_tokens: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred_tokens, target_tokens.detach())


def class_weighted_jepa_smooth_l1_loss(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    patch_logits: torch.Tensor,
    mask_indices: torch.Tensor,
    image_labels: torch.Tensor,
    class_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_logits = gather_tokens(patch_logits.detach(), mask_indices)
    probs = masked_logits.softmax(dim=-1)
    labels = image_labels.to(probs.dtype).unsqueeze(1)
    class_weights = class_weights.to(probs.device, probs.dtype).view(1, 1, -1)

    present_mass = (probs * labels).sum(dim=-1).clamp_min(1e-6)
    token_weights = (probs * labels * class_weights).sum(dim=-1) / present_mass
    fallback_weights = (probs * class_weights).sum(dim=-1)
    has_present = image_labels.sum(dim=1, keepdim=True) > 0
    token_weights = torch.where(has_present, token_weights, fallback_weights)
    raw_weight_mean = token_weights.mean().detach()
    token_weights = token_weights / raw_weight_mean.clamp_min(1e-6)

    per_token_loss = F.smooth_l1_loss(pred_tokens, target_tokens.detach(), reduction="none").mean(dim=-1)
    loss = (per_token_loss * token_weights).mean()
    return loss, raw_weight_mean


def jepa_prototype_affinity_loss(
    student_prototype_sims: torch.Tensor,
    teacher_prototype_sims: torch.Tensor,
    mask_indices: torch.Tensor,
    image_labels: torch.Tensor,
    prototype_valid: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    teacher_temp: float = 1.0,
    student_temp: float = 1.0,
    confidence_threshold: float = 0.0,
    class_confidence_threshold: float = 0.0,
    agreement_threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, num_classes, prototypes_per_class = student_prototype_sims.shape
    gather_idx = mask_indices.view(batch, -1, 1, 1).expand(-1, -1, num_classes, prototypes_per_class)
    student = student_prototype_sims.gather(dim=1, index=gather_idx)
    teacher = teacher_prototype_sims.detach()

    valid = prototype_valid.to(student.device).view(1, 1, num_classes, prototypes_per_class)
    label_valid = image_labels.to(torch.bool).view(batch, 1, num_classes, 1)
    present_valid = valid & label_valid
    has_present = present_valid.flatten(2).any(dim=-1, keepdim=True).view(batch, 1, 1, 1)
    valid = torch.where(has_present, present_valid, valid.expand(batch, mask_indices.shape[1], -1, -1))

    teacher_temp = max(float(teacher_temp), 1e-6)
    student_temp = max(float(student_temp), 1e-6)
    teacher_logits = (teacher / teacher_temp).masked_fill(~valid, -torch.inf)
    student_logits = (student / student_temp).masked_fill(~valid, -torch.inf)
    target = teacher_logits.flatten(2).softmax(dim=-1)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    log_student = student_logits.flatten(2).log_softmax(dim=-1)
    log_student = torch.nan_to_num(log_student, nan=0.0, posinf=0.0, neginf=0.0)

    confidence = target.max(dim=-1).values
    gate = confidence >= float(confidence_threshold)
    if class_confidence_threshold > 0.0 or agreement_threshold > 0.0:
        target_proto = target.view(batch, mask_indices.shape[1], num_classes, prototypes_per_class)
        teacher_class_probs = target_proto.sum(dim=-1)
        teacher_class_conf, teacher_class = teacher_class_probs.max(dim=-1)

        student_probs = student_logits.flatten(2).detach().softmax(dim=-1)
        student_probs = torch.nan_to_num(student_probs, nan=0.0, posinf=0.0, neginf=0.0)
        student_class_probs = student_probs.view(batch, mask_indices.shape[1], num_classes, prototypes_per_class).sum(dim=-1)
        agreement = student_class_probs.gather(dim=-1, index=teacher_class.unsqueeze(-1)).squeeze(-1)
        gate = gate & (teacher_class_conf >= float(class_confidence_threshold)) & (agreement >= float(agreement_threshold))
    if not gate.any():
        return student.sum() * 0.0, gate.float().mean(), torch.ones((), device=student.device, dtype=student.dtype)

    per_token_loss = (target * (target.clamp_min(1e-8).log() - log_student)).sum(dim=-1)
    token_weights = torch.ones_like(per_token_loss)
    if class_weights is not None:
        class_weights = class_weights.to(student.device, student.dtype).view(1, 1, num_classes)
        class_probs = target.view(batch, mask_indices.shape[1], num_classes, prototypes_per_class).sum(dim=-1)
        token_weights = (class_probs * class_weights).sum(dim=-1)
        token_weights = token_weights / token_weights[gate].mean().detach().clamp_min(1e-6)

    loss = (per_token_loss[gate] * token_weights[gate]).mean()
    return loss, gate.float().mean(), token_weights[gate].mean().detach()


def jepa_semantic_affinity_consistency_loss(
    patch_tokens: torch.Tensor,
    pred_tokens: torch.Tensor,
    patch_logits: torch.Tensor,
    mask_indices: torch.Tensor,
    image_labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    affinity_temp: float = 0.2,
    visible_confidence_threshold: float = 0.6,
    target_confidence_threshold: float = 0.5,
    topk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, num_patches, num_classes = patch_logits.shape
    num_mask = mask_indices.shape[1]
    affinity_temp = max(float(affinity_temp), 1e-6)
    topk = max(1, min(int(topk), num_patches - num_mask))

    token_is_masked = torch.zeros(batch, num_patches, dtype=torch.bool, device=patch_logits.device)
    token_is_masked.scatter_(1, mask_indices, True)
    visible = ~token_is_masked

    probs = patch_logits.detach().softmax(dim=-1)
    label_mask = image_labels.to(probs.dtype).view(batch, 1, num_classes)
    present_probs = probs * label_mask
    present_mass = present_probs.sum(dim=-1, keepdim=True)
    probs = torch.where(present_mass > 0, present_probs / present_mass.clamp_min(1e-6), probs)
    visible_conf = probs.max(dim=-1).values
    visible = visible & (visible_conf >= float(visible_confidence_threshold))

    keys = F.normalize(patch_tokens.detach(), dim=-1)
    queries = F.normalize(pred_tokens, dim=-1)
    affinity_logits = torch.einsum("bmd,bnd->bmn", queries, keys) / affinity_temp
    affinity_logits = affinity_logits.masked_fill(~visible.unsqueeze(1), -torch.inf)

    finite = torch.isfinite(affinity_logits).any(dim=-1)
    safe_logits = torch.where(finite.unsqueeze(-1), affinity_logits, torch.zeros_like(affinity_logits))
    k = min(topk, safe_logits.shape[-1])
    top_values, top_indices = safe_logits.topk(k, dim=-1)
    top_values = top_values.masked_fill(~finite.unsqueeze(-1), -torch.inf)
    affinity = top_values.softmax(dim=-1)
    affinity = torch.nan_to_num(affinity, nan=0.0, posinf=0.0, neginf=0.0)

    gather_probs = probs.gather(1, top_indices.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, num_classes))
    gather_probs = gather_probs.reshape(batch, num_mask, k, num_classes)
    target = (affinity.unsqueeze(-1) * gather_probs).sum(dim=2)
    target_sum = target.sum(dim=-1, keepdim=True)
    target = target / target_sum.clamp_min(1e-6)
    target_conf = target.max(dim=-1).values
    gate = finite & (target_sum.squeeze(-1) > 0) & (target_conf >= float(target_confidence_threshold))

    if not gate.any():
        return patch_logits.sum() * 0.0, gate.float().mean(), torch.ones((), device=patch_logits.device, dtype=patch_logits.dtype)

    masked_logits = gather_tokens(patch_logits, mask_indices)
    log_student = masked_logits.log_softmax(dim=-1)
    per_token_loss = (target * (target.clamp_min(1e-8).log() - log_student)).sum(dim=-1)

    token_weights = torch.ones_like(per_token_loss)
    if class_weights is not None:
        class_weights = class_weights.to(patch_logits.device, patch_logits.dtype).view(1, 1, num_classes)
        token_weights = (target * class_weights).sum(dim=-1)
        token_weights = token_weights / token_weights[gate].mean().detach().clamp_min(1e-6)

    loss = (per_token_loss[gate] * token_weights[gate]).mean()
    return loss, gate.float().mean(), token_weights[gate].mean().detach()


def _neighbor_table(num_patches: int, radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    grid = int(num_patches**0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Expected square token grid, got {num_patches}")
    neighbors: list[list[int]] = []
    max_len = 0
    for idx in range(num_patches):
        row, col = divmod(idx, grid)
        current: list[int] = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = row + dr, col + dc
                if 0 <= rr < grid and 0 <= cc < grid:
                    current.append(rr * grid + cc)
        neighbors.append(current)
        max_len = max(max_len, len(current))
    table = torch.zeros(num_patches, max_len, dtype=torch.long, device=device)
    valid = torch.zeros(num_patches, max_len, dtype=torch.bool, device=device)
    for idx, current in enumerate(neighbors):
        if current:
            table[idx, : len(current)] = torch.tensor(current, dtype=torch.long, device=device)
            valid[idx, : len(current)] = True
    return table, valid


def local_context_targets(
    patch_logits: torch.Tensor,
    mask_indices: torch.Tensor,
    image_labels: torch.Tensor,
    radius: int = 1,
    confidence_threshold: float = 0.6,
    agreement_threshold: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build pseudo targets from confident visible neighbors around masked tokens."""
    batch, num_patches, num_classes = patch_logits.shape
    probs = patch_logits.softmax(dim=-1)
    class_mask = image_labels.to(probs.dtype).unsqueeze(1)
    masked_probs = probs * class_mask
    denom = masked_probs.sum(dim=-1, keepdim=True)
    probs = torch.where(denom > 0, masked_probs / denom.clamp_min(1e-6), probs)

    token_is_masked = torch.zeros(batch, num_patches, dtype=torch.bool, device=patch_logits.device)
    token_is_masked.scatter_(1, mask_indices, True)

    table, table_valid = _neighbor_table(num_patches, radius, patch_logits.device)
    neighbor_idx = table[mask_indices]
    neighbor_valid = table_valid[mask_indices]
    neighbor_masked = token_is_masked.gather(1, neighbor_idx.reshape(batch, -1)).reshape_as(neighbor_idx)
    valid = neighbor_valid & (~neighbor_masked)

    gather_idx = neighbor_idx.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, num_classes)
    neighbor_probs = probs.gather(1, gather_idx).reshape(batch, mask_indices.shape[1], neighbor_idx.shape[-1], num_classes)
    neighbor_probs = neighbor_probs * valid.unsqueeze(-1).to(neighbor_probs.dtype)
    counts = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(neighbor_probs.dtype)
    context_probs = neighbor_probs.sum(dim=2) / counts

    neighbor_labels = neighbor_probs.argmax(dim=-1)
    one_hot = F.one_hot(neighbor_labels, num_classes=num_classes).to(neighbor_probs.dtype)
    one_hot = one_hot * valid.unsqueeze(-1).to(one_hot.dtype)
    agreement_probs = one_hot.sum(dim=2) / counts

    visible = probs.masked_fill(token_is_masked.unsqueeze(-1), 0.0)
    visible_counts = (~token_is_masked).sum(dim=1, keepdim=True).clamp_min(1).to(probs.dtype)
    global_context = visible.sum(dim=1) / visible_counts
    no_neighbors = valid.sum(dim=-1) == 0
    context_probs = torch.where(no_neighbors.unsqueeze(-1), global_context.unsqueeze(1), context_probs)
    agreement_probs = torch.where(no_neighbors.unsqueeze(-1), torch.zeros_like(agreement_probs), agreement_probs)

    confidence, target = context_probs.max(dim=-1)
    agreement = agreement_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    gate = (confidence >= confidence_threshold) & (agreement >= agreement_threshold)
    return target, gate, agreement


def propagation_loss(pred_patch_logits: torch.Tensor, target: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    if gate.any():
        return F.cross_entropy(pred_patch_logits[gate], target[gate])
    return pred_patch_logits.sum() * 0.0
