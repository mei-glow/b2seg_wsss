# JEPA-WSSS

Weakly supervised semantic segmentation experiments for histopathology.

The current final BCSS recipe is a **2-stage teacher-initialized dual-router pipeline**:

1. train a semantic router teacher from the raw DeiT backbone
2. train the final online-pseudo model from that teacher

Classes: `tumor`, `stroma`, `lymphocyte`, `necrosis`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected server layout:

```text
data/
jepa_wsss/pretrained/deit_base_patch16_224-b5f2ef4d.pth
scripts/
runs/
```

All commands below assume:

```bash
cd /home/ubuntu/24ngoc.nk
source .venv/bin/activate
```

## Stage 1: Train Semantic Teacher

This trains the raw DeiT + single semantic router teacher.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_dual_router.py \
  --data-root data --dataset bcss \
  --model deit_base_patch16_224 \
  --checkpoint /home/ubuntu/24ngoc.nk/jepa_wsss/pretrained/deit_base_patch16_224-b5f2ef4d.pth \
  --output-dir runs/dual_v1_single_semantic_all12 \
  --variant single_semantic \
  --route-layers all \
  --epochs 5 --batch-size 32 --val-batch-size 64 --num-workers 8 \
  --lr 1e-4 --scheduler warmup_cosine --warmup-epochs 1 --min-lr-ratio 0.1 \
  --semantic-init final \
  --amp --grad-checkpointing --log-every 25
```

## Stage 2: Train Final Model

This trains the final fused model using the semantic teacher from Stage 1.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_online_pseudo.py \
  --data-root data --dataset bcss \
  --model deit_base_patch16_224 \
  --checkpoint /home/ubuntu/24ngoc.nk/jepa_wsss/pretrained/deit_base_patch16_224-b5f2ef4d.pth \
  --teacher-checkpoint runs/dual_v1_single_semantic_all12/best.pt \
  --output-dir runs/phase2_A2b_t1a_adaptcons_ratio_min03 \
  --route-layers all --epochs 10 --batch-size 24 --val-batch-size 64 --num-workers 8 \
  --lr 1e-4 --scheduler warmup_cosine --warmup-epochs 1 --min-lr-ratio 0.1 \
  --teacher-mode ema --ema-decay 0.99 --init-from-teacher \
  --teacher-logit-source adaptive_topk --teacher-source-temperature 0.5 --teacher-source-topk-frac 0.05 \
  --output-mode learned_fuse --output-fuse-alpha 0.35 \
  --pseudo-logit-target output --pseudo-score softmax --pseudo-thresholds 0.85 \
  --adaptive-pseudo-thresholds --adaptive-threshold-strength 0.5 --adaptive-threshold-min 0.50 --adaptive-threshold-max 0.90 \
  --pseudo-weight 0.2 --pseudo-expand-mode fixed --pseudo-expand-min-frac 0.03 --pseudo-expand-max-frac 0.06 \
  --strong-consistency-weight 0.1 --consistency-class-weights 1,1,1,1 \
  --consistency-weight-mode ratio --consistency-weight-min 0.3 --consistency-weight-max 1.0 \
  --strong-brightness 0.15 --strong-contrast 0.25 --strong-noise 0.03 \
  --restrict-present --amp --grad-checkpointing --log-every 25
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_crf.py \
  --checkpoint runs/phase2_A2b_t1a_adaptcons_ratio_min03/best.pt \
  --split test \
  --batch-size 64 \
  --num-workers 8 \
  --amp \
  --prediction-head coarse \
  --output runs/phase2_A2b_t1a_adaptcons_ratio_min03/test_metrics_crf.json
```

## Notes

- This is a 2-stage training recipe, not a pure one-stage method.
- `runs/dual_v1_single_semantic_all12/best.pt` is the Stage 1 teacher.
- `runs/phase2_A2b_t1a_adaptcons_ratio_min03/best.pt` is the final Stage 2 checkpoint.
- Keep Phikon runs as ablations only; the final recipe above uses the original DeiT-B backbone.
