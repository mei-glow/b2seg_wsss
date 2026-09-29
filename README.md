# B2Seg

## Stage 1

Set the dataset root, output directory, and pretrained DeiT checkpoint paths:

```bash
export DATA_ROOT=/path/to/data
export RUN_ROOT=/path/to/runs
export PRETRAIN=/path/to/deit_base_patch16_224-b5f2ef4d.pth
```

### BCSS

```bash
SEED=0; CUDA_VISIBLE_DEVICES=0 python scripts/train_dual_router.py \
  --preset f --dataset bcss --data-root $DATA_ROOT --checkpoint $PRETRAIN \
  --seed $SEED --output-dir $RUN_ROOT/bcss_s1_F_seed$SEED
```

### GCSS

```bash
SEED=0; CUDA_VISIBLE_DEVICES=1 python scripts/train_dual_router.py \
  --preset f --dataset gcss --data-root $DATA_ROOT --checkpoint $PRETRAIN \
  --seed $SEED --output-dir $RUN_ROOT/gcss_s1_F_seed$SEED
```

### LUAD

```bash
SEED=0; CUDA_VISIBLE_DEVICES=2 python scripts/train_dual_router.py \
  --preset f --dataset luad --data-root $DATA_ROOT --checkpoint $PRETRAIN \
  --seed $SEED --output-dir $RUN_ROOT/luad_s1_F_seed$SEED
```

## Stage 2

Set the following paths:

- `DATA_ROOT`: path to the dataset directory.
- `RUN_ROOT`: path to the output directory.
- `PRETRAIN`: path to the pretrained DeiT checkpoint.

```bash
export DATA_ROOT=/path/to/data
export RUN_ROOT=/path/to/runs
export PRETRAIN=/path/to/deit_base_patch16_224-b5f2ef4d.pth

CUDA_VISIBLE_DEVICES=1 python scripts/train_online_pseudo.py \
  --preset n2 --dataset bcss --data-root $DATA_ROOT --checkpoint $PRETRAIN \
  --teacher-checkpoint $RUN_ROOT/s1_F_augment_seed0/best.pt \
  --seed 0 --output-dir $RUN_ROOT/s2_N2_seed0
```

## Evaluate

Set the following paths:

- `DATA_ROOT`: path to the dataset directory.
- `RUN_ROOT`: path to the output directory.
- `R`: directory containing the Stage 2 `best.pt` checkpoint.

```bash
export DATA_ROOT=/path/to/data
export RUN_ROOT=/path/to/runs
export R=$RUN_ROOT/s2_N3_all_seed0
```

### CRF without TTA

Evaluate the checkpoint on the test set with CRF post-processing and no test-time augmentation.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_crf.py \
  --checkpoint $R/best.pt --data-root $DATA_ROOT --dataset bcss \
  --split test --batch-size 16 --num-workers 8 --amp \
  --tta none --prediction-head coarse --output $R/test_metrics_crf.json
```

### CRF with flip TTA

Evaluate the checkpoint on the test set with CRF post-processing and flip test-time augmentation.

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_crf.py \
  --checkpoint $R/best.pt --data-root $DATA_ROOT --dataset bcss \
  --split test --batch-size 16 --num-workers 8 --amp \
  --tta flip --prediction-head coarse \
  --output $R/test_metrics_crf_tta_flip.json
```
