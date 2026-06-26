# JEPA-WSSS

Weakly supervised semantic segmentation experiments for histopathology using JEPA auxiliary regularization and adaptive prototype learning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put datasets under `data/` and pretrained checkpoints under `jepa_wsss/pretrained/`.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_jepa.py --help
```

## Evaluate

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --help
```
