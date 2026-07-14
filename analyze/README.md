# Analyze BCSS-WSSS Training Labels

This folder contains small analysis utilities for the local BCSS-WSSS data.

## Count Training Labels

BCSS-WSSS training labels are encoded in each PNG filename:

```text
TCGA-...+0[1101].png
```

The label bits are:

| bit | class |
|---:|---|
| 1 | tumor |
| 2 | stroma |
| 3 | lymphocyte |
| 4 | necrosis |

Run from the repository root:

```bash
python analyze/count_bcss_training_labels.py
```

Run with the explicit local training folder:

```bash
python analyze/count_bcss_training_labels.py \
  --data-dir /mnt/f40d9457-9b54-4677-beb8-559e6c90a5a4/AI_Team/minhtoan/jepa-wsss/data/BCSS-WSSS/training
```

Write the report to a Markdown file:

```bash
python analyze/count_bcss_training_labels.py \
  --output analyze/bcss_training_label_counts.md
```

The generated report in this folder is:

- `bcss_training_label_counts.md`

The per-class table counts how many files have each positive class bit. Because
this is a multi-label dataset, one file can be counted under multiple classes.
