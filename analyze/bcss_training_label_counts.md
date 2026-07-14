# BCSS-WSSS Training Label Counts

- Data dir: `/mnt/f40d9457-9b54-4677-beb8-559e6c90a5a4/AI_Team/minhtoan/jepa-wsss/data/BCSS-WSSS/training`
- File pattern: `*.png`
- Files scanned: `23422`
- Files with valid labels: `23422`
- Files without valid labels: `0`

## Per-Class Positive Labels

| index | bit | class | positive files | percent of valid files |
|---:|---:|---|---:|---:|
| 0 | 1 | tumor | 15232 | 65.03% |
| 1 | 2 | stroma | 16254 | 69.40% |
| 2 | 3 | lymphocyte | 5598 | 23.90% |
| 3 | 4 | necrosis | 2884 | 12.31% |

## Label Combinations

| label | tumor | stroma | lymphocyte | necrosis | files | percent of valid files |
|---|---:|---:|---:|---:|---:|---:|
| `0001` | 0 | 0 | 0 | 1 | 1058 | 4.52% |
| `0010` | 0 | 0 | 1 | 0 | 679 | 2.90% |
| `0011` | 0 | 0 | 1 | 1 | 5 | 0.02% |
| `0100` | 0 | 1 | 0 | 0 | 2903 | 12.39% |
| `0101` | 0 | 1 | 0 | 1 | 247 | 1.05% |
| `0110` | 0 | 1 | 1 | 0 | 3268 | 13.95% |
| `0111` | 0 | 1 | 1 | 1 | 30 | 0.13% |
| `1000` | 1 | 0 | 0 | 0 | 4738 | 20.23% |
| `1001` | 1 | 0 | 0 | 1 | 558 | 2.38% |
| `1010` | 1 | 0 | 1 | 0 | 130 | 0.56% |
| `1100` | 1 | 1 | 0 | 0 | 7343 | 31.35% |
| `1101` | 1 | 1 | 0 | 1 | 977 | 4.17% |
| `1110` | 1 | 1 | 1 | 0 | 1477 | 6.31% |
| `1111` | 1 | 1 | 1 | 1 | 9 | 0.04% |

## Labels Per File

| positive labels in file | files | percent of valid files |
|---:|---:|---:|
| 1 | 9378 | 40.04% |
| 2 | 11551 | 49.32% |
| 3 | 2484 | 10.61% |
| 4 | 9 | 0.04% |
