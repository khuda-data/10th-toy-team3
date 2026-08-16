# Train/Valid/Test Split Info

Generated: 2026-08-16 21:21

---

## Split Method

- **Time-based split at 60 : 20 : 20** (train : valid : test)
- Based on `rcDate` order — NOT shuffled
- Split boundaries fall on race-date edges, so a single race never spans two folds
- Prevents data leakage from future races into training

## Split Sizes (identical across all versions)

| Fold | Rows | Ratio | Win Rate | Date Range |
|------|------|-------|----------|------------|
| train | 19,637 | 59.7% | 9.65% | 20230805 ~ 20250511 |
| valid | 6,591 | 20.0% | 9.76% | 20250517 ~ 20251227 |
| test | 6,660 | 20.3% | 9.56% | 20251228 ~ 20260809 |
| **Total** | **32,888** | 100% | - | - |

## Directory Structure

```
data/splits/
├── v1_base/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v2_standard/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v3_minmax/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v4_robust/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v5_base_no_outlier/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v6_standard_no_outlier/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v7_minmax_no_outlier/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── v8_robust_no_outlier/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
└── split_info.md
```

## Usage

```python
import pandas as pd

# Choose your version
VERSION = "v2_standard"  # or v1_base, v3_minmax, v4_robust

train = pd.read_csv(f"data/splits/{VERSION}/train.csv", encoding="utf-8-sig")
valid = pd.read_csv(f"data/splits/{VERSION}/valid.csv", encoding="utf-8-sig")
test  = pd.read_csv(f"data/splits/{VERSION}/test.csv",  encoding="utf-8-sig")

# Separate X and y
y_train = train["win"]
# Drop ID/market/outcome columns for X (see src/pipeline/config.py)
```

## Notes

- Each split file still contains the `fold` column (redundant but harmless)
- Encoding is `utf-8-sig` so Excel opens Korean text correctly
- **valid** is for hyperparameter/threshold tuning
- **test** is for final evaluation only — do not tune on it
