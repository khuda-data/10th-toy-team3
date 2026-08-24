"""Train the compact, market-free winner-probability model for v1~v8.

Usage:
    .venv\\Scripts\\python.exe train_market_free_model.py --version v1
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# No current-market amount, odds, popularity, or an odds-derived feature belongs here.
FEATURES_30 = [
    "meet", "rcDist", "rank", "n_run", "track", "chulNo",  # race context
    "age", "sex", "rating", "rating_na", "wgBudam", "wg_diff", "is_debut",
    "hr_starts", "hr_winrate", "hr_plcrate", "hr_rest_days", "hr_last_ord",
    "hr_last_dist", "hr_dist_starts", "hr_dist_winrate",  # horse form
    "jk_starts", "jk_winrate", "jk_plcrate", "tr_starts", "tr_winrate",
    "tr_plcrate",  # jockey / trainer
    "train_runs_14", "jkhr_winrate", "pace_conflict",  # race-day / pairing
]
FEATURES_50 = FEATURES_30 + [
    # Additional pre-race context, preparation, and running-style information.
    "ageCond", "budam", "prizeCond", "weather", "wgJk",
    "waterRate", "spRating", "stRating", "born", "is_new_horse",
    "tool_n", "ill_n", "clinic_30d", "train_days_14", "start_delay",
    "hr_style", "hr_style_sd", "race_front_n", "style_vs_race", "wgBudam_chg",
]
# Backward-compatible name used by the RF selection script.
FEATURES = FEATURES_30
CATEGORICAL = {
    "meet", "rank", "track", "sex", "rating_na", "ageCond", "budam", "prizeCond", "weather", "born",
    "tool_set", "wgBudamBigo", "rcName", "rcDay", "name", "jkName", "trName", "owName",
    "hrName", "hrNo", "jkNo", "trNo", "owNo",
}
KEY_COLUMNS = ["race_id", "entry_id", "win"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data" / "전처리_데이터셋"
OUTPUT_ROOT = REPO_ROOT / "results" / "market_free_feature_search"


def resolve_version(name: str) -> Path:
    requested = Path(name)
    candidates = [requested, DATASET_ROOT / name, DATASET_ROOT / f"{name}_base"]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Dataset version not found: {name}")


def split_path(version_dir: Path, split: str) -> Path:
    for candidate in (version_dir / f"{split}.csv.gz", version_dir / f"{split}.csv"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {split} split under {version_dir}")


def read_rows(path: Path):
    handle = (
        gzip.open(path, "rt", encoding="utf-8-sig", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8-sig", newline="")
    )
    with handle as f:
        return list(csv.DictReader(f))


def write_selected(rows, path: Path, available):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [c for c in KEY_COLUMNS if c in available] + available
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(({c: row.get(c, "") for c in fields} for row in rows))


def fit_transform(train_rows, valid_rows, features):
    numeric = [c for c in features if c not in CATEGORICAL]
    medians = {}
    for col in numeric:
        values = []
        for row in train_rows:
            try:
                value = float(row.get(col, ""))
                if np.isfinite(value):
                    values.append(value)
            except ValueError:
                pass
        medians[col] = float(np.median(values)) if values else 0.0

    category_codes = {}
    for col in features:
        if col in CATEGORICAL:
            values = sorted({row.get(col, "") or "__MISSING__" for row in train_rows})
            category_codes[col] = {value: index for index, value in enumerate(values)}

    def convert(rows):
        matrix = np.empty((len(rows), len(features)), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, col in enumerate(features):
                raw = row.get(col, "")
                if col in CATEGORICAL:
                    matrix[i, j] = category_codes[col].get(raw or "__MISSING__", -1)
                else:
                    try:
                        value = float(raw)
                        matrix[i, j] = value if np.isfinite(value) else medians[col]
                    except ValueError:
                        matrix[i, j] = medians[col]
        return matrix

    return convert(train_rows), convert(valid_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="e.g. v1")
    parser.add_argument("--model", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--feature-set", choices=["30", "50"], default="30")
    args = parser.parse_args()
    version_dir = resolve_version(args.version)
    train_path, valid_path = split_path(version_dir, "train"), split_path(version_dir, "valid")

    train_rows, valid_rows = read_rows(train_path), read_rows(valid_path)
    configured_features = FEATURES_30 if args.feature_set == "30" else FEATURES_50
    available = [c for c in configured_features if c in train_rows[0] and c in valid_rows[0]]
    missing = [c for c in configured_features if c not in available]
    if not available:
        raise ValueError("No configured features are present in this version.")

    output_dir = OUTPUT_ROOT / args.version / f"market_free_{args.feature_set}_features"
    write_selected(train_rows, output_dir / "train_selected.csv", available)
    write_selected(valid_rows, output_dir / "valid_selected.csv", available)
    test_rows = read_rows(split_path(version_dir, "test"))
    write_selected(test_rows, output_dir / "test_selected.csv", available)

    x_train, x_valid = fit_transform(train_rows, valid_rows, available)
    y_train = np.asarray([int(float(row["win"])) for row in train_rows])
    y_valid = np.asarray([int(float(row["win"])) for row in valid_rows])

    if args.model == "xgboost":
        # v1/v5 base data: boosted trees are the primary recommendation in the version guide.
        model = XGBClassifier(
            n_estimators=700, max_depth=4, learning_rate=0.035,
            min_child_weight=8, subsample=0.85, colsample_bytree=0.9,
            reg_lambda=8.0, reg_alpha=0.1, objective="binary:logistic",
            eval_metric="auc", random_state=42, n_jobs=4, tree_method="hist",
        )
        model_name = "XGBoost"
    else:
        model = RandomForestClassifier(
            n_estimators=700, max_features="sqrt", min_samples_leaf=4,
            class_weight="balanced_subsample", random_state=42, n_jobs=4,
        )
        model_name = "Random Forest"
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_valid)[:, 1]
    auc = roc_auc_score(y_valid, probability)
    metrics = {
        "version": args.version,
        "model": f"{model_name} (market-free compact features)",
        "valid_roc_auc": round(float(auc), 6),
        "train_rows": int(len(train_rows)), "valid_rows": int(len(valid_rows)),
        "feature_count": len(available), "features": available, "missing_configured_features": missing,
    }
    (output_dir / f"metrics_{args.model}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
