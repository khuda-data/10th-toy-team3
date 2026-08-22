"""Recursive backward selection for the market-free RF model.

The valid set is used only to compare the 30-to-20 feature candidates.  The
test set remains untouched.  Usage:
    .venv\\Scripts\\python.exe select_rf_features.py --version v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from train_market_free_model import FEATURES, fit_transform, read_rows, write_selected


def rf_model():
    return RandomForestClassifier(
        n_estimators=700, max_features="sqrt", min_samples_leaf=4,
        class_weight="balanced_subsample", random_state=42, n_jobs=4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--min-features", type=int, default=20)
    args = parser.parse_args()

    version = Path(args.version)
    train_rows = read_rows(version / "train.csv")
    valid_rows = read_rows(version / "valid.csv")
    header = set(train_rows[0]) & set(valid_rows[0])
    current = [feature for feature in FEATURES if feature in header]
    if args.min_features < 1 or args.min_features > len(current):
        raise ValueError(f"min-features must be between 1 and {len(current)}")

    y_train = [int(float(row["win"])) for row in train_rows]
    y_valid = [int(float(row["win"])) for row in valid_rows]
    history = []
    best = None

    while len(current) >= args.min_features:
        x_train, x_valid = fit_transform(train_rows, valid_rows, current)
        model = rf_model()
        model.fit(x_train, y_train)
        auc = float(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        record = {
            "feature_count": len(current),
            "valid_roc_auc": round(auc, 6),
            "features": list(current),
        }
        history.append(record)
        if best is None or auc > best["valid_roc_auc"]:
            best = record
        print(f"{len(current):2d} features | valid ROC-AUC {auc:.6f}")

        if len(current) == args.min_features:
            break
        least_important_index = min(range(len(current)), key=lambda i: model.feature_importances_[i])
        removed = current.pop(least_important_index)
        record["removed_for_next_round"] = removed
        print(f"   remove: {removed}")

    out = version / "market_free_rf_feature_selection"
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "version": args.version,
        "model": "Random Forest recursive backward selection",
        "selection_range": [len(FEATURES), args.min_features],
        "best": best,
        "history": history,
    }
    (out / "selection_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_features = best["features"]
    write_selected(train_rows, out / "train_best.csv", best_features)
    write_selected(valid_rows, out / "valid_best.csv", best_features)
    for name in ("test.csv", "test (1).csv"):
        test_path = version / name
        if test_path.exists():
            write_selected(read_rows(test_path), out / "test_best.csv", best_features)
            break
    print("BEST", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
