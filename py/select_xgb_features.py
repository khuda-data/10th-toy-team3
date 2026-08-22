"""Find the best 30-to-50 feature count for the market-free XGBoost model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from train_market_free_model import FEATURES_50, fit_transform, read_rows, write_selected


def xgb_model():
    return XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.035,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=8.0, reg_alpha=0.1, objective="binary:logistic",
        eval_metric="auc", random_state=42, n_jobs=4, tree_method="hist",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--min-features", type=int, default=30)
    args = parser.parse_args()

    version = Path(args.version)
    train_rows, valid_rows = read_rows(version / "train.csv"), read_rows(version / "valid.csv")
    available_headers = set(train_rows[0]) & set(valid_rows[0])
    current = [feature for feature in FEATURES_50 if feature in available_headers]
    if args.min_features < 1 or args.min_features > len(current):
        raise ValueError(f"min-features must be from 1 to {len(current)}")
    y_train = [int(float(row["win"])) for row in train_rows]
    y_valid = [int(float(row["win"])) for row in valid_rows]

    history, best = [], None
    while len(current) >= args.min_features:
        x_train, x_valid = fit_transform(train_rows, valid_rows, current)
        model = xgb_model()
        model.fit(x_train, y_train)
        auc = float(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        record = {"feature_count": len(current), "valid_roc_auc": round(auc, 6), "features": list(current)}
        history.append(record)
        if best is None or auc > best["valid_roc_auc"]:
            best = record
        print(f"{len(current):2d} features | valid ROC-AUC {auc:.6f}")
        if len(current) == args.min_features:
            break
        least = min(range(len(current)), key=lambda i: model.feature_importances_[i])
        record["removed_for_next_round"] = current.pop(least)
        print(f"   remove: {record['removed_for_next_round']}")

    out = version / "market_free_xgb_feature_selection"
    out.mkdir(parents=True, exist_ok=True)
    result = {"version": args.version, "model": "XGBoost recursive backward selection", "best": best, "history": history}
    (out / "selection_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for split, rows in (("train_best.csv", train_rows), ("valid_best.csv", valid_rows)):
        write_selected(rows, out / split, best["features"])
    for name in ("test.csv", "test (1).csv"):
        path = version / name
        if path.exists():
            write_selected(read_rows(path), out / "test_best.csv", best["features"])
            break
    print("BEST", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
