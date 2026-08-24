"""Evaluate curated, cumulative 5-feature expansions from 50 to 100 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from train_market_free_model import (
    FEATURES_50, OUTPUT_ROOT, fit_transform, read_rows, resolve_version, split_path,
)


# Ordered from strongest expected incremental pre-race signal to weaker/contextual signal.
# Current odds, betting amounts, and all post-race targets are deliberately absent.
ADDITIONAL_FEATURES = [
    "hr_resid", "hr_last_resid", "hr_last_poppct", "hr_dist_chg", "ow_winrate",
    "rating__z", "hr_winrate__z", "hr_resid__z", "jk_winrate__z", "tr_winrate__z",
    "wgBudam__z", "wg_diff__z", "hr_rest_days__z", "train_runs_14__z", "race_front_ratio",
    "race_style_mean", "race_style_sd", "is_front", "tr_multi", "ow_plcrate",
    "ow_starts", "ow_resid", "jk_resid", "tr_resid", "age__pr",
    "train_sec_14", "bleed", "tool_set", "wgBudamBigo", "chaksun1",
    "rcName", "rcDay", "rcNo", "ilsu", "meet_cd",
    "birthday", "name", "jkName", "trName", "owName",
    "hrName", "hrNo", "jkNo", "trNo", "owNo",
    "wg", "wg__z", "jkhr_starts", "buga1", "rcDate",
]


def make_model():
    return XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.035,
        min_child_weight=8, subsample=0.85, colsample_bytree=0.9,
        reg_lambda=8.0, reg_alpha=0.1, objective="binary:logistic",
        eval_metric="auc", random_state=42, n_jobs=4, tree_method="hist",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    base = resolve_version(args.version)
    train = read_rows(split_path(base, "train"))
    valid = read_rows(split_path(base, "valid"))
    headers = set(train[0]) & set(valid[0])
    y_train = [int(float(row["win"])) for row in train]
    y_valid = [int(float(row["win"])) for row in valid]
    results = []

    for extra_count in range(0, 51, 5):
        features = [f for f in FEATURES_50 + ADDITIONAL_FEATURES[:extra_count] if f in headers]
        x_train, x_valid = fit_transform(train, valid, features)
        model = make_model()
        model.fit(x_train, y_train)
        auc = float(roc_auc_score(y_valid, model.predict_proba(x_valid)[:, 1]))
        result = {
            "feature_count": len(features),
            "added_features": ADDITIONAL_FEATURES[:extra_count],
            "valid_roc_auc": round(auc, 6),
        }
        results.append(result)
        print(f"{len(features):3d} features | valid ROC-AUC {auc:.6f}")

    best = max(results, key=lambda r: r["valid_roc_auc"])
    output = {"version": args.version, "model": "XGBoost curated 50-to-100 feature expansion", "best": best, "results": results}
    output_path = OUTPUT_ROOT / args.version / "xgb_50_to_100_feature_search.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BEST", json.dumps(best, ensure_ascii=False))


if __name__ == "__main__":
    main()
