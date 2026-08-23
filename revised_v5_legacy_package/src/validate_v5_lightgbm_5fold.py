"""Five-fold race-group cross-validation for the v5 LightGBM rank+binary model.

The final test split remains untouched.  This script uses only v5's original
training interval and keeps every race intact in one validation fold.
Market probability, odds, and every known outcome/leakage column are excluded.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from src.common.tabular_data import TARGET, select_features


DATA_PATH = Path("data/revised_v5/train_revised_v5.csv")
OUT = Path("outputs/reports")
RACE_KEY = "race_id"


def encode_fold(train: pd.DataFrame, valid: pd.DataFrame, numeric: list[str], categorical: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit categorical maps on the fold's training races only."""
    train_x = train[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    valid_x = valid[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    for column in categorical:
        values = train[column].astype("string").fillna("<NA>")
        mapping = {value: index for index, value in enumerate(sorted(values.unique()))}
        train_x[column + "__code"] = values.map(mapping).astype(int)
        valid_x[column + "__code"] = valid[column].astype("string").fillna("<NA>").map(mapping).fillna(-1).astype(int)
    return train_x, valid_x


def group_sizes(frame: pd.DataFrame) -> np.ndarray:
    return frame.groupby(RACE_KEY, sort=False).size().to_numpy()


def race_softmax(frame: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    result = np.zeros(len(frame), dtype=float)
    for _, indices in frame.groupby(RACE_KEY, sort=False).indices.items():
        local = scores[indices]
        local = local - np.max(local)
        exp = np.exp(local)
        result[indices] = exp / exp.sum()
    return result


def calibration_mae(y: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    groups = pd.qcut(probabilities, bins, labels=False, duplicates="drop")
    return float(np.mean([abs(y[groups == group].mean() - probabilities[groups == group].mean()) for group in np.unique(groups)]))


def top1_hit_rate(frame: pd.DataFrame, probabilities: np.ndarray) -> float:
    hits = total = 0
    for _, indices in frame.groupby(RACE_KEY, sort=False).indices.items():
        hits += int(frame.iloc[indices][TARGET].iloc[np.argmax(probabilities[indices])] == 1)
        total += 1
    return hits / total


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, low_memory=False).sort_values(["rcDate", RACE_KEY, "entry_id"]).reset_index(drop=True)
    numeric, categorical = select_features(data)
    splitter = GroupKFold(n_splits=5)
    fold_rows, oof_parts = [], []
    common = dict(learning_rate=0.03, num_leaves=127, max_depth=-1, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l1=0.1, lambda_l2=1.0, verbosity=-1, seed=42)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(data, data[TARGET], groups=data[RACE_KEY]), start=1):
        train = data.iloc[train_idx].sort_values([RACE_KEY, "entry_id"]).reset_index(drop=True)
        valid = data.iloc[valid_idx].sort_values([RACE_KEY, "entry_id"]).reset_index(drop=True)
        x_train, x_valid = encode_fold(train, valid, numeric, categorical)
        y_train, y_valid = train[TARGET].to_numpy(dtype=int), valid[TARGET].to_numpy(dtype=int)
        cat_codes = [column + "__code" for column in categorical]
        rank_train = lgb.Dataset(x_train, label=y_train, group=group_sizes(train), categorical_feature=cat_codes, free_raw_data=False)
        rank_valid = lgb.Dataset(x_valid, label=y_valid, group=group_sizes(valid), reference=rank_train, free_raw_data=False)
        rank_model = lgb.train({**common, "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1], "lambdarank_truncation_level": 20},
                               rank_train, num_boost_round=3000, valid_sets=[rank_valid], callbacks=[lgb.early_stopping(150, verbose=False)])
        binary_train = lgb.Dataset(x_train, label=y_train, categorical_feature=cat_codes, free_raw_data=False)
        binary_valid = lgb.Dataset(x_valid, label=y_valid, reference=binary_train, free_raw_data=False)
        scale = float((y_train == 0).sum() / (y_train == 1).sum())
        binary_model = lgb.train({**common, "objective": "binary", "metric": "auc", "scale_pos_weight": scale},
                                 binary_train, num_boost_round=3000, valid_sets=[binary_valid], callbacks=[lgb.early_stopping(150, verbose=False)])
        rank_score = rank_model.predict(x_valid)
        binary_score = binary_model.predict(x_valid, raw_score=True)
        rank_score = (rank_score - rank_score.mean()) / (rank_score.std() + 1e-12)
        binary_score = (binary_score - binary_score.mean()) / (binary_score.std() + 1e-12)
        weight, best_auc = 0.5, -np.inf
        for candidate in np.linspace(0, 1, 21):
            score = candidate * rank_score + (1 - candidate) * binary_score
            auc = roc_auc_score(y_valid, score)
            if auc > best_auc:
                weight, best_auc = float(candidate), float(auc)
        probabilities = race_softmax(valid, weight * rank_score + (1 - weight) * binary_score)
        fold_rows.append({"fold": fold, "train_rows": len(train), "valid_rows": len(valid), "train_races": train[RACE_KEY].nunique(),
                          "valid_races": valid[RACE_KEY].nunique(), "rank_weight": weight,
                          "rank_best_iteration": int(rank_model.best_iteration), "binary_best_iteration": int(binary_model.best_iteration),
                          "roc_auc": roc_auc_score(y_valid, probabilities), "pr_auc": average_precision_score(y_valid, probabilities),
                          "calibration_mae": calibration_mae(y_valid, probabilities), "top1_hit_rate": top1_hit_rate(valid, probabilities)})
        oof_parts.append(valid[[RACE_KEY, "entry_id", "rcDate", TARGET]].assign(fold=fold, model_probability=probabilities))
        print(f"fold {fold}/5: ROC={fold_rows[-1]['roc_auc']:.4f}, PR={fold_rows[-1]['pr_auc']:.4f}, top1={fold_rows[-1]['top1_hit_rate']:.4f}")

    folds = pd.DataFrame(fold_rows)
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(["rcDate", RACE_KEY, "entry_id"])
    summary = {"method": "5-fold GroupKFold by race_id on original v5 training interval", "data_scope": "train_revised_v5.csv only; final valid/test never used",
               "model": "LightGBM LambdaRank + binary blend; fold-local categorical encoding; no market inputs",
               "folds": fold_rows,
               "mean": {column: float(folds[column].mean()) for column in ["roc_auc", "pr_auc", "calibration_mae", "top1_hit_rate"]},
               "std": {column: float(folds[column].std(ddof=1)) for column in ["roc_auc", "pr_auc", "calibration_mae", "top1_hit_rate"]},
               "oof": {"rows": int(len(oof)), "races": int(oof[RACE_KEY].nunique()), "roc_auc": float(roc_auc_score(oof[TARGET], oof["model_probability"])),
                       "pr_auc": float(average_precision_score(oof[TARGET], oof["model_probability"])), "top1_hit_rate": float(top1_hit_rate(oof.reset_index(drop=True), oof["model_probability"].to_numpy()))}}
    folds.to_csv(OUT / "v5_lightgbm_5fold_results.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT / "v5_lightgbm_5fold_oof_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "v5_lightgbm_5fold_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["mean"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
