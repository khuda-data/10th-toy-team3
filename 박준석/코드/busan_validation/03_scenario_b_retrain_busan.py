"""
03_scenario_b_retrain_busan.py — 시나리오 B: 부경 데이터로 처음부터 재학습

서울에서 확정한 방식(RandomForest, n_estimators=600, min_samples_leaf=50,
class_weight 없음, 다크호스/인기마붕괴 분리 학습)을 그대로, 부경 데이터에만
적용해 처음부터 새로 학습한다. 부경 자체를 시간순 6:2:2로 분할해
train/valid/test를 만들고, test는 재학습 과정에서 한 번도 보지 않다가
마지막에 한 번만 평가한다(서울과 동일한 원칙).

실행:
    python src/busan_validation/03_scenario_b_retrain_busan.py

출력:
    results/busan_validation/03_scenario_b_results.csv
    results/busan_validation/03_scenario_b_roi_bootstrap.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE,
    assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/busan_validation")
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
TOP_PCT = 0.10

MODEL_SPECS = {
    "darkhorse": {"subset_query": "pop_pct >= 0.50", "target": "upset_B", "max_depth": 12, "odds_col": "plcOdds"},
    "bust": {"subset_query": "pop_pct <= 0.25", "target": "upset_A", "max_depth": 8, "odds_col": None},
}


def load_busan():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "부경"].reset_index(drop=True)
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)
    return df


def prep_features(df):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "fold"]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    train_mask = df["fold"] == "train"
    medians = df.loc[train_mask, num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        df[col] = LabelEncoder().fit(df[col].unique()).transform(df[col])

    return df, feature_cols


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_busan()
    logger.info(f"부경 전체: {len(df):,}행")

    summary_rows = []

    for name, spec in MODEL_SPECS.items():
        logger.info("=" * 60)
        logger.info(f"[시나리오 B: {name}] 부경 데이터로 재학습")

        sub = df.query(spec["subset_query"]).reset_index(drop=True)
        sub, feature_cols = prep_features(sub.copy())

        train = sub[sub["fold"] == "train"]
        test = sub[sub["fold"] == "test"]

        X_train, y_train = train[feature_cols], train[spec["target"]]
        X_test, y_test = test[feature_cols], test[spec["target"]].values

        logger.info(f"  train {len(train):,} | test {len(test):,} | 기저율(test) {y_test.mean():.4f}")

        model = RandomForestClassifier(
            n_estimators=600, max_depth=spec["max_depth"], min_samples_leaf=50,
            class_weight=None, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, proba)
        k = max(1, int(len(y_test) * TOP_PCT))
        order = np.argsort(-proba)[:k]
        top_rate = y_test[order].mean()
        base_rate = y_test.mean()
        lift = top_rate / base_rate if base_rate > 0 else np.nan

        logger.info(f"  test AUC {auc:.4f} | 상위{TOP_PCT:.0%}({k}건) Lift {lift:.2f}")

        row = {
            "model": name, "scenario": "B_retrain_busan", "n_train": len(train), "n_test": len(test),
            "base_rate": base_rate, "auc": auc, "k10": k, "lift_at_10pct": lift,
        }

        if spec["odds_col"]:
            odds = test[spec["odds_col"]].values[order]
            hit = y_test[order]
            point_roi = roi_of(hit, odds)

            boot_rois = np.empty(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                idx = RNG.integers(0, k, size=k)
                boot_rois[i] = roi_of(hit[idx], odds[idx])
            ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])

            logger.info(f"  ROI 점추정 {point_roi:+.1f}% | 부트스트랩 95% CI [{ci_low:+.1f}%, {ci_high:+.1f}%] "
                        f"| 0 포함: {'예' if ci_low <= 0 <= ci_high else '아니오'}")
            row.update({
                "roi_pct": point_roi, "roi_ci_low": ci_low, "roi_ci_high": ci_high,
                "roi_ci_includes_zero": ci_low <= 0 <= ci_high,
            })

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "03_scenario_b_results.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/busan_validation/03_scenario_b_results.csv")


if __name__ == "__main__":
    main()
