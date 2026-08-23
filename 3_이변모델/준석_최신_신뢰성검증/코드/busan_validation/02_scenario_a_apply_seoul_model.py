"""
02_scenario_a_apply_seoul_model.py — 시나리오 A: 서울 모델을 부경에 그대로 적용

재학습 없이, 서울 train으로 학습된 모델(results/final_validation/*_model.pkl)을
부경 전체 데이터에 그대로 적용한다.

주의(방법론): 모델뿐 아니라 전처리(결측치 중앙값·범주형 인코딩)도 "서울 train
기준으로 고정"해야 진짜 "그대로 적용"이 된다. 01_build_models.py는 이 전처리
객체를 저장하지 않았으므로, 여기서는 서울 train으로 medians·LabelEncoder를
다시 fit한 뒤 그 고정된 변환을 부경 데이터에 적용한다. 부경에만 있고 서울
train에는 없던 범주값은 "미지 범주"로 별도 처리한다(코드 주석 참고).

실행:
    python src/busan_validation/02_scenario_a_apply_seoul_model.py

출력:
    results/busan_validation/02_scenario_a_results.csv
    results/busan_validation/02_scenario_a_roi_bootstrap.csv
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, CATEGORICAL_COLS, assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/busan_validation")
MODEL_DIR = Path("results/final_validation")
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
TOP_PCT = 0.10

MODEL_SPECS = {
    "darkhorse": {"subset_query": "pop_pct >= 0.50", "target": "upset_B", "odds_col": "plcOdds"},
    "bust": {"subset_query": "pop_pct <= 0.25", "target": "upset_A", "odds_col": None},
}


def fit_seoul_preprocessing(full_df, feature_cols):
    """서울 train으로 medians·LabelEncoder를 고정 fit한다 (부경에 재사용)."""
    seoul_train = full_df[(full_df["meet"] == "서울") & (full_df["fold"] == "train")]

    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or full_df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    medians = seoul_train[num_cols].median()

    encoders = {}
    for col in cat_cols:
        classes = sorted(seoul_train[col].fillna("MISSING").astype(str).unique())
        encoders[col] = {v: i for i, v in enumerate(classes)}

    return medians, encoders, cat_cols, num_cols


def apply_frozen_preprocessing(df, medians, encoders, cat_cols, num_cols):
    df = df.copy()
    df[num_cols] = df[num_cols].fillna(medians)

    unseen_report = {}
    for col in cat_cols:
        vals = df[col].fillna("MISSING").astype(str)
        mapping = encoders[col]
        unseen_mask = ~vals.isin(mapping.keys())
        unseen_report[col] = int(unseen_mask.sum())
        # 서울 train에 없던 범주값은 "미지 범주" 코드(기존 클래스 수)로 매핑
        unknown_code = len(mapping)
        df[col] = vals.map(mapping).fillna(unknown_code).astype(int)

    return df, unseen_report


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("final.csv", low_memory=False)
    df = df.sort_values("rcDate").reset_index(drop=True)
    # 서울 fold는 서울 데이터만으로, 부경 fold는 부경 데이터만으로 각자 시간순 분할
    # (여기서는 서울 train 정의만 필요하므로 서울에 대해서만 계산)
    seoul_mask = df["meet"] == "서울"
    df.loc[seoul_mask, "fold"] = assign_time_split(
        df[seoul_mask].reset_index(drop=True), date_col="rcDate", ratios=SPLIT_RATIOS
    ).values

    busan = df[df["meet"] == "부경"].reset_index(drop=True)
    logger.info(f"부경 전체: {len(busan):,}행")

    summary_rows = []

    for name, spec in MODEL_SPECS.items():
        logger.info("=" * 60)
        logger.info(f"[시나리오 A: {name}] 서울 모델을 부경에 그대로 적용")

        with open(MODEL_DIR / f"{name}_model.pkl", "rb") as f:
            saved = pickle.load(f)
        model, feature_cols = saved["model"], saved["feature_cols"]

        medians, encoders, cat_cols, num_cols = fit_seoul_preprocessing(df, feature_cols)

        busan_sub = busan.query(spec["subset_query"]).reset_index(drop=True)
        busan_sub_prep, unseen = apply_frozen_preprocessing(busan_sub, medians, encoders, cat_cols, num_cols)

        total_unseen = sum(unseen.values())
        if total_unseen > 0:
            logger.warning(f"  서울 train에 없던 범주값 발견: {total_unseen}건 "
                            f"(컬럼별: { {k: v for k, v in unseen.items() if v > 0} })")

        X = busan_sub_prep[feature_cols]
        y = busan_sub_prep[spec["target"]].values
        proba = model.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, proba)
        k = max(1, int(len(y) * TOP_PCT))
        order = np.argsort(-proba)[:k]
        top_rate = y[order].mean()
        base_rate = y.mean()
        lift = top_rate / base_rate if base_rate > 0 else np.nan

        logger.info(f"  부경 {spec['subset_query']} 대상 {len(y):,}건 | 기저율 {base_rate:.4f} | "
                    f"AUC {auc:.4f} | 상위{TOP_PCT:.0%}({k}건) Lift {lift:.2f}")

        row = {
            "model": name, "scenario": "A_apply_seoul_asis", "n": len(y),
            "base_rate": base_rate, "auc": auc, "k10": k, "lift_at_10pct": lift,
            "n_unseen_categories": total_unseen,
        }

        if spec["odds_col"]:
            odds = busan_sub_prep[spec["odds_col"]].values[order]
            hit = y[order]
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

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "02_scenario_a_results.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/busan_validation/02_scenario_a_results.csv")


if __name__ == "__main__":
    main()
