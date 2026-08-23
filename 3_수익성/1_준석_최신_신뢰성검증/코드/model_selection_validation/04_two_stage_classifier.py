"""
04_two_stage_classifier.py — "1단계 유형 분류기 + 2단계 특화 모델" 비교 실험

현재 방식(01_build_models.py)은 배당률 기반 인기도(pop_pct)로 먼저
비인기마(>=0.50)/인기마(<=0.25)를 하드코딩 규칙으로 나눈 뒤, 각 부분집합
안에서만 전용 이진분류기를 학습한다.

이 스크립트는 대안을 만든다: pop_pct 문턱값으로 미리 자르지 않고, 전체
말을 대상으로 "이 말이 다크호스형/인기마붕괴형/해당없음 중 무엇인지"를
바로 예측하는 분류기를 학습한다(부분집합 없이 통째로 학습). 그리고
같은 개수의 후보를 뽑았을 때(다크호스 352건, 인기마붕괴 190건 — 기존과
동일 예산) 실제 적중 건수가 기존 방식보다 많은지 적은지 비교한다.

레이블:
    darkhorse_type = 1 if (pop_pct >= 0.50) & (upset_B == 1) else 0
    bust_type      = 1 if (pop_pct <= 0.25) & (upset_A == 1) else 0

피처: 기존과 동일하게 시장/배당 컬럼(pop_pct 포함) 전부 제외. "오늘 이
말이 인기가 있는지 없는지도 모르는 상태에서" 과거 성적만으로 두 유형을
가려낼 수 있는지를 보는 실험이다.

실행:
    python src/model_selection_validation/04_two_stage_classifier.py

출력:
    results/final_validation/two_stage_vs_baseline.csv
    results/final_validation/two_stage_feature_importance.csv
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

OUTPUT_DIR = Path("results/final_validation")
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])

# 기존(01_build_models.py) 방식의 예산과 정확히 맞춘다 — 같은 개수를 뽑아 비교
BASELINE_BUDGET = {"darkhorse_type": 352, "bust_type": 190}
BASELINE_HITS = {"darkhorse_type": 81, "bust_type": 64}  # lift_summary.csv 기준 (190*0.3368≈64)


def load_data():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)

    df["darkhorse_type"] = ((df["pop_pct"] >= 0.50) & (df["upset_B"] == 1)).astype(int)
    df["bust_type"] = ((df["pop_pct"] <= 0.25) & (df["upset_A"] == 1)).astype(int)
    return df


def prep_features(df):
    feature_cols = [c for c in df.columns
                     if c not in EXCLUDE_COLS and c not in ("fold", "darkhorse_type", "bust_type")]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    train_mask = df["fold"] == "train"
    medians = df.loc[train_mask, num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        df[col] = LabelEncoder().fit(df[col].unique()).transform(df[col])

    return df, feature_cols


def main():
    df = load_data()
    df, feature_cols = prep_features(df)

    train = df[df["fold"] == "train"]
    test = df[df["fold"] == "test"]
    X_train, X_test = train[feature_cols], test[feature_cols]

    logger.info(f"전체(부분집합 없음) train {len(train):,} | test {len(test):,}")
    logger.info(f"  darkhorse_type 기저율(test): {test['darkhorse_type'].mean():.4f} "
                f"({test['darkhorse_type'].sum()}건)")
    logger.info(f"  bust_type 기저율(test):      {test['bust_type'].mean():.4f} "
                f"({test['bust_type'].sum()}건)")

    rows = []
    fi_rows = []

    for label in ["darkhorse_type", "bust_type"]:
        logger.info("=" * 60)
        logger.info(f"[1단계 통합 분류기] target={label} (전체 {len(train):,}행으로 학습, 부분집합 필터 없음)")

        y_train = train[label]
        y_test = test[label].values

        model = RandomForestClassifier(
            n_estimators=600, max_depth=12, min_samples_leaf=50,
            class_weight=None, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)

        budget = BASELINE_BUDGET[label]
        order = np.argsort(-proba)[:budget]
        hits_new = int(y_test[order].sum())
        base_rate_full = y_test.mean()
        lift_new = (hits_new / budget) / base_rate_full if base_rate_full > 0 else np.nan

        hits_baseline = BASELINE_HITS[label]

        logger.info(f"  AUC(전체 모집단 기준) {auc:.4f}")
        logger.info(f"  같은 예산({budget}건) 적중 비교 — 기존(부분집합+전용모델): {hits_baseline}건 "
                    f"vs 1단계 통합분류기: {hits_new}건")

        rows.append({
            "type": label, "budget": budget,
            "baseline_hits": hits_baseline, "baseline_hit_rate": hits_baseline / budget,
            "two_stage_hits": hits_new, "two_stage_hit_rate": hits_new / budget,
            "two_stage_lift_vs_full_population": lift_new,
            "two_stage_auc_full_population": auc,
            "diff_hits": hits_new - hits_baseline,
        })

        fi = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(15)
        for feat, imp in fi.items():
            fi_rows.append({"type": label, "feature": feat, "importance": imp})

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "two_stage_vs_baseline.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fi_rows).to_csv(OUTPUT_DIR / "two_stage_feature_importance.csv", index=False, encoding="utf-8-sig")

    logger.info("=" * 60)
    logger.info("결론:")
    for _, r in result.iterrows():
        verdict = "개선" if r["diff_hits"] > 0 else ("동일" if r["diff_hits"] == 0 else "악화")
        logger.info(f"  {r['type']}: 기존 {int(r['baseline_hits'])}건 -> 통합분류기 {int(r['two_stage_hits'])}건 "
                    f"({verdict}, {r['diff_hits']:+d}건)")
    logger.info("완료: results/final_validation/two_stage_vs_baseline.csv")


if __name__ == "__main__":
    main()
