"""
01_build_models.py — 이변 탐지 모델 재현 (다크호스 · 인기마 붕괴)

reports/5_이변모델_선정/01_모델선정.docx 에 기록된 스펙을 그대로 재현한다.
이 보고서의 원본 코드는 어디에도 저장되어 있지 않아, 문서에 적힌
하이퍼파라미터·서브셋 정의·피처 제외 목록을 그대로 따라 새로 작성했다.

모델:
    다크호스   — subset: pop_pct >= 0.50, target: upset_B (비인기마 입상)
    인기마 붕괴 — subset: pop_pct <= 0.25, target: upset_A (인기마 부진)

공통 스펙 (보고서 10.1절 그대로):
    RandomForestClassifier(n_estimators=600, min_samples_leaf=50,
                            class_weight=None, random_state=42)
    다크호스 max_depth=12, 인기마 붕괴 max_depth=8

실행:
    python src/model_selection_validation/01_build_models.py

출력:
    results/final_validation/{darkhorse,bust}_model.pkl
    results/final_validation/{darkhorse,bust}_test_predictions.csv
    results/final_validation/lift_summary.csv
"""

import logging
import pickle
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

# 보고서 DROP 목록과 동일 (타겟 정의 + 시장 정보 + 결과 + 식별자)
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])

MODEL_SPECS = {
    "darkhorse": {
        "subset_query": "pop_pct >= 0.50",
        "target": "upset_B",
        "max_depth": 12,
        "odds_col": "plcOdds",  # 연승 배당 (입상 기준이므로)
    },
    "bust": {
        "subset_query": "pop_pct <= 0.25",
        "target": "upset_A",
        "max_depth": 8,
        "odds_col": None,  # 인기마 붕괴는 "베팅 대상"이 아니라 스크리닝 지표
    },
}


def load_data():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)
    return df


def prep_features(df, target):
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


def lift_at_k(y_true, scores, k_pct):
    n = len(y_true)
    k = max(1, int(n * k_pct))
    order = np.argsort(-scores)
    top_rate = y_true[order[:k]].mean()
    base_rate = y_true.mean()
    return top_rate / base_rate if base_rate > 0 else np.nan, top_rate, base_rate, k


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    logger.info(f"서울 전체: {len(df):,}행")

    summary_rows = []

    for name, spec in MODEL_SPECS.items():
        logger.info("=" * 60)
        logger.info(f"[{name}] subset: {spec['subset_query']} | target: {spec['target']}")

        sub = df.query(spec["subset_query"]).reset_index(drop=True)
        sub, feature_cols = prep_features(sub.copy(), spec["target"])

        train = sub[sub["fold"] == "train"]
        valid = sub[sub["fold"] == "valid"]
        test = sub[sub["fold"] == "test"]

        X_train, y_train = train[feature_cols], train[spec["target"]]
        X_test, y_test = test[feature_cols], test[spec["target"]].values

        logger.info(f"  train {len(train):,} | valid {len(valid):,} | test {len(test):,} | "
                    f"기저율(test) {y_test.mean():.4f}")

        model = RandomForestClassifier(
            n_estimators=600,
            max_depth=spec["max_depth"],
            min_samples_leaf=50,
            class_weight=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        lift10, top_rate10, base_rate, k10 = lift_at_k(y_test, proba, 0.10)

        logger.info(f"  test AUC {auc:.4f} | Lift@10% {lift10:.2f} (top {top_rate10:.4f} vs base {base_rate:.4f})")

        # 저장: 모델
        with open(OUTPUT_DIR / f"{name}_model.pkl", "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

        # 저장: test 예측 (다음 스크립트에서 부트스트랩/민감도 분석에 사용)
        out = test[["rcDate", spec["target"]]].copy()
        out["proba"] = proba
        if spec["odds_col"]:
            out["odds"] = df.loc[test.index, spec["odds_col"]].values
        out.to_csv(OUTPUT_DIR / f"{name}_test_predictions.csv", index=False, encoding="utf-8-sig")

        summary_rows.append({
            "model": name, "target": spec["target"], "n_train": len(train),
            "n_test": len(test), "base_rate": base_rate, "test_auc": auc,
            "lift_at_10pct": lift10, "top10_hit_rate": top_rate10, "k10": k10,
        })

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "lift_summary.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/final_validation/lift_summary.csv")


if __name__ == "__main__":
    main()
