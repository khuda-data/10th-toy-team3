"""
01_build_c_model.py — 계보2(C모델, 배당률 q 포함) 재현 + 최종 전략 확정

12_final_strategy.py가 실제로 쓴 스펙(모델·피처·전처리)을 그대로 재현한다.
타겟은 upset(1착), 최종 전략은 winOdds 10~50배 + 상위10%(ROI+32.0%로 이미
확인된 조합)로 고정한다.

계보1(다크호스/인기마붕괴, q 제외)과의 차이:
    - 타겟: upset_B(입상) 대신 upset(1착)
    - 피처: q(배당률) 포함, 고상관 23개 컬럼 추가 제거
    - 결측치: 구조적 결측 26개 컬럼 0-fill 먼저 적용
    - 스케일링: StandardScaler 적용
    - 모델: n_estimators=300, max_depth=10, min_samples_leaf=20, class_weight='balanced'
    - 인기마 붕괴에 해당하는 두 번째 모델 없음(이 계보는 다크호스만 다룸)

실행:
    python src/c_model_validation/01_build_c_model.py

출력:
    results/c_model_validation/c_model.pkl
    results/c_model_validation/final_strategy_predictions.csv
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS,
    RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/c_model_validation")

ODDS_DROP = [
    "winOdds", "plcOdds", "p_raw", "logit_q", "log_q",
    "pop_rank", "is_fav", "book_sum", "takeout",
    "pl_harville", "pl_disc", "q_plc",
]
HIGH_CORR_DROP = [
    "chaksun2", "chaksun3", "chaksun4", "chaksun5",
    "buga2", "buga3", "dusu", "hr_style_n", "hr_prev_rating",
    "hr_last_finpct", "age__z", "train_runs_14__pr", "hr_last_wg",
    "wg__pr", "wg_diff__pr", "wgBudam__pr",
    "hr_winrate__pr", "hr_resid__pr",
    "jk_winrate__pr", "tr_winrate__pr",
    "hr_rest_days__pr", "bleed__pr", "rating__pr",
]
STRUCTURAL_MISSING = [
    "rating", "rating__z", "hr_winrate", "hr_plcrate", "hr_resid",
    "hr_starts", "hr_winrate__z", "hr_resid__z",
    "hr_last_ord", "hr_last_poppct", "hr_last_resid",
    "hr_last_dist", "hr_dist_chg", "hr_rest_days", "hr_rest_days__z",
    "hr_dist_winrate", "hr_dist_starts",
    "hr_style", "hr_style_sd", "style_vs_race",
    "race_style_mean", "race_style_sd", "race_front_ratio",
    "jkhr_winrate", "wgBudam_chg",
]
FULL_EXCLUDE = set(
    ID_COLS + OUTCOME_COLS + [TARGET_COL, "upset", "fold", "pop_pct"]
    + ODDS_DROP
    + ["winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
    + ["gap_h", "gap_d"]
)

FINAL_FILTER = (10, 50)  # winOdds 10~50배
FINAL_TOP_PCT = 0.10


def load_and_prepare():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)
    df["_winOdds"] = df["winOdds"].copy()
    df = df[df["pop_pct"] >= 0.5].reset_index(drop=True)

    drop_cols = [c for c in HIGH_CORR_DROP + ODDS_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, "rcDate", SPLIT_RATIOS)

    for col in STRUCTURAL_MISSING:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    c_features = [c for c in df.columns if c not in FULL_EXCLUDE and c != "_winOdds"]
    train_mask = df["fold"] == "train"
    cat_cols = [c for c in c_features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in c_features if c not in cat_cols]

    remaining = [c for c in num_cols if df[c].isnull().any()]
    if remaining:
        medians = df.loc[train_mask, remaining].median()
        df[remaining] = df[remaining].fillna(medians)

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        df[col] = LabelEncoder().fit(df[col].unique()).transform(df[col])

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    return df, c_features


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, c_features = load_and_prepare()

    train = df[df["fold"] == "train"]
    test = df[df["fold"] == "test"]
    logger.info(f"train {len(train):,} | test {len(test):,} | 기저율(test) {test['upset'].mean():.4f}")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train[c_features].values.astype(np.float32), train["upset"].values)

    proba = model.predict_proba(test[c_features].values.astype(np.float32))[:, 1]
    auc = roc_auc_score(test["upset"].values, proba)
    logger.info(f"test AUC {auc:.4f}")

    with open(OUTPUT_DIR / "c_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": c_features}, f)

    out = test[["rcDate", "upset", "_winOdds"]].copy()
    out["proba"] = proba
    out.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")

    # 최종 전략(winOdds 10~50배 + 상위10%) 서브셋도 별도 저장
    lo, hi = FINAL_FILTER
    seg = out[(out["_winOdds"] >= lo) & (out["_winOdds"] < hi)]
    k = max(1, int(len(seg) * FINAL_TOP_PCT))
    strategy = seg.nlargest(k, "proba")
    strategy.to_csv(OUTPUT_DIR / "final_strategy_predictions.csv", index=False, encoding="utf-8-sig")

    roi = (strategy["upset"] * strategy["_winOdds"]).sum() / len(strategy) * 100 - 100
    logger.info(f"최종 전략(winOdds {lo}~{hi}배 + 상위{FINAL_TOP_PCT:.0%}): "
                f"{len(strategy)}건 베팅, {int(strategy['upset'].sum())}건 적중, ROI {roi:+.1f}%")
    logger.info("완료: results/c_model_validation/")


if __name__ == "__main__":
    main()
