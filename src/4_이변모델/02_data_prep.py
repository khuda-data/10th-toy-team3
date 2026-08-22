"""
02_data_prep.py — 데이터 준비 및 타겟 생성

- 서울만 필터
- pop_pct >= 0.5 (비인기마)만 사용
- upset 타겟 생성: (비인기마) & (win==1) → 1
- 시간순 6:2:2 분할
- 피처셋 A/B/C 정의 및 저장

실행:
    python src/4_이변모델/02_data_prep.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE,
    assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")

# 배당률 파생 14개 (q 제외한 13개를 피처에서 제외)
ODDS_EXCLUDE = [
    "winOdds", "plcOdds", "p_raw", "logit_q", "log_q",
    "pop_rank", "pop_pct", "is_fav", "book_sum", "takeout",
    "pl_harville", "pl_disc", "q_plc",
]

# 피처에서 완전 제외 (식별자 + 결과 + 타겟 + 배당률13개)
FULL_EXCLUDE = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL, "upset"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[Step 2] Data Preparation & Target Generation")
    logger.info("=" * 60)

    # Load full data
    df = pd.read_csv("race_entries.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    logger.info(f"  Seoul: {len(df):,} rows")

    # upset target: (pop_pct >= 0.5) & (win == 1)
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)

    # Filter: only longshots (pop_pct >= 0.5)
    df = df[df["pop_pct"] >= 0.5].reset_index(drop=True)
    logger.info(f"  Longshots only (pop_pct >= 0.5): {len(df):,} rows")
    logger.info(f"  Upset rate: {df['upset'].mean():.4f} ({df['upset'].sum():,} upsets)")

    # Time-based 6:2:2 split
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)

    for fold in ["train", "valid", "test"]:
        sub = df[df["fold"] == fold]
        logger.info(f"  {fold:5s}: {len(sub):>6,} rows | {sub['rcDate'].min()}~{sub['rcDate'].max()} | upset={sub['upset'].mean():.4f}")

    # Define feature sets
    all_cols = set(df.columns)

    # B features: no odds at all
    b_features = sorted([c for c in all_cols if c not in FULL_EXCLUDE and c != "q" and c != "fold"])

    # A features: q only
    a_features = ["q"]

    # C features: B + q
    c_features = sorted(b_features + ["q"])

    logger.info(f"\n  Feature sets:")
    logger.info(f"    A (odds only): {len(a_features)} features  -> ['q']")
    logger.info(f"    B (no odds):   {len(b_features)} features")
    logger.info(f"    C (combined):  {len(c_features)} features  -> B + q")

    # Preprocess: encode categoricals + fill missing
    train_mask = df["fold"] == "train"

    # Identify categorical columns
    cat_cols = [c for c in c_features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in c_features if c not in cat_cols]

    logger.info(f"\n  Preprocessing:")
    logger.info(f"    Numeric: {len(num_cols)} | Categorical: {len(cat_cols)}")

    # Fill numeric missing with train median
    medians = df.loc[train_mask, num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    # Encode categoricals
    label_encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])
        label_encoders[col] = le

    remaining_nan = df[c_features].isnull().sum().sum()
    logger.info(f"    Remaining NaN: {remaining_nan}")

    # Save prepared data + metadata
    prep = {
        "df": df,
        "a_features": a_features,
        "b_features": b_features,
        "c_features": c_features,
        "label_encoders": label_encoders,
        "medians": medians,
    }
    with open(OUTPUT_DIR / "prepared_data.pkl", "wb") as f:
        pickle.dump(prep, f)
    logger.info(f"\n  Saved: results/upset_with_odds/prepared_data.pkl")

    # Summary CSV
    summary = pd.DataFrame([{
        "total_rows": len(df),
        "upset_count": df["upset"].sum(),
        "upset_rate": round(df["upset"].mean(), 4),
        "train_rows": (df["fold"] == "train").sum(),
        "valid_rows": (df["fold"] == "valid").sum(),
        "test_rows": (df["fold"] == "test").sum(),
        "a_features": len(a_features),
        "b_features": len(b_features),
        "c_features": len(c_features),
    }])
    summary.to_csv(OUTPUT_DIR / "data_summary.csv", index=False)

    logger.info("\n" + "=" * 60)
    logger.info("02_data_prep.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
