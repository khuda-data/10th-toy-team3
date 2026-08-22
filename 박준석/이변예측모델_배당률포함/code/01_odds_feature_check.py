"""
01_odds_feature_check.py — 배당률 파생 14개 컬럼 상관관계 확인

배당률 컬럼끼리 상관관계가 극도로 높은지 확인하고,
대표 피처로 q(정규화 암묵적 확률)를 선정한다.

실행:
    python src/upset_with_odds/01_odds_feature_check.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")

ODDS_COLS = [
    "winOdds", "plcOdds", "p_raw", "q", "logit_q", "log_q",
    "pop_rank", "pop_pct", "is_fav", "book_sum", "takeout",
    "pl_harville", "pl_disc", "q_plc",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[Step 1] Odds Feature Correlation Check")
    logger.info("=" * 60)

    # Load
    df = pd.read_csv("final.csv", usecols=["meet"] + ODDS_COLS, low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    logger.info(f"  Seoul data: {len(df):,} rows")

    # Correlation
    corr = df[ODDS_COLS].corr()
    corr.to_csv(OUTPUT_DIR / "odds_correlation.csv")
    logger.info(f"  Saved: results/upset_with_odds/odds_correlation.csv")

    # q vs others
    logger.info(f"\n  --- Correlation with 'q' ---")
    q_corr = corr["q"].drop("q").sort_values(key=abs, ascending=False)
    for col, r in q_corr.items():
        marker = " *LOW*" if abs(r) < 0.9 else ""
        logger.info(f"    q <-> {col:15s}  r = {r:.4f}{marker}")

    # Pairs with |r| < 0.9 (candidates to keep alongside q)
    low_corr = q_corr[q_corr.abs() < 0.9]
    if len(low_corr) > 0:
        logger.info(f"\n  Columns with |r| < 0.9 vs q (potential additional features):")
        for col, r in low_corr.items():
            logger.info(f"    {col:15s}  r = {r:.4f}")
    else:
        logger.info(f"\n  All 13 columns have |r| >= 0.9 with q. Only q is needed.")

    # Summary
    high_count = (q_corr.abs() >= 0.9).sum()
    logger.info(f"\n  Conclusion: {high_count}/13 columns have |r| >= 0.9 with q")
    logger.info(f"  -> Use 'q' as the single representative odds feature")

    logger.info("\n" + "=" * 60)
    logger.info("01_odds_feature_check.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
