"""
01_missing_check.py — 결측치 현황 파악 + 구조적/랜덤 패턴 분류 (1~2단계)

1단계: 156개 컬럼 전체 결측치 현황
2단계: 결측률 5%+ 컬럼의 구조적/랜덤 분류

실행:
    python src/eda/01_missing_check.py
    python src/eda/01_missing_check.py --input race_entries.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/eda")


# ============================================================
# 1단계: 전체 결측치 현황
# ============================================================

def compute_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """전체 컬럼의 결측치 개수/비율을 계산하여 결측률 높은 순으로 정렬."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    report = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values,
        "dtype": df.dtypes.astype(str).values,
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    return report


def summarize_missing_bands(report: pd.DataFrame):
    """결측률 구간별 컬럼 개수를 요약."""
    bands = {
        "0% (no missing)": (report["missing_pct"] == 0).sum(),
        "0~10%": ((report["missing_pct"] > 0) & (report["missing_pct"] <= 10)).sum(),
        "10~30%": ((report["missing_pct"] > 10) & (report["missing_pct"] <= 30)).sum(),
        "30%+": (report["missing_pct"] > 30).sum(),
    }

    logger.info("\n  --- Missing Rate Summary ---")
    for band, count in bands.items():
        logger.info(f"    {band:20s}: {count} columns")
    logger.info(f"    {'Total':20s}: {len(report)} columns")

    return bands


# ============================================================
# 2단계: 구조적 vs 랜덤 결측 패턴 분류
# ============================================================

def classify_missing_pattern(
    df: pd.DataFrame,
    target_cols: list[str],
    group_cols: list[str] = None,
) -> pd.DataFrame:
    """결측률 5%+ 컬럼에 대해 그룹별 교차 집계로 구조적/랜덤을 분류.

    구조적 판정: 특정 그룹에서 결측률 90%+
    랜덤 판정: 그룹 간 결측률 차이 max-min < 30%p
    """
    if group_cols is None:
        group_cols = ["is_debut", "rank", "ageCond"]

    # 실제 존재하는 그룹 컬럼만 사용
    group_cols = [c for c in group_cols if c in df.columns]

    results = []

    for col in target_cols:
        overall_pct = df[col].isnull().mean() * 100
        classification = "random"
        concentrated_group = ""
        group_missing_pct = 0.0
        analysis_group = ""

        for gcol in group_cols:
            # 그룹별 결측률
            group_missing = df.groupby(gcol)[col].apply(lambda x: x.isnull().mean() * 100)

            max_pct = group_missing.max()
            min_pct = group_missing.min()
            max_group = group_missing.idxmax()

            # 구조적 판정: 특정 그룹에서 90%+ 결측
            if max_pct >= 90:
                classification = "structural"
                concentrated_group = f"{gcol}={max_group}"
                group_missing_pct = max_pct
                analysis_group = gcol
                break

            # 차이가 큰 경우도 구조적으로 간주 (60%p 이상 차이)
            if max_pct - min_pct >= 60:
                classification = "structural"
                concentrated_group = f"{gcol}={max_group}"
                group_missing_pct = max_pct
                analysis_group = gcol
                break

        results.append({
            "column": col,
            "overall_missing_pct": round(overall_pct, 2),
            "classification": classification,
            "concentrated_group": concentrated_group,
            "group_missing_pct": round(group_missing_pct, 2),
            "analysis_group_col": analysis_group,
        })

    return pd.DataFrame(results)


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EDA: Missing value check")
    parser.add_argument("--input", default="race_entries.csv", help="Input CSV path")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[1 Stage] Missing Value Report")
    logger.info("=" * 60)

    # 데이터 로드 + 서울 필터링
    df = pd.read_csv(args.input, low_memory=False)
    logger.info(f"  Loaded: {args.input} ({len(df):,} rows x {len(df.columns)} cols)")

    df = df[df["meet"] == "서울"].reset_index(drop=True)
    logger.info(f"  Seoul only: {len(df):,} rows")

    # 1단계: 결측 현황
    report = compute_missing_report(df)
    report.to_csv(OUTPUT_DIR / "missing_report.csv", index=False)
    logger.info(f"  Saved: results/eda/missing_report.csv")

    # 구간 요약
    summarize_missing_bands(report)

    # 상위 15개 출력
    logger.info("\n  --- Top 15 Missing Columns ---")
    top15 = report[report["missing_pct"] > 0].head(15)
    for _, row in top15.iterrows():
        logger.info(f"    {row['column']:30s} {row['missing_pct']:6.2f}% ({row['missing_count']:,})")

    # 2단계: 패턴 분류
    logger.info("\n" + "=" * 60)
    logger.info("[2 Stage] Missing Pattern Classification")
    logger.info("=" * 60)

    # 결측률 5%+ 컬럼
    high_missing_cols = report[report["missing_pct"] >= 5]["column"].tolist()
    logger.info(f"  Target columns (missing >= 5%): {len(high_missing_cols)}")

    if high_missing_cols:
        # meet 컬럼도 그룹으로 추가 (부경 관련 확인용 — 이미 서울만이지만 is_debut 등)
        pattern_report = classify_missing_pattern(
            df, high_missing_cols, group_cols=["is_debut", "rank", "ageCond"]
        )
        pattern_report.to_csv(OUTPUT_DIR / "missing_pattern_report.csv", index=False)
        logger.info(f"  Saved: results/eda/missing_pattern_report.csv")

        # 결과 출력
        structural = pattern_report[pattern_report["classification"] == "structural"]
        random_m = pattern_report[pattern_report["classification"] == "random"]

        logger.info(f"\n  Structural missing: {len(structural)} columns")
        for _, row in structural.iterrows():
            logger.info(f"    {row['column']:30s} {row['overall_missing_pct']:5.1f}% -> {row['concentrated_group']} ({row['group_missing_pct']:.1f}%)")

        logger.info(f"\n  Random missing: {len(random_m)} columns")
        for _, row in random_m.iterrows():
            logger.info(f"    {row['column']:30s} {row['overall_missing_pct']:5.1f}%")
    else:
        logger.info("  No columns with missing >= 5%")

    logger.info("\n" + "=" * 60)
    logger.info("01_missing_check.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
