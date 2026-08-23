"""
02_correlation_check.py — 상관관계 히트맵 + 고상관 쌍 추출 + VIF (3~4단계)

3단계: 수치형 피처 상관행렬 + 히트맵 + |r|>=0.8 쌍 추출
4단계: (--vif 옵션) 고상관 피처 대상 VIF 계산

실행:
    python src/1_전처리/02_correlation_check.py
    python src/1_전처리/02_correlation_check.py --vif
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, setup_plot_style,
)

# ── 저장소 어디서 실행해도 원천 데이터를 찾도록 ─────────────
#    src/<단계폴더>/<script>.py 이므로 parents[2] 가 저장소 루트
_REPO_ROOT = Path(__file__).resolve().parents[2]
RACE_ENTRIES = _REPO_ROOT / "data" / "race_entries.csv.gz"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/eda")

# 제외할 컬럼 (식별자 + 시장 + 결과 + 타겟)
EXCLUDE_FROM_CORR = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])


# ============================================================
# 3단계: 상관관계 히트맵
# ============================================================

def get_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """수치형 피처만 추출 (제외 컬럼 제거)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in EXCLUDE_FROM_CORR]
    return df[feature_cols]


def plot_full_heatmap(corr: pd.DataFrame):
    """전체 상관행렬 히트맵."""
    setup_plot_style()
    n = len(corr)
    fig_size = max(16, n * 0.2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0, ax=ax,
        cbar_kws={"shrink": 0.5},
        xticklabels=True, yticklabels=True,
    )
    ax.set_title(f"Correlation Heatmap (all {n} numeric features)", fontsize=14)
    ax.tick_params(axis="both", labelsize=5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", bbox_inches="tight", dpi=100)
    plt.close()
    logger.info(f"  Saved: results/eda/correlation_heatmap.png ({n}x{n})")


def plot_top40_heatmap(corr: pd.DataFrame):
    """상관계수 절대값 평균 상위 40개 피처만 히트맵."""
    # 각 피처의 평균 |상관계수| (자기 자신 제외)
    mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
    top40 = mean_abs_corr.head(40).index.tolist()

    corr_top = corr.loc[top40, top40]

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_top, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=False, square=True, linewidths=0.3, ax=ax,
        cbar_kws={"shrink": 0.7},
    )
    ax.set_title("Correlation Heatmap (Top 40 features by avg |r|)", fontsize=13)
    ax.tick_params(axis="both", labelsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap_top40.png", bbox_inches="tight", dpi=150)
    plt.close()
    logger.info(f"  Saved: results/eda/correlation_heatmap_top40.png")


def extract_high_correlation_pairs(
    corr: pd.DataFrame, df: pd.DataFrame, threshold: float = 0.8
) -> pd.DataFrame:
    """상관계수 절대값 >= threshold인 피처 쌍을 추출."""
    pairs = []
    cols = corr.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) >= threshold:
                c1, c2 = cols[i], cols[j]
                m1 = df[c1].isnull().mean() * 100
                m2 = df[c2].isnull().mean() * 100
                more_missing = c1 if m1 > m2 else (c2 if m2 > m1 else "same")

                pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": round(r, 4),
                    "feature_1_missing_pct": round(m1, 2),
                    "feature_2_missing_pct": round(m2, 2),
                    "more_missing_side": more_missing,
                })

    result = pd.DataFrame(pairs).sort_values("correlation", key=abs, ascending=False)
    return result.reset_index(drop=True)


# ============================================================
# 4단계: VIF (선택)
# ============================================================

def compute_vif(df_numeric: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
    """
    고상관 쌍에 포함된 피처만 대상으로 VIF 계산.

    # 참고: VIF는 교재 범위 밖의 보조 지표입니다.
    # "파이썬 머신러닝 완벽가이드"에서는 다루지 않으며, 참고용으로만 제공합니다.
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        logger.warning("  statsmodels not installed. Skipping VIF.")
        return pd.DataFrame()

    # 대상 피처만 추출 + 결측 제거
    cols = [c for c in target_cols if c in df_numeric.columns]
    sub = df_numeric[cols].dropna()

    if len(sub) < 10 or len(cols) < 2:
        logger.warning("  Not enough data for VIF calculation.")
        return pd.DataFrame()

    logger.info(f"  VIF calculation: {len(cols)} features, {len(sub):,} rows (after dropna)")

    vif_data = []
    X = sub.values.astype(np.float64)

    for i, col in enumerate(cols):
        try:
            vif = variance_inflation_factor(X, i)
            vif_data.append({"feature": col, "VIF": round(vif, 2)})
        except Exception:
            vif_data.append({"feature": col, "VIF": np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values("VIF", ascending=False).reset_index(drop=True)
    return vif_df


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EDA: Correlation check")
    parser.add_argument("--input", default=str(RACE_ENTRIES), help="입력 CSV 경로 (기본: data/race_entries.csv.gz)")
    parser.add_argument("--vif", action="store_true", help="Include VIF analysis (beyond textbook)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[3 Stage] Correlation Analysis")
    logger.info("=" * 60)

    # 데이터 로드 + 서울 필터링
    df = pd.read_csv(args.input, low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    logger.info(f"  Seoul data: {len(df):,} rows x {len(df.columns)} cols")

    # 수치형 피처 추출
    df_numeric = get_numeric_features(df)
    logger.info(f"  Numeric features (excl. ID/market/outcome): {len(df_numeric.columns)}")

    # 상관행렬
    corr = df_numeric.corr()

    # 히트맵
    plot_full_heatmap(corr)
    plot_top40_heatmap(corr)

    # 고상관 쌍 추출
    pairs = extract_high_correlation_pairs(corr, df, threshold=0.8)
    pairs.to_csv(OUTPUT_DIR / "high_correlation_pairs.csv", index=False)
    logger.info(f"  High correlation pairs (|r|>=0.8): {len(pairs)}")
    logger.info(f"  Saved: results/eda/high_correlation_pairs.csv")

    if len(pairs) > 0:
        logger.info(f"\n  --- Top 10 High Correlation Pairs ---")
        for _, row in pairs.head(10).iterrows():
            logger.info(
                f"    {row['feature_1']:25s} <-> {row['feature_2']:25s}  r={row['correlation']:.4f}"
            )

    # 4단계: VIF (선택)
    if args.vif:
        logger.info("\n" + "=" * 60)
        logger.info("[4 Stage] VIF (supplementary - beyond textbook scope)")
        logger.info("  # Note: VIF is NOT covered in the textbook.")
        logger.info("  # This is a supplementary reference only.")
        logger.info("=" * 60)

        # 고상관 쌍에 포함된 피처만
        vif_targets = list(set(pairs["feature_1"].tolist() + pairs["feature_2"].tolist()))
        logger.info(f"  VIF target features: {len(vif_targets)}")

        vif_df = compute_vif(df_numeric, vif_targets)

        if not vif_df.empty:
            vif_df.to_csv(OUTPUT_DIR / "vif_report.csv", index=False)
            logger.info(f"  Saved: results/eda/vif_report.csv")

            high_vif = vif_df[vif_df["VIF"] >= 10]
            logger.info(f"  VIF >= 10: {len(high_vif)} features")
            for _, row in high_vif.head(15).iterrows():
                logger.info(f"    {row['feature']:30s} VIF={row['VIF']:.1f}")

    logger.info("\n" + "=" * 60)
    logger.info("02_correlation_check.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
