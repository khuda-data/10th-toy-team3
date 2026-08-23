"""
05_make_versions.py — 전처리 버전별 CSV 생성 (이상치 제거/미제거 × 스케일링)

이상치 탐지: IQR 방식 (Q1 - 1.5*IQR ~ Q3 + 1.5*IQR 범위 밖)
배당률 피처는 이상치 탐지 대상에서 제외 (이미 model_features에서 제외됨)

버전:
    [이상치 미제거]
    v1_base.csv         : 결측 처리 + 고상관 제거 (스케일링 없음)
    v2_standard.csv     : v1 + StandardScaler
    v3_minmax.csv       : v1 + MinMaxScaler
    v4_robust.csv       : v1 + RobustScaler

    [이상치 제거]
    v5_base_no_outlier.csv      : IQR 이상치 제거 + 결측 처리 + 고상관 제거
    v6_standard_no_outlier.csv  : v5 + StandardScaler
    v7_minmax_no_outlier.csv    : v5 + MinMaxScaler
    v8_robust_no_outlier.csv    : v5 + RobustScaler

실행:
    python src/1_전처리/05_make_versions.py

출력:
    data/versions/v1~v8.csv
    results/eda/outlier_report.html (이상치 탐지 보고서 + 그래프)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, CATEGORICAL_COLS,
    assign_time_split, SPLIT_RATIOS, setup_plot_style,
    EXCLUDE_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/versions")
REPORT_DIR = Path("results/eda")

# ============================================================
# 제거할 고상관 피처 (EDA 결과 기반)
# ============================================================
DROP_HIGH_CORR = [
    "chaksun2", "chaksun3", "chaksun4", "chaksun5",
    "buga2", "buga3",
    "dusu", "hr_style_n", "hr_prev_rating", "hr_last_finpct",
    "age__z", "train_runs_14__pr", "hr_last_wg",
    "wg__pr", "wg_diff__pr", "wgBudam__pr",
    "hr_winrate__pr", "hr_resid__pr",
    "jk_winrate__pr", "tr_winrate__pr",
    "hr_rest_days__pr", "bleed__pr", "rating__pr",
]


# ============================================================
# 결측치 처리
# ============================================================
def handle_missing(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    df = df.copy()
    structural_cols = [
        "rating", "rating__z",
        "hr_winrate", "hr_plcrate", "hr_resid", "hr_starts",
        "hr_winrate__z", "hr_resid__z",
        "hr_last_ord", "hr_last_poppct", "hr_last_resid",
        "hr_last_dist", "hr_dist_chg",
        "hr_rest_days", "hr_rest_days__z",
        "hr_dist_winrate", "hr_dist_starts",
        "hr_prev_rating",
        "hr_style", "hr_style_sd", "style_vs_race",
        "race_style_mean", "race_style_sd", "race_front_ratio",
        "jkhr_winrate", "wgBudam_chg",
    ]
    for col in structural_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    remaining = [c for c in num_cols if df[c].isnull().any()]
    if remaining:
        medians = df.loc[train_mask, remaining].median()
        df[remaining] = df[remaining].fillna(medians)

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING")

    return df


# ============================================================
# 고상관 피처 제거
# ============================================================
def drop_high_corr_features(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_HIGH_CORR if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"  Dropped {len(cols_to_drop)} high-correlation features")
    return df


# ============================================================
# 이상치 탐지 (IQR)
# ============================================================
def detect_outliers_iqr(df: pd.DataFrame, train_mask: pd.Series) -> dict:
    """수치형 피처의 IQR 이상치를 탐지. 배당률/식별자/타겟 제외.

    Returns:
        {col: {"q1", "q3", "iqr", "lower", "upper", "n_outliers", "pct"}}
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # 이상치 탐지 대상에서 제외할 컬럼
    skip = EXCLUDE_COLS | set(MARKET_COLS) | set(DUAL_MARKET_COLS)
    target_cols = [c for c in num_cols if c not in skip]

    train_df = df[train_mask]
    outlier_info = {}

    for col in target_cols:
        q1 = train_df[col].quantile(0.25)
        q3 = train_df[col].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue  # 분산 0인 컬럼은 스킵

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        is_outlier = (df[col] < lower) | (df[col] > upper)
        n_outliers = is_outlier.sum()

        if n_outliers > 0:
            outlier_info[col] = {
                "q1": q1, "q3": q3, "iqr": iqr,
                "lower": lower, "upper": upper,
                "n_outliers": n_outliers,
                "pct": n_outliers / len(df) * 100,
            }

    return outlier_info


def remove_outliers(df: pd.DataFrame, outlier_info: dict, train_mask: pd.Series) -> pd.DataFrame:
    """IQR 이상치 행을 제거 (너무 많이 제거되지 않도록 상위 이상치만).

    전략: 이상치 비율 30% 이상인 컬럼은 제거 대상에서 제외
    (경마 데이터 특성상 정상적인 분포가 넓은 컬럼이 있으므로)
    """
    mask = pd.Series(True, index=df.index)

    for col, info in outlier_info.items():
        if info["pct"] > 30:
            continue  # 30% 넘는 건 이상치가 아니라 분포 자체가 넓은 것

        lower = info["lower"]
        upper = info["upper"]
        mask &= (df[col] >= lower) & (df[col] <= upper)

    removed = (~mask).sum()
    logger.info(f"  Outlier removal: {removed:,} rows removed ({removed/len(df):.1%})")
    return df[mask].reset_index(drop=True)


# ============================================================
# 이상치 시각화 (boxplot)
# ============================================================
def save_outlier_plots(df: pd.DataFrame, outlier_info: dict):
    """이상치 상위 피처의 boxplot과 비율 그래프를 PNG로 저장."""
    setup_plot_style()

    sorted_cols = sorted(
        [(col, info) for col, info in outlier_info.items() if info["pct"] <= 30],
        key=lambda x: x[1]["n_outliers"],
        reverse=True,
    )[:12]

    if not sorted_cols:
        return

    # Boxplot
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    for i, (col, info) in enumerate(sorted_cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.boxplot(data, vert=True)
        ax.set_title(f"{col}\noutliers={info['n_outliers']:,} ({info['pct']:.1f}%)", fontsize=9)
        ax.axhline(info["upper"], color="red", linestyle="--", linewidth=0.8)
        ax.axhline(info["lower"], color="red", linestyle="--", linewidth=0.8)
        ax.tick_params(axis="x", labelbottom=False)
    for j in range(len(sorted_cols), len(axes)):
        fig.delaxes(axes[j])
    plt.suptitle("Outlier Detection (IQR) — Top 12 features", fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "outlier_boxplots.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: results/eda/outlier_boxplots.png")

    # 비율 막대 그래프
    fig, ax = plt.subplots(figsize=(12, 6))
    cols = [c for c, _ in sorted_cols]
    pcts = [info["pct"] for _, info in sorted_cols]
    ax.barh(range(len(cols)), pcts, color="coral")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Outlier %")
    ax.set_title("Outlier Rate by Feature (IQR method)")
    ax.axvline(5, color="gray", linestyle="--", linewidth=0.8, label="5%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "outlier_barplot.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: results/eda/outlier_barplot.png")


# ============================================================
# 스케일링
# ============================================================
def apply_scaling(df, train_mask, scaler_class, scaler_name):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scale_cols = [c for c in num_cols if c not in EXCLUDE_COLS and c != TARGET_COL]
    scaler = scaler_class()
    scaler.fit(df.loc[train_mask, scale_cols])
    df[scale_cols] = scaler.transform(df[scale_cols])
    logger.info(f"  Applied {scaler_name} to {len(scale_cols)} features")
    return df


# ============================================================
# 메인
# ============================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Making preprocessed versions (with/without outlier removal)")
    logger.info("=" * 60)

    # 데이터 로드
    input_path = Path("data/processed/model_features.csv")
    if not input_path.exists():
        logger.error(f"  {input_path} not found.")
        sys.exit(1)

    df = pd.read_csv(input_path, dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str})
    logger.info(f"  Loaded: {input_path} ({len(df):,} rows x {len(df.columns)} cols)")

    # fold 확인/재배정
    df = df.sort_values("rcDate").reset_index(drop=True)
    train_ratio = (df["fold"] == "train").mean() if "fold" in df.columns else 0
    if abs(train_ratio - SPLIT_RATIOS[0]) > 0.03:
        df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)

    train_mask = df["fold"] == "train"
    logger.info(f"  Split: train {train_mask.sum():,} | valid {(df['fold']=='valid').sum():,} | test {(df['fold']=='test').sum():,}")

    # 결측치 처리
    logger.info("\n  [Step 1] Missing value treatment")
    df = handle_missing(df, train_mask)

    # 고상관 제거
    logger.info("\n  [Step 2] Drop high-correlation features")
    df = drop_high_corr_features(df)

    # ====== 이상치 탐지 ======
    logger.info("\n  [Step 3] Outlier detection (IQR)")
    outlier_info = detect_outliers_iqr(df, train_mask)
    logger.info(f"  Features with outliers: {len(outlier_info)}")

    top5 = sorted(outlier_info.items(), key=lambda x: x[1]["n_outliers"], reverse=True)[:5]
    for col, info in top5:
        logger.info(f"    {col:25s} {info['n_outliers']:>5,} ({info['pct']:.1f}%)")

    # 이상치 시각화 + 저장
    save_outlier_plots(df, outlier_info)

    # ====== 이상치 미제거 트랙 (v1~v4) ======
    logger.info("\n  === Track A: WITH outliers ===")
    n_before = len(df)

    save_versions(df, train_mask, prefix="", suffix="")

    # ====== 이상치 제거 트랙 (v5~v8) ======
    logger.info("\n  === Track B: WITHOUT outliers (IQR removed) ===")
    df_clean = remove_outliers(df, outlier_info, train_mask)
    n_after = len(df_clean)

    # 제거 후 train_mask 재계산
    train_mask_clean = df_clean["fold"] == "train"

    save_versions(df_clean, train_mask_clean, prefix="v5", suffix="_no_outlier", start_num=5)

    # 보고서용 이상치 데이터 저장 (04_eda_report.py, 06_version_report.py에서 읽음)
    outlier_df = pd.DataFrame([
        {"feature": col, "n_outliers": info["n_outliers"], "pct": round(info["pct"], 2),
         "lower": round(info["lower"], 2), "upper": round(info["upper"], 2),
         "removed": info["pct"] <= 30}
        for col, info in sorted(outlier_info.items(), key=lambda x: x[1]["n_outliers"], reverse=True)
    ])
    outlier_df.to_csv(REPORT_DIR / "outlier_summary.csv", index=False)
    logger.info(f"  Saved: results/eda/outlier_summary.csv")

    # 이상치 boxplot 이미지 저장 (보고서에서 삽입)
    save_outlier_plots(df, outlier_info)

    # version_info.md 갱신
    write_version_info(df, df_clean)

    logger.info("\n" + "=" * 60)
    logger.info("All versions created!")
    logger.info("=" * 60)


def save_versions(df, train_mask, prefix="", suffix="", start_num=1):
    """4가지 스케일링 버전을 저장."""
    configs = [
        (f"v{start_num}_base{suffix}.csv", None, "No scaling"),
        (f"v{start_num+1}_standard{suffix}.csv", StandardScaler, "StandardScaler"),
        (f"v{start_num+2}_minmax{suffix}.csv", MinMaxScaler, "MinMaxScaler"),
        (f"v{start_num+3}_robust{suffix}.csv", RobustScaler, "RobustScaler"),
    ]

    for filename, scaler_cls, scaler_name in configs:
        if scaler_cls:
            out = apply_scaling(df, train_mask, scaler_cls, scaler_name)
        else:
            out = df.copy()
            logger.info(f"  No scaling")

        out.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")
        logger.info(f"  Saved: data/versions/{filename} ({out.shape})")


def write_version_info(df_full, df_clean):
    """버전 설명 마크다운."""
    info = f"""# Preprocessed Data Versions

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

## Versions

### With Outliers (v1~v4): {len(df_full):,} rows x {len(df_full.columns)} cols
| File | Scaling | Use |
|------|---------|-----|
| v1_base.csv | None | Tree models (RF, XGB) |
| v2_standard.csv | StandardScaler | Logistic, SVM |
| v3_minmax.csv | MinMaxScaler | KNN, K-means |
| v4_robust.csv | RobustScaler | Robust to outliers |

### Without Outliers (v5~v8): {len(df_clean):,} rows x {len(df_clean.columns)} cols
| File | Scaling | Use |
|------|---------|-----|
| v5_base_no_outlier.csv | None | Tree models (clean) |
| v6_standard_no_outlier.csv | StandardScaler | Logistic (clean) |
| v7_minmax_no_outlier.csv | MinMaxScaler | KNN (clean) |
| v8_robust_no_outlier.csv | RobustScaler | Comparison |

## Outlier Removal Method
- IQR: Q1 - 1.5*IQR to Q3 + 1.5*IQR (trained on train set only)
- Features with >30% outlier rate are excluded from removal (natural wide distribution)
- Rows removed: {len(df_full) - len(df_clean):,} ({(len(df_full)-len(df_clean))/len(df_full):.1%})

## Detailed report
- See: results/eda/outlier_report.html
"""
    (OUTPUT_DIR / "version_info.md").write_text(info, encoding="utf-8")
    logger.info(f"  Saved: data/versions/version_info.md")


if __name__ == "__main__":
    main()
