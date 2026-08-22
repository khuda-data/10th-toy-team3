"""
02_market_gap.py — 배당률 괴리 계산 + Feature Importance 재해석 (4~5단계)

4단계: test set 예측확률 vs 배당률 암묵적확률 → 괴리(gap) 계산
5단계: feature importance + 괴리 상위 서브셋 피처 분포 분석

실행:
    python src/pipeline/02_market_gap.py

의존:
    - 01_train_model.py 실행 후 생성된 results/models/ 파일들
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    EXCLUDE_COLS,
    CATEGORICAL_COLS,
    TARGET_COL,
    ID_COLS,
    get_feature_cols,
    setup_logging,
    setup_plot_style,
    ensure_dirs,
)

logger = setup_logging()


# ============================================================
# 유틸리티
# ============================================================

def load_model_and_encoders():
    """best 모델과 전처리 인코더를 로드한다."""
    with open(MODELS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "best_threshold.pkl", "rb") as f:
        threshold_info = pickle.load(f)
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, threshold_info, encoders


def apply_preprocessing(df: pd.DataFrame, encoders: dict) -> np.ndarray:
    """저장된 인코더를 사용하여 피처를 전처리한다."""
    feature_cols = encoders["feature_cols"]
    cat_cols = encoders["cat_cols"]
    num_cols = encoders["num_cols"]
    medians = encoders["medians"]
    label_encoders = encoders["label_encoders"]

    X_df = df[feature_cols].copy()

    # 수치형 결측 채움
    X_df[num_cols] = X_df[num_cols].fillna(medians)

    # 범주형 처리
    for col in cat_cols:
        X_df[col] = X_df[col].fillna("MISSING").astype(str)
        le = label_encoders[col]
        # 미지 값 처리: le.classes_에 없는 값은 'MISSING'으로 대체
        known = set(le.classes_)
        X_df[col] = X_df[col].apply(lambda x: x if x in known else "MISSING")
        X_df[col] = le.transform(X_df[col])

    return X_df.values.astype(np.float32)


# ============================================================
# 4단계: 배당률 괴리 계산
# ============================================================

def compute_market_gap(
    df_features: pd.DataFrame,
    df_odds: pd.DataFrame,
    model,
    encoders: dict,
) -> pd.DataFrame:
    """test set에서 모델 예측확률과 시장 암묵적확률의 괴리를 계산한다."""
    logger.info("=" * 60)
    logger.info("[4단계] 배당률 괴리 계산")
    logger.info("=" * 60)

    # test set만 추출
    test = df_features[df_features["fold"] == "test"].copy()
    logger.info(f"  test set: {len(test):,}행")

    # 예측확률 계산
    X_test = apply_preprocessing(test, encoders)
    model_prob = model.predict_proba(X_test)[:, 1]
    test["model_prob"] = model_prob

    # market_odds 조인
    odds = df_odds[["entry_id", "winOdds"]].copy()
    test = test.merge(odds, on="entry_id", how="left")

    # 경주 내 1/winOdds 정규화 → 시장 암묵적 확률
    test["inv_odds"] = 1.0 / test["winOdds"]
    test["market_prob"] = test.groupby("race_id")["inv_odds"].transform(
        lambda x: x / x.sum()
    )

    # 괴리 계산
    test["gap"] = test["model_prob"] - test["market_prob"]

    # 결과 정리
    result = test[["entry_id", "race_id", "hrName", "model_prob", "market_prob", "gap"]].copy()

    # 기술통계 출력
    logger.info(f"\n  괴리(gap) 기술통계:")
    logger.info(f"    mean:   {result['gap'].mean():.6f}")
    logger.info(f"    std:    {result['gap'].std():.6f}")
    logger.info(f"    min:    {result['gap'].min():.6f}")
    logger.info(f"    25%:    {result['gap'].quantile(0.25):.6f}")
    logger.info(f"    50%:    {result['gap'].quantile(0.50):.6f}")
    logger.info(f"    75%:    {result['gap'].quantile(0.75):.6f}")
    logger.info(f"    max:    {result['gap'].max():.6f}")

    return result


# ============================================================
# 5단계: Feature Importance 재해석
# ============================================================

def plot_feature_importance(model, encoders: dict):
    """전체 feature importance 상위 20개를 막대그래프로 저장한다."""
    logger.info("=" * 60)
    logger.info("[5단계] Feature Importance 재해석")
    logger.info("=" * 60)

    feature_cols = encoders["feature_cols"]

    # importance 추출 (트리 모델)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # 로지스틱 회귀의 경우 계수 절대값
        importances = np.abs(model.coef_[0])

    fi_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    # 상위 20개 그래프
    top20 = fi_df.head(20)

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top20)), top20["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close()

    logger.info(f"  저장: results/feature_importance.png")
    logger.info(f"\n  Top 10 피처:")
    for _, row in fi_df.head(10).iterrows():
        logger.info(f"    {row['feature']:30s} {row['importance']:.6f}")

    return fi_df


def analyze_gap_subset(
    gap_df: pd.DataFrame,
    df_features: pd.DataFrame,
    encoders: dict,
):
    """괴리 절대값 상위 20% 서브셋의 피처 분포를 전체와 비교한다."""
    logger.info("\n  --- 괴리 상위 20% 서브셋 피처 분석 ---")

    # 상위 20% 추출
    threshold = gap_df["gap"].abs().quantile(0.80)
    top_entries = gap_df[gap_df["gap"].abs() >= threshold]["entry_id"]
    logger.info(f"  |gap| >= {threshold:.4f} → {len(top_entries):,}건 (상위 20%)")

    # 원본 피처 가져오기
    test = df_features[df_features["fold"] == "test"].copy()
    feature_cols = encoders["feature_cols"]
    num_cols = encoders["num_cols"]

    # 수치형만 비교 (범주형은 평균 비교 의미 없음)
    compare_cols = [c for c in num_cols if c in test.columns]

    full_mean = test[compare_cols].mean()
    subset = test[test["entry_id"].isin(top_entries)]
    subset_mean = subset[compare_cols].mean()

    comparison = pd.DataFrame({
        "전체_평균": full_mean,
        "괴리상위20%_평균": subset_mean,
        "차이": subset_mean - full_mean,
    })
    comparison["차이_비율"] = comparison["차이"] / (comparison["전체_평균"].abs() + 1e-10)
    comparison = comparison.sort_values("차이_비율", key=abs, ascending=False)

    comparison.to_csv(RESULTS_DIR / "feature_comparison.csv")
    logger.info(f"  저장: results/feature_comparison.csv")

    # 상위 5개 피처 분포 비교 시각화
    top5_features = comparison.head(5).index.tolist()
    logger.info(f"  분포 비교 피처: {top5_features}")

    setup_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat in enumerate(top5_features):
        ax = axes[i]
        ax.hist(test[feat].dropna(), bins=30, alpha=0.5, label="All", density=True)
        ax.hist(subset[feat].dropna(), bins=30, alpha=0.5, label="Top 20% gap", density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)

    # 빈 서브플롯 제거
    for j in range(len(top5_features), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Feature Distribution: All vs Top 20% Gap", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_distribution.png", bbox_inches="tight")
    plt.close()

    logger.info(f"  저장: results/feature_distribution.png")


# ============================================================
# 메인 실행
# ============================================================

def main():
    ensure_dirs()

    # 모델 및 인코더 로드
    model, threshold_info, encoders = load_model_and_encoders()
    logger.info(f"모델 로드: {threshold_info['model_name']} (threshold={threshold_info['threshold']})")

    # 데이터 로드
    df_features = pd.read_csv(
        DATA_DIR / "model_features.csv",
        dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str},
    )
    df_odds = pd.read_csv(DATA_DIR / "market_odds.csv")

    # 4단계: 괴리 계산
    gap_df = compute_market_gap(df_features, df_odds, model, encoders)
    gap_df.to_csv(RESULTS_DIR / "market_gap.csv", index=False)
    logger.info(f"  저장: results/market_gap.csv ({len(gap_df):,}행)")

    # 5단계: Feature Importance
    plot_feature_importance(model, encoders)
    analyze_gap_subset(gap_df, df_features, encoders)

    logger.info("\n" + "=" * 60)
    logger.info("02_market_gap.py 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
