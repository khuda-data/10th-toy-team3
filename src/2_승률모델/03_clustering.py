"""
03_clustering.py — K-means 클러스터링 (6단계)

괴리 절대값 상위 20% 서브셋을 대상으로 K-means 수행.
최적 k를 실루엣 계수로 탐색하고, 군집별 피처 특성을 분석한다.

실행:
    python src/1_전처리/03_clustering.py

의존:
    - 02_market_gap.py 실행 후 생성된 results/market_gap.csv
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DATA_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    setup_logging,
    setup_plot_style,
    ensure_dirs,
)

logger = setup_logging()


# ============================================================
# 6단계: K-means 클러스터링
# ============================================================

def load_gap_subset() -> tuple[pd.DataFrame, list[str]]:
    """괴리 상위 20% 서브셋의 entry_id를 추출하고 피처를 준비한다."""
    logger.info("=" * 60)
    logger.info("[6단계] K-means 클러스터링")
    logger.info("=" * 60)

    # 괴리 데이터 로드
    gap_df = pd.read_csv(RESULTS_DIR / "market_gap.csv")
    threshold = gap_df["gap"].abs().quantile(0.80)
    top_entries = gap_df[gap_df["gap"].abs() >= threshold]["entry_id"].tolist()
    logger.info(f"  괴리 상위 20%: {len(top_entries):,}건 (|gap| >= {threshold:.4f})")

    # 피처 데이터 로드
    df_features = pd.read_csv(
        DATA_DIR / "model_features.csv",
        dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str},
    )

    # 인코더 로드
    with open(MODELS_DIR / "encoders.pkl", "rb") as f:
        encoders = pickle.load(f)

    num_cols = encoders["num_cols"]

    # test set에서 상위 20% 추출, 수치형 피처만
    test = df_features[df_features["fold"] == "test"].copy()
    subset = test[test["entry_id"].isin(top_entries)].copy()

    # 수치형 피처만 사용 (클러스터링에 범주형은 부적합)
    available_num = [c for c in num_cols if c in subset.columns]
    subset_num = subset[available_num].copy()

    # 결측 채움 (중앙값)
    subset_num = subset_num.fillna(subset_num.median())

    logger.info(f"  클러스터링 대상: {len(subset_num):,}행 × {len(available_num)}피처 (수치형)")

    return subset_num, available_num


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> tuple[int, list, list]:
    """실루엣 계수와 inertia로 최적 k를 탐색한다."""
    inertias = []
    silhouettes = []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouettes.append(sil)
        logger.info(f"    k={k}: inertia={km.inertia_:.0f}, silhouette={sil:.4f}")

    best_k = k_range[np.argmax(silhouettes)]
    logger.info(f"  최적 k: {best_k} (silhouette={max(silhouettes):.4f})")

    return best_k, inertias, silhouettes


def plot_elbow_silhouette(k_range: range, inertias: list, silhouettes: list):
    """엘보우 + 실루엣 그래프를 저장한다."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 엘보우
    ax1.plot(list(k_range), inertias, "bo-", linewidth=2)
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.set_xticks(list(k_range))

    # 실루엣
    ax2.plot(list(k_range), silhouettes, "rs-", linewidth=2)
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score")
    ax2.set_xticks(list(k_range))

    plt.suptitle("K-means Optimal k Search", fontsize=13)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kmeans_elbow.png", bbox_inches="tight")
    plt.close()

    logger.info(f"  저장: results/kmeans_elbow.png")


def analyze_clusters(
    X_df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
):
    """군집별 피처 평균을 분석하고 저장한다."""
    X_df = X_df.copy()
    X_df["cluster"] = labels

    # 군집별 피처 평균
    profiles = X_df.groupby("cluster")[feature_cols].mean()
    profiles.to_csv(RESULTS_DIR / "cluster_profiles.csv")
    logger.info(f"  저장: results/cluster_profiles.csv")

    # 군집 크기
    cluster_sizes = X_df["cluster"].value_counts().sort_index()
    logger.info(f"\n  --- 군집 요약 ---")
    for cluster_id, size in cluster_sizes.items():
        logger.info(f"  군집 {cluster_id}: {size:,}건")

    # 각 군집별 전체 평균 대비 차이가 큰 피처 상위 5개
    overall_mean = X_df[feature_cols].mean()

    for cluster_id in sorted(X_df["cluster"].unique()):
        cluster_mean = profiles.loc[cluster_id]
        diff = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
        top5 = diff.head(5)
        logger.info(f"\n  군집 {cluster_id} 특징 (전체 평균 대비 차이 큰 피처):")
        for feat, val in top5.items():
            direction = "↑" if cluster_mean[feat] > overall_mean[feat] else "↓"
            logger.info(f"    {feat:30s} {direction} (차이: {val:.4f})")


# ============================================================
# 메인 실행
# ============================================================

def main():
    ensure_dirs()

    # 데이터 준비
    subset_num, feature_cols = load_gap_subset()

    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset_num.values)

    # 최적 k 탐색
    k_range = range(2, 7)  # k = 2~6
    logger.info("\n  k 탐색 중...")
    best_k, inertias, silhouettes = find_optimal_k(X_scaled, k_range)

    # 시각화
    plot_elbow_silhouette(k_range, inertias, silhouettes)

    # 최종 클러스터링
    logger.info(f"\n  최적 k={best_k}로 최종 클러스터링 수행")
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    labels = km_final.fit_predict(X_scaled)

    # 군집 분석
    analyze_clusters(subset_num, labels, feature_cols)

    logger.info("\n" + "=" * 60)
    logger.info("03_clustering.py 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
