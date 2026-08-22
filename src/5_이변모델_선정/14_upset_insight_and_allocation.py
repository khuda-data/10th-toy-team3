"""
14_upset_insight_and_allocation.py

Part A: 이변마 공통점 분석 (인사이트 도출)
  - 이변마 vs 비이변마 피처 비교
  - K-means 군집화 (이변마 유형 분류)
  - 의사결정나무 (이변 규칙 추출)
  - 배당 구간별 이변마 프로필

Part B: 최적 베팅 배분 전략 탐색
  - Flat / 기대값 비례 / 구간별 차등 / Top-only
  - valid에서 최적 찾고 test에서 검증

실행:
    python src/5_이변모델_선정/14_upset_insight_and_allocation.py
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS,
    RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
    setup_plot_style, translate_feature_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_insights")

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
FULL_EXCLUDE = set(
    ID_COLS + OUTCOME_COLS + [TARGET_COL, "upset", "fold", "pop_pct"]
    + ODDS_DROP
    + ["winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
    + ["gap_h", "gap_d"]
)


def load_data():
    """데이터 로드 + 전처리 (12_final_strategy와 동일)."""
    df = pd.read_csv("race_entries.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)
    df["_winOdds"] = df["winOdds"].copy()
    df = df[df["pop_pct"] >= 0.5].reset_index(drop=True)

    drop_cols = [c for c in HIGH_CORR_DROP + ODDS_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, "rcDate", SPLIT_RATIOS)

    structural = [
        "rating", "rating__z", "hr_winrate", "hr_plcrate", "hr_resid",
        "hr_starts", "hr_winrate__z", "hr_resid__z",
        "hr_last_ord", "hr_last_poppct", "hr_last_resid",
        "hr_last_dist", "hr_dist_chg", "hr_rest_days", "hr_rest_days__z",
        "hr_dist_winrate", "hr_dist_starts",
        "hr_style", "hr_style_sd", "style_vs_race",
        "race_style_mean", "race_style_sd", "race_front_ratio",
        "jkhr_winrate", "wgBudam_chg",
    ]
    for col in structural:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    features = [c for c in df.columns if c not in FULL_EXCLUDE and c != "_winOdds"]
    train_mask = df["fold"] == "train"
    cat_cols = [c for c in features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    remaining = [c for c in num_cols if df[c].isnull().any()]
    if remaining:
        medians = df.loc[train_mask, remaining].median()
        df[remaining] = df[remaining].fillna(medians)

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    return df, features, num_cols


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    df, features, num_cols = load_data()
    train = df[df["fold"] == "train"]
    valid = df[df["fold"] == "valid"]
    test = df[df["fold"] == "test"]

    logger.info("=" * 70)
    logger.info("PART A: Upset Horse Insight Analysis")
    logger.info("=" * 70)

    # ====== A1: 이변마 vs 비이변마 피처 비교 ======
    logger.info("\n[A1] Upset vs Normal — Feature Comparison")

    # test set 기준
    upset_df = test[test["upset"] == 1]
    normal_df = test[test["upset"] == 0]
    logger.info(f"  Upset: {len(upset_df)} | Normal: {len(normal_df)}")

    # 수치형 피처만 비교
    num_feats = [c for c in num_cols if c in features]
    comparison = pd.DataFrame({
        "upset_mean": upset_df[num_feats].mean(),
        "normal_mean": normal_df[num_feats].mean(),
        "diff": upset_df[num_feats].mean() - normal_df[num_feats].mean(),
    })
    comparison["abs_diff"] = comparison["diff"].abs()
    comparison = comparison.sort_values("abs_diff", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "upset_vs_normal_comparison.csv")

    logger.info(f"\n  Top 10 features where upset horses differ most:")
    for i, (feat, row) in enumerate(comparison.head(10).iterrows(), 1):
        direction = "higher" if row["diff"] > 0 else "lower"
        logger.info(f"    {i:>2}. {feat:25s} upset is {direction} by {row['abs_diff']:.3f} std")

    # 시각화: Top 5 분포 비교
    top5 = comparison.head(5).index.tolist()
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, feat in zip(axes, top5):
        ax.hist(normal_df[feat].dropna(), bins=20, alpha=0.5, label="Normal", density=True)
        ax.hist(upset_df[feat].dropna(), bins=20, alpha=0.5, label="Upset", density=True)
        ax.set_title(feat, fontsize=9)
        ax.legend(fontsize=7)
    plt.suptitle("Feature Distribution: Upset vs Normal (Top 5 differences)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "upset_vs_normal_distribution.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: upset_vs_normal_distribution.png")

    # ====== A2: K-means 군집화 ======
    logger.info("\n[A2] K-means Clustering of Upset Horses")

    # train+valid+test 전체에서 이변마 추출 (더 많은 표본)
    all_upset = df[df["upset"] == 1]
    logger.info(f"  Total upset horses (all folds): {len(all_upset)}")

    X_upset = all_upset[num_feats].values.astype(np.float32)

    # k 탐색
    sil_scores = []
    inertias = []
    k_range = range(2, 7)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_upset)
        sil = silhouette_score(X_upset, labels)
        sil_scores.append(sil)
        inertias.append(km.inertia_)
        logger.info(f"    k={k}: silhouette={sil:.4f}")

    best_k = list(k_range)[np.argmax(sil_scores)]
    logger.info(f"  Best k: {best_k} (silhouette={max(sil_scores):.4f})")

    # 최종 군집화
    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    all_upset_copy = all_upset.copy()
    all_upset_copy["cluster"] = km_final.fit_predict(X_upset)

    # 군집별 프로필
    profiles = all_upset_copy.groupby("cluster")[num_feats].mean()
    overall_mean = all_upset_copy[num_feats].mean()

    logger.info(f"\n  --- Cluster Profiles ---")
    for cluster_id in range(best_k):
        cluster_size = (all_upset_copy["cluster"] == cluster_id).sum()
        cluster_mean = profiles.loc[cluster_id]
        diff = (cluster_mean - overall_mean).abs().sort_values(ascending=False)
        top3 = diff.head(3)

        logger.info(f"\n  Cluster {cluster_id} ({cluster_size} horses):")
        for feat, val in top3.items():
            direction = "UP" if cluster_mean[feat] > overall_mean[feat] else "DOWN"
            logger.info(f"    {feat:25s} {direction} ({val:.3f} from mean)")

    profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv")

    # 엘보우/실루엣 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_xlabel("k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax2.plot(list(k_range), sil_scores, "rs-")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Score")
    plt.suptitle("K-means: Optimal k for Upset Horses")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kmeans_elbow.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ====== A3: 의사결정나무 (규칙 추출) ======
    logger.info("\n[A3] Decision Tree — Rule Extraction")

    X_train_dt = train[features].values.astype(np.float32)
    y_train_dt = train["upset"].values

    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=RANDOM_STATE)
    dt.fit(X_train_dt, y_train_dt)

    # 규칙 텍스트
    tree_rules = export_text(dt, feature_names=features, max_depth=4)
    logger.info(f"\n  Decision Tree Rules (depth=4):\n{tree_rules[:2000]}")

    # 저장
    with open(OUTPUT_DIR / "decision_tree_rules.txt", "w", encoding="utf-8") as f:
        f.write(tree_rules)

    # test AUC
    dt_proba = dt.predict_proba(test[features].values.astype(np.float32))[:, 1]
    dt_auc = roc_auc_score(test["upset"].values, dt_proba)
    logger.info(f"\n  Decision Tree test AUC: {dt_auc:.4f} (RF was 0.742)")
    logger.info(f"  (DT is worse but interpretable — rules above show 'why')")

    # ====== A4: 배당 구간별 이변마 프로필 ======
    logger.info("\n[A4] Upset Horse Profile by Odds Segment")

    all_upset_copy["odds_seg"] = pd.cut(
        all_upset_copy["_winOdds"],
        bins=[0, 15, 30, 50, 99999],
        labels=["10-15x", "15-30x", "30-50x", "50x+"]
    )

    seg_profiles = all_upset_copy.groupby("odds_seg")[num_feats].mean()
    top_feats = comparison.head(5).index.tolist()

    logger.info(f"\n  Top 5 features by odds segment (upset horses only):")
    logger.info(f"  {'Feature':<25s} {'10-15x':>8s} {'15-30x':>8s} {'30-50x':>8s} {'50x+':>8s}")
    for feat in top_feats:
        vals = [f"{seg_profiles.loc[seg, feat]:.3f}" if seg in seg_profiles.index else "N/A"
                for seg in ["10-15x", "15-30x", "30-50x", "50x+"]]
        logger.info(f"  {feat:<25s} {vals[0]:>8s} {vals[1]:>8s} {vals[2]:>8s} {vals[3]:>8s}")

    seg_profiles.to_csv(OUTPUT_DIR / "upset_profile_by_odds.csv")

    # ====================================================================
    logger.info("\n\n" + "=" * 70)
    logger.info("PART B: Optimal Bet Allocation Strategy")
    logger.info("=" * 70)

    # 모델 학습 (C model RF)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(train[features].values.astype(np.float32), train["upset"].values)

    # valid/test 예측
    valid_df = valid.copy()
    test_df = test.copy()
    valid_df["proba"] = model.predict_proba(valid[features].values.astype(np.float32))[:, 1]
    test_df["proba"] = model.predict_proba(test[features].values.astype(np.float32))[:, 1]

    # 배당 필터 (10~50배)
    valid_filtered = valid_df[(valid_df["_winOdds"] >= 10) & (valid_df["_winOdds"] < 50)]
    test_filtered = test_df[(test_df["_winOdds"] >= 10) & (test_df["_winOdds"] < 50)]

    logger.info(f"\n  Valid (10-50x): {len(valid_filtered)} | Test (10-50x): {len(test_filtered)}")

    # ====== B1: 베팅 전략 비교 ======
    logger.info("\n[B1] Allocation Strategy Comparison")

    TOTAL_CAPITAL = 1000  # 1000 units total

    def evaluate_strategy(df_sel, strategy_name, weights):
        """주어진 가중치로 ROI 계산."""
        total_bet = weights.sum()
        returns = weights * df_sel["upset"].values * df_sel["_winOdds"].values
        total_return = returns.sum()
        roi = (total_return - total_bet) / total_bet * 100
        return roi

    # 상위 10% 선택
    for dataset_name, dataset in [("valid", valid_filtered), ("test", test_filtered)]:
        n_bets = max(1, int(len(dataset) * 0.1))
        top = dataset.nlargest(n_bets, "proba").copy()

        strategies = {}

        # 1. Flat
        w_flat = np.ones(n_bets) * (TOTAL_CAPITAL / n_bets)
        strategies["Flat (equal)"] = w_flat

        # 2. 기대값 비례: bet ~ proba * odds (expected return)
        ev = top["proba"].values * top["_winOdds"].values
        w_ev = ev / ev.sum() * TOTAL_CAPITAL
        strategies["EV-proportional"] = w_ev

        # 3. 확률 비례: bet ~ proba
        w_prob = top["proba"].values / top["proba"].values.sum() * TOTAL_CAPITAL
        strategies["Prob-proportional"] = w_prob

        # 4. 구간별 차등: 상위 1/3에 50%, 중간 1/3에 30%, 하위 1/3에 20%
        n3 = n_bets // 3
        w_tiered = np.zeros(n_bets)
        w_tiered[:n3] = (TOTAL_CAPITAL * 0.5) / max(n3, 1)
        w_tiered[n3:2*n3] = (TOTAL_CAPITAL * 0.3) / max(n3, 1)
        w_tiered[2*n3:] = (TOTAL_CAPITAL * 0.2) / max(n_bets - 2*n3, 1)
        strategies["Tiered (50/30/20)"] = w_tiered

        # 5. Top-heavy: 상위 20%에 70%, 나머지에 30%
        n_top = max(1, n_bets // 5)
        w_heavy = np.zeros(n_bets)
        w_heavy[:n_top] = (TOTAL_CAPITAL * 0.7) / n_top
        w_heavy[n_top:] = (TOTAL_CAPITAL * 0.3) / max(n_bets - n_top, 1)
        strategies["Top-heavy (70/30)"] = w_heavy

        # 6. Kelly-inspired: bet ~ max(0, proba * odds - 1) / (odds - 1)
        p = top["proba"].values
        b = top["_winOdds"].values
        kelly = np.clip((p * b - 1) / (b - 1), 0, None)
        if kelly.sum() > 0:
            w_kelly = kelly / kelly.sum() * TOTAL_CAPITAL
        else:
            w_kelly = w_flat.copy()
        strategies["Kelly-inspired"] = w_kelly

        logger.info(f"\n  [{dataset_name.upper()}] Top 10% = {n_bets} bets, capital = {TOTAL_CAPITAL} units")
        logger.info(f"  {'Strategy':<25s} {'ROI':>8s} {'Max bet':>10s} {'Min bet':>10s}")
        logger.info(f"  {'-'*55}")

        for name, weights in strategies.items():
            roi = evaluate_strategy(top, name, weights)
            logger.info(f"  {name:<25s} {roi:>+7.1f}% {weights.max():>10.2f} {weights.min():>10.2f}")

    # ====== B2: valid에서 최적 → test 검증 ======
    logger.info("\n[B2] Best strategy on valid → verify on test")

    # valid에서 각 전략 ROI
    n_v = max(1, int(len(valid_filtered) * 0.1))
    top_v = valid_filtered.nlargest(n_v, "proba").copy()

    best_strat = None
    best_roi_v = -999

    for name, make_weights in [
        ("Flat", lambda n, p, o: np.ones(n)),
        ("EV-proportional", lambda n, p, o: (p * o) / (p * o).sum() * n),
        ("Prob-proportional", lambda n, p, o: p / p.sum() * n),
        ("Tiered", lambda n, p, o: np.array([1.5]*max(1,n//3) + [1.0]*max(1,n//3) + [0.5]*(n - 2*max(1,n//3)))),
        ("Kelly-inspired", lambda n, p, o: np.clip((p*o-1)/(o-1), 0, None) / max(np.clip((p*o-1)/(o-1), 0, None).sum(), 1e-8) * n),
    ]:
        p = top_v["proba"].values
        o = top_v["_winOdds"].values
        w = make_weights(n_v, p, o)
        if w.sum() == 0:
            w = np.ones(n_v)
        ret = (w * top_v["upset"].values * top_v["_winOdds"].values).sum()
        roi_v = (ret - w.sum()) / w.sum() * 100

        if roi_v > best_roi_v:
            best_roi_v = roi_v
            best_strat = name

    logger.info(f"  Best strategy on valid: {best_strat} (ROI={best_roi_v:+.1f}%)")

    # test 검증
    n_t = max(1, int(len(test_filtered) * 0.1))
    top_t = test_filtered.nlargest(n_t, "proba").copy()
    p_t = top_t["proba"].values
    o_t = top_t["_winOdds"].values

    # 재계산 (best)
    if best_strat == "Flat":
        w_best = np.ones(n_t)
    elif best_strat == "EV-proportional":
        ev = p_t * o_t
        w_best = ev / ev.sum() * n_t
    elif best_strat == "Prob-proportional":
        w_best = p_t / p_t.sum() * n_t
    elif best_strat == "Tiered":
        w_best = np.array([1.5]*max(1,n_t//3) + [1.0]*max(1,n_t//3) + [0.5]*(n_t - 2*max(1,n_t//3)))
    else:  # Kelly
        kelly = np.clip((p_t * o_t - 1) / (o_t - 1), 0, None)
        w_best = kelly / max(kelly.sum(), 1e-8) * n_t

    ret_t = (w_best * top_t["upset"].values * top_t["_winOdds"].values).sum()
    roi_t = (ret_t - w_best.sum()) / w_best.sum() * 100

    # Flat 비교
    w_flat_t = np.ones(n_t)
    ret_flat = (w_flat_t * top_t["upset"].values * top_t["_winOdds"].values).sum()
    roi_flat_t = (ret_flat - n_t) / n_t * 100

    logger.info(f"\n  TEST verification:")
    logger.info(f"    Flat ROI:          {roi_flat_t:+.1f}%")
    logger.info(f"    {best_strat} ROI:  {roi_t:+.1f}%")
    logger.info(f"    Improvement:       {roi_t - roi_flat_t:+.1f}%p")

    # Bootstrap CI for best
    np.random.seed(RANDOM_STATE)
    boot_rois = []
    upsets = top_t["upset"].values
    odds = top_t["_winOdds"].values
    for _ in range(1000):
        idx = np.random.choice(n_t, size=n_t, replace=True)
        ret = (w_best[idx] * upsets[idx] * odds[idx]).sum()
        boot_rois.append((ret - w_best[idx].sum()) / w_best[idx].sum() * 100)

    ci = np.percentile(boot_rois, [2.5, 97.5])
    p_profit = (np.array(boot_rois) > 0).mean() * 100

    logger.info(f"    95% CI: [{ci[0]:+.1f}%, {ci[1]:+.1f}%]")
    logger.info(f"    P(profit): {p_profit:.1f}%")

    # ====== 최종 요약 ======
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n  [Insight] Upset horses differ most in:")
    for i, (feat, row) in enumerate(comparison.head(5).iterrows(), 1):
        direction = "higher" if row["diff"] > 0 else "lower"
        logger.info(f"    {i}. {feat} ({translate_feature_name(feat)}) — {direction}")

    logger.info(f"\n  [Clustering] {best_k} types of upset horses identified")
    logger.info(f"\n  [Best Strategy] {best_strat} on winOdds 10-50x, top 10%")
    logger.info(f"    ROI: {roi_t:+.1f}% | CI: [{ci[0]:+.1f}%, {ci[1]:+.1f}%] | P(profit): {p_profit:.1f}%")
    logger.info(f"    vs Flat: {roi_flat_t:+.1f}% (improvement: {roi_t - roi_flat_t:+.1f}%p)")

    logger.info("\n" + "=" * 70)
    logger.info("DONE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
