"""
11_improvement_experiments.py — 이변 모델 6단계 개선 실험

각 단계를 순차 실행하고 결과를 출력한다.
1단계: place 타겟 + plcOdds ROI
2단계: min_samples_leaf=50 + balanced 제거
3단계: 고배당 제거 로버스트니스
4단계: 배당 구간 필터 (10배+ only)
5단계: 피처 추가 (hr_trend_3, jk_recent_form)
6단계: 붕괴+다크호스 조합

실행:
    python src/5_이변모델_선정/11_improvement_experiments.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sklearn.base

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS,
    RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
    EXCLUDE_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_improvements")

# 배당률 14개 중 q만 남기고 나머지 제거
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
    ID_COLS + OUTCOME_COLS + [TARGET_COL, "upset", "upset_place", "fold", "pop_pct"]
    + ODDS_DROP
    + ["winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
    + ["gap_h", "gap_d"]
)


def load_and_prepare():
    """데이터 로드 + 공통 전처리."""
    df = pd.read_csv("race_entries.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)

    # 타겟 생성
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)
    df["upset_place"] = ((df["pop_pct"] >= 0.5) & (df["place"] == 1)).astype(int)

    # 배당 보존 (ROI 계산용)
    df["_winOdds"] = df["winOdds"].copy()
    df["_plcOdds"] = df["plcOdds"].copy()

    # 비인기마만
    df = df[df["pop_pct"] >= 0.5].reset_index(drop=True)

    # 고상관 제거
    drop_cols = [c for c in HIGH_CORR_DROP + ODDS_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 분할
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)

    # 결측치
    structural = [
        "rating", "rating__z", "hr_winrate", "hr_plcrate", "hr_resid",
        "hr_starts", "hr_winrate__z", "hr_resid__z",
        "hr_last_ord", "hr_last_poppct", "hr_last_resid",
        "hr_last_dist", "hr_dist_chg", "hr_rest_days", "hr_rest_days__z",
        "hr_dist_winrate", "hr_dist_starts", "hr_prev_rating",
        "hr_style", "hr_style_sd", "style_vs_race",
        "race_style_mean", "race_style_sd", "race_front_ratio",
        "jkhr_winrate", "wgBudam_chg",
    ]
    for col in structural:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    train_mask = df["fold"] == "train"
    # 피처
    features = [c for c in df.columns if c not in FULL_EXCLUDE and c != "_winOdds" and c != "_plcOdds"]
    cat_cols = [c for c in features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]

    # 남은 결측
    remaining = [c for c in num_cols if df[c].isnull().any()]
    if remaining:
        medians = df.loc[train_mask, remaining].median()
        df[remaining] = df[remaining].fillna(medians)

    # 인코딩
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])

    # 스케일링
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    return df, features


def calc_roi(test_df, proba_col, odds_col, target_col, top_pct=20):
    """상위 N%의 ROI 계산."""
    n = max(1, int(len(test_df) * top_pct / 100))
    top = test_df.nlargest(n, proba_col)
    total_return = (top[target_col] * top[odds_col]).sum()
    roi = (total_return - n) / n * 100
    hit_rate = top[target_col].mean()
    return roi, hit_rate, n


def bootstrap_roi(test_df, proba_col, odds_col, target_col, top_pct=20, n_boot=500):
    """Bootstrap ROI CI."""
    n = max(1, int(len(test_df) * top_pct / 100))
    top = test_df.nlargest(n, proba_col)
    upsets = top[target_col].values
    odds = top[odds_col].values

    rois = []
    np.random.seed(RANDOM_STATE)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        ret = (upsets[idx] * odds[idx]).sum()
        rois.append((ret - n) / n * 100)

    ci = np.percentile(rois, [2.5, 97.5])
    p_profit = (np.array(rois) > 0).mean() * 100
    return ci, p_profit


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("IMPROVEMENT EXPERIMENTS (6 stages)")
    logger.info("=" * 70)

    df, features = load_and_prepare()
    train = df[df["fold"] == "train"]
    valid = df[df["fold"] == "valid"]
    test = df[df["fold"] == "test"]

    logger.info(f"Data: train={len(train):,} | valid={len(valid):,} | test={len(test):,}")
    logger.info(f"Features: {len(features)}")
    logger.info(f"upset(1착) rate: {test['upset'].mean():.4f} | upset_place(입상) rate: {test['upset_place'].mean():.4f}")

    X_train = train[features].values.astype(np.float32)
    X_test = test[features].values.astype(np.float32)

    all_results = []

    # ==================== STAGE 1 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[STAGE 1] Place target + plcOdds ROI")
    logger.info("=" * 70)

    # 기존 (단승)
    rf_base = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_base.fit(X_train, train["upset"].values)
    proba_win = rf_base.predict_proba(X_test)[:, 1]
    auc_win = roc_auc_score(test["upset"].values, proba_win)

    test_df = test.copy()
    test_df["proba_win"] = proba_win
    roi_win, hit_win, n_win = calc_roi(test_df, "proba_win", "_winOdds", "upset", 20)

    # 새로운 (연승/입상)
    rf_place = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_place.fit(X_train, train["upset_place"].values)
    proba_place = rf_place.predict_proba(X_test)[:, 1]
    auc_place = roc_auc_score(test["upset_place"].values, proba_place)

    test_df["proba_place"] = proba_place
    roi_place, hit_place, n_place = calc_roi(test_df, "proba_place", "_plcOdds", "upset_place", 20)

    ci_win, pp_win = bootstrap_roi(test_df, "proba_win", "_winOdds", "upset", 20)
    ci_place, pp_place = bootstrap_roi(test_df, "proba_place", "_plcOdds", "upset_place", 20)

    logger.info(f"\n  {'':30s} {'단승(win)':>15s} {'연승(place)':>15s}")
    logger.info(f"  {'AUC':30s} {auc_win:>15.4f} {auc_place:>15.4f}")
    logger.info(f"  {'Top 20% Hit Rate':30s} {hit_win:>15.4f} {hit_place:>15.4f}")
    logger.info(f"  {'Top 20% ROI':30s} {roi_win:>+14.1f}% {roi_place:>+14.1f}%")
    logger.info(f"  {'95% CI':30s} [{ci_win[0]:+.1f}%, {ci_win[1]:+.1f}%] [{ci_place[0]:+.1f}%, {ci_place[1]:+.1f}%]")
    logger.info(f"  {'P(profit)':30s} {pp_win:>14.1f}% {pp_place:>14.1f}%")

    all_results.append({"stage": "1_place_target", "target": "win", "AUC": auc_win, "ROI_20": roi_win, "CI_low": ci_win[0], "CI_high": ci_win[1], "P_profit": pp_win})
    all_results.append({"stage": "1_place_target", "target": "place", "AUC": auc_place, "ROI_20": roi_place, "CI_low": ci_place[0], "CI_high": ci_place[1], "P_profit": pp_place})

    # ==================== STAGE 2 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[STAGE 2] min_samples_leaf=50 + class_weight=None")
    logger.info("=" * 70)

    rf_tuned = RandomForestClassifier(
        n_estimators=600, max_depth=12, min_samples_leaf=50,
        class_weight=None, random_state=RANDOM_STATE, n_jobs=-1)
    rf_tuned.fit(X_train, train["upset_place"].values)
    proba_tuned = rf_tuned.predict_proba(X_test)[:, 1]
    auc_tuned = roc_auc_score(test["upset_place"].values, proba_tuned)

    test_df["proba_tuned"] = proba_tuned
    roi_tuned, hit_tuned, _ = calc_roi(test_df, "proba_tuned", "_plcOdds", "upset_place", 20)
    ci_tuned, pp_tuned = bootstrap_roi(test_df, "proba_tuned", "_plcOdds", "upset_place", 20)

    logger.info(f"  Before (leaf=20, balanced): AUC={auc_place:.4f} ROI={roi_place:+.1f}% P(profit)={pp_place:.1f}%")
    logger.info(f"  After  (leaf=50, None):     AUC={auc_tuned:.4f} ROI={roi_tuned:+.1f}% P(profit)={pp_tuned:.1f}%")
    logger.info(f"  Change:                     AUC {auc_tuned-auc_place:+.4f} ROI {roi_tuned-roi_place:+.1f}%p")

    all_results.append({"stage": "2_tuned_rf", "target": "place", "AUC": auc_tuned, "ROI_20": roi_tuned, "CI_low": ci_tuned[0], "CI_high": ci_tuned[1], "P_profit": pp_tuned})

    # ==================== STAGE 3 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[STAGE 3] Robustness — remove top N high-odds wins")
    logger.info("=" * 70)

    # 사용할 모델: stage 2의 tuned 모델
    best_proba = "proba_tuned"
    n_top = max(1, int(len(test_df) * 0.2))
    top20 = test_df.nlargest(n_top, best_proba)

    # 적중마만 배당 높은 순 정렬
    winners = top20[top20["upset_place"] == 1].sort_values("_plcOdds", ascending=False)
    base_roi = roi_tuned

    logger.info(f"  Top 20% bets: {n_top} | Wins: {len(winners)} | Top winner odds: {winners['_plcOdds'].iloc[0]:.1f}x")
    logger.info(f"\n  {'Removed':>8s} {'ROI':>10s} {'Change':>10s} {'Still positive?':>16s}")

    robustness = []
    for n_remove in [0, 1, 3, 5, 10]:
        if n_remove >= len(winners):
            break
        remaining = top20.drop(winners.index[:n_remove])
        ret = (remaining["upset_place"] * remaining["_plcOdds"]).sum()
        n_bets = len(remaining)
        roi_r = (ret - n_bets) / n_bets * 100
        positive = "YES" if roi_r > 0 else "NO"
        logger.info(f"  {n_remove:>8d} {roi_r:>+9.1f}% {roi_r - base_roi:>+9.1f}%p {positive:>16s}")
        robustness.append({"removed": n_remove, "ROI": roi_r, "positive": roi_r > 0})

    pd.DataFrame(robustness).to_csv(OUTPUT_DIR / "robustness.csv", index=False)

    # ==================== STAGE 4 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[STAGE 4] Odds segment filter (10x+ only)")
    logger.info("=" * 70)

    for seg_name, lo, hi in [("All", 0, 99999), ("5x+", 5, 99999), ("10x+", 10, 99999), ("20x+", 20, 99999)]:
        seg = test_df[(test_df["_plcOdds"] >= lo) & (test_df["_plcOdds"] < hi)]
        if len(seg) < 20:
            continue
        n_seg = max(1, int(len(seg) * 0.2))
        top_seg = seg.nlargest(n_seg, best_proba)
        ret = (top_seg["upset_place"] * top_seg["_plcOdds"]).sum()
        roi_seg = (ret - n_seg) / n_seg * 100
        hit_seg = top_seg["upset_place"].mean()
        logger.info(f"  {seg_name:>5s}: n={len(seg):>5} | top20%={n_seg:>4} | hit={hit_seg:.3f} | ROI={roi_seg:+.1f}%")

    # ==================== STAGE 5 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[STAGE 5] Feature engineering (hr_trend_3, jk_recent_form)")
    logger.info("=" * 70)
    logger.info("  Note: These features need to be computed from raw data with")
    logger.info("  expanding window. Currently not available in race_entries.csv.")
    logger.info("  Skipping — requires separate feature pipeline.")
    logger.info("  (Would need rolling(3) on hr_last_ord grouped by hrNo)")

    # ==================== STAGE 6 ====================
    logger.info("\n" + "=" * 70)
    logger.info("[STAGE 6] Collapse + Darkhorse combination strategy")
    logger.info("=" * 70)

    # 인기마 붕괴 모델 (전체 데이터에서 인기마 대상)
    df_full = pd.read_csv("race_entries.csv", low_memory=False)
    df_full = df_full[df_full["meet"] == "서울"].reset_index(drop=True)
    df_full = df_full.sort_values("rcDate").reset_index(drop=True)
    df_full["fold"] = assign_time_split(df_full, "rcDate", SPLIT_RATIOS)

    # upset_A = 인기 상위 25%인데 하위 50% 착순 = 붕괴
    favorites = df_full[df_full["pop_pct"] <= 0.25].copy()
    favorites["collapse"] = (favorites["ord"] > favorites["n_run"] * 0.5).astype(int)

    fav_features_available = [c for c in features if c in favorites.columns]
    fav_train = favorites[favorites["fold"] == "train"]
    fav_test = favorites[favorites["fold"] == "test"]

    if len(fav_train) > 100 and len(fav_features_available) > 5:
        # 범주형 인코딩 (인기마 데이터)
        fav_cat = [c for c in fav_features_available if favorites[c].dtype == "object"]
        for col in fav_cat:
            favorites[col] = favorites[col].fillna("MISSING").astype(str)
            le = LabelEncoder()
            le.fit(favorites[col].unique())
            favorites[col] = le.transform(favorites[col])

        fav_train = favorites[favorites["fold"] == "train"]
        fav_test = favorites[favorites["fold"] == "test"]

        # 수치형만 사용 (인코딩 완료)
        fav_num_features = [c for c in fav_features_available if c in fav_train.columns]

        # 간단한 붕괴 모델
        X_fav_train = fav_train[fav_num_features].fillna(0).values.astype(np.float32)
        X_fav_test = fav_test[fav_num_features].fillna(0).values.astype(np.float32)

        rf_collapse = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=50,
            random_state=RANDOM_STATE, n_jobs=-1)
        rf_collapse.fit(X_fav_train, fav_train["collapse"].values)
        fav_test_proba = rf_collapse.predict_proba(X_fav_test)[:, 1]

        # 경주별 최대 붕괴 확률
        fav_test = fav_test.copy()
        fav_test["collapse_prob"] = fav_test_proba
        race_risk = fav_test.groupby("race_id")["collapse_prob"].max().reset_index()
        race_risk.columns = ["race_id", "race_collapse_risk"]

        # 다크호스 test에 조인
        test_combo = test_df.merge(race_risk, on="race_id", how="left")
        test_combo["race_collapse_risk"] = test_combo["race_collapse_risk"].fillna(0)

        # 필터: 붕괴 위험 상위 50% 경주에서만 베팅
        risk_threshold = test_combo["race_collapse_risk"].median()
        high_risk = test_combo[test_combo["race_collapse_risk"] >= risk_threshold]

        if len(high_risk) > 20:
            n_combo = max(1, int(len(high_risk) * 0.2))
            top_combo = high_risk.nlargest(n_combo, best_proba)
            ret_combo = (top_combo["upset_place"] * top_combo["_plcOdds"]).sum()
            roi_combo = (ret_combo - n_combo) / n_combo * 100
            hit_combo = top_combo["upset_place"].mean()

            logger.info(f"  Collapse model trained on {len(fav_train):,} favorites")
            logger.info(f"  High-risk races (collapse risk >= median): {len(high_risk):,} horses")
            logger.info(f"  Combo top 20%: {n_combo} bets | hit={hit_combo:.3f} | ROI={roi_combo:+.1f}%")
            logger.info(f"  vs Simple top 20%: ROI={roi_tuned:+.1f}%")
            logger.info(f"  Improvement: {roi_combo - roi_tuned:+.1f}%p")

            all_results.append({"stage": "6_combo", "target": "place", "AUC": 0, "ROI_20": roi_combo, "CI_low": 0, "CI_high": 0, "P_profit": 0})
        else:
            logger.info("  Not enough data in high-risk races for combo strategy")
    else:
        logger.info("  Not enough favorite data for collapse model")

    # ==================== SUMMARY ====================
    logger.info("\n" + "=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "improvement_results.csv", index=False)
    logger.info(f"\n{df_results.to_string(index=False)}")

    logger.info(f"\n  Saved: results/upset_improvements/improvement_results.csv")
    logger.info(f"  Saved: results/upset_improvements/robustness.csv")
    logger.info("\n" + "=" * 70)
    logger.info("ALL EXPERIMENTS DONE!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
