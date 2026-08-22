"""
12_final_strategy.py — 최종 전략: C모델(단승) + 배당 필터

기존 08_full_pipeline의 C모델 결과를 활용하여,
winOdds 구간별 필터를 적용한 ROI를 계산한다.

실행:
    python src/5_이변모델_선정/12_final_strategy.py
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS,
    RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_improvements")


def bootstrap_roi(upsets, odds, n_boot=1000):
    """Bootstrap ROI CI + P(profit)."""
    n = len(upsets)
    rois = []
    np.random.seed(42)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        ret = (upsets[idx] * odds[idx]).sum()
        rois.append((ret - n) / n * 100)
    ci = np.percentile(rois, [2.5, 97.5])
    p_profit = (np.array(rois) > 0).mean() * 100
    median_roi = np.median(rois)
    return ci, p_profit, median_roi


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FINAL STRATEGY: C model (win target) + winOdds filter")
    logger.info("=" * 70)

    # Load prepared data from 08_full_pipeline
    # 08이 pkl을 저장하지 않으므로, 직접 모델을 재학습한다 (빠름)
    prep_path = Path("results/upset_with_odds_v2/prepared_data.pkl")
    models_path = Path("results/upset_with_odds_v2/trained_models.pkl")

    if prep_path.exists() and models_path.exists():
        with open(prep_path, "rb") as f:
            prep = pickle.load(f)
        with open(models_path, "rb") as f:
            trained = pickle.load(f)
        df = prep["df"]
        c_features = prep["c_features"]
        model_comp = pd.read_csv(Path("results/upset_with_odds_v2/model_comparison.csv"))
        c_rows = model_comp[model_comp["feature_set"] == "C (q + features)"]
        best_name = c_rows.loc[c_rows["ROC_AUC"].idxmax(), "model"]
        model = trained[f"C (q + features)_{best_name}"]
    else:
        # pkl 없으면 직접 빠르게 재학습
        logger.info("  pkl not found — rebuilding model quickly...")
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
        from config import (
            ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS,
            RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
        )

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

        df = pd.read_csv("race_entries.csv", low_memory=False)
        df = df[df["meet"] == "서울"].reset_index(drop=True)
        df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)
        df["_winOdds"] = df["winOdds"].copy()
        df["_plcOdds"] = df["plcOdds"].copy()
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

        c_features = [c for c in df.columns if c not in FULL_EXCLUDE and c != "_winOdds" and c != "_plcOdds"]
        train_mask = df["fold"] == "train"
        cat_cols = [c for c in c_features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
        num_cols = [c for c in c_features if c not in cat_cols]

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

        train = df[df["fold"] == "train"]
        model = RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(train[c_features].values.astype(np.float32), train["upset"].values)
        best_name = "RF"
        logger.info("  Model rebuilt.")

    logger.info(f"  Model: C + {best_name}")

    # Test set predictions
    test = df[df["fold"] == "test"].copy()
    X_test = test[c_features].values.astype(np.float32)
    test["proba"] = model.predict_proba(X_test)[:, 1]

    logger.info(f"  Test set: {len(test):,} horses | upset rate: {test['upset'].mean():.4f}")
    logger.info(f"  Total upsets in test: {test['upset'].sum()}")

    # ====== Strategy comparison ======
    logger.info("\n" + "=" * 70)
    logger.info("ROI by winOdds filter + top N% selection (WIN target, winOdds payout)")
    logger.info("=" * 70)
    logger.info(f"  Note: winOdds already includes ~20% takeout. ROI is net of commission.\n")

    segments = [
        ("No filter (all)", 0, 99999),
        ("winOdds >= 5x", 5, 99999),
        ("winOdds >= 10x", 10, 99999),
        ("winOdds >= 15x", 15, 99999),
        ("winOdds >= 20x", 20, 99999),
        ("winOdds >= 30x", 30, 99999),
        ("winOdds 10-30x", 10, 30),
        ("winOdds 10-50x", 10, 50),
    ]

    all_results = []

    for seg_name, lo, hi in segments:
        seg = test[(test["_winOdds"] >= lo) & (test["_winOdds"] < hi)]
        if len(seg) < 10:
            continue

        for top_pct in [5, 10, 20]:
            n_bets = max(1, int(len(seg) * top_pct / 100))
            top = seg.nlargest(n_bets, "proba")

            n_wins = top["upset"].sum()
            total_return = (top["upset"] * top["_winOdds"]).sum()
            roi = (total_return - n_bets) / n_bets * 100
            hit_rate = n_wins / n_bets

            # Bootstrap
            ci, p_profit, median = bootstrap_roi(
                top["upset"].values, top["_winOdds"].values, n_boot=1000
            )

            all_results.append({
                "filter": seg_name,
                "top_pct": f"{top_pct}%",
                "n_pool": len(seg),
                "n_bets": n_bets,
                "n_wins": int(n_wins),
                "hit_rate": round(hit_rate, 4),
                "avg_odds": round(top["_winOdds"].mean(), 1),
                "ROI": round(roi, 1),
                "CI_low": round(ci[0], 1),
                "CI_high": round(ci[1], 1),
                "P_profit": round(p_profit, 1),
            })

    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_DIR / "final_strategy.csv", index=False)

    # Print results
    logger.info(f"\n{'Filter':<20s} {'Top%':>5s} {'Pool':>5s} {'Bets':>5s} {'Wins':>5s} "
                f"{'Hit':>6s} {'Odds':>6s} {'ROI':>8s} {'CI_low':>8s} {'CI_hi':>8s} {'P(+)':>6s}")
    logger.info("-" * 100)

    for _, r in df_results.iterrows():
        roi_color = "+" if r["ROI"] > 0 else ""
        logger.info(
            f"  {r['filter']:<18s} {r['top_pct']:>5s} {r['n_pool']:>5d} {r['n_bets']:>5d} {r['n_wins']:>5d} "
            f"{r['hit_rate']:>6.3f} {r['avg_odds']:>6.1f} {roi_color}{r['ROI']:>7.1f}% "
            f"[{r['CI_low']:>+7.1f}%,{r['CI_high']:>+7.1f}%] {r['P_profit']:>5.1f}%"
        )

    # ====== Best strategies ======
    logger.info("\n" + "=" * 70)
    logger.info("TOP 5 STRATEGIES (by P(profit))")
    logger.info("=" * 70)

    top5 = df_results.sort_values("P_profit", ascending=False).head(5)
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        logger.info(
            f"  #{i}: {r['filter']} top{r['top_pct']} → "
            f"ROI {r['ROI']:+.1f}% | CI [{r['CI_low']:+.1f}%, {r['CI_high']:+.1f}%] | "
            f"P(profit)={r['P_profit']:.1f}% | {r['n_bets']} bets, {r['n_wins']} wins"
        )

    # ====== Risk-adjusted best ======
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDED STRATEGY (balance ROI, CI, sample size)")
    logger.info("=" * 70)

    # Filter: P(profit) > 70% AND n_bets >= 30 (enough sample)
    viable = df_results[(df_results["P_profit"] >= 70) & (df_results["n_bets"] >= 30)]
    if len(viable) > 0:
        best = viable.sort_values("ROI", ascending=False).iloc[0]
        logger.info(f"\n  >>> {best['filter']} + top {best['top_pct']} <<<")
        logger.info(f"      ROI: {best['ROI']:+.1f}%")
        logger.info(f"      95% CI: [{best['CI_low']:+.1f}%, {best['CI_high']:+.1f}%]")
        logger.info(f"      P(profit): {best['P_profit']:.1f}%")
        logger.info(f"      Bets: {int(best['n_bets'])} | Wins: {int(best['n_wins'])} | Hit rate: {best['hit_rate']:.3f}")
        logger.info(f"      Avg winOdds: {best['avg_odds']:.1f}x")
    else:
        logger.info("  No strategy meets criteria (P>70% and n>=30)")

    # Random baseline
    random_roi = (test["upset"] * test["_winOdds"]).sum() / len(test) * 100 - 100
    logger.info(f"\n  Random baseline (bet all): ROI = {random_roi:+.1f}%")

    logger.info("\n" + "=" * 70)
    logger.info("DONE! Results saved to: results/upset_improvements/final_strategy.csv")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
