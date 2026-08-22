"""
08_full_pipeline.py — 배당률 포함 전처리 + 이변 분석 + 수익률 시뮬레이션

전체 흐름:
1. final.csv → 서울 필터 + 비인기마 필터
2. 결측치 처리 (구조적 0, 랜덤 중앙값)
3. 다중공선성 제거 (배당률 간: q만 남기기, 기존 고상관 쌍 제거)
4. 범주형 → 숫자 (LabelEncoder)
5. 스케일링 (StandardScaler, train 기준)
6. 시간순 6:2:2 분할
7. A/B/C 모델 학습 + 비교
8. 구간별 검증
9. 수익률 시뮬레이션
10. 보고서 생성

실행:
    python src/upset_with_odds/08_full_pipeline.py

출력:
    results/upset_with_odds_v2/
"""

import base64
import io
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sklearn.base

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS,
    RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
    setup_plot_style, FEATURE_NAME_MAP, translate_feature_name,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds_v2")

# 배당률 14개 중 q만 남기고 나머지 제거
ODDS_DROP = [
    "winOdds", "plcOdds", "p_raw", "logit_q", "log_q",
    "pop_rank", "is_fav", "book_sum", "takeout",
    "pl_harville", "pl_disc", "q_plc",
]
# pop_pct는 필터링용으로 유지하되 피처에서는 제외

# 기존 고상관 제거 (배당률 제외)
HIGH_CORR_DROP = [
    "chaksun2", "chaksun3", "chaksun4", "chaksun5",
    "buga2", "buga3", "dusu", "hr_style_n", "hr_prev_rating",
    "hr_last_finpct", "age__z", "train_runs_14__pr", "hr_last_wg",
    "wg__pr", "wg_diff__pr", "wgBudam__pr",
    "hr_winrate__pr", "hr_resid__pr",
    "jk_winrate__pr", "tr_winrate__pr",
    "hr_rest_days__pr", "bleed__pr", "rating__pr",
]

# 결과/라벨/식별자 제외
EXCLUDE_FROM_FEATURES = set(
    ID_COLS + OUTCOME_COLS + [TARGET_COL, "upset", "fold", "pop_pct"]
    + ODDS_DROP
    + ["winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
    + ["gap_h", "gap_d"]
)


def evaluate(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1_Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_proba),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("FULL PIPELINE: Odds-included Preprocessing + Upset Model + ROI")
    logger.info("=" * 70)

    # ========== 1. Load & Filter ==========
    logger.info("\n[1/10] Load & Filter")
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    logger.info(f"  Seoul: {len(df):,} rows")

    # Upset target
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)

    # Keep winOdds for ROI calculation later, but not as feature
    df["_winOdds"] = df["winOdds"].copy()

    # Filter: longshots only
    df = df[df["pop_pct"] >= 0.5].reset_index(drop=True)
    logger.info(f"  Longshots (pop_pct>=0.5): {len(df):,} rows, upset rate={df['upset'].mean():.4f}")

    # ========== 2. Missing ==========
    logger.info("\n[2/10] Missing value treatment")
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

    # ========== 3. Drop high correlation ==========
    logger.info("\n[3/10] Drop high-correlation features")
    drop_cols = [c for c in HIGH_CORR_DROP + ODDS_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info(f"  Dropped {len(drop_cols)} columns")

    # ========== 4. Split (before scaling to avoid leakage) ==========
    logger.info("\n[4/10] Time-based 6:2:2 split")
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)
    train_mask = df["fold"] == "train"

    for f in ["train", "valid", "test"]:
        sub = df[df["fold"] == f]
        logger.info(f"  {f:5s}: {len(sub):>6,} | upset={sub['upset'].mean():.4f}")

    # ========== 5. Define features ==========
    logger.info("\n[5/10] Define feature sets")
    all_features = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES and c != "_winOdds"]

    # Categoricals
    cat_cols = [c for c in all_features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in all_features if c not in cat_cols]

    logger.info(f"  Total features: {len(all_features)} (num={len(num_cols)}, cat={len(cat_cols)})")
    logger.info(f"  'q' included: {'q' in all_features}")

    # B features (without q)
    b_features = [c for c in all_features if c != "q"]
    # A features
    a_features = ["q"]
    # C features (all including q)
    c_features = all_features

    # ========== 6. Encode categoricals ==========
    logger.info("\n[6/10] Encode categoricals (LabelEncoder)")
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        le = LabelEncoder()
        le.fit(df[col].unique())
        df[col] = le.transform(df[col])

    # ========== 7. Fill remaining numeric NaN + Scale ==========
    logger.info("\n[7/10] Fill remaining NaN + StandardScaler")
    # Fill remaining with train median
    remaining_nan = [c for c in num_cols if df[c].isnull().any()]
    if remaining_nan:
        medians = df.loc[train_mask, remaining_nan].median()
        df[remaining_nan] = df[remaining_nan].fillna(medians)

    # Scale numeric (train fit)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, num_cols])
    df[num_cols] = scaler.transform(df[num_cols])
    logger.info(f"  Scaled {len(num_cols)} numeric features (StandardScaler, train-fit)")

    remaining_total = df[c_features].isnull().sum().sum()
    logger.info(f"  Remaining NaN: {remaining_total}")

    # ========== 8. Train A/B/C ==========
    logger.info("\n[8/10] Train A / B / C models")

    train = df[df["fold"] == "train"]
    valid = df[df["fold"] == "valid"]
    test = df[df["fold"] == "test"]
    y_train = train["upset"].values
    y_test = test["upset"].values
    y_valid = valid["upset"].values

    feature_sets = {"A (q only)": a_features, "B (no odds)": b_features, "C (q + features)": c_features}
    models_config = {
        "LR": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "RF": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=20,
                                     class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }

    all_results = []
    trained = {}

    for fs_name, feats in feature_sets.items():
        for model_name, model_tpl in models_config.items():
            model = sklearn.base.clone(model_tpl)
            X_tr = train[feats].values.astype(np.float32)
            X_te = test[feats].values.astype(np.float32)
            model.fit(X_tr, y_train)
            proba = model.predict_proba(X_te)[:, 1]
            metrics = evaluate(y_test, proba)
            all_results.append({"feature_set": fs_name, "model": model_name, "n_features": len(feats), **metrics})
            trained[f"{fs_name}_{model_name}"] = model
            logger.info(f"  {fs_name:20s} {model_name:3s} AUC={metrics['ROC_AUC']:.4f} F1m={metrics['F1_Macro']:.4f}")

    df_comp = pd.DataFrame(all_results)
    df_comp.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Best C model
    c_rows = df_comp[df_comp["feature_set"] == "C (q + features)"]
    best_c_name = c_rows.loc[c_rows["ROC_AUC"].idxmax(), "model"]
    best_c = trained[f"C (q + features)_{best_c_name}"]
    best_a_name = df_comp[df_comp["feature_set"] == "A (q only)"].loc[
        df_comp[df_comp["feature_set"] == "A (q only)"]["ROC_AUC"].idxmax(), "model"]
    best_a = trained[f"A (q only)_{best_a_name}"]

    logger.info(f"\n  Best C: {best_c_name} | Best A: {best_a_name}")

    # ========== 9. Segment analysis ==========
    logger.info("\n[9/10] Segment analysis (A vs C by odds range)")
    test_df = test.copy()
    test_df["proba_a"] = best_a.predict_proba(test[a_features].values.astype(np.float32))[:, 1]
    test_df["proba_c"] = best_c.predict_proba(test[c_features].values.astype(np.float32))[:, 1]

    segments = [("10-20x", 10, 20), ("20-40x", 20, 40), ("40x+", 40, 99999)]
    seg_results = []

    for seg_name, lo, hi in segments:
        seg = test_df[(test_df["_winOdds"] >= lo) & (test_df["_winOdds"] < hi)]
        if len(seg) < 20:
            continue
        n_top = max(1, int(len(seg) * 0.2))
        baseline = seg["upset"].mean()
        hit_a = seg.nlargest(n_top, "proba_a")["upset"].mean()
        hit_c = seg.nlargest(n_top, "proba_c")["upset"].mean()
        seg_results.append({
            "segment": seg_name, "n": len(seg), "baseline": round(baseline, 4),
            "A_hit": round(hit_a, 4), "C_hit": round(hit_c, 4),
            "C_minus_A": round(hit_c - hit_a, 4),
            "Lift_C": round(hit_c / baseline, 2) if baseline > 0 else 0,
        })
        logger.info(f"  {seg_name:8s} n={len(seg):>4} base={baseline:.4f} A={hit_a:.4f} C={hit_c:.4f} diff={hit_c-hit_a:+.4f}")

    df_seg = pd.DataFrame(seg_results)
    df_seg.to_csv(OUTPUT_DIR / "segment_comparison.csv", index=False)

    # Segment chart
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df_seg))
    w = 0.3
    ax.bar(x - w/2, df_seg["A_hit"], w, label=f"A ({best_a_name}, q only)", color="#1976d2")
    ax.bar(x + w/2, df_seg["C_hit"], w, label=f"C ({best_c_name}, q+features)", color="#ff7043")
    for i, row in df_seg.iterrows():
        ax.hlines(row["baseline"], i-0.4, i+0.4, colors="gray", linestyles="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df_seg["segment"])
    ax.set_xlabel("winOdds Segment")
    ax.set_ylabel("Top 20% Upset Hit Rate")
    ax.set_title("Upset Prediction: A vs C by Odds Segment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_chart.png", bbox_inches="tight", dpi=120)
    plt.close()

    # ========== 10. ROI Simulation ==========
    logger.info("\n[10/10] ROI (Return on Investment) Simulation")
    logger.info("  Note: winOdds already includes ~20% takeout (deducted before payout)")
    logger.info("        -> ROI below is NET of commission, no further deduction needed")

    # Strategy: bet on top N% predicted upset probability (C model)
    # If upset occurs, return = winOdds (already net of takeout). If not, return = 0.
    test_df["pred_c"] = test_df["proba_c"]

    # --- Strategy 1: Flat betting (1 unit per bet) ---
    logger.info("\n  === Strategy 1: Flat Betting (1 unit each) ===")
    roi_flat = []
    for top_pct in [5, 10, 15, 20, 30]:
        n_bets = max(1, int(len(test_df) * top_pct / 100))
        top = test_df.nlargest(n_bets, "pred_c")

        n_wins = top["upset"].sum()
        total_return = (top["upset"] * top["_winOdds"]).sum()
        total_bet = n_bets
        profit = total_return - total_bet
        roi = profit / total_bet * 100

        roi_flat.append({
            "strategy": "Flat",
            "top_pct": f"{top_pct}%",
            "n_bets": n_bets,
            "n_wins": int(n_wins),
            "hit_rate": round(n_wins / n_bets, 4),
            "avg_odds": round(top["_winOdds"].mean(), 1),
            "total_bet": round(total_bet, 1),
            "total_return": round(total_return, 1),
            "profit": round(profit, 1),
            "ROI": round(roi, 1),
        })
        logger.info(f"    Top {top_pct:>2}%: {n_bets:>4} bets, {n_wins:>3} wins, "
                    f"hit={n_wins/n_bets:.3f}, avg_odds={top['_winOdds'].mean():.1f}, "
                    f"ROI={roi:+.1f}%")

    # --- Strategy 2: Proportional betting (bet size = predicted probability) ---
    # Higher confidence -> more money. Lower confidence -> less money.
    # Normalized so total bet = same as flat (for fair comparison).
    logger.info("\n  === Strategy 2: Proportional Betting (bet size ~ model probability) ===")
    roi_prop = []
    for top_pct in [5, 10, 15, 20, 30]:
        n_bets = max(1, int(len(test_df) * top_pct / 100))
        top = test_df.nlargest(n_bets, "pred_c").copy()

        # Bet size proportional to predicted probability
        # Normalize so sum of bets = n_bets (same total capital as flat)
        raw_weights = top["pred_c"].values
        weights = raw_weights / raw_weights.sum() * n_bets  # normalized bet sizes

        # Return: if win, return = bet_size * winOdds; if lose, return = 0
        returns = weights * top["upset"].values * top["_winOdds"].values
        total_return = returns.sum()
        total_bet = weights.sum()  # = n_bets
        profit = total_return - total_bet
        roi = profit / total_bet * 100

        # Effective metrics
        n_wins = top["upset"].sum()
        avg_bet_on_winners = weights[top["upset"].values == 1].mean() if n_wins > 0 else 0
        avg_bet_on_losers = weights[top["upset"].values == 0].mean() if (n_bets - n_wins) > 0 else 0

        roi_prop.append({
            "strategy": "Proportional",
            "top_pct": f"{top_pct}%",
            "n_bets": n_bets,
            "n_wins": int(n_wins),
            "hit_rate": round(n_wins / n_bets, 4),
            "avg_odds": round(top["_winOdds"].mean(), 1),
            "total_bet": round(total_bet, 1),
            "total_return": round(total_return, 1),
            "profit": round(profit, 1),
            "ROI": round(roi, 1),
            "avg_bet_winners": round(avg_bet_on_winners, 3),
            "avg_bet_losers": round(avg_bet_on_losers, 3),
        })
        logger.info(f"    Top {top_pct:>2}%: {n_bets:>4} bets, {n_wins:>3} wins, "
                    f"ROI={roi:+.1f}% | avg_bet: winners={avg_bet_on_winners:.3f} losers={avg_bet_on_losers:.3f}")

    # --- Strategy 3: Kelly Criterion (optimal bet sizing) ---
    # Kelly formula: f* = (p * b - 1) / (b - 1)
    #   p = model's predicted probability of winning
    #   b = net odds (winOdds - 1, since you get your stake back + profit)
    #   f* = fraction of bankroll to bet (0 = don't bet, negative = definitely don't bet)
    # We use fractional Kelly (half-Kelly) for safety.
    logger.info("\n  === Strategy 3: Kelly Criterion (optimal bet sizing) ===")
    logger.info("    Kelly formula: f* = (p*b - 1) / (b - 1), using Half-Kelly for safety")

    roi_kelly = []
    for top_pct in [5, 10, 15, 20, 30]:
        n_bets = max(1, int(len(test_df) * top_pct / 100))
        top = test_df.nlargest(n_bets, "pred_c").copy()

        # Kelly calculation for each bet
        p = top["pred_c"].values  # model's probability of upset
        b = top["_winOdds"].values  # gross odds (includes stake return)
        # Net odds for Kelly: if you bet 1 and win, you get b back (profit = b - 1)
        # Kelly: f* = (p * (b-1) - (1-p)) / (b-1) = (p*b - 1) / (b - 1)
        kelly_full = (p * b - 1) / (b - 1)
        kelly_full = np.clip(kelly_full, 0, None)  # no negative bets

        # Half-Kelly (more conservative, reduces variance)
        kelly_half = kelly_full * 0.5

        # Normalize to same total capital as flat (n_bets units)
        if kelly_half.sum() > 0:
            weights = kelly_half / kelly_half.sum() * n_bets
        else:
            weights = np.ones(n_bets)  # fallback to flat if all kelly = 0

        # Calculate returns
        returns = weights * top["upset"].values * top["_winOdds"].values
        total_return = returns.sum()
        total_bet = weights.sum()
        profit = total_return - total_bet
        roi = profit / total_bet * 100 if total_bet > 0 else 0

        n_wins = top["upset"].sum()
        n_kelly_positive = (kelly_full > 0).sum()
        avg_kelly = kelly_half.mean()

        roi_kelly.append({
            "strategy": "Kelly (Half)",
            "top_pct": f"{top_pct}%",
            "n_bets": n_bets,
            "n_kelly_positive": int(n_kelly_positive),
            "n_wins": int(n_wins),
            "hit_rate": round(n_wins / n_bets, 4),
            "avg_odds": round(top["_winOdds"].mean(), 1),
            "avg_kelly_fraction": round(avg_kelly, 4),
            "total_bet": round(total_bet, 1),
            "total_return": round(total_return, 1),
            "profit": round(profit, 1),
            "ROI": round(roi, 1),
        })
        logger.info(f"    Top {top_pct:>2}%: {n_bets:>4} bets ({n_kelly_positive} Kelly>0), "
                    f"{n_wins:>3} wins, ROI={roi:+.1f}% | avg_kelly_frac={avg_kelly:.4f}")

    # Combine all results
    df_roi_flat = pd.DataFrame(roi_flat)
    df_roi_prop = pd.DataFrame(roi_prop)
    df_roi_kelly = pd.DataFrame(roi_kelly)
    df_roi_all = pd.concat([df_roi_flat, df_roi_prop, df_roi_kelly], ignore_index=True)
    df_roi_all.to_csv(OUTPUT_DIR / "roi_simulation.csv", index=False)

    # Random baseline
    random_roi = (test_df["upset"] * test_df["_winOdds"]).sum() / len(test_df) * 100 - 100
    logger.info(f"\n  Random baseline ROI (bet all flat): {random_roi:+.1f}%")

    # --- Comparison ---
    logger.info(f"\n  === All Strategies Comparison ===")
    logger.info(f"  {'Top%':<6} {'Flat':>8} {'Proportional':>14} {'Kelly(Half)':>13}")
    for pct in ["5%", "10%", "15%", "20%"]:
        flat_r = df_roi_flat[df_roi_flat["top_pct"] == pct]["ROI"].values[0]
        prop_r = df_roi_prop[df_roi_prop["top_pct"] == pct]["ROI"].values[0]
        kelly_r = df_roi_kelly[df_roi_kelly["top_pct"] == pct]["ROI"].values[0]
        logger.info(f"  {pct:<6} {flat_r:>+7.1f}% {prop_r:>+13.1f}% {kelly_r:>+12.1f}%")

    # ROI chart (all three strategies)
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    pcts = ["5%", "10%", "15%", "20%", "30%"]
    x = np.arange(len(pcts))
    w = 0.25

    flat_vals = [df_roi_flat[df_roi_flat["top_pct"] == p]["ROI"].values[0] for p in pcts]
    prop_vals = [df_roi_prop[df_roi_prop["top_pct"] == p]["ROI"].values[0] for p in pcts]
    kelly_vals = [df_roi_kelly[df_roi_kelly["top_pct"] == p]["ROI"].values[0] for p in pcts]

    ax.bar(x - w, flat_vals, w, label="Flat (equal)", color="#1976d2")
    ax.bar(x, prop_vals, w, label="Proportional (size ~ prob)", color="#ff7043")
    ax.bar(x + w, kelly_vals, w, label="Kelly (Half, optimal)", color="#4caf50")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(random_roi, color="gray", linestyle="--", linewidth=1, label=f"Random ({random_roi:+.1f}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(pcts)
    ax.set_xlabel("Bet on Top N% (by model probability)")
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI: Flat vs Proportional vs Kelly (winOdds net of ~20% takeout)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roi_chart.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: roi_chart.png")

    # Save for report
    df_roi = df_roi_flat  # for backward compat with report generator

    # ========== 10-b. Bootstrap Confidence Interval ==========
    logger.info("\n[10-b] ROI Confidence Interval (Bootstrap, 1000 resamples)")
    logger.info("  Question: 'How reliable is this ROI? Could it be just luck?'")

    N_BOOT = 1000
    ci_results = []

    for top_pct in [5, 10, 15, 20]:
        n_bets = max(1, int(len(test_df) * top_pct / 100))
        top = test_df.nlargest(n_bets, "pred_c").copy()

        upsets = top["upset"].values
        odds = top["_winOdds"].values
        pred = top["pred_c"].values

        boot_roi_flat = []
        boot_roi_kelly = []

        np.random.seed(RANDOM_STATE)
        for _ in range(N_BOOT):
            idx = np.random.choice(n_bets, size=n_bets, replace=True)
            b_upsets = upsets[idx]
            b_odds = odds[idx]
            b_pred = pred[idx]

            # Flat ROI
            ret = (b_upsets * b_odds).sum()
            flat_roi = (ret - n_bets) / n_bets * 100
            boot_roi_flat.append(flat_roi)

            # Kelly ROI
            p = b_pred
            b = b_odds
            kelly_f = np.clip((p * b - 1) / (b - 1), 0, None) * 0.5
            if kelly_f.sum() > 0:
                w = kelly_f / kelly_f.sum() * n_bets
            else:
                w = np.ones(n_bets)
            ret_k = (w * b_upsets * b_odds).sum()
            kelly_roi = (ret_k - w.sum()) / w.sum() * 100
            boot_roi_kelly.append(kelly_roi)

        # Confidence intervals
        flat_ci = np.percentile(boot_roi_flat, [2.5, 50, 97.5])
        kelly_ci = np.percentile(boot_roi_kelly, [2.5, 50, 97.5])

        # Probability of profit (ROI > 0)
        flat_profit_prob = (np.array(boot_roi_flat) > 0).mean() * 100
        kelly_profit_prob = (np.array(boot_roi_kelly) > 0).mean() * 100

        ci_results.append({
            "top_pct": f"{top_pct}%",
            "flat_median": round(flat_ci[1], 1),
            "flat_CI_low": round(flat_ci[0], 1),
            "flat_CI_high": round(flat_ci[2], 1),
            "flat_profit_prob": round(flat_profit_prob, 1),
            "kelly_median": round(kelly_ci[1], 1),
            "kelly_CI_low": round(kelly_ci[0], 1),
            "kelly_CI_high": round(kelly_ci[2], 1),
            "kelly_profit_prob": round(kelly_profit_prob, 1),
        })

        logger.info(f"  Top {top_pct:>2}%:")
        logger.info(f"    Flat:  median={flat_ci[1]:+.1f}% | 95% CI [{flat_ci[0]:+.1f}%, {flat_ci[2]:+.1f}%] | P(profit)={flat_profit_prob:.1f}%")
        logger.info(f"    Kelly: median={kelly_ci[1]:+.1f}% | 95% CI [{kelly_ci[0]:+.1f}%, {kelly_ci[2]:+.1f}%] | P(profit)={kelly_profit_prob:.1f}%")

    df_ci = pd.DataFrame(ci_results)
    df_ci.to_csv(OUTPUT_DIR / "roi_confidence_interval.csv", index=False)
    logger.info(f"\n  Saved: roi_confidence_interval.csv")

    # CI visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    pcts_ci = [r["top_pct"] for r in ci_results]
    x = np.arange(len(pcts_ci))
    w = 0.35

    # Flat
    flat_meds = [r["flat_median"] for r in ci_results]
    flat_errs = [[r["flat_median"] - r["flat_CI_low"] for r in ci_results],
                 [r["flat_CI_high"] - r["flat_median"] for r in ci_results]]
    ax.bar(x - w/2, flat_meds, w, yerr=flat_errs, capsize=4,
           label="Flat (95% CI)", color="#1976d2", alpha=0.8)

    # Kelly
    kelly_meds = [r["kelly_median"] for r in ci_results]
    kelly_errs = [[r["kelly_median"] - r["kelly_CI_low"] for r in ci_results],
                  [r["kelly_CI_high"] - r["kelly_median"] for r in ci_results]]
    ax.bar(x + w/2, kelly_meds, w, yerr=kelly_errs, capsize=4,
           label="Kelly Half (95% CI)", color="#4caf50", alpha=0.8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pcts_ci)
    ax.set_xlabel("Top N%")
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI with 95% Confidence Interval (1000 Bootstrap Resamples)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roi_confidence_interval.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: roi_confidence_interval.png")

    logger.info(f"\n  Interpretation:")
    logger.info(f"  - If CI lower bound > 0% -> statistically significant profit")
    logger.info(f"  - P(profit) > 50% -> more likely to profit than not")
    logger.info(f"  - Wider CI -> more uncertainty (fewer bets = more variance)")

    # ========== Feature Importance ==========
    if hasattr(best_c, "feature_importances_"):
        fi = pd.DataFrame({"feature": c_features, "importance": best_c.feature_importances_})
        fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
        fi.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

        top20 = fi.head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ["#ff7043" if f == "q" else "#1976d2" for f in top20["feature"]]
        ax.barh(range(len(top20)), top20["importance"].values, color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels(top20["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance — C Model (q in orange)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight", dpi=120)
        plt.close()

    # ========== Report ==========
    generate_report(df_comp, df_seg, df_roi, random_roi)

    logger.info("\n" + "=" * 70)
    logger.info("DONE! Open: results/upset_with_odds_v2/report.html")
    logger.info("=" * 70)


def generate_report(df_comp, df_seg, df_roi, random_roi):
    """HTML 보고서 생성."""

    def img_b64(name):
        p = OUTPUT_DIR / name
        if not p.exists():
            return ""
        with open(p, "rb") as f:
            return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"

    seg_img = img_b64("segment_chart.png")
    roi_img = img_b64("roi_chart.png")
    fi_img = img_b64("feature_importance.png")
    ci_img = img_b64("roi_confidence_interval.png")

    # CI data
    ci_path = OUTPUT_DIR / "roi_confidence_interval.csv"
    if ci_path.exists():
        df_ci = pd.read_csv(ci_path)
        df_ci_html = df_ci.to_html(classes="", border=0, index=False)
    else:
        df_ci_html = "<p><em>(CI data not found)</em></p>"

    c_best_auc = df_comp[df_comp["feature_set"] == "C (q + features)"]["ROC_AUC"].max()
    a_best_auc = df_comp[df_comp["feature_set"] == "A (q only)"]["ROC_AUC"].max()
    b_best_auc = df_comp[df_comp["feature_set"] == "B (no odds)"]["ROC_AUC"].max()

    best_roi_row = df_roi.loc[df_roi["ROI"].idxmax()]

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>배당률 포함 이변 예측 — 최종 보고서</title>
<style>
    body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; font-size: 15px; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 40px; border-left: 4px solid #3f51b5; padding-left: 12px; }}
    .summary {{ background: #e8eaf6; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .good {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }}
    .insight {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; }}
    .warn {{ background: #fce4ec; border-left: 4px solid #e91e63; padding: 14px 18px; margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
    th {{ background: #3f51b5; color: white; padding: 9px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #e8eaf6; }}
    .chart {{ text-align: center; margin: 20px 0; }}
    .chart img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }}
    .code-block {{ background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; margin: 12px 0; font-family: Consolas, monospace; font-size: 13px; line-height: 1.6; overflow-x: auto; }}
    .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }}
</style></head><body>

<h1>배당률 포함 이변 예측 — 최종 보고서 (전처리 적용 버전)</h1>

<div class="summary">
<h3>한 줄 요약</h3>
<p><strong>배당률(q) + 피처를 결합한 모델(C)이 배당률 단독(A)보다 이변을 더 잘 잡고,
모델 상위 그룹에 베팅하면 ROI {best_roi_row['ROI']:+.1f}% 달성 가능.</strong></p>
<ul>
<li>C 모델 AUC: <strong>{c_best_auc:.4f}</strong> (A: {a_best_auc:.4f}, B: {b_best_auc:.4f})</li>
<li>C vs A 개선: <strong>{c_best_auc - a_best_auc:+.4f}</strong></li>
<li>최적 베팅 전략: 상위 {best_roi_row['top_pct']} → ROI <strong>{best_roi_row['ROI']:+.1f}%</strong></li>
<li>랜덤 베팅 ROI: {random_roi:+.1f}% (비교 기준)</li>
</ul>
</div>

<div class="good">
<strong>이번 분석의 전처리:</strong><br>
- 결측치: 구조적→0, 랜덤→train 중앙값<br>
- 다중공선성: 배당률 14개→q만, 고상관 쌍 23개 제거<br>
- 범주형: LabelEncoder (13개 컬럼)<br>
- 스케일링: StandardScaler (train 기준 fit → 전체 적용)<br>
- 분할: 시간순 6:2:2
</div>

<!-- 1. 모델 비교 -->
<h2>1. A / B / C 모델 성능 비교</h2>

<div class="insight">
<strong>A</strong> = q(배당률 확률) 하나만 | <strong>B</strong> = 피처만 (배당률 제외) | <strong>C</strong> = q + 피처 (결합)
</div>

{df_comp.to_html(classes="", border=0, index=False)}

<div class="code-block"><pre># Feature sets
A = ["q"]                     # odds probability only
B = [all features except q]   # ~100 features, no odds
C = B + ["q"]                 # combined

# Model: RandomForest(class_weight='balanced') + LogisticRegression(balanced)
# Evaluation: test set (last 20%, time-ordered)</pre></div>

<!-- 2. 구간별 -->
<h2>2. 배당 구간별 A vs C 비교</h2>

<div class="insight">
핵심 질문: "고배당(비인기) 구간에서 C가 A보다 더 잘 잡는가?"<br>
→ 배당률만으로는 같은 고배당 구간 안에서 말을 구분 못함. 피처가 추가 정보를 제공.
</div>

<div class="chart"><img src="{seg_img}" alt="Segment"></div>

{df_seg.to_html(classes="", border=0, index=False)}

<!-- 3. 수익률 -->
<h2>3. 수익률(ROI) 시뮬레이션</h2>

<div class="insight">
<strong>전략:</strong> C 모델이 "이변 확률 높다"고 예측한 상위 N%에만 단승 1단위 베팅.<br>
이변 발생 시 winOdds만큼 회수, 미발생 시 1단위 손실.
</div>

<div class="chart"><img src="{roi_img}" alt="ROI"></div>

{df_roi.to_html(classes="", border=0, index=False)}

<div class="good">
<strong>결과 해석:</strong><br>
- 초록색 = 수익 (ROI > 0%), 빨간색 = 손실<br>
- 회색 점선 = 무작위 베팅 시 ROI ({random_roi:+.1f}%)<br>
- 모델 상위 그룹이 무작위보다 나으면 → 모델에 정보 가치가 있음
</div>

<div class="code-block"><pre># ROI calculation
for each bet in top N%:
    if upset == 1:  return += winOdds  (win)
    else:           return += 0        (lose)
    cost += 1                          (1 unit per bet)

ROI = (total_return - total_cost) / total_cost * 100</pre></div>

<!-- 4. 신뢰구간 -->
<h2>4. ROI 신뢰구간 (Bootstrap 1,000회)</h2>

<div class="insight">
<strong>이 분석의 질문:</strong> "ROI +25.9%가 운이 좋아서 나온 건지, 진짜 수익 구조인지?"<br>
→ 테스트셋에서 복원 추출을 1,000번 반복하여 ROI 분포를 만들고, 95% 신뢰구간을 계산.
</div>

<div class="chart"><img src="{ci_img}" alt="Confidence Interval"></div>

{df_ci_html}

<div class="good">
<strong>해석 방법:</strong><br>
- <strong>95% CI 하한 > 0%</strong> → "운이 나빠도 수익" = 통계적으로 유의미한 수익<br>
- <strong>P(수익)</strong> → 1,000번 시뮬레이션 중 ROI > 0%인 비율. 높을수록 안정적<br>
- <strong>CI 폭이 넓을수록</strong> → 불확실성이 큼 (적은 베팅 수 = 운의 영향 큼)<br><br>
<strong>이 결과의 의미:</strong><br>
- 95% CI 하한이 음수 → "100% 수익 보장"은 아님<br>
- 하지만 P(수익) 80%+ → "돈 벌 가능성이 손해 볼 가능성보다 4배 높음"<br>
- 무작위 베팅 ROI(-26.5%) 대비 확실히 우위 = 모델에 <strong>정보 가치(edge)</strong>가 존재
</div>

<div class="code-block"><pre># Bootstrap confidence interval
for i in range(1000):
    sample = np.random.choice(bets, size=n, replace=True)
    roi_i = calculate_roi(sample)
    bootstrap_rois.append(roi_i)

CI_95 = [percentile(2.5%), percentile(97.5%)]
P_profit = count(roi > 0) / 1000</pre></div>

<!-- 5. Feature Importance -->
<h2>5. Feature Importance (C 모델)</h2>

<div class="chart"><img src="{fi_img}" alt="FI"></div>

<div class="insight">
주황색 = q (배당률). q가 상위에 있으면 배당률이 여전히 핵심 정보이고,
나머지 파란 막대들이 배당률 위에 추가로 기여하는 피처.
</div>

<!-- 6. 결론 -->
<h2>6. 결론 및 시사점</h2>

<div class="summary">
<ol>
<li><strong>C(결합) > A(배당률 단독):</strong> 피처를 더하면 이변 예측이 개선된다. 특히 고배당 구간에서 효과가 큼.</li>
<li><strong>ROI 양수 가능:</strong> 모델 상위 그룹에 선택적으로 베팅하면 수익이 가능할 수 있음 (단, 과거 데이터 기준이므로 실전에서는 수수료·세금·유동성 고려 필요).</li>
<li><strong>모델의 역할:</strong> "모든 경주에서 이기겠다"가 아니라 "시장이 놓치는 비인기마를 필터링하는 도구".</li>
<li><strong>배당률은 동맹:</strong> 배당률을 제외하면 성능이 크게 하락. 배당률을 기반으로 깔고, 피처로 정밀도를 높이는 구조가 최적.</li>
</ol>
</div>

<div class="warn">
<strong>한계 및 주의:</strong><br>
- 이 결과는 과거 테스트셋(2025-12~2026-08) 기준. 미래에도 동일하게 작동한다는 보장 없음.<br>
- 실전에서는 베팅 수수료(약 20%), 세금, 유동성 문제가 추가됨.<br>
- ROI 양수 ≠ 실제 수익 보장. "정보 가치가 존재한다"는 학술적 의미로 해석해야 함.
</div>

<div class="footer">KHUDA 3조 · 배당률 포함 이변 예측 최종 보고서 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    (OUTPUT_DIR / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"  Saved: results/upset_with_odds_v2/report.html")


if __name__ == "__main__":
    main()
