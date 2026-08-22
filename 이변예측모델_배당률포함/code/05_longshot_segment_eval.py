"""
05_longshot_segment_eval.py — 비인기마 배당 구간별 A vs C 비교

핵심 검증: "C가 A보다 어느 배당 구간에서 얼마나 더 나은지"
인기마 구간에서는 개선폭이 거의 없고, 비인기마(고배당) 구간에서
개선폭이 커지는지 확인하는 게 목적.

실행:
    python src/upset_with_odds/05_longshot_segment_eval.py
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import setup_plot_style

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")

# Odds segments for longshots
SEGMENTS = [
    ("10-20x", 10, 20),
    ("20-40x", 20, 40),
    ("40x+", 40, 99999),
]


def main():
    logger.info("=" * 60)
    logger.info("[Step 5] Longshot Segment Evaluation: A vs C")
    logger.info("=" * 60)

    # Load
    with open(OUTPUT_DIR / "prepared_data.pkl", "rb") as f:
        prep = pickle.load(f)
    with open(OUTPUT_DIR / "trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

    df = prep["df"]
    a_features = prep["a_features"]
    c_features = prep["c_features"]

    # Determine best models (RF for both A and C)
    comp = pd.read_csv(OUTPUT_DIR / "model_comparison.csv")
    best_a_model_name = comp[comp["feature_set"] == "A (q only)"].loc[
        comp[comp["feature_set"] == "A (q only)"]["ROC_AUC"].idxmax(), "model"
    ]
    best_c_model_name = comp[comp["feature_set"] == "C (q + features)"].loc[
        comp[comp["feature_set"] == "C (q + features)"]["ROC_AUC"].idxmax(), "model"
    ]

    model_a = trained_models[f"A (q only)_{best_a_model_name}"]
    model_c = trained_models[f"C (q + features)_{best_c_model_name}"]

    # Test set
    test = df[df["fold"] == "test"].copy()
    X_test_a = test[a_features].values.astype(np.float32)
    X_test_c = test[c_features].values.astype(np.float32)

    test["proba_a"] = model_a.predict_proba(X_test_a)[:, 1]
    test["proba_c"] = model_c.predict_proba(X_test_c)[:, 1]

    logger.info(f"  Test set: {len(test):,} rows")
    logger.info(f"  Model A: {best_a_model_name} | Model C: {best_c_model_name}")

    # Segment analysis
    # For each segment:
    #   - Take top 20% by predicted probability
    #   - Calculate actual upset hit rate in that top group
    results = []

    for seg_name, low, high in SEGMENTS:
        seg = test[(test["winOdds"] >= low) & (test["winOdds"] < high)]
        if len(seg) < 20:
            continue

        n_top = max(1, int(len(seg) * 0.2))

        # A model: top 20% by proba_a
        top_a = seg.nlargest(n_top, "proba_a")
        hit_rate_a = top_a["upset"].mean()

        # C model: top 20% by proba_c
        top_c = seg.nlargest(n_top, "proba_c")
        hit_rate_c = top_c["upset"].mean()

        # Baseline: overall upset rate in this segment
        baseline = seg["upset"].mean()

        improvement = hit_rate_c - hit_rate_a
        lift_a = hit_rate_a / baseline if baseline > 0 else 0
        lift_c = hit_rate_c / baseline if baseline > 0 else 0

        results.append({
            "segment": seg_name,
            "n_horses": len(seg),
            "n_top20pct": n_top,
            "baseline_upset_rate": round(baseline, 4),
            "A_top20_hit_rate": round(hit_rate_a, 4),
            "C_top20_hit_rate": round(hit_rate_c, 4),
            "C_minus_A": round(improvement, 4),
            "Lift_A": round(lift_a, 2),
            "Lift_C": round(lift_c, 2),
        })

        logger.info(
            f"  {seg_name:8s} | n={len(seg):>4} | baseline={baseline:.4f} | "
            f"A top20%={hit_rate_a:.4f} (Lift {lift_a:.2f}) | "
            f"C top20%={hit_rate_c:.4f} (Lift {lift_c:.2f}) | "
            f"C-A = {improvement:+.4f}"
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_DIR / "segment_comparison.csv", index=False)
    logger.info(f"\n  Saved: segment_comparison.csv")

    # Visualization
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df_results))
    width = 0.3

    ax.bar(x - width/2, df_results["A_top20_hit_rate"], width, label=f"A ({best_a_model_name}, q only)", color="#1976d2")
    ax.bar(x + width/2, df_results["C_top20_hit_rate"], width, label=f"C ({best_c_model_name}, q + features)", color="#ff7043")

    # Baseline line per segment
    for i, (_, row) in enumerate(df_results.iterrows()):
        ax.hlines(row["baseline_upset_rate"], i - 0.4, i + 0.4, colors="gray", linestyles="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df_results["segment"])
    ax.set_xlabel("winOdds Segment")
    ax.set_ylabel("Top 20% Upset Hit Rate")
    ax.set_title("Upset Prediction: A (odds only) vs C (odds + features) by Odds Segment")
    ax.legend()

    # Annotation: improvement
    for i, (_, row) in enumerate(df_results.iterrows()):
        if row["C_minus_A"] > 0:
            ax.annotate(f"+{row['C_minus_A']:.3f}",
                        xy=(i + width/2, row["C_top20_hit_rate"]),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9, color="green", fontweight="bold")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "segment_chart.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: segment_chart.png")

    # Summary
    logger.info(f"\n  --- Summary ---")
    if len(df_results) > 0:
        best_seg = df_results.loc[df_results["C_minus_A"].idxmax()]
        logger.info(f"  Biggest C improvement over A: {best_seg['segment']} ({best_seg['C_minus_A']:+.4f})")
        logger.info(f"  C Lift in that segment: {best_seg['Lift_C']:.2f}x vs baseline")

    logger.info("\n" + "=" * 60)
    logger.info("05_longshot_segment_eval.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
