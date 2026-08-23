"""
06_feature_importance.py — C 모델 Feature Importance + B와 순위 비교

- C 모델(RF)의 FI Top 20 막대그래프
- q가 몇 위인지 확인
- B 모델 vs C 모델 FI 순위 비교표

실행:
    python src/4_이변모델/06_feature_importance.py
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))  # config.py 는 같은 폴더
from config import setup_plot_style, FEATURE_NAME_MAP, translate_feature_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")


def main():
    logger.info("=" * 60)
    logger.info("[Step 6] Feature Importance Analysis")
    logger.info("=" * 60)

    # Load
    with open(OUTPUT_DIR / "prepared_data.pkl", "rb") as f:
        prep = pickle.load(f)
    with open(OUTPUT_DIR / "trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

    b_features = prep["b_features"]
    c_features = prep["c_features"]

    # Get RF models (B and C)
    model_b = trained_models.get("B (no odds)_RF")
    model_c = trained_models.get("C (q + features)_RF")

    if model_c is None:
        logger.error("  C RF model not found!")
        return

    # --- C model FI ---
    fi_c = pd.DataFrame({
        "feature": c_features,
        "importance": model_c.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_c["rank_c"] = fi_c.index + 1

    # q rank
    q_rank = fi_c[fi_c["feature"] == "q"]["rank_c"].values[0]
    q_importance = fi_c[fi_c["feature"] == "q"]["importance"].values[0]
    logger.info(f"\n  q rank in C model: #{q_rank} (importance={q_importance:.4f})")

    # --- B model FI ---
    fi_b = None
    if model_b is not None:
        fi_b = pd.DataFrame({
            "feature": b_features,
            "importance": model_b.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        fi_b["rank_b"] = fi_b.index + 1

    # --- Top 20 bar chart (C model) ---
    setup_plot_style()
    top20 = fi_c.head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#ff7043" if f == "q" else "#1976d2" for f in top20["feature"]]
    ax.barh(range(len(top20)), top20["importance"].values, color=colors)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — C Model (RF)\n'q' highlighted in orange (rank #{q_rank})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight", dpi=120)
    plt.close()
    logger.info(f"  Saved: feature_importance.png")

    # --- B vs C rank comparison ---
    if fi_b is not None:
        # Merge on feature (q not in B, skip)
        comparison = fi_c[["feature", "rank_c", "importance"]].merge(
            fi_b[["feature", "rank_b"]], on="feature", how="left"
        )
        comparison["rank_change"] = comparison["rank_b"] - comparison["rank_c"]
        comparison = comparison.sort_values("rank_c")

        # Top 15 with translation
        comp_top = comparison.head(15).copy()
        comp_top["description"] = comp_top["feature"].apply(translate_feature_name)

        logger.info(f"\n  --- B vs C Rank Comparison (Top 15 in C) ---")
        logger.info(f"  {'Feature':<25s} {'Desc':<25s} {'C_rank':>6s} {'B_rank':>6s} {'Change':>7s}")
        for _, row in comp_top.iterrows():
            b_rank_str = f"#{int(row['rank_b'])}" if pd.notna(row["rank_b"]) else "N/A"
            change_str = f"{int(row['rank_change']):+d}" if pd.notna(row["rank_change"]) else "NEW"
            logger.info(f"  {row['feature']:<25s} {row['description']:<25s} #{int(row['rank_c']):>4d} {b_rank_str:>6s} {change_str:>7s}")

        comparison.to_csv(OUTPUT_DIR / "fi_comparison.csv", index=False)
        logger.info(f"\n  Saved: fi_comparison.csv")

    # --- Summary ---
    logger.info(f"\n  --- Key Findings ---")
    logger.info(f"  1. q (odds probability) ranks #{q_rank} in C model")
    if q_rank <= 5:
        logger.info(f"     -> q is a TOP feature. Odds info is dominant even with other features.")
    elif q_rank <= 15:
        logger.info(f"     -> q is important but NOT dominant. Other features contribute meaningfully.")
    else:
        logger.info(f"     -> q is relatively low. Model relies more on non-odds features.")

    if fi_b is not None:
        # Check if top features shifted
        b_top5 = fi_b.head(5)["feature"].tolist()
        c_top5 = fi_c.head(5)["feature"].tolist()
        overlap = set(b_top5) & set(c_top5)
        logger.info(f"  2. Top 5 overlap (B vs C): {len(overlap)}/5 features in common")
        logger.info(f"     B top 5: {b_top5}")
        logger.info(f"     C top 5: {c_top5}")

    logger.info("\n" + "=" * 60)
    logger.info("06_feature_importance.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
