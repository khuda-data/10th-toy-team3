"""
04_validation.py — 괴리 구간별 검증 + 시장 편향(FLB) 분석 (7~8단계)

7단계: 괴리 구간별 모델확률·시장확률·실제 승률 비교
8단계: Favorite-Longshot Bias 분석 (기본 + 조절변수별)

실행:
    python src/1_전처리/04_validation.py

의존:
    - 02_market_gap.py 실행 후 생성된 results/market_gap.csv
    - data/processed/ 의 market_odds.csv, race_outcome.csv, model_features.csv
"""

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
    MODERATORS,
    ODDS_BINS,
    ODDS_LABELS,
    TARGET_COL,
    setup_logging,
    setup_plot_style,
    ensure_dirs,
)

logger = setup_logging()


# ============================================================
# 7단계: 괴리 구간별 실제 승률 비교
# ============================================================

def validate_gap_deciles():
    """괴리를 10분위로 나누고 구간별 모델/시장/실제 승률을 비교한다."""
    logger.info("=" * 60)
    logger.info("[7단계] 괴리 구간별 실제 승률 비교")
    logger.info("=" * 60)

    # 데이터 로드
    gap_df = pd.read_csv(RESULTS_DIR / "market_gap.csv")
    outcome_df = pd.read_csv(DATA_DIR / "race_outcome.csv")

    # 실제 win 결과 조인 (race_outcome에 win이 없으면 model_features에서)
    if TARGET_COL in outcome_df.columns:
        gap_df = gap_df.merge(outcome_df[["entry_id", TARGET_COL]], on="entry_id", how="left")
    else:
        features_df = pd.read_csv(DATA_DIR / "model_features.csv", usecols=["entry_id", "fold", TARGET_COL])
        test_win = features_df[features_df["fold"] == "test"][["entry_id", TARGET_COL]]
        gap_df = gap_df.merge(test_win, on="entry_id", how="left")

    # 10분위 구간화
    gap_df["gap_decile"] = pd.qcut(gap_df["gap"], q=10, labels=False, duplicates="drop")

    # 구간별 집계
    agg = gap_df.groupby("gap_decile").agg(
        model_prob_mean=("model_prob", "mean"),
        market_prob_mean=("market_prob", "mean"),
        actual_winrate=(TARGET_COL, "mean"),
        gap_mean=("gap", "mean"),
        count=("entry_id", "count"),
    ).reset_index()

    agg.to_csv(RESULTS_DIR / "gap_validation.csv", index=False)
    logger.info(f"  저장: results/gap_validation.csv")
    logger.info(f"\n{agg.to_string(index=False)}")

    # 시각화
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    x = agg["gap_decile"]
    ax.plot(x, agg["model_prob_mean"], "b-o", label="Model Prob", linewidth=2)
    ax.plot(x, agg["market_prob_mean"], "r-s", label="Market Prob", linewidth=2)
    ax.plot(x, agg["actual_winrate"], "g-^", label="Actual Win Rate", linewidth=2)
    ax.set_xlabel("Gap Decile (0=model<market -> 9=model>market)")
    ax.set_ylabel("Probability")
    ax.set_title("Gap Decile: Model vs Market vs Actual Win Rate")
    ax.legend()
    ax.set_xticks(range(len(agg)))
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "gap_validation.png", bbox_inches="tight")
    plt.close()

    logger.info(f"  저장: results/gap_validation.png")

    # 해석 요약
    high_gap = agg[agg["gap_decile"] >= 8]
    low_gap = agg[agg["gap_decile"] <= 1]

    logger.info("\n  --- 해석 ---")
    if len(high_gap) > 0:
        model_err = abs(high_gap["model_prob_mean"].mean() - high_gap["actual_winrate"].mean())
        market_err = abs(high_gap["market_prob_mean"].mean() - high_gap["actual_winrate"].mean())
        winner = "모델" if model_err < market_err else "시장"
        logger.info(f"  괴리 상위(모델>시장) 구간: {winner}이 실제에 더 가까움 "
                    f"(모델 오차: {model_err:.4f}, 시장 오차: {market_err:.4f})")

    if len(low_gap) > 0:
        model_err = abs(low_gap["model_prob_mean"].mean() - low_gap["actual_winrate"].mean())
        market_err = abs(low_gap["market_prob_mean"].mean() - low_gap["actual_winrate"].mean())
        winner = "모델" if model_err < market_err else "시장"
        logger.info(f"  괴리 하위(모델<시장) 구간: {winner}이 실제에 더 가까움 "
                    f"(모델 오차: {model_err:.4f}, 시장 오차: {market_err:.4f})")


# ============================================================
# 8단계: 시장 편향 (Favorite-Longshot Bias)
# ============================================================

def compute_flb_overall(df: pd.DataFrame):
    """배당 구간별 실제 승률 vs 암묵적확률 기본 분석."""
    logger.info("=" * 60)
    logger.info("[8단계] 시장 편향 (Favorite-Longshot Bias) 분석")
    logger.info("=" * 60)

    # 배당 구간 분류
    df["odds_group"] = pd.cut(
        df["winOdds"], bins=ODDS_BINS, labels=ODDS_LABELS, right=False
    )

    # 경주 내 정규화 시장 확률
    df["inv_odds"] = 1.0 / df["winOdds"]
    df["market_prob"] = df.groupby("race_id")["inv_odds"].transform(lambda x: x / x.sum())

    # 구간별 집계
    flb = df.groupby("odds_group", observed=True).agg(
        actual_winrate=(TARGET_COL, "mean"),
        market_prob_mean=("market_prob", "mean"),
        count=("entry_id", "count"),
    ).reset_index()
    flb["bias"] = flb["actual_winrate"] - flb["market_prob_mean"]

    flb.to_csv(RESULTS_DIR / "flb_overall.csv", index=False)
    logger.info(f"  저장: results/flb_overall.csv")
    logger.info(f"\n{flb.to_string(index=False)}")

    # 시각화
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = range(len(flb))
    width = 0.35
    ax.bar([p - width / 2 for p in x_pos], flb["actual_winrate"], width, label="Actual Win Rate", color="steelblue")
    ax.bar([p + width / 2 for p in x_pos], flb["market_prob_mean"], width, label="Market Prob", color="coral")
    ax.set_xlabel("Odds Group")
    ax.set_ylabel("Probability")
    ax.set_title("Favorite-Longshot Bias: Actual Win Rate vs Market Prob")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(flb["odds_group"].values)
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "flb_overall.png", bbox_inches="tight")
    plt.close()

    logger.info(f"  저장: results/flb_overall.png")

    # FLB 해석
    logger.info("\n  --- FLB 해석 ---")
    fav = flb[flb["odds_group"] == "1배대"]
    long = flb[flb["odds_group"] == "10배+"]
    if len(fav) > 0 and len(long) > 0:
        fav_bias = fav["bias"].values[0]
        long_bias = long["bias"].values[0]
        logger.info(f"  인기마(1배대) bias: {fav_bias:+.4f} (양수=과소평가)")
        logger.info(f"  비인기마(10배+) bias: {long_bias:+.4f} (음수=과대평가)")
        if fav_bias > 0 and long_bias < 0:
            logger.info("  → 전형적인 Favorite-Longshot Bias 확인됨")

    return df


def analyze_flb_by_moderator(df: pd.DataFrame, mod_name: str, mod_col: str):
    """조절변수별 FLB 분석."""
    logger.info(f"\n  --- 조절변수: {mod_name} ({mod_col}) ---")

    # 조절변수 구간화
    if mod_col == "rcDist":
        df["mod_group"] = pd.cut(
            df[mod_col],
            bins=[0, 1200, 1600, 2500],
            labels=["Short(~1200)", "Mid(1300~1600)", "Long(1700~)"],
        )
    elif mod_col == "n_run":
        df["mod_group"] = pd.cut(
            df[mod_col],
            bins=[0, 8, 11, 20],
            labels=["Small(6-8)", "Mid(9-11)", "Large(12-16)"],
        )
    elif mod_col == "totalAmt":
        df["mod_group"] = pd.qcut(df[mod_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    elif mod_col == "track":
        # 주요 상태만
        def simplify_track(val):
            if pd.isna(val):
                return "Other"
            val = str(val)
            if "건조" in val:
                return "Dry"
            elif "양호" in val:
                return "Good"
            elif "다습" in val:
                return "Humid"
            elif "포화" in val or "불량" in val:
                return "Wet/Bad"
            return "Other"
        df["mod_group"] = df[mod_col].apply(simplify_track)
    else:
        # rank 등 범주형 — 그대로 사용
        df["mod_group"] = df[mod_col].astype(str)

    # 조절변수 × 배당구간 집계
    flb_mod = df.groupby(["mod_group", "odds_group"], observed=True).agg(
        actual_winrate=(TARGET_COL, "mean"),
        market_prob_mean=("market_prob", "mean"),
        count=("entry_id", "count"),
    ).reset_index()
    flb_mod["bias"] = flb_mod["actual_winrate"] - flb_mod["market_prob_mean"]

    # 저장
    safe_name = mod_col.replace("/", "_")
    flb_mod.to_csv(RESULTS_DIR / f"flb_by_{safe_name}.csv", index=False)
    logger.info(f"  저장: results/flb_by_{safe_name}.csv ({len(flb_mod)}행)")

    # 시각화
    setup_plot_style()
    groups = sorted(df["mod_group"].dropna().unique())
    n_groups = len(groups)

    if n_groups <= 6:
        fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 5), sharey=True)
        if n_groups == 1:
            axes = [axes]

        for ax, grp in zip(axes, groups):
            sub = flb_mod[flb_mod["mod_group"] == grp]
            if len(sub) == 0:
                continue
            x_pos = range(len(sub))
            width = 0.35
            ax.bar([p - width / 2 for p in x_pos], sub["actual_winrate"], width, label="Actual", color="steelblue")
            ax.bar([p + width / 2 for p in x_pos], sub["market_prob_mean"], width, label="Market", color="coral")
            ax.set_title(f"{mod_col}={grp}", fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(sub["odds_group"].values, rotation=45, ha="right", fontsize=8)
            if ax == axes[0]:
                ax.set_ylabel("Probability")
                ax.legend(fontsize=8)

        plt.suptitle(f"FLB by {mod_col}", fontsize=13)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"flb_by_{safe_name}.png", bbox_inches="tight")
        plt.close()
        logger.info(f"  저장: results/flb_by_{safe_name}.png")
    else:
        # 그룹이 너무 많으면 히트맵으로
        pivot = flb_mod.pivot_table(index="mod_group", columns="odds_group", values="bias")
        fig, ax = plt.subplots(figsize=(10, max(6, n_groups * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0, ax=ax)
        ax.set_title(f"FLB Bias by {mod_col}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"flb_by_{safe_name}.png", bbox_inches="tight")
        plt.close()
        logger.info(f"  저장: results/flb_by_{safe_name}.png (히트맵)")

    # 정리
    df.drop(columns=["mod_group"], inplace=True, errors="ignore")


# ============================================================
# 메인 실행
# ============================================================

def main():
    ensure_dirs()

    # --- 7단계 ---
    validate_gap_deciles()

    # --- 8단계 ---
    logger.info("\n")

    # 데이터 준비 (test set 전체)
    df_features = pd.read_csv(
        DATA_DIR / "model_features.csv",
        dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str},
    )
    df_odds = pd.read_csv(DATA_DIR / "market_odds.csv")
    df_outcome = pd.read_csv(DATA_DIR / "race_outcome.csv")

    # test set 필터
    test = df_features[df_features["fold"] == "test"].copy()

    # 조인
    test = test.merge(df_odds[["entry_id", "winOdds"]], on="entry_id", how="left")
    if TARGET_COL not in test.columns:
        test = test.merge(df_outcome[["entry_id", TARGET_COL]], on="entry_id", how="left")

    # FLB 기본 분석
    test = compute_flb_overall(test)

    # 조절변수별 분석
    for mod_name, mod_col in MODERATORS.items():
        if mod_col in test.columns:
            analyze_flb_by_moderator(test, mod_name, mod_col)
        else:
            logger.warning(f"  조절변수 '{mod_col}' 컬럼이 데이터에 없습니다. 건너뜁니다.")

    logger.info("\n" + "=" * 60)
    logger.info("04_validation.py 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
