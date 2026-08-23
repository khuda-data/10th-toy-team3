"""
01_odds_filter_strategy.py — 배당률 제외 다크호스 모델의 최종 전략 확정

results/final_validation/darkhorse_model.pkl(q 미포함, upset_B 타겟)의 test
예측값을 갖고, plcOdds 구간 필터 × 상위 N% 그리드서치로 최적 전략을 찾는다.
방법론은 12_final_strategy.py(C모델 계보)와 동일하되, 모델은 완전히 다르다
(q 없음, plcOdds는 결과 정산에만 사용).

실행:
    python src/darkhorse_final/01_odds_filter_strategy.py

출력:
    results/darkhorse_final/strategy_grid.csv
    results/darkhorse_final/best_strategy.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

PRED_PATH = Path("results/final_validation/darkhorse_test_predictions.csv")
OUTPUT_DIR = Path("results/darkhorse_final")
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 1000


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def bootstrap_roi(hit, odds, n_boot=N_BOOTSTRAP):
    n = len(hit)
    rois = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        rois[i] = roi_of(hit[idx], odds[idx])
    ci_low, ci_high = np.percentile(rois, [2.5, 97.5])
    p_profit = (rois > 0).mean() * 100
    return ci_low, ci_high, p_profit


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PRED_PATH)
    logger.info(f"test {len(df):,}건 | 기저율 {df['upset_B'].mean():.4f}")

    segments = [
        ("전체 (필터 없음)", 0, 99999),
        ("plcOdds >= 3배", 3, 99999),
        ("plcOdds >= 5배", 5, 99999),
        ("plcOdds >= 10배", 10, 99999),
        ("plcOdds 3~10배", 3, 10),
        ("plcOdds 5~15배", 5, 15),
        ("plcOdds 5~20배", 5, 20),
    ]

    rows = []
    for seg_name, lo, hi in segments:
        seg = df[(df["odds"] >= lo) & (df["odds"] < hi)]
        if len(seg) < 10:
            continue
        for top_pct in [5, 10, 20, 30]:
            n_bets = max(1, int(len(seg) * top_pct / 100))
            top = seg.nlargest(n_bets, "proba")
            hit = top["upset_B"].values
            odds = top["odds"].values
            roi = roi_of(hit, odds)
            ci_low, ci_high, p_profit = bootstrap_roi(hit, odds)
            rows.append({
                "filter": seg_name, "top_pct": top_pct, "n_pool": len(seg),
                "n_bets": n_bets, "n_hits": int(hit.sum()), "hit_rate": hit.mean(),
                "avg_odds": odds.mean(), "roi_pct": roi,
                "ci_low": ci_low, "ci_high": ci_high, "p_profit": p_profit,
                "ci_excludes_zero": ci_low > 0,
            })

    grid = pd.DataFrame(rows)
    grid.to_csv(OUTPUT_DIR / "strategy_grid.csv", index=False, encoding="utf-8-sig")

    logger.info(f"\n{'필터':<20s}{'top%':>6s}{'베팅':>6s}{'적중':>6s}{'ROI':>9s}{'CI_low':>9s}{'CI_hi':>9s}{'P(+)':>7s}")
    for _, r in grid.sort_values("roi_pct", ascending=False).iterrows():
        logger.info(f"  {r['filter']:<18s}{r['top_pct']:>5.0f}%{r['n_bets']:>6.0f}{r['n_hits']:>6.0f}"
                    f"{r['roi_pct']:>+8.1f}%{r['ci_low']:>+8.1f}%{r['ci_high']:>+8.1f}%{r['p_profit']:>6.1f}%")

    # 실무 기준: P(수익) >= 70%, 베팅 수 >= 30 인 것 중 ROI 최고
    viable = grid[(grid["p_profit"] >= 70) & (grid["n_bets"] >= 30)]
    if len(viable) == 0:
        viable = grid[grid["n_bets"] >= 30]
    best = viable.sort_values("roi_pct", ascending=False).iloc[0]
    logger.info(f"\n>>> 채택 전략: {best['filter']} + 상위{best['top_pct']:.0f}% <<<")
    logger.info(f"    ROI {best['roi_pct']:+.1f}% | CI [{best['ci_low']:+.1f}%, {best['ci_high']:+.1f}%] | "
                f"P(수익) {best['p_profit']:.1f}% | 베팅 {best['n_bets']:.0f}건 적중 {best['n_hits']:.0f}건")

    pd.DataFrame([best]).to_csv(OUTPUT_DIR / "best_strategy.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/darkhorse_final/strategy_grid.csv, best_strategy.csv")


if __name__ == "__main__":
    main()
