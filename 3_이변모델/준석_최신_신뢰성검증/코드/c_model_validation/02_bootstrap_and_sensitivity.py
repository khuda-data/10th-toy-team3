"""
02_bootstrap_and_sensitivity.py — C모델 최종 전략(winOdds 10~50배+상위10%)의
ROI 부트스트랩 신뢰구간 + 극단값 민감도 분석

계보1(다크호스 재현, 02번 스크립트)과 동일한 방법론을 C모델에 적용한다.

실행:
    python src/c_model_validation/02_bootstrap_and_sensitivity.py

출력:
    results/c_model_validation/roi_bootstrap.csv
    results/c_model_validation/roi_bootstrap_hist.png
    results/c_model_validation/sensitivity.csv
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("results/c_model_validation")
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def main():
    df = pd.read_csv(OUTPUT_DIR / "final_strategy_predictions.csv")
    hit = df["upset"].values
    odds = df["_winOdds"].values
    k = len(df)

    point_roi = roi_of(hit, odds)
    logger.info(f"최종 전략 {k}건 베팅 | 적중 {int(hit.sum())}건 | 점추정 ROI {point_roi:+.1f}%")

    # --- 부트스트랩 CI ---
    boot_rois = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = RNG.integers(0, k, size=k)
        boot_rois[i] = roi_of(hit[idx], odds[idx])

    ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])
    includes_zero = ci_low <= 0 <= ci_high
    p_profit = (boot_rois > 0).mean() * 100
    logger.info(f"부트스트랩 95% CI [{ci_low:+.1f}%, {ci_high:+.1f}%] | "
                f"0 포함: {'예' if includes_zero else '아니오'} | P(수익)={p_profit:.1f}%")

    pd.DataFrame([{
        "n_bets": k, "n_hits": int(hit.sum()), "point_roi": point_roi,
        "ci_low_95": ci_low, "ci_high_95": ci_high,
        "ci_includes_zero": includes_zero, "p_profit_pct": p_profit,
        "n_bootstrap": N_BOOTSTRAP,
    }]).to_csv(OUTPUT_DIR / "roi_bootstrap.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boot_rois, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="ROI = 0%")
    ax.axvline(point_roi, color="#DD8452", linestyle="-", linewidth=2, label=f"점추정 {point_roi:+.1f}%")
    ax.axvline(ci_low, color="gray", linestyle=":", linewidth=1)
    ax.axvline(ci_high, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("부트스트랩 ROI (%)")
    ax.set_ylabel("빈도")
    ax.set_title(f"C모델 최종 전략 ROI 부트스트랩 분포 (n={N_BOOTSTRAP})\n"
                 f"95% CI [{ci_low:+.1f}%, {ci_high:+.1f}%] — 0 {'포함' if includes_zero else '미포함'}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roi_bootstrap_hist.png", dpi=120)
    plt.close(fig)

    # --- 극단값 민감도 ---
    hit_idx = np.where(hit == 1)[0]
    hit_odds_sorted = hit_idx[np.argsort(-odds[hit_idx])]
    total_return = np.where(hit == 1, odds, 0.0).sum()

    rows = []
    for n_removed in [0, 1, 3, 5]:
        removed = set(hit_odds_sorted[:n_removed])
        keep = [i for i in range(k) if i not in removed]
        roi_kept = roi_of(hit[keep], odds[keep])
        removed_return = sum(odds[i] for i in hit_odds_sorted[:n_removed])
        pct_of_total_return = removed_return / total_return * 100 if total_return > 0 else 0
        rows.append({
            "n_removed": n_removed, "roi_pct": roi_kept,
            "removed_odds": [round(float(odds[i]), 1) for i in hit_odds_sorted[:n_removed]],
            "pct_of_total_return_from_removed": pct_of_total_return,
        })
        logger.info(f"상위 고배당 {n_removed}건 제외 -> ROI {roi_kept:+.1f}% "
                    f"(제외분이 전체 회수액의 {pct_of_total_return:.1f}%)")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig")
    logger.info("완료")


if __name__ == "__main__":
    main()
