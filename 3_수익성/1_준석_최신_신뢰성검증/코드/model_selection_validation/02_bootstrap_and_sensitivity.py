"""
02_bootstrap_and_sensitivity.py — ROI 부트스트랩 신뢰구간 + 극단값 민감도 분석

01_build_models.py가 만든 test 예측을 이용해:
  - 다크호스: 상위10% 베팅의 ROI를 부트스트랩(2,000회)으로 95% CI 계산,
    최고 배당 적중 1건/3건을 제외했을 때 ROI가 어떻게 바뀌는지 비교
  - 인기마 붕괴: "베팅 대상"이 아니라 스크리닝 지표이므로 ROI 대신
    Lift@10%(선별력) 자체를 부트스트랩으로 검증

실행:
    python src/model_selection_validation/02_bootstrap_and_sensitivity.py

출력:
    results/final_validation/darkhorse_roi_bootstrap.csv
    results/final_validation/darkhorse_roi_bootstrap_hist.png
    results/final_validation/darkhorse_sensitivity.csv
    results/final_validation/bust_lift_bootstrap.csv
    results/final_validation/bust_lift_bootstrap_hist.png
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

OUTPUT_DIR = Path("results/final_validation")
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
TOP_PCT = 0.10


def roi_of(hit, odds):
    """플랫 베팅 ROI(%) — 적중하면 odds배, 실패하면 -100%."""
    returns = np.where(hit == 1, odds, 0.0) - 1.0
    return returns.mean() * 100


def darkhorse_analysis():
    logger.info("=" * 60)
    logger.info("[다크호스] ROI 부트스트랩 + 극단값 민감도")

    df = pd.read_csv(OUTPUT_DIR / "darkhorse_test_predictions.csv")
    k = max(1, int(len(df) * TOP_PCT))
    top = df.nlargest(k, "proba").reset_index(drop=True)

    hit = top["upset_B"].values
    odds = top["odds"].values
    point_roi = roi_of(hit, odds)
    logger.info(f"  상위{TOP_PCT:.0%} 베팅 {k}건 | 적중 {int(hit.sum())}건 | 점추정 ROI {point_roi:+.1f}%")

    # --- 부트스트랩 CI ---
    boot_rois = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = RNG.integers(0, k, size=k)
        boot_rois[i] = roi_of(hit[idx], odds[idx])

    ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])
    includes_zero = ci_low <= 0 <= ci_high
    p_profit = (boot_rois > 0).mean() * 100

    logger.info(f"  부트스트랩 95% CI: [{ci_low:+.1f}%, {ci_high:+.1f}%] | "
                f"0 포함: {'예' if includes_zero else '아니오'} | P(수익)={p_profit:.1f}%")

    pd.DataFrame([{
        "n_bets": k, "n_hits": int(hit.sum()), "point_roi": point_roi,
        "ci_low_95": ci_low, "ci_high_95": ci_high,
        "ci_includes_zero": includes_zero, "p_profit_pct": p_profit,
        "n_bootstrap": N_BOOTSTRAP,
    }]).to_csv(OUTPUT_DIR / "darkhorse_roi_bootstrap.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boot_rois, bins=60, color="#4C72B0", alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="ROI = 0%")
    ax.axvline(point_roi, color="#DD8452", linestyle="-", linewidth=2, label=f"점추정 {point_roi:+.1f}%")
    ax.axvline(ci_low, color="gray", linestyle=":", linewidth=1)
    ax.axvline(ci_high, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("부트스트랩 ROI (%)")
    ax.set_ylabel("빈도")
    ax.set_title(f"다크호스 상위{TOP_PCT:.0%} 베팅 ROI 부트스트랩 분포 (n={N_BOOTSTRAP})\n"
                 f"95% CI [{ci_low:+.1f}%, {ci_high:+.1f}%] — 0 {'포함' if includes_zero else '미포함'}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "darkhorse_roi_bootstrap_hist.png", dpi=120)
    plt.close(fig)

    # --- 극단값 민감도: 최고 배당 적중 1건/3건 제외 ---
    hit_idx = np.where(hit == 1)[0]
    hit_odds_sorted = hit_idx[np.argsort(-odds[hit_idx])]  # 배당 높은 순

    total_return = np.where(hit == 1, odds, 0.0).sum()
    total_bet = k
    rows = []
    for n_removed in [0, 1, 3, 5]:
        removed = set(hit_odds_sorted[:n_removed])
        keep = [i for i in range(k) if i not in removed]
        roi_kept = roi_of(hit[keep], odds[keep])
        removed_return = sum(odds[i] for i in hit_odds_sorted[:n_removed])
        pct_of_total_return = removed_return / total_return * 100 if total_return > 0 else 0
        rows.append({
            "n_removed": n_removed,
            "roi_pct": roi_kept,
            "removed_odds": [round(float(odds[i]), 1) for i in hit_odds_sorted[:n_removed]],
            "pct_of_total_return_from_removed": pct_of_total_return,
        })
        logger.info(f"  상위 고배당 {n_removed}건 제외 -> ROI {roi_kept:+.1f}% "
                    f"(제외분이 전체 회수액의 {pct_of_total_return:.1f}%)")

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "darkhorse_sensitivity.csv", index=False, encoding="utf-8-sig")


def bust_analysis():
    logger.info("=" * 60)
    logger.info("[인기마 붕괴] Lift@10% 부트스트랩 (베팅 대상이 아니므로 ROI 대신 선별력 검증)")

    df = pd.read_csv(OUTPUT_DIR / "bust_test_predictions.csv")
    y = df["upset_A"].values
    proba = df["proba"].values
    n = len(df)
    k = max(1, int(n * TOP_PCT))

    def lift(y_arr, p_arr):
        order = np.argsort(-p_arr)
        top_rate = y_arr[order[:k]].mean()
        base_rate = y_arr.mean()
        return top_rate / base_rate if base_rate > 0 else np.nan

    point_lift = lift(y, proba)
    logger.info(f"  전체 {n}건 | 상위{TOP_PCT:.0%} Lift 점추정 {point_lift:.2f}")

    boot_lifts = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = RNG.integers(0, n, size=n)
        boot_lifts[i] = lift(y[idx], proba[idx])

    ci_low, ci_high = np.percentile(boot_lifts, [2.5, 97.5])
    includes_one = ci_low <= 1.0 <= ci_high  # Lift=1.0이 "무작위와 같음" 기준선
    p_above_one = (boot_lifts > 1.0).mean() * 100

    logger.info(f"  부트스트랩 95% CI: [{ci_low:.2f}, {ci_high:.2f}] | "
                f"1.0(무작위) 포함: {'예' if includes_one else '아니오'} | P(Lift>1)={p_above_one:.1f}%")

    pd.DataFrame([{
        "n_total": n, "point_lift": point_lift,
        "ci_low_95": ci_low, "ci_high_95": ci_high,
        "ci_includes_random_baseline": includes_one, "p_lift_above_1_pct": p_above_one,
        "n_bootstrap": N_BOOTSTRAP,
    }]).to_csv(OUTPUT_DIR / "bust_lift_bootstrap.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(boot_lifts, bins=60, color="#55A868", alpha=0.85)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Lift = 1.0 (무작위)")
    ax.axvline(point_lift, color="#DD8452", linestyle="-", linewidth=2, label=f"점추정 {point_lift:.2f}")
    ax.axvline(ci_low, color="gray", linestyle=":", linewidth=1)
    ax.axvline(ci_high, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("부트스트랩 Lift@10%")
    ax.set_ylabel("빈도")
    ax.set_title(f"인기마 붕괴 상위{TOP_PCT:.0%} Lift 부트스트랩 분포 (n={N_BOOTSTRAP})\n"
                 f"95% CI [{ci_low:.2f}, {ci_high:.2f}] — 1.0 {'포함' if includes_one else '미포함'}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "bust_lift_bootstrap_hist.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    darkhorse_analysis()
    bust_analysis()
    logger.info("완료")
