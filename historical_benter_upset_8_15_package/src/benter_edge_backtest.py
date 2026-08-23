"""시장기준선 결합확률로 실제 베팅 ROI를 검증한다 — 정책 선택까지 워크포워드로 잠근다.

benter_market_anchored_walkforward.py가 만든 표본외 결합확률을 받아서,
"어떤 배당구간에서 엣지 상위 몇 %를 사는가" 라는 **정책 자체를 과거 fold에서만
고르고** 다음 fold에 그대로 적용한다.

왜 이렇게까지 하는가:
  단순히 전체 표본에서 배당구간별 ROI를 훑으면 "8~15배 구간에서 ROI +32.7%,
  CI가 0을 배제"같은 결과가 나온다. 하지만 그건 **테스트 데이터를 보고 구간을
  고른 것**이라 그대로 믿으면 안 된다(data snooping). 배당구간 6개 × 엣지비율
  5개 = 30개 정책을 훑으면, 진짜 edge가 하나도 없어도 5% 유의수준에서 평균
  1.5개는 우연히 "유의"하게 나온다.

  그래서 이 스크립트는 정책 선택을 과거 데이터 안에 가둔다. fold k를 베팅할 때
  fold<k 만 보고 정책을 고른다. 이렇게 나온 수익률이 진짜 실전 수익률이다.

추가 안전장치:
  - 정책 스캔 개수를 명시하고 Bonferroni/BH 보정을 함께 보고한다.
  - 무작위 정책(플라시보) 대조군을 같이 돌려 "이 정도 ROI는 우연으로도 나오는가"를 본다.
  - 경주일 블록부트스트랩 CI(같은 날 경주 간 상관 보존).
  - 25% fractional Kelly 자금관리 시뮬레이션(뱅크롤 2% 상한)으로 MDD/Sharpe까지.

실행: python -m src.training.core.benter_edge_backtest
(먼저 python -m src.training.core.benter_market_anchored_walkforward 을 실행할 것)
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.finance_stats import annualization_factor  # noqa: E402

ROOT = _PROJECT_ROOT
BASE = ROOT / "outputs/reports/benter_market_anchored_20260820"
PROBS = BASE / "walkforward_combined_probabilities.csv"

ODDS_BINS = [(1.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, 30.0), (30.0, 9999.0)]
EDGE_FRACTIONS = [0.05, 0.10, 0.20, 0.35, 0.50]
N_BOOT = 4000
RNG = np.random.default_rng(20260820)
KELLY_FRACTION = 0.25
MAX_STAKE_FRACTION = 0.02
STARTING_BANKROLL = 100.0


def block_bootstrap_ci(values: np.ndarray, dates: np.ndarray, n_boot: int = N_BOOT, seed: int = 20260820):
    if len(values) < 10:
        return None, None
    rng = np.random.default_rng(seed)
    unique = np.unique(dates)
    if len(unique) < 5:
        return None, None
    index = {d: np.flatnonzero(dates == d) for d in unique}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        boots[b] = values[np.concatenate([index[d] for d in sampled])].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def apply_policy(frame: pd.DataFrame, odds_lo: float, odds_hi: float, fraction: float) -> pd.DataFrame:
    sub = frame[(frame.winOdds >= odds_lo) & (frame.winOdds < odds_hi)]
    if sub.empty:
        return sub
    n = max(1, int(np.ceil(len(sub) * fraction)))
    return sub.nlargest(n, "edge")


def evaluate(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {"n_bets": 0, "hit_rate": np.nan, "unit_roi": np.nan,
                "roi_ci_95_lo": None, "roi_ci_95_hi": None, "ci_confirmed_positive": False}
    lo, hi = block_bootstrap_ci(sub["ret"].to_numpy(), sub["rcDate"].to_numpy())
    return {"n_bets": int(len(sub)), "hit_rate": float(sub["win"].mean()),
            "unit_roi": float(sub["ret"].mean()), "roi_ci_95_lo": lo, "roi_ci_95_hi": hi,
            "ci_confirmed_positive": bool(lo is not None and lo > 0)}


def fractional_kelly(sub: pd.DataFrame) -> dict:
    frame = sub.sort_values(["rcDate", "race_id"], kind="stable")
    bankroll, rows = STARTING_BANKROLL, []
    for row in frame.itertuples(index=False):
        odds = float(row.winOdds)
        prob = float(row.p_combined)
        kelly = (prob * odds - 1.0) / (odds - 1.0) if odds > 1.0 else 0.0
        frac = min(max(kelly, 0.0) * KELLY_FRACTION, MAX_STAKE_FRACTION)
        if frac <= 0.0:
            continue
        stake = bankroll * frac
        profit = stake * float(row.ret)
        bankroll += profit
        rows.append({"rcDate": row.rcDate, "after": bankroll, "profit": profit,
                     "before": bankroll - profit})
    log = pd.DataFrame(rows)
    if log.empty:
        return {"kelly_bets": 0, "ending_bankroll": STARTING_BANKROLL,
                "bankroll_return": 0.0, "max_drawdown": 0.0, "sharpe": None,
                "sharpe_annualization_days": None}
    wealth = np.r_[STARTING_BANKROLL, log["after"].to_numpy(dtype=float)]
    peaks = np.maximum.accumulate(wealth)
    daily = log.groupby("rcDate").agg(start=("before", "first"), profit=("profit", "sum"))
    vals = (daily["profit"] / daily["start"]).to_numpy(dtype=float)
    # 주의: sqrt(252)는 미국 주식시장 연간거래일 관행이다. 이 프로젝트는 실제
    # 베팅일수가 연 100일 안팎(실측 시점마다 다름)이라 252를 쓰면 Sharpe가
    # 약 1.4~1.6배 과대평가된다. 실제 관측된 베팅일 빈도로 연환산한다.
    ann = annualization_factor(daily.index.to_numpy())
    sharpe = None
    sharpe_ann_days = None
    if len(vals) >= 2 and np.std(vals, ddof=1) > 1e-15 and ann is not None:
        sharpe = float(np.mean(vals) / np.std(vals, ddof=1) * ann)
        sharpe_ann_days = float(ann ** 2)
    return {"kelly_bets": int(len(log)), "ending_bankroll": float(log["after"].iloc[-1]),
            "bankroll_return": float(log["after"].iloc[-1] / STARTING_BANKROLL - 1.0),
            "max_drawdown": float(-(wealth / peaks - 1.0).min()), "sharpe": sharpe,
            "sharpe_annualization_days": sharpe_ann_days}


def main() -> None:
    if not PROBS.is_file():
        print("=" * 70)
        print(f"[중단] {PROBS} 가 없습니다.")
        print("먼저 실행: python -m src.training.core.benter_market_anchored_walkforward")
        print("=" * 70)
        sys.exit(1)

    b = pd.read_csv(PROBS)
    b = b[b.winOdds.between(1.0, 9999.0, inclusive="neither")].copy()
    b["ret"] = b["win"] * b["winOdds"] - 1.0
    folds = sorted(b["fold"].unique())
    n_policies = len(ODDS_BINS) * len(EDGE_FRACTIONS)
    print("=" * 78)
    print("결합확률 EDGE 백테스트 — 정책 선택까지 워크포워드로 잠금")
    print("=" * 78)
    print(f"  표본외 {b.race_id.nunique()}경주 / {len(b)}출전 / fold {folds}")
    print(f"  스캔 정책 수 = 배당구간 {len(ODDS_BINS)} x 엣지비율 {len(EDGE_FRACTIONS)} = {n_policies}")

    # ---------- (A) 참고용: 전체 표본 스캔 (data snooping 위험 있음, 그대로 믿으면 안 됨) ----------
    scan_rows = []
    for lo_o, hi_o in ODDS_BINS:
        for frac in EDGE_FRACTIONS:
            sub = apply_policy(b, lo_o, hi_o, frac)
            scan_rows.append({"odds_lo": lo_o, "odds_hi": hi_o, "fraction": frac, **evaluate(sub)})
    scan = pd.DataFrame(scan_rows)
    n_hits = int(scan["ci_confirmed_positive"].sum())
    print(f"\n[A] 전체표본 스캔(참고용, snooping 위험): CI 양수 정책 {n_hits}/{n_policies}개")
    print(f"    ※ 진짜 edge가 0이어도 5% 수준에서 평균 {n_policies*0.05:.1f}개는 우연히 나온다.")
    top = scan.sort_values("unit_roi", ascending=False).head(6)
    print(top[["odds_lo", "odds_hi", "fraction", "n_bets", "hit_rate", "unit_roi",
               "roi_ci_95_lo", "roi_ci_95_hi", "ci_confirmed_positive"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    scan.to_csv(BASE / "policy_scan_full_sample.csv", index=False, encoding="utf-8-sig")

    # ---------- (B) 진짜 시험: 정책을 과거 fold에서만 고르고 다음 fold에 적용 ----------
    print(f"\n[B] 검증잠금 워크포워드 — fold<k 로 정책 선택 -> fold k 에 그대로 적용")
    locked_rows, all_bets = [], []
    for i, k in enumerate(folds):
        if i == 0:
            continue
        past = b[b["fold"].isin(folds[:i])]
        current = b[b["fold"] == k]
        best, best_roi = None, -np.inf
        for lo_o, hi_o in ODDS_BINS:
            for frac in EDGE_FRACTIONS:
                sub = apply_policy(past, lo_o, hi_o, frac)
                if len(sub) < 30:
                    continue
                roi = sub["ret"].mean()
                if roi > best_roi:
                    best_roi, best = roi, (lo_o, hi_o, frac)
        if best is None:
            continue
        lo_o, hi_o, frac = best
        bets = apply_policy(current, lo_o, hi_o, frac)
        all_bets.append(bets)
        locked_rows.append({"fold": k, "policy_odds": f"{lo_o:g}-{hi_o:g}", "policy_fraction": frac,
                            "past_roi": best_roi, **evaluate(bets)})
        print(f"    fold {k}: 과거선택 정책=배당 {lo_o:g}~{hi_o:g}, 상위 {frac:.0%} "
              f"(과거ROI {best_roi:+.1%}) -> 실전 {len(bets)}건 ROI {bets['ret'].mean():+.1%}")

    locked = pd.DataFrame(locked_rows)
    locked.to_csv(BASE / "walkforward_locked_policy.csv", index=False, encoding="utf-8-sig")

    pooled = pd.concat(all_bets, ignore_index=True) if all_bets else pd.DataFrame()
    print("\n" + "=" * 78)
    print("검증잠금 통합 결과 — 이게 실전에서 기대할 수 있는 진짜 수익률")
    print("=" * 78)
    if pooled.empty:
        print("  베팅 없음")
        summary = {"pooled": None}
    else:
        ev = evaluate(pooled)
        kelly = fractional_kelly(pooled)
        print(f"  베팅 {ev['n_bets']}건, 적중률 {ev['hit_rate']:.2%}")
        print(f"  단위 ROI = {ev['unit_roi']:+.2%}   95% CI = "
              f"[{ev['roi_ci_95_lo']:+.2%}, {ev['roi_ci_95_hi']:+.2%}]")
        print(f"  판정: {'★ CI 확정 양수' if ev['ci_confirmed_positive'] else 'CI가 0을 포함 — 수익 확정 불가'}")
        ann_days = kelly['sharpe_annualization_days']
        ann_note = f" (연환산 {ann_days:.0f}베팅일 기준, sqrt(252) 아님)" if ann_days else ""
        print(f"  25% Kelly: 최종뱅크롤 {kelly['ending_bankroll']:.1f} "
              f"(수익률 {kelly['bankroll_return']:+.1%}), MDD {kelly['max_drawdown']:.1%}, "
              f"Sharpe {kelly['sharpe'] if kelly['sharpe'] is None else round(kelly['sharpe'],2)}{ann_note}")

        # ---------- (C) 플라시보 대조군: 엣지를 무작위로 섞어 같은 정책을 돌린다 ----------
        placebo = []
        for seed in range(200):
            rng = np.random.default_rng(1000 + seed)
            shuffled = []
            for i, k in enumerate(folds):
                if i == 0 or len(locked) == 0:
                    continue
                row = locked[locked.fold == k]
                if row.empty:
                    continue
                lo_o, hi_o, frac = (row.policy_odds.iloc[0].split("-")[0],
                                    row.policy_odds.iloc[0].split("-")[1], row.policy_fraction.iloc[0])
                cur = b[b["fold"] == k].copy()
                cur["edge"] = rng.permutation(cur["edge"].to_numpy())
                shuffled.append(apply_policy(cur, float(lo_o), float(hi_o), float(frac)))
            if shuffled:
                placebo.append(pd.concat(shuffled, ignore_index=True)["ret"].mean())
        placebo = np.array(placebo)
        pval = float((placebo >= ev["unit_roi"]).mean())
        print(f"\n  [플라시보] 엣지를 무작위로 섞은 200회 대조군 ROI: "
              f"평균 {placebo.mean():+.2%}, 95분위 {np.percentile(placebo,95):+.2%}")
        print(f"  실제 ROI가 플라시보를 넘을 확률(경험적 p값) = {pval:.3f} -> "
              f"{'신호가 우연이 아님' if pval < 0.05 else '우연과 구분 안 됨'}")
        summary = {"pooled": ev, "kelly": kelly, "placebo_mean_roi": float(placebo.mean()),
                   "placebo_p_value": pval, "n_policies_scanned": n_policies,
                   "full_sample_scan_ci_positive": n_hits}
        pooled.to_csv(BASE / "walkforward_locked_bets.csv", index=False, encoding="utf-8-sig")

    (BASE / "backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n  결과 저장 -> {BASE}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        print("\n" + "=" * 70)
        print("[오류] 스크립트가 예외로 중단됐습니다 — 아래 내용을 그대로 복사해서 물어보세요:")
        print("=" * 70)
        traceback.print_exc()
        input("\n엔터를 누르면 창이 닫힙니다...")
        raise
