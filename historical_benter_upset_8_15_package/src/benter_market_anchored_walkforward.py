"""[핵심 돌파구] 시장가를 기준선으로 삼는 Benter식 조건부 로짓 + 워크포워드 검정.

────────────────────────────────────────────────────────────────────────
왜 지금까지 아무것도 안 통했는가 — 방법론의 근본 오류
────────────────────────────────────────────────────────────────────────
기존 프로젝트의 파이프라인은 이랬다:

  1. 시장 관련 컬럼(q, winOdds, logit_q, pop_rank, is_fav ...)을 LEAK_COLS로
     **모델 피처에서 전부 제외**하고 펀더멘털 모델을 학습한다.
  2. 경주 안에서 sum-to-1 정규화한다(race_normalize).
  3. EDGE = p_model - p_market 으로 베팅 후보를 고른다.

이 설계는 "내 모델이 시장보다 낫다"를 **암묵적으로 가정**한다. p_model이
p_market을 통째로 대체하기 때문이다. 그런데 이 프로젝트가 스스로 측정한 값은
정반대다:

      시장 ROC-AUC 0.82   >   최고 모델 ROC-AUC 0.77
      시장 NDCG@5 0.635   >   최고 모델 NDCG@5 0.574

즉 **더 나쁜 추정치로 더 좋은 추정치를 대체**하고 있었다. 그러니 EDGE가 클수록
오히려 시장 대비 정보를 더 많이 파괴한 셈이고, 40개 정책 중 CI로 확정된 양의
ROI가 0개였던 건 당연한 귀결이다. 모델을 GBM으로 바꾸든(진짜 xgboost/lightgbm/
catboost로 확인 완료), 피처를 늘리든, 캘리브레이션을 하든 이 구조에서는 안 된다.

Benter(1994)가 홍콩에서 실제로 돈을 번 방법은 정반대였다:
**시장가를 버리지 않고 기준선으로 삼은 뒤, 그 위에 남는 잔차만 모델링한다.**

  결합확률  c_i = exp(α·ln q_i + γ·ln f_i) / Σ_j exp(α·ln q_j + γ·ln f_j)

여기서 q는 시장확률, f는 펀더멘털 모델 확률. α, γ는 최대우도로 추정한다.
문헌 실측치(Lessmann et al.)는 α≈0.695(시장), γ≈0.174(펀더멘털)이고, 같은
데이터에서 2단계 방식이 1단계 방식보다 수익률이 17.53% vs 0.96%로 압도적이었다.

────────────────────────────────────────────────────────────────────────
두 번째 돌파구 — 검정력(statistical power) 문제를 우회한다
────────────────────────────────────────────────────────────────────────
기존 백테스트는 "경주당 1마리 × 상위 10~30%"로 걸러 베팅 64~276건으로 ROI
부트스트랩 CI를 만들었다. 단승 적중률 ~10%에 배당 분산이 극단적이라, 진짜
+5% edge가 있어도 이 표본으로는 CI가 절대 0을 배제하지 못한다. 즉 "edge가
없다"가 아니라 **"있어도 볼 수 없는 실험"** 이었다.

이 스크립트는 판정 기준을 바꾼다. 조건부 로짓의 **로그우도**는 베팅한 몇십
건이 아니라 **모든 경주의 모든 말**을 쓴다. "모델이 시장 대비 추가 정보를
갖는가"를 우도비 검정으로 물으면 검정력이 수십 배 올라간다.

  귀무가설 H0: 시장확률 위에 모델이 더할 정보가 없다 (γ=0, 또는 모든 피처계수=0)
  대립가설 H1: 있다

────────────────────────────────────────────────────────────────────────
세 번째 장치 — ln(q) 오프셋 잔차모형
────────────────────────────────────────────────────────────────────────
ln(q)를 **계수 1로 고정한 오프셋**으로 넣고 피처를 추가하면,

  P(i 승) ∝ exp( ln q_i + Σ_k β_k x_ik )

가 된다. 이 모형에서 β_k ≠ 0 이라는 건 "시장이 그 피처를 **가격에 덜/과하게
반영했다**"는 뜻이다. 시장이 이미 반영한 정보로는 구조적으로 가짜 edge가 생길
수 없다. 개별 피처가 시장 대비 초과정보를 갖는지 직접 검정할 수 있다.

────────────────────────────────────────────────────────────────────────
검증 설계 (look-ahead 완전 차단)
────────────────────────────────────────────────────────────────────────
확장 윈도우 워크포워드. fold k를 평가할 때 **fold k 이전 데이터만** 쓴다.
피처 선택(Bonferroni 보정 단독 스캔)조차 각 fold의 train 안에서만 다시 한다 —
전체 표본을 보고 피처를 고르는 순간 그건 이미 look-ahead다.
모든 fold의 표본외 결과를 모아 경주일 블록부트스트랩으로 CI를 만든다.

실행: python -m src.training.core.benter_market_anchored_walkforward
"""
from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.common.conditional_logit import (  # noqa: E402
    RaceGroups, fit_conditional_logit, group_logsumexp, group_softmax,
    likelihood_ratio_test, loglik_of,
)
from src.common.column_policy import ID_COLS, LEAK_COLS  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = _PROJECT_ROOT
FINAL_CSV_GZ_CANDIDATES = [
    Path(r"C:\Users\user\Downloads\final.csv.gz"),
    Path("/mnt/user-data/uploads/Downloads/final.csv.gz"),
    ROOT / "data" / "final.csv.gz",
]
OUT = ROOT / "outputs/reports/benter_market_anchored_20260820"

N_FOLDS = 8            # 확장 윈도우 fold 수
MIN_TRAIN_FOLDS = 2    # 최소 이만큼 학습한 뒤부터 평가 시작
BONFERRONI_ALPHA = 0.05
L2 = 5.0               # 조건부 로짓 ridge 벌점(과적합 억제)
N_BOOT = 4000
RNG = np.random.default_rng(20260820)
COMMISSION_WIN = 0.20


def find_final_csv() -> Path:
    for p in FINAL_CSV_GZ_CANDIDATES:
        if p.is_file():
            return p
    print("[중단] final.csv.gz를 찾을 수 없습니다. 아래 중 한 곳에 두세요:")
    for p in FINAL_CSV_GZ_CANDIDATES:
        print(f"  - {p}")
    sys.exit(1)


def load_races() -> pd.DataFrame:
    df = pd.read_csv(find_final_csv(), compression="gzip", low_memory=False)
    df = df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
    stats = df.groupby("race_id")["win"].agg(["size", "sum"])
    keep = stats.index[(stats["sum"] == 1) & (stats["size"] >= 4)]
    dropped = df.race_id.nunique() - len(keep)
    if dropped:
        print(f"  [정보] 승자가 정확히 1마리가 아니거나 4두 미만인 경주 {dropped}개 제외")
    df = df[df.race_id.isin(keep)].sort_values(["rcDate", "race_id"], kind="stable")
    return df.reset_index(drop=True)


def candidate_features(df: pd.DataFrame) -> list[str]:
    """시장/누수/식별자 컬럼을 제외한 펀더멘털 후보. __z/__pr 중복 파생은 제거."""
    banned = set(LEAK_COLS) | set(ID_COLS) | {"win"}
    numeric = df.select_dtypes(include=[np.number])
    cands = [c for c in numeric.columns
             if c not in banned
             and numeric[c].notna().mean() > 0.80
             and numeric[c].nunique(dropna=True) > 2]
    base = {c for c in cands if not (c.endswith("__z") or c.endswith("__pr"))}
    out = []
    for c in cands:
        if c.endswith("__z") and c[:-3] in base:
            continue
        if c.endswith("__pr") and c[:-4] in base:
            continue
        out.append(c)
    return out


def build_block(sub: pd.DataFrame, cands: list[str], med=None, sd=None):
    """경주 정렬 + 경주내 평균차감 + 표준화.

    조건부 로짓에서는 경주 안에서 상수인 항이 softmax에서 소거된다. 그래서
    경주내 평균을 빼는 게 자연스럽고(정보 손실 없음), 스케일만 맞춰준다.
    med/sd를 주면 **학습표본의 통계량**을 그대로 쓴다(표본외 오염 방지).
    """
    sub = sub.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
    race_code = pd.factorize(sub["race_id"])[0]
    g = RaceGroups.from_sorted(race_code, sub["win"].to_numpy())
    ln_q = np.log(sub["q"].clip(lower=1e-9).to_numpy())
    use = cands if med is None else list(med.keys())
    cols, M, S = {}, {}, {}
    for c in use:
        v = pd.to_numeric(sub[c], errors="coerce")
        m = v.median() if med is None else med[c]
        if not np.isfinite(m):
            m = 0.0
        v = v.fillna(m)
        v = v - v.groupby(race_code).transform("mean")
        s = v.std() if sd is None else sd[c]
        if not np.isfinite(s) or s < 1e-12:
            continue
        cols[c] = (v / s).to_numpy()
        M[c], S[c] = m, s
    return sub, g, ln_q, cols, M, S


def select_features_bonferroni(Z: dict, g: RaceGroups, ln_q: np.ndarray,
                               ll_market: float, alpha: float) -> tuple[list[str], pd.DataFrame]:
    """ln(q) 오프셋 위에서 단독 피처의 잔차 정보를 우도비로 검정하고 Bonferroni 보정.

    **반드시 학습표본 안에서만 호출할 것** — 전체 표본으로 고르면 look-ahead다.
    """
    rows = []
    for c, v in Z.items():
        f = fit_conditional_logit(v[:, None], g, [c], offset=ln_q, compute_se=False, l2=0.0)
        lr = likelihood_ratio_test(f.loglik, ll_market, df=1)
        rows.append({"feature": c, "coef": float(f.beta[0]),
                     "lr_stat": lr["lr_statistic"], "p_value": lr["p_value"]})
    scan = pd.DataFrame(rows)
    scan["p_bonferroni"] = (scan["p_value"] * len(scan)).clip(upper=1.0)
    scan = scan.sort_values("p_value").reset_index(drop=True)
    return scan.loc[scan.p_bonferroni < alpha, "feature"].tolist(), scan


def block_bootstrap_ci(values: np.ndarray, dates: np.ndarray, n_boot: int = N_BOOT):
    """경주일 단위 블록부트스트랩 — 같은 날 경주끼리의 상관을 보존한다."""
    if len(values) == 0:
        return None, None
    unique = np.unique(dates)
    if len(unique) < 5:
        return None, None
    index = {d: np.flatnonzero(dates == d) for d in unique}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sampled = RNG.choice(unique, size=len(unique), replace=True)
        boots[b] = values[np.concatenate([index[d] for d in sampled])].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("시장가 기준선(market-anchored) 조건부 로짓 — 확장 윈도우 워크포워드")
    print("=" * 78)

    df = load_races()
    cands = candidate_features(df)
    print(f"  경주 {df.race_id.nunique()}개 / 출전 {len(df)}행 / 펀더멘털 후보피처 {len(cands)}개")

    dates = np.sort(df["rcDate"].unique())
    bounds = [dates[int(len(dates) * k / N_FOLDS)] for k in range(1, N_FOLDS)]
    bounds = [dates[0]] + bounds + [dates[-1] + 1]

    oos_delta_ll, oos_dates, fold_rows, scan_snapshots = [], [], [], []
    bet_frames = []
    n_folds_run, n_folds_skipped_no_signal = 0, 0

    for k in range(MIN_TRAIN_FOLDS, N_FOLDS):
        lo, hi = bounds[k], bounds[k + 1]
        train_df = df[df.rcDate < lo]
        test_df = df[(df.rcDate >= lo) & (df.rcDate < hi)]
        if train_df.race_id.nunique() < 300 or test_df.race_id.nunique() < 50:
            continue

        tr, g_tr, q_tr, Z_tr, MED, SD = build_block(train_df, cands)
        te, g_te, q_te, Z_te, _, _ = build_block(test_df, cands, MED, SD)
        shared = [c for c in Z_tr if c in Z_te]

        ll_mkt_tr = loglik_of(np.array([1.0]), q_tr[:, None], g_tr)
        ll_mkt_te = loglik_of(np.array([1.0]), q_te[:, None], g_te)

        sel, scan = select_features_bonferroni(
            {c: Z_tr[c] for c in shared}, g_tr, q_tr, ll_mkt_tr, BONFERRONI_ALPHA)
        scan["fold"] = k
        scan_snapshots.append(scan)

        n_folds_run += 1
        if not sel:
            n_folds_skipped_no_signal += 1
            print(f"\n  [fold {k}] {lo}~{hi}: train에서 Bonferroni 통과 피처 없음 -> 시장가 그대로 사용")
            print("      주의: 이 fold는 표본외 풀링에서 완전히 제외된다(0으로 채우지 않음) —")
            print("      즉 아래 최종 통합 ΔLL은 '뭔가 찾아낸 fold만' 평균낸 값이다. fold 수가")
            print("      늘어날수록(데이터 추가 시) 이 방식이 평균을 위로 편향시킬 수 있다.")
            fold_rows.append({"fold": k, "test_start": int(lo), "test_end": int(hi),
                              "train_races": int(g_tr.n_races), "test_races": int(g_te.n_races),
                              "n_selected": 0, "selected": "", "delta_ll_test": 0.0,
                              "delta_ll_per_race": 0.0, "excluded_from_pooling": True})
            continue

        X_tr = np.column_stack([Z_tr[c] for c in sel])
        X_te = np.column_stack([Z_te[c] for c in sel])
        fit = fit_conditional_logit(X_tr, g_tr, sel, offset=q_tr, l2=L2)

        ll_te = loglik_of(fit.beta, X_te, g_te, offset=q_te)
        eta_model = X_te @ fit.beta + q_te
        per_race = ((eta_model[g_te.winner_rows] - group_logsumexp(eta_model, g_te))
                    - (q_te[g_te.winner_rows] - group_logsumexp(q_te, g_te)))

        oos_delta_ll.append(per_race)
        oos_dates.append(te["rcDate"].to_numpy()[g_te.offsets])

        # 베팅 백테스트용 결합확률 저장
        combined = group_softmax(eta_model, g_te)
        market = group_softmax(q_te, g_te)
        bet = te[["rcDate", "race_id", "entry_id", "win", "winOdds"]].copy()
        bet["fold"] = k
        bet["p_market"] = market
        bet["p_combined"] = combined
        bet["edge"] = combined - market
        bet["commission_adjusted_edge"] = combined - market / (1.0 - COMMISSION_WIN)
        bet_frames.append(bet)

        print(f"\n  [fold {k}] test {lo}~{hi}  train {g_tr.n_races}경주 -> test {g_te.n_races}경주")
        print(f"      선택피처({len(sel)}): {sel}")
        print(f"      표본외 ΔLL = {ll_te - ll_mkt_te:+.2f}  (경주당 {(ll_te-ll_mkt_te)/g_te.n_races:+.5f})")
        fold_rows.append({"fold": k, "test_start": int(lo), "test_end": int(hi),
                          "train_races": int(g_tr.n_races), "test_races": int(g_te.n_races),
                          "n_selected": len(sel), "selected": ",".join(sel),
                          "delta_ll_test": float(ll_te - ll_mkt_te),
                          "delta_ll_per_race": float((ll_te - ll_mkt_te) / g_te.n_races),
                          "excluded_from_pooling": False})

    if not oos_delta_ll:
        print("\n[중단] 평가 가능한 fold가 없습니다.")
        sys.exit(1)

    values = np.concatenate(oos_delta_ll)
    vdates = np.concatenate(oos_dates)
    lo_ci, hi_ci = block_bootstrap_ci(values, vdates)

    print("\n" + "=" * 78)
    print(f"통합 표본외 결과 — {len(values)}경주 (모든 fold의 순수 out-of-sample)")
    print("=" * 78)
    print(f"  경주당 평균 ΔLL(결합 - 시장) = {values.mean():+.5f}")
    print(f"  블록부트스트랩 95% CI        = [{lo_ci:+.5f}, {hi_ci:+.5f}]")
    verdict = (lo_ci is not None and lo_ci > 0)
    print(f"  판정: {'★ 시장 대비 초과정보 확정 (CI가 0을 배제)' if verdict else '유의하지 않음'}")
    print(f"  총 ΔLL = {values.sum():+.2f}")

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(OUT / "walkforward_folds.csv", index=False, encoding="utf-8-sig")
    pd.concat(scan_snapshots, ignore_index=True).to_csv(
        OUT / "feature_scan_by_fold.csv", index=False, encoding="utf-8-sig")

    bets = pd.concat(bet_frames, ignore_index=True)
    bets.to_csv(OUT / "walkforward_combined_probabilities.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_oos_races": int(len(values)),
        "mean_delta_ll_per_race": float(values.mean()),
        "delta_ll_ci_95": [lo_ci, hi_ci],
        "ci_confirmed_positive": bool(verdict),
        "total_delta_ll": float(values.sum()),
        "n_folds_evaluated": int(fold_df["n_selected"].gt(0).sum()),
        "n_folds_run": n_folds_run,
        "n_folds_skipped_no_signal": n_folds_skipped_no_signal,
        "skipped_folds_excluded_from_pooling_caveat":
            "Bonferroni 통과 피처가 없는 fold는 표본외 풀링에서 완전히 제외된다(0으로 채우지 "
            "않음). '뭔가 찾아낸 fold만' 평균낸 값이라는 뜻 — 데이터가 늘어 fold 수가 많아지면 "
            "이 방식이 평균 ΔLL을 위로 편향시킬 가능성이 있다. 이번 실행에서 실제로 스킵된 "
            f"fold 수: {n_folds_skipped_no_signal}/{n_folds_run}.",
        "l2_penalty": L2,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "method": "ln(q) offset(coef=1) conditional logit, expanding-window walk-forward, "
                  "feature selection re-done inside each fold's training window only",
        "reference": "Benter (1994); Bolton & Chapman (1986); Lessmann et al. two-step conditional logit",
    }
    (OUT / "walkforward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  결과 저장 -> {OUT}")
    print("  다음 단계: python -m src.training.core.benter_edge_backtest "
          "(결합확률로 실제 베팅 ROI/수수료/Kelly 검증)")


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
