"""Benter 2단계 모형 정식 구현 — 펀더멘털 모델 → 시장가 결합.

────────────────────────────────────────────────────────────────────────
왜 이걸 새로 만드나
────────────────────────────────────────────────────────────────────────
현행 `benter_market_anchored_walkforward.py`는 **1단계 변형**이다. 원시 피처를
ln(q) 오프셋 위에 직접 얹는다:

    η = ln(q) + Σ_k β_k · x_k

Benter(1994)의 정식 구조는 2단계다:

    [1단계] 시장정보를 완전히 배제하고 펀더멘털 확률 P_fund 를 만든다
            P_fund(i) = exp(β'x_i) / Σ_j exp(β'x_j)
    [2단계] 시장확률과 펀더멘털 확률을 로그오즈에서 결합한다
            η = α·ln(q_i) + γ·ln(P_fund(i))

차이가 중요한 이유: 1단계 변형은 fold별 Bonferroni를 통과한 피처만 쓰는데 실제로
통과하는 건 fold당 1~4개뿐이다. 나머지 40여 개 피처의 정보를 통째로 버린다.
2단계는 그 피처들을 P_fund 하나로 **압축**해서 넣으므로, 개별로는 유의하지 않지만
합치면 의미 있는 약한 신호들을 살릴 수 있다.

`최종보고서_20260820.md` 84행에 이미 2단계 결과가 인용돼 있다
(펀더멘털 GBM 83피처, γ 0.050 → 0.111, z 0.62 → 2.93, ΔLL +0.00244 확정).
그런데 **그 코드가 저장소에 없다** — 재현 불가능한 인용 상태였다. 이 파일이 그걸 메운다.

────────────────────────────────────────────────────────────────────────
가장 중요한 구현 세부 — 교차적합(cross-fitting)
────────────────────────────────────────────────────────────────────────
Benter가 명시적으로 경고한 지점이다. 2단계에서 γ를 추정할 때 쓰는 P_fund 는
**펀더멘털 모델이 학습에 쓰지 않은 표본**에서 나와야 한다.

이유: 같은 데이터로 펀더멘털 모델을 적합한 뒤 그 데이터에서 P_fund 를 뽑으면,
P_fund 가 이미 그 경주의 결과를 부분적으로 외운 상태다. 그 위에서 γ를 추정하면
γ가 **위로 크게 편향된다** — "펀더멘털 모델이 실제보다 훨씬 유용해 보이는" 착시가
생기고, 표본외에서는 그만큼 무너진다.

그래서 각 워크포워드 fold의 학습창 안에서 다시 K겹으로 나눠,
  · 내부 K-1겹으로 펀더멘털 모델 적합 → 남은 1겹에 예측
  · 이걸 K번 반복해 학습창 전체에 대한 **표본외 P_fund** 를 만든다
  · 그 P_fund 로만 (α, γ)를 추정한다
  · 테스트 fold 예측용으로는 학습창 **전체**로 펀더멘털 모델을 다시 적합한다
분할은 경주 단위다(같은 경주의 말들이 학습/검증으로 쪼개지면 누수다).

비교를 위해 교차적합을 **끄고** 돌린 결과도 같이 보고한다. 두 γ의 차이가 이
장치가 실제로 필요한지를 그 자리에서 보여준다.

────────────────────────────────────────────────────────────────────────
검증 설계 (현행과 동일하게 유지)
────────────────────────────────────────────────────────────────────────
확장 윈도우 8-fold 워크포워드, 경주일 블록부트스트랩 95% CI, 시장 대비 ΔLL.
하이퍼파라미터는 현행 파이프라인 값을 그대로 쓴다(N_FOLDS=8, MIN_TRAIN_FOLDS=2).

실행: python -m src.training.core.benter_two_step
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
    RaceGroups, fit_conditional_logit, group_logsumexp, group_softmax, loglik_of,
)
from src.training.core.benter_market_anchored_walkforward import (  # noqa: E402
    L2, MIN_TRAIN_FOLDS, N_FOLDS, build_block, candidate_features, load_races,
)
from src.training.core.final_leakproof_model import block_ci, kelly_sim  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = _PROJECT_ROOT
OUT = ROOT / "outputs/reports/benter_two_step_20260821"

N_INNER = 4          # 교차적합 내부 겹 수
PROB_FLOOR = 1e-6    # ln(P_fund) 안전 하한
ODDS_BINS = [(1.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, 30.0), (30.0, 9999.0)]
EDGE_FRACTIONS = [0.05, 0.10, 0.20, 0.35, 0.50]


# ══════════════════════════════════════════════════════════════════════
# 1단계 — 펀더멘털 모델 (시장정보 완전 배제)
# ══════════════════════════════════════════════════════════════════════
class FundamentalCL:
    """조건부 로짓 펀더멘털 모델 — Benter 원논문의 1단계에 해당."""

    name = "conditional_logit"

    def __init__(self, feats: list[str], l2: float = L2):
        self.feats, self.l2 = feats, l2
        self.beta = self.med = self.sd = self.used = None

    def fit(self, df: pd.DataFrame) -> "FundamentalCL":
        sub, g, _lnq, Z, med, sd = build_block(df, self.feats)
        self.used = sorted(Z.keys())
        X = np.column_stack([Z[c] for c in self.used])
        fit = fit_conditional_logit(X, g, self.used, offset=None, l2=self.l2, compute_se=False)
        self.beta, self.med, self.sd = fit.beta, med, sd
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        sub, g, _lnq, Z, _m, _s = build_block(df, self.feats, self.med, self.sd)
        X = np.column_stack([Z[c] if c in Z else np.zeros(len(sub)) for c in self.used])
        return group_softmax(X @ self.beta, g)


class FundamentalGBM:
    """그래디언트 부스팅 펀더멘털 모델 — 비선형·상호작용까지 담는 큰 용량 버전.

    xgboost/lightgbm/catboost 는 이 환경에서 설치가 차단돼 있어(round4 보고서에서
    조직 네트워크 정책으로 확인) sklearn 의 HistGradientBoosting 을 쓴다.
    말 단위로 승/패를 학습한 뒤 경주 내에서 합이 1이 되도록 정규화한다."""

    name = "hist_gbm"

    def __init__(self, feats: list[str], seed: int = 20260821):
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.feats = feats
        self.clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=4,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed)

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def fit(self, df: pd.DataFrame) -> "FundamentalGBM":
        d = df.sort_values(["rcDate", "race_id"], kind="stable")
        self.clf.fit(self._X(d), d["win"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
        raw = self.clf.predict_proba(self._X(d))[:, 1]
        s = pd.Series(raw).groupby(pd.factorize(d["race_id"])[0]).transform("sum").to_numpy()
        return np.clip(raw / np.where(s > 0, s, 1.0), PROB_FLOOR, 1.0)


def make_model(kind: str, feats: list[str]):
    return FundamentalCL(feats) if kind == "cl" else FundamentalGBM(feats)


# ══════════════════════════════════════════════════════════════════════
# 교차적합 — 학습창 안에서 표본외 P_fund 생성
# ══════════════════════════════════════════════════════════════════════
def crossfit_pfund(tr_df: pd.DataFrame, feats: list[str], kind: str,
                    n_inner: int = N_INNER, seed: int = 20260821) -> np.ndarray:
    """학습창 전체에 대한 표본외 P_fund. 분할은 경주 단위(같은 경주는 통째로 이동)."""
    d = tr_df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
    races = d["race_id"].unique()
    rng = np.random.default_rng(seed)
    assign = {r: i for r, i in zip(races, rng.integers(0, n_inner, size=len(races)))}
    blk = d["race_id"].map(assign).to_numpy()

    out = np.full(len(d), np.nan)
    for k in range(n_inner):
        inner_tr, inner_te = d[blk != k], d[blk == k]
        if inner_tr.race_id.nunique() < 100 or len(inner_te) == 0:
            continue
        m = make_model(kind, feats).fit(inner_tr)
        # predict()는 내부적으로 정렬하므로 같은 순서로 맞춰 되돌린다
        te_sorted = inner_te.sort_values(["rcDate", "race_id"], kind="stable")
        out[te_sorted.index.to_numpy()] = m.predict(inner_te)
    return out


def _lnp(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, PROB_FLOOR, 1.0))


# ══════════════════════════════════════════════════════════════════════
# 워크포워드
# ══════════════════════════════════════════════════════════════════════
def run(df: pd.DataFrame, feats: list[str], kind: str, crossfit: bool,
        free_alpha: bool, label: str) -> dict:
    dates = np.sort(df["rcDate"].unique())
    bounds = ([dates[0]] + [dates[int(len(dates) * k / N_FOLDS)] for k in range(1, N_FOLDS)]
              + [dates[-1] + 1])

    oos, oos_dates, fold_rows, bets = [], [], [], []
    for k in range(MIN_TRAIN_FOLDS, N_FOLDS):
        lo, hi = bounds[k], bounds[k + 1]
        tr_df = df[df.rcDate < lo]
        te_df = df[(df.rcDate >= lo) & (df.rcDate < hi)]
        if tr_df.race_id.nunique() < 300 or te_df.race_id.nunique() < 50:
            continue

        # --- 1단계: 학습창 안에서 P_fund (교차적합 여부에 따라 다름) ---
        tr_sorted = tr_df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
        if crossfit:
            p_tr = crossfit_pfund(tr_sorted, feats, kind)
        else:
            p_tr = make_model(kind, feats).fit(tr_sorted).predict(tr_sorted)
        ok = np.isfinite(p_tr)
        tr_use = tr_sorted.loc[ok].reset_index(drop=True)
        p_tr = p_tr[ok]

        # --- 2단계: (α, γ) 추정 ---
        code = pd.factorize(tr_use["race_id"])[0]
        g_tr = RaceGroups.from_sorted(code, tr_use["win"].to_numpy())
        lnq_tr = np.log(tr_use["q"].clip(lower=PROB_FLOOR).to_numpy())
        lnf_tr = _lnp(p_tr)

        if free_alpha:
            X2 = np.column_stack([lnq_tr, lnf_tr])
            fit2 = fit_conditional_logit(X2, g_tr, ["alpha_lnq", "gamma_lnfund"],
                                          offset=None, l2=0.0)
            alpha, gamma = float(fit2.beta[0]), float(fit2.beta[1])
            z_gamma = (float(fit2.beta[1] / fit2.std_errors[1])
                       if fit2.std_errors is not None else None)
        else:  # α=1 고정 (현행 오프셋 방식과 같은 제약)
            fit2 = fit_conditional_logit(lnf_tr[:, None], g_tr, ["gamma_lnfund"],
                                          offset=lnq_tr, l2=0.0)
            alpha, gamma = 1.0, float(fit2.beta[0])
            z_gamma = (float(fit2.beta[0] / fit2.std_errors[0])
                       if fit2.std_errors is not None else None)

        # --- 테스트 fold 적용: 펀더멘털은 학습창 전체로 재적합 ---
        m_full = make_model(kind, feats).fit(tr_sorted)
        te_sorted = te_df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
        p_te = m_full.predict(te_sorted)

        code_te = pd.factorize(te_sorted["race_id"])[0]
        g_te = RaceGroups.from_sorted(code_te, te_sorted["win"].to_numpy())
        lnq_te = np.log(te_sorted["q"].clip(lower=PROB_FLOOR).to_numpy())
        eta = alpha * lnq_te + gamma * _lnp(p_te)

        ll_mkt = loglik_of(np.array([1.0]), lnq_te[:, None], g_te)
        ll_comb = float(eta[g_te.winner_rows].sum() - group_logsumexp(eta, g_te).sum())
        per = ((eta[g_te.winner_rows] - group_logsumexp(eta, g_te))
               - (lnq_te[g_te.winner_rows] - group_logsumexp(lnq_te, g_te)))
        oos.append(per)
        oos_dates.append(te_sorted["rcDate"].to_numpy()[g_te.offsets])

        comb, mkt = group_softmax(eta, g_te), group_softmax(lnq_te, g_te)
        b = te_sorted[["rcDate", "race_id", "entry_id", "win", "winOdds"]].copy()
        b["fold"], b["p_market"], b["p_combined"] = k, mkt, comb
        b["edge"] = comb - mkt
        b["ret"] = b["win"] * b["winOdds"] - 1.0
        bets.append(b)

        dll = (ll_comb - ll_mkt) / g_te.n_races
        fold_rows.append({"fold": k, "train_races": int(g_tr.n_races),
                          "test_races": int(g_te.n_races), "alpha": alpha,
                          "gamma": gamma, "z_gamma": z_gamma, "delta_ll_per_race": dll})
        zs = f"{z_gamma:+.2f}" if z_gamma is not None else "N/A"
        print(f"    [fold {k}] α={alpha:.3f} γ={gamma:+.4f} (z={zs})  "
              f"경주당 ΔLL {dll:+.5f}")

    if not oos:
        return {"label": label, "n_oos_races": 0}
    values = np.concatenate(oos)
    lo_ci, hi_ci = block_ci(values, np.concatenate(oos_dates))
    confirmed = bool(lo_ci is not None and lo_ci > 0)
    print(f"  [{label}] 통합 {len(values)}경주  ΔLL {values.mean():+.5f} "
          f"CI [{lo_ci:+.5f}, {hi_ci:+.5f}] -> {'★확정' if confirmed else '미확정'}")
    return {"label": label, "n_oos_races": int(len(values)),
            "delta_ll_per_race": float(values.mean()), "delta_ll_ci": [lo_ci, hi_ci],
            "ci_confirmed_positive": confirmed,
            "mean_gamma": float(np.mean([r["gamma"] for r in fold_rows])),
            "folds": fold_rows, "_bets": pd.concat(bets, ignore_index=True)}


def backtest(bets: pd.DataFrame) -> dict:
    b = bets[bets.winOdds.between(1.0, 9999.0, inclusive="neither")]
    folds = sorted(b.fold.unique())

    def pol(d, lo_o, hi_o, fr):
        s = d[(d.winOdds >= lo_o) & (d.winOdds < hi_o)]
        return s.nlargest(max(1, int(np.ceil(len(s) * fr))), "edge") if len(s) else s

    picked = []
    for i, k in enumerate(folds):
        if i == 0:
            continue
        past, cur = b[b.fold.isin(folds[:i])], b[b.fold == k]
        best, br = None, -np.inf
        for lo_o, hi_o in ODDS_BINS:
            for fr in EDGE_FRACTIONS:
                s = pol(past, lo_o, hi_o, fr)
                if len(s) < 30:
                    continue
                if s.ret.mean() > br:
                    br, best = s.ret.mean(), (lo_o, hi_o, fr)
        if best:
            picked.append(pol(cur, *best))
    if not picked:
        return {"n_bets": 0}
    pool = pd.concat(picked, ignore_index=True)
    r_lo, r_hi = block_ci(pool.ret.to_numpy(), pool.rcDate.to_numpy())
    kel = kelly_sim(pool)
    return {"n_bets": int(len(pool)), "hit_rate": float(pool.win.mean()),
            "unit_roi": float(pool.ret.mean()), "roi_ci": [r_lo, r_hi],
            "ci_confirmed_positive": bool(r_lo is not None and r_lo > 0),
            "kelly_sharpe": kel["sharpe"], "kelly_max_drawdown": kel["max_drawdown"]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("Benter 2단계 모형 — 펀더멘털 → 시장가 결합")
    print("=" * 78)

    df = load_races()
    feats = candidate_features(df)
    print(f"  경주 {df.race_id.nunique()} / 출전 {len(df)} / 펀더멘털 피처 {len(feats)}개 "
          f"(시장·누수·식별자 전부 제외)")

    results, all_bets = {}, {}
    configs = [
        ("cl",  True,  True,  "2단계 CL · 교차적합 · α자유"),
        ("cl",  True,  False, "2단계 CL · 교차적합 · α=1"),
        ("cl",  False, True,  "2단계 CL · 교차적합없음 · α자유  ← 대조군"),
        ("gbm", True,  True,  "2단계 GBM · 교차적합 · α자유"),
        ("gbm", True,  False, "2단계 GBM · 교차적합 · α=1"),
        ("gbm", False, True,  "2단계 GBM · 교차적합없음 · α자유  ← 대조군"),
    ]
    for kind, cf, fa, label in configs:
        print(f"\n[{label}]")
        r = run(df, feats, kind, cf, fa, label)
        if r.get("n_oos_races"):
            all_bets[label] = r.pop("_bets")
            results[label] = r
        else:
            r.pop("_bets", None)

    print("\n" + "=" * 78)
    print("종합 — 시장 대비 표본외 ΔLL")
    print("=" * 78)
    print(f"  {'조건':<38} {'평균 γ':>8} {'ΔLL/경주':>11} {'판정':>7}")
    print(f"  {'1단계(현행, 참고)':<38} {'—':>8} {'+0.00491':>11} {'확정':>7}")
    for label, r in results.items():
        print(f"  {label:<38} {r['mean_gamma']:>+8.4f} {r['delta_ll_per_race']:>+11.5f} "
              f"{'확정' if r['ci_confirmed_positive'] else '미확정':>7}")

    # 최선 조건으로 베팅 백테스트
    best_label = max(results, key=lambda L: results[L]["delta_ll_per_race"]) if results else None
    bt = {}
    if best_label:
        print(f"\n[베팅 백테스트] 최선 조건: {best_label}")
        bt = backtest(all_bets[best_label])
        if bt.get("n_bets"):
            print(f"    {bt['n_bets']}건, 적중률 {bt['hit_rate']:.2%}, "
                  f"ROI {bt['unit_roi']:+.2%} CI [{bt['roi_ci'][0]:+.2%}, {bt['roi_ci'][1]:+.2%}]"
                  f" -> {'★확정' if bt['ci_confirmed_positive'] else '0 포함'}")
            print(f"    Sharpe {bt['kelly_sharpe']}, 최대낙폭 {bt['kelly_max_drawdown']:.1%}")
        all_bets[best_label].to_csv(OUT / "two_step_probabilities.csv",
                                     index=False, encoding="utf-8-sig")

    (OUT / "summary.json").write_text(json.dumps({
        "n_fundamental_features": len(feats),
        "one_step_reference": {"delta_ll_per_race": 0.00491,
                               "delta_ll_ci": [0.00126, 0.00864]},
        "results": results, "best_label": best_label, "backtest_of_best": bt,
        "crossfit_note": "교차적합을 끄면 γ가 위로 편향된다 — 대조군과 비교할 것.",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장 -> {OUT}")


if __name__ == "__main__":
    main()
