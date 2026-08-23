"""Base Margin 부스팅 — 시장 로그오즈를 출발점으로 두고 트리가 잔차만 학습한다.

────────────────────────────────────────────────────────────────────────
기존 두 방식과 무엇이 다른가
────────────────────────────────────────────────────────────────────────
    1단계 (현행)   η = ln(q) + Σ_k β_k·x_k        ← 시장 위에 **선형** 항만
    2단계          η = α·ln(q) + γ·ln(P_fund)     ← 모든 피처를 γ **스칼라 하나**로 압축
    Base Margin    η = ln(q) + f_tree(x)          ← 시장 위에 **비선형 함수** 전체를

1단계는 상호작용을 못 잡는다("비 오는 날 × 특정 트랙 × 특정 조교사" 같은 결합 조건).
2단계는 트리를 쓰긴 하지만 결과를 `ln(P_fund)` 하나로 눌러 담은 뒤 계수 하나(γ)만
붙이므로, "시장이 **어떤 조건에서** 틀리는가"라는 특정성이 사라진다. 실제로 2단계는
1단계보다 약했다(+0.00220 vs +0.00491) — 압축이 손해였다는 해석과 맞는다.

Base Margin은 그 압축 단계를 아예 없앤다. 부스팅의 **초기값(base margin / init score)**
을 ln(q)로 두면, 트리는 처음부터 "시장이 이미 맞힌 부분"을 다시 배울 이유가 없고
**시장 예측의 오차만** 파고든다. XGBoost의 `base_margin`, LightGBM의 `init_score`가
바로 이것이다.

────────────────────────────────────────────────────────────────────────
왜 직접 구현했나
────────────────────────────────────────────────────────────────────────
이 환경은 xgboost/lightgbm/catboost 설치가 조직 네트워크 정책으로 차단돼 있다
(`round4_followup_20260820` 보고서에서 tqdm·cowsay 같은 사소한 패키지도 동일하게
막히는 것으로 확인). sklearn 의 부스팅 구현은 base margin 을 지원하지 않는다.

그래서 **경주단위 조건부 로짓 손실에 대한 Newton 부스팅**을 직접 짰다. 어차피
경마의 손실함수(경주 안에서 정확히 1마리가 이김)는 일반 이진분류와 달라서, 기성
라이브러리를 쓰더라도 커스텀 손실을 넣어야 한다.

  손실:      LL = Σ_경주 [ η_승자 − logsumexp(η) ]
  1차 미분:  g_i = y_i − p_i           (y=승리여부, p=경주내 softmax)
  2차 미분:  h_i = p_i(1 − p_i)
  잎 값:     w_leaf = Σg / (Σh + λ)    ← XGBoost 와 같은 Newton 스텝
  갱신:      η ← η + lr · w

트리는 sklearn `DecisionTreeRegressor` 를 구조 학습에만 쓰고, 잎 값은 위 Newton 식으로
직접 덮어쓴다(회귀 트리의 기본 잎 값은 단순 평균이라 2차 정보를 못 쓴다).

────────────────────────────────────────────────────────────────────────
검증 설계
────────────────────────────────────────────────────────────────────────
· 현행과 동일한 8-fold 확장윈도우 워크포워드, 경주일 블록부트스트랩 95% CI
· 트리 개수는 학습창 **내부**의 시간순 뒤쪽 20% 를 검증용으로 떼어 조기중단으로 정한다
  (테스트 fold 는 절대 보지 않는다)
· **대조군**: 같은 부스터를 base margin 없이(η 를 0에서 시작) 돌린다. 시장 앵커링이
  실제로 기여하는지를 같은 코드에서 직접 비교하기 위함이다.

실행: python -m src.training.core.market_base_margin_boosting
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

from src.common.conditional_logit import RaceGroups, group_logsumexp, group_softmax  # noqa: E402
from src.training.core.benter_market_anchored_walkforward import (  # noqa: E402
    MIN_TRAIN_FOLDS, N_FOLDS, candidate_features, load_races,
)
from src.training.core.final_leakproof_model import block_ci, kelly_sim  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = _PROJECT_ROOT
OUT = ROOT / "outputs/reports/base_margin_boosting_20260821"

LR = 0.05
MAX_TREES = 300
MAX_DEPTH = 3
MIN_LEAF = 60
LAMBDA = 20.0          # 잎 값 Newton 스텝의 정규화 (신호가 작아 크게 잡는다)
PATIENCE = 20          # 내부 검증 LL 이 이만큼 개선 없으면 중단
# 분할마다 피처 부분집합만 본다. XGBoost 의 colsample_bynode 에 해당하며,
# 정규화 역할(트리 간 상관 감소)과 속도(측정상 약 7배) 두 가지를 동시에 준다.
MAX_FEATURES = "sqrt"
INNER_VALID_FRAC = 0.20
ODDS_BINS = [(1.0, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 15.0), (15.0, 30.0), (30.0, 9999.0)]
EDGE_FRACTIONS = [0.05, 0.10, 0.20, 0.35, 0.50]


def race_groups(df: pd.DataFrame) -> RaceGroups:
    return RaceGroups.from_sorted(pd.factorize(df["race_id"])[0], df["win"].to_numpy())


def loglik(eta: np.ndarray, g: RaceGroups) -> float:
    return float(eta[g.winner_rows].sum() - group_logsumexp(eta, g).sum())


class BaseMarginBooster:
    """경주단위 조건부 로짓 손실에 대한 Newton 부스팅. 초기값(base margin) 지정 가능."""

    def __init__(self, lr: float = LR, max_trees: int = MAX_TREES,
                 max_depth: int = MAX_DEPTH, min_leaf: int = MIN_LEAF,
                 lam: float = LAMBDA, patience: int = PATIENCE, seed: int = 20260821):
        self.lr, self.max_trees, self.max_depth = lr, max_trees, max_depth
        self.min_leaf, self.lam, self.patience, self.seed = min_leaf, lam, patience, seed
        self.trees: list = []
        self.leaf_values: list[np.ndarray] = []
        self.n_used = 0

    def _leaf_newton(self, tree, X, g_arr, h_arr) -> np.ndarray:
        """회귀트리의 기본 잎 값(평균)을 Newton 스텝 Σg/(Σh+λ) 로 덮어쓴다."""
        leaf_id = tree.apply(X)
        vals = np.zeros(tree.tree_.node_count)
        for lid in np.unique(leaf_id):
            m = leaf_id == lid
            vals[lid] = g_arr[m].sum() / (h_arr[m].sum() + self.lam)
        return vals

    def fit(self, X_tr, g_tr, y_tr, base_tr,
            X_va=None, g_va=None, base_va=None) -> "BaseMarginBooster":
        from sklearn.tree import DecisionTreeRegressor

        eta_tr = base_tr.copy()
        eta_va = base_va.copy() if X_va is not None else None
        best_ll, best_n, since = -np.inf, 0, 0

        for t in range(self.max_trees):
            p = group_softmax(eta_tr, g_tr)
            grad = y_tr - p                      # dLL/dη
            hess = np.clip(p * (1.0 - p), 1e-8, None)

            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_leaf=self.min_leaf,
                                          max_features=MAX_FEATURES,
                                          random_state=self.seed + t)
            tree.fit(X_tr, grad)
            vals = self._leaf_newton(tree, X_tr, grad, hess)
            self.trees.append(tree)
            self.leaf_values.append(vals)

            eta_tr = eta_tr + self.lr * vals[tree.apply(X_tr)]
            if X_va is not None:
                eta_va = eta_va + self.lr * vals[tree.apply(X_va)]
                ll = loglik(eta_va, g_va) / g_va.n_races
                if ll > best_ll + 1e-9:
                    best_ll, best_n, since = ll, t + 1, 0
                else:
                    since += 1
                    if since >= self.patience:
                        break

        self.n_used = best_n if X_va is not None else len(self.trees)
        return self

    def margin(self, X: np.ndarray) -> np.ndarray:
        """base margin 위에 더할 값 f_tree(x)."""
        out = np.zeros(len(X))
        for tree, vals in zip(self.trees[:self.n_used], self.leaf_values[:self.n_used]):
            out += self.lr * vals[tree.apply(X)]
        return out


def prep(df: pd.DataFrame, feats: list[str]):
    d = df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
    X = d[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    g = race_groups(d)
    lnq = np.log(d["q"].clip(lower=1e-9).to_numpy())
    return d, X, g, lnq, d["win"].to_numpy(dtype=float)


def run(df: pd.DataFrame, feats: list[str], anchored: bool, label: str) -> dict:
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

        # 학습창 내부를 시간순으로 갈라 조기중단용 검증 확보 (테스트는 안 봄)
        tr_dates = np.sort(tr_df["rcDate"].unique())
        cut = tr_dates[int(len(tr_dates) * (1 - INNER_VALID_FRAC))]
        inner_tr, inner_va = tr_df[tr_df.rcDate < cut], tr_df[tr_df.rcDate >= cut]
        if inner_va.race_id.nunique() < 50:
            inner_tr, inner_va = tr_df, tr_df

        d_it, X_it, g_it, lnq_it, y_it = prep(inner_tr, feats)
        d_iv, X_iv, g_iv, lnq_iv, _ = prep(inner_va, feats)
        base_it = lnq_it if anchored else np.zeros_like(lnq_it)
        base_iv = lnq_iv if anchored else np.zeros_like(lnq_iv)

        bst = BaseMarginBooster().fit(X_it, g_it, y_it, base_it, X_iv, g_iv, base_iv)
        n_trees = bst.n_used

        # 확정된 트리 수로 학습창 전체 재학습
        d_tr, X_tr, g_tr, lnq_tr, y_tr = prep(tr_df, feats)
        base_tr = lnq_tr if anchored else np.zeros_like(lnq_tr)
        final = BaseMarginBooster(max_trees=max(n_trees, 1))
        final.fit(X_tr, g_tr, y_tr, base_tr)
        final.n_used = max(n_trees, 1)

        d_te, X_te, g_te, lnq_te, _ = prep(te_df, feats)
        eta = (lnq_te if anchored else np.zeros_like(lnq_te)) + final.margin(X_te)

        ll_mkt = loglik(lnq_te, g_te)
        per = ((eta[g_te.winner_rows] - group_logsumexp(eta, g_te))
               - (lnq_te[g_te.winner_rows] - group_logsumexp(lnq_te, g_te)))
        oos.append(per)
        oos_dates.append(d_te["rcDate"].to_numpy()[g_te.offsets])

        comb, mkt = group_softmax(eta, g_te), group_softmax(lnq_te, g_te)
        b = d_te[["rcDate", "race_id", "entry_id", "win", "winOdds"]].copy()
        b["fold"], b["p_market"], b["p_combined"] = k, mkt, comb
        b["edge"] = comb - mkt
        b["ret"] = b["win"] * b["winOdds"] - 1.0
        bets.append(b)

        dll = (loglik(eta, g_te) - ll_mkt) / g_te.n_races
        fold_rows.append({"fold": k, "n_trees": int(n_trees),
                          "test_races": int(g_te.n_races), "delta_ll_per_race": dll})
        print(f"    [fold {k}] 트리 {n_trees:>3}개  경주당 ΔLL {dll:+.5f}")

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
            "mean_trees": float(np.mean([r["n_trees"] for r in fold_rows])),
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
    print("Base Margin 부스팅 — 시장 로그오즈를 출발점으로, 트리는 잔차만")
    print("=" * 78)

    df = load_races()
    feats = candidate_features(df)
    print(f"  경주 {df.race_id.nunique()} / 출전 {len(df)} / 피처 {len(feats)}개")
    print(f"  설정: lr={LR}, depth={MAX_DEPTH}, min_leaf={MIN_LEAF}, λ={LAMBDA}, "
          f"max_features={MAX_FEATURES}, 조기중단 patience={PATIENCE}")

    results, all_bets = {}, {}
    for anchored, label in [(True, "Base Margin (시장 앵커)"),
                             (False, "앵커 없음 (0에서 시작)  ← 대조군")]:
        print(f"\n[{label}]")
        r = run(df, feats, anchored, label)
        if r.get("n_oos_races"):
            all_bets[label] = r.pop("_bets")
            results[label] = r
        else:
            r.pop("_bets", None)

    print("\n" + "=" * 78)
    print("종합 — 시장 대비 표본외 ΔLL")
    print("=" * 78)
    print(f"  {'방식':<34} {'ΔLL/경주':>11} {'95% CI':>26} {'판정':>7}")
    ref = [("1단계 (현행, 선형)", 0.00491, [0.00126, 0.00864], True),
           ("2단계 (γ 압축)", 0.00220, [0.00044, 0.00396], True)]
    for nm, d, ci, ok in ref:
        print(f"  {nm:<34} {d:>+11.5f} {f'[{ci[0]:+.5f}, {ci[1]:+.5f}]':>26} "
              f"{'확정' if ok else '미확정':>7}")
    for label, r in results.items():
        ci = r["delta_ll_ci"]
        print(f"  {label:<34} {r['delta_ll_per_race']:>+11.5f} "
              f"{f'[{ci[0]:+.5f}, {ci[1]:+.5f}]':>26} "
              f"{'확정' if r['ci_confirmed_positive'] else '미확정':>7}")

    bt = {}
    if "Base Margin (시장 앵커)" in all_bets:
        print("\n[베팅 백테스트] Base Margin")
        bt = backtest(all_bets["Base Margin (시장 앵커)"])
        if bt.get("n_bets"):
            print(f"    {bt['n_bets']}건, 적중률 {bt['hit_rate']:.2%}, "
                  f"ROI {bt['unit_roi']:+.2%} CI [{bt['roi_ci'][0]:+.2%}, {bt['roi_ci'][1]:+.2%}]"
                  f" -> {'★확정' if bt['ci_confirmed_positive'] else '0 포함'}")
            print(f"    Sharpe {bt['kelly_sharpe']}, 최대낙폭 {bt['kelly_max_drawdown']:.1%}")
        all_bets["Base Margin (시장 앵커)"].to_csv(
            OUT / "base_margin_probabilities.csv", index=False, encoding="utf-8-sig")

    (OUT / "summary.json").write_text(json.dumps({
        "config": {"lr": LR, "max_depth": MAX_DEPTH, "min_leaf": MIN_LEAF,
                   "lambda": LAMBDA, "patience": PATIENCE,
                   "max_features": MAX_FEATURES, "max_trees": MAX_TREES},
        "reference": {"one_step": 0.00491, "two_step": 0.00220},
        "results": results, "backtest_base_margin": bt,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n저장 -> {OUT}")


if __name__ == "__main__":
    main()
