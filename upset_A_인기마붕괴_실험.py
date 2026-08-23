# -*- coding: utf-8 -*-
"""
인기마 붕괴(upset_A) 예측 모델 실험
===================================
「이변 예측모델 선정 보고서」 6장 · 부록 B.2 · B.3 을 만들어낸 코드입니다.

인기마 붕괴란
    인기 상위 25% 안에 드는 말이 하위 50%로 들어오는 경우입니다.
    시장이 유력하다고 본 말이 무너지는 현상이라, 다크호스(upset_B)와 정확히 반대입니다.

        upset_A = (pop_pct <= 0.25) & (fin_pct >= 0.50)

    pop_pct 는 인기 백분위(작을수록 인기마), fin_pct 는 착순 백분위(클수록 뒤쪽)입니다.
    저장소 데이터에 이미 upset_A 열이 있고, 위 식으로 100% 재현됩니다.

실행
    python upset_A_인기마붕괴_실험.py
    python upset_A_인기마붕괴_실험.py --quick     # 깊이 탐색·부트스트랩 축소

출력
    results/upset_A/  아래에 표 5개와 콘솔 로그
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────
# 저장소 루트를 기준으로 원천 데이터를 찾습니다.
_ROOT = Path(__file__).resolve().parent
for _ in range(4):                       # 어디에 두어도 data/ 를 찾도록
    if (_ROOT / "data" / "race_entries.csv.gz").exists():
        break
    _ROOT = _ROOT.parent
RACE_ENTRIES = _ROOT / "data" / "race_entries.csv.gz"
OUT = Path("results/upset_A")

SEED = 42
SPLIT = (0.6, 0.2, 0.2)                  # train : valid : test (시간순)

# ── 누출 열 ───────────────────────────────────────────────
# 타겟이 인기도로 정의되어 있어서, 배당률 계열이 하나라도 남으면
# 정답을 그대로 보는 셈이 됩니다. 실제로 남기면 ROC-AUC 가 0.98 까지 올라갑니다.
DROP_TARGET = ["pop_pct", "fin_pct", "upset_A", "upset_B", "surprise_score",
               "darkhorse", "favorite_bust", "upset"]
DROP_MARKET = ["winOdds", "plcOdds", "p_raw", "q", "q_plc", "log_q", "logit_q",
               "pop_rank", "is_fav", "book_sum", "takeout", "pl_harville",
               "pl_disc", "gap_h", "gap_d",
               "winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
DROP_RESULT = ["ord", "fin_rank", "win", "place", "resid",
               "chaksun1", "chaksun2", "chaksun3", "chaksun4", "chaksun5",
               "rcTime", "diffUnit"]
DROP_ID = ["race_id", "entry_id", "hrName", "jkName", "trName", "owName",
           "hrNo", "jkNo", "trNo", "owNo", "meet", "rcDate", "fold"]


# ══════════════════════════════════════════════════════════
# 1. 데이터 준비
# ══════════════════════════════════════════════════════════
def load_subset() -> pd.DataFrame:
    """서울 데이터에서 인기 상위 25% 만 남기고 시간순으로 분할한다."""
    df = pd.read_csv(RACE_ENTRIES, low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)

    # 타겟 (데이터에 이미 있으면 그대로, 없으면 정의대로 계산)
    if "upset_A" not in df.columns:
        df["upset_A"] = ((df["pop_pct"] <= 0.25) & (df["fin_pct"] >= 0.50)).astype(int)

    # ── 부분집합 추출 — 이게 핵심입니다 ──────────────────
    # 인기마 붕괴는 '인기마'에게만 물을 수 있는 질문입니다.
    # 전체 데이터로 학습하면 모델이 '이 말이 인기마인가'만 배우고 무너집니다(7.2절).
    sub = df[df["pop_pct"] <= 0.25].reset_index(drop=True)

    # ── 시간순 분할 — 날짜 단위로 통째로 ────────────────
    sub = sub.sort_values("rcDate", kind="stable").reset_index(drop=True)
    cnt = sub["rcDate"].value_counts().sort_index()
    cum = cnt.cumsum() / len(sub)
    tr_d = set(cum[cum <= SPLIT[0]].index)
    va_d = set(cum[(cum > SPLIT[0]) & (cum <= SPLIT[0] + SPLIT[1])].index)
    sub["fold"] = np.where(sub["rcDate"].isin(tr_d), "train",
                  np.where(sub["rcDate"].isin(va_d), "valid", "test"))
    return sub


def make_xy(sub: pd.DataFrame):
    """누출 열을 걷어내고 학습용 행렬을 만든다."""
    drop = [c for c in DROP_TARGET + DROP_MARKET + DROP_RESULT + DROP_ID
            if c in sub.columns]
    X = sub.drop(columns=drop)

    # 범주형은 코드로 (트리 모델용)
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category").cat.codes
    X = X.apply(pd.to_numeric, errors="coerce")

    # 구조적 결측(데뷔전 등)은 0, 나머지는 train 중앙값
    m = sub["fold"] == "train"
    X = X.fillna(X[m].median()).fillna(0)

    y = sub["upset_A"].values
    return X, y, m.values, (sub["fold"] == "valid").values, (sub["fold"] == "test").values


# ══════════════════════════════════════════════════════════
# 2. 평가 지표
# ══════════════════════════════════════════════════════════
def lift_at(y, p, k):
    """상위 k 비율 안에서의 적중률 ÷ 전체 적중률."""
    n = max(1, int(len(y) * k))
    idx = np.argsort(p)[::-1][:n]
    base = y.mean()
    return (y[idx].mean() / base) if base > 0 else np.nan


def evaluate(y, p) -> dict:
    return {
        "ROC-AUC": roc_auc_score(y, p),
        "PR-AUC": average_precision_score(y, p),
        "L@5%": lift_at(y, p, 0.05),
        "L@10%": lift_at(y, p, 0.10),
        "L@20%": lift_at(y, p, 0.20),
        "L@30%": lift_at(y, p, 0.30),
    }


def bootstrap_lift(y, p, k=0.10, n_boot=2000, seed=SEED):
    """Lift@k 의 부트스트랩 95% 신뢰구간."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        yi, pi = y[i], p[i]
        if yi.sum() == 0:
            continue
        out.append(lift_at(yi, pi, k))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ══════════════════════════════════════════════════════════
# 3. 모델 정의
# ══════════════════════════════════════════════════════════
def build_models(depth_rf=8):
    """보고서 3장에서 고른 네 종류 + 무작위 기준선."""
    models = {}

    # 표본이 적을 때 가장 안정적인 기준선. 정규화를 세게 건다.
    models["로지스틱 회귀"] = ("linear", LogisticRegression(
        C=0.1, max_iter=2000, random_state=SEED))

    # min_samples_leaf=50 이 성능을 좌우한다. 잎마다 최소 50개를 요구해
    # 나무가 깊어져도 한두 사례를 외우는 일이 구조적으로 막힌다.
    models[f"랜덤포레스트 (d{depth_rf})"] = ("tree", RandomForestClassifier(
        n_estimators=600, max_depth=depth_rf, min_samples_leaf=50,
        random_state=SEED, n_jobs=-1))          # class_weight 지정하지 않음

    try:
        from xgboost import XGBClassifier
        # 신호가 약해 정규화를 최대로. 얕게(3) + L2 페널티(10).
        models["XGBoost (d3)"] = ("tree", XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            reg_lambda=10, subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=SEED, n_jobs=-1))
    except ImportError:
        print("  [건너뜀] xgboost 미설치")

    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = ("tree", LGBMClassifier(
            n_estimators=400, max_depth=3, num_leaves=8, learning_rate=0.05,
            reg_lambda=10, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, n_jobs=-1, verbose=-1))
    except ImportError:
        print("  [건너뜀] lightgbm 미설치")

    return models


def fit_predict(kind, model, Xtr, ytr, Xte):
    """선형 모델만 표준화한다 (트리는 스케일에 무관)."""
    if kind == "linear":
        sc = StandardScaler().fit(Xtr)
        model.fit(sc.transform(Xtr), ytr)
        return model.predict_proba(sc.transform(Xte))[:, 1]
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


# ══════════════════════════════════════════════════════════
# 4. 실험
# ══════════════════════════════════════════════════════════
def main(quick=False):
    OUT.mkdir(parents=True, exist_ok=True)
    n_boot = 300 if quick else 2000
    depths = [3, 8] if quick else [3, 4, 6, 8, 12]

    print("=" * 72)
    print("인기마 붕괴(upset_A) 예측 실험")
    print("=" * 72)

    sub = load_subset()
    X, y, m_tr, m_va, m_te = make_xy(sub)
    Xtr, ytr = X[m_tr], y[m_tr]
    Xva, yva = X[m_va], y[m_va]
    Xte, yte = X[m_te], y[m_te]

    print(f"  입력      {RACE_ENTRIES}")
    print(f"  부분집합  pop_pct <= 0.25 → {len(sub):,}행 (전체 서울의 {len(sub)/32888*100:.0f}%)")
    print(f"  피처      {X.shape[1]}개  (누출 열 제거 후)")
    print(f"  train     {len(ytr):,}행 · 기저율 {ytr.mean()*100:.2f}%")
    print(f"  valid     {len(yva):,}행 · 기저율 {yva.mean()*100:.2f}%")
    print(f"  test      {len(yte):,}행 · 기저율 {yte.mean()*100:.2f}%")

    # ── 4-1. 랜덤포레스트 깊이 선택 (valid 기준) ──────────
    print("\n" + "=" * 72)
    print("[1] 랜덤포레스트 깊이 선택  — valid ROC-AUC 로만 고른다")
    print("=" * 72)
    rows = []
    for d in depths:
        mdl = RandomForestClassifier(n_estimators=600, max_depth=d,
                                     min_samples_leaf=50, random_state=SEED, n_jobs=-1)
        pv = fit_predict("tree", mdl, Xtr, ytr, Xva)
        pt = fit_predict("tree", mdl, Xtr, ytr, Xte)
        rows.append({"depth": d, "valid_AUC": roc_auc_score(yva, pv),
                     "test_AUC": roc_auc_score(yte, pt),
                     "test_L@10%": lift_at(yte, pt, 0.10)})
        print(f"  depth={d:<3} valid {rows[-1]['valid_AUC']:.4f}   "
              f"test {rows[-1]['test_AUC']:.4f}   L@10% {rows[-1]['test_L@10%']:.2f}")
    df_d = pd.DataFrame(rows)
    df_d.to_csv(OUT / "depth_sweep.csv", index=False, encoding="utf-8-sig")
    best_d = int(df_d.loc[df_d["valid_AUC"].idxmax(), "depth"])
    print(f"  → 선택: depth={best_d}")

    # ── 4-2. 모델 비교 (test) ─────────────────────────────
    print("\n" + "=" * 72)
    print("[2] 모델 비교  — test set")
    print("=" * 72)
    rng = np.random.default_rng(SEED)
    preds = {"무작위": rng.random(len(yte))}
    for name, (kind, mdl) in build_models(best_d).items():
        preds[name] = fit_predict(kind, mdl, Xtr, ytr, Xte)

    res = pd.DataFrame({n: evaluate(yte, p) for n, p in preds.items()}).T
    print(res.round(4).to_string())
    res.to_csv(OUT / "model_comparison.csv", encoding="utf-8-sig")

    # ── 4-3. 부트스트랩 신뢰구간 ─────────────────────────
    print("\n" + "=" * 72)
    print(f"[3] Lift@10% 부트스트랩 95% 신뢰구간  ({n_boot:,}회)")
    print("=" * 72)
    ci_rows = []
    for n, p in preds.items():
        lo, hi = bootstrap_lift(yte, p, 0.10, n_boot)
        ci_rows.append({"model": n, "Lift@10%": lift_at(yte, p, 0.10),
                        "CI_low": lo, "CI_high": hi, "1_초과": lo > 1})
        print(f"  {n:<20} {ci_rows[-1]['Lift@10%']:.2f}  [{lo:.2f}, {hi:.2f}]"
              f"{'   ← 하한이 1을 넘음' if lo > 1 else ''}")
    pd.DataFrame(ci_rows).to_csv(OUT / "lift_ci.csv", index=False, encoding="utf-8-sig")

    # ── 4-4. 피처 중요도 ─────────────────────────────────
    print("\n" + "=" * 72)
    print("[4] 피처 중요도 상위 10  — 랜덤포레스트")
    print("=" * 72)
    rf = RandomForestClassifier(n_estimators=600, max_depth=best_d,
                                min_samples_leaf=50, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    imp = (pd.DataFrame({"feature": X.columns, "importance": rf.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    for i, r in imp.head(10).iterrows():
        print(f"  {i+1:>2}. {r['feature']:<24} {r['importance']:.4f}")
    imp.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")

    # ── 4-5. 불균형 처리 — 하지 말아야 할 것 ①  ──────────
    print("\n" + "=" * 72)
    print("[5] 불균형 처리를 걸면 어떻게 되나")
    print("=" * 72)
    print("  순위만 쓰는 문제에서는 확률 전체를 위로 미는 것이 도움이 되지 않습니다.")
    rows = []
    for name, plain, weighted in [
        ("랜덤포레스트",
         RandomForestClassifier(n_estimators=600, max_depth=best_d, min_samples_leaf=50,
                                random_state=SEED, n_jobs=-1),
         RandomForestClassifier(n_estimators=600, max_depth=best_d, min_samples_leaf=50,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ("로지스틱 회귀",
         LogisticRegression(C=0.1, max_iter=2000, random_state=SEED),
         LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=SEED)),
    ]:
        kind = "linear" if "로지스틱" in name else "tree"
        a = roc_auc_score(yte, fit_predict(kind, plain, Xtr, ytr, Xte))
        b = roc_auc_score(yte, fit_predict(kind, weighted, Xtr, ytr, Xte))
        rows.append({"model": name, "기본": a, "balanced": b, "차이": b - a})
        print(f"  {name:<16} {a:.4f}  →  {b:.4f}   ({b-a:+.4f})")
    pd.DataFrame(rows).to_csv(OUT / "imbalance.csv", index=False, encoding="utf-8-sig")

    # ── 4-6. 학습 범위 — 하지 말아야 할 것 ②  ────────────
    print("\n" + "=" * 72)
    print("[6] 전체 데이터로 학습하면 어떻게 되나")
    print("=" * 72)
    full = pd.read_csv(RACE_ENTRIES, low_memory=False)
    full = full[full["meet"] == "서울"].reset_index(drop=True)
    if "upset_A" not in full.columns:
        full["upset_A"] = ((full["pop_pct"] <= 0.25) & (full["fin_pct"] >= 0.50)).astype(int)
    full = full.sort_values("rcDate", kind="stable").reset_index(drop=True)
    cnt = full["rcDate"].value_counts().sort_index()
    cum = cnt.cumsum() / len(full)
    tr_d = set(cum[cum <= SPLIT[0]].index)
    va_d = set(cum[(cum > SPLIT[0]) & (cum <= SPLIT[0] + SPLIT[1])].index)
    full["fold"] = np.where(full["rcDate"].isin(tr_d), "train",
                   np.where(full["rcDate"].isin(va_d), "valid", "test"))
    is_fav = (full["pop_pct"] <= 0.25).values

    Xf, yf, fm_tr, _, fm_te = make_xy(full)
    scope = []
    for label, ytrain, mask_tr, mask_te in [
        ("전체 + upset_A 라벨", yf, fm_tr, fm_te & is_fav),
        ("전체 + 1착 라벨", full["win"].values, fm_tr, fm_te & is_fav),
    ]:
        mdl = RandomForestClassifier(n_estimators=600, max_depth=best_d,
                                     min_samples_leaf=50, random_state=SEED, n_jobs=-1)
        mdl.fit(Xf[mask_tr], ytrain[mask_tr])
        p = mdl.predict_proba(Xf[mask_te])[:, 1]
        a = roc_auc_score(yf[mask_te], p)
        scope.append({"학습 범위": label, "test_AUC": a})
        print(f"  {label:<24} {a:.4f}")
    a_sub = res.loc[[c for c in res.index if "랜덤포레스트" in c][0], "ROC-AUC"]
    scope.append({"학습 범위": "부분집합 + upset_A 라벨 (채택)", "test_AUC": a_sub})
    print(f"  {'부분집합 + upset_A 라벨 (채택)':<24} {a_sub:.4f}   ← 채택")
    pd.DataFrame(scope).to_csv(OUT / "training_scope.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print(f"완료 — 표 5개를 {OUT}/ 에 저장했습니다")
    print("=" * 72)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="인기마 붕괴(upset_A) 예측 실험")
    ap.add_argument("--quick", action="store_true", help="깊이 탐색·부트스트랩 축소")
    main(**vars(ap.parse_args()))
