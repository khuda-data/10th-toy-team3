# -*- coding: utf-8 -*-
"""
upset_A_odds_ablation.py — 인기마 붕괴(upset_A) 예측에 배당률(q)을 넣으면 어떻게 되나

원본 upset_A_인기마붕괴_실험.py는 배당·시장 정보를 전부 제외하고 학습한다
(피처 108개, q 없음). 다크호스에서 q를 넣었을 때 AUC가 0.657→0.742로
크게 오른 것과 같은 실험을 인기마 붕괴에도 적용해, A(q만)/B(기존, q 없음)/
C(B+q) 세 피처셋으로 비교한다.

타겟·서브셋·전처리·모델 하이퍼파라미터는 원본 upset_A 스크립트와 완전히
동일하게 맞췄고, 피처셋만 다르다.

실행:
    python src/model_selection_validation/upset_A_odds_ablation.py

출력:
    results/upset_A/odds_ablation.csv
    results/upset_A/odds_ablation_bootstrap.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RACE_ENTRIES = _PROJECT_ROOT / "final.csv"
OUT = Path("results/upset_A")

SEED = 42
SPLIT = (0.6, 0.2, 0.2)
N_BOOT = 2000

# 원본과 동일 — q 딱 하나만 예외적으로 살려둔다
DROP_TARGET = ["pop_pct", "fin_pct", "upset_A", "upset_B", "surprise_score",
               "darkhorse", "favorite_bust", "upset"]
DROP_MARKET_NO_Q = ["winOdds", "plcOdds", "p_raw", "q_plc", "log_q", "logit_q",
                     "pop_rank", "is_fav", "book_sum", "takeout", "pl_harville",
                     "pl_disc", "gap_h", "gap_d",
                     "winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
DROP_RESULT = ["ord", "fin_rank", "win", "place", "resid",
               "chaksun1", "chaksun2", "chaksun3", "chaksun4", "chaksun5",
               "rcTime", "diffUnit"]
DROP_ID = ["race_id", "entry_id", "hrName", "jkName", "trName", "owName",
           "hrNo", "jkNo", "trNo", "owNo", "meet", "rcDate", "fold"]


def load_subset() -> pd.DataFrame:
    df = pd.read_csv(RACE_ENTRIES, low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    if "upset_A" not in df.columns:
        df["upset_A"] = ((df["pop_pct"] <= 0.25) & (df["fin_pct"] >= 0.50)).astype(int)
    sub = df[df["pop_pct"] <= 0.25].reset_index(drop=True)
    sub = sub.sort_values("rcDate", kind="stable").reset_index(drop=True)
    cnt = sub["rcDate"].value_counts().sort_index()
    cum = cnt.cumsum() / len(sub)
    tr_d = set(cum[cum <= SPLIT[0]].index)
    va_d = set(cum[(cum > SPLIT[0]) & (cum <= SPLIT[0] + SPLIT[1])].index)
    sub["fold"] = np.where(sub["rcDate"].isin(tr_d), "train",
                  np.where(sub["rcDate"].isin(va_d), "valid", "test"))
    return sub


def make_xy(sub: pd.DataFrame, feature_set: str):
    """feature_set: 'A' (q만) / 'B' (기존, q 없음) / 'C' (B+q)"""
    drop = [c for c in DROP_TARGET + DROP_MARKET_NO_Q + DROP_RESULT + DROP_ID
            if c in sub.columns]
    X_full = sub.drop(columns=drop)  # q가 살아있는 상태 (B는 여기서 q만 더 뺌)

    for c in X_full.columns:
        if X_full[c].dtype == "object":
            X_full[c] = X_full[c].astype("category").cat.codes
    X_full = X_full.apply(pd.to_numeric, errors="coerce")

    m = sub["fold"] == "train"
    X_full = X_full.fillna(X_full[m].median()).fillna(0)

    if feature_set == "A":
        X = X_full[["q"]]
    elif feature_set == "B":
        X = X_full.drop(columns=["q"])
    else:  # C
        X = X_full

    y = sub["upset_A"].values
    return X, y, m.values, (sub["fold"] == "valid").values, (sub["fold"] == "test").values


def lift_at(y, p, k):
    n = max(1, int(len(y) * k))
    idx = np.argsort(p)[::-1][:n]
    base = y.mean()
    return (y[idx].mean() / base) if base > 0 else np.nan


def bootstrap_lift(y, p, k=0.10, n_boot=N_BOOT, seed=SEED):
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


def fit_predict(kind, model, Xtr, ytr, Xte):
    if kind == "linear":
        sc = StandardScaler().fit(Xtr)
        model.fit(sc.transform(Xtr), ytr)
        return model.predict_proba(sc.transform(Xte))[:, 1]
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sub = load_subset()

    print("=" * 72)
    print("인기마 붕괴(upset_A) — 배당률(q) 포함 여부 ablation (A/B/C)")
    print("=" * 72)

    rows = []
    boot_rows = []

    for fs, label in [("A", "A (q만)"), ("B", "B (기존, q 없음)"), ("C", "C (B+q)")]:
        X, y, m_tr, m_va, m_te = make_xy(sub, fs)
        Xtr, ytr = X[m_tr], y[m_tr]
        Xte, yte = X[m_te], y[m_te]

        print(f"\n[{label}] 피처 {X.shape[1]}개")

        for name, kind, model in [
            ("Logistic", "linear", LogisticRegression(C=0.1, max_iter=2000, random_state=SEED)),
            ("RF", "tree", RandomForestClassifier(
                n_estimators=600, max_depth=12, min_samples_leaf=50,
                random_state=SEED, n_jobs=-1)),
        ]:
            p = fit_predict(kind, model, Xtr, ytr, Xte)
            auc = roc_auc_score(yte, p)
            lift10 = lift_at(yte, p, 0.10)
            lo, hi = bootstrap_lift(yte, p, 0.10)

            print(f"  {name:<10} AUC {auc:.4f}  Lift@10% {lift10:.2f}  "
                  f"CI [{lo:.2f}, {hi:.2f}]{'  ← 1 초과' if lo > 1 else ''}")

            rows.append({"feature_set": label, "model": name, "n_features": X.shape[1],
                        "AUC": auc, "Lift@10%": lift10})
            boot_rows.append({"feature_set": label, "model": name,
                             "Lift@10%": lift10, "CI_low": lo, "CI_high": hi,
                             "ci_excludes_1": lo > 1})

    res = pd.DataFrame(rows)
    boot = pd.DataFrame(boot_rows)
    res.to_csv(OUT / "odds_ablation.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(OUT / "odds_ablation_bootstrap.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 72)
    print("요약 (RF 기준)")
    print("=" * 72)
    rf_res = res[res["model"] == "RF"].set_index("feature_set")
    print(rf_res[["n_features", "AUC", "Lift@10%"]].to_string())

    a_auc = rf_res.loc["A (q만)", "AUC"]
    b_auc = rf_res.loc["B (기존, q 없음)", "AUC"]
    c_auc = rf_res.loc["C (B+q)", "AUC"]
    print(f"\nB->C (q 추가) AUC 변화: {b_auc:.4f} -> {c_auc:.4f} ({c_auc-b_auc:+.4f})")
    print(f"A (q 단독) vs B (기존 피처만): {a_auc:.4f} vs {b_auc:.4f}")

    print(f"\n완료: {OUT}/odds_ablation.csv, odds_ablation_bootstrap.csv")


if __name__ == "__main__":
    main()
