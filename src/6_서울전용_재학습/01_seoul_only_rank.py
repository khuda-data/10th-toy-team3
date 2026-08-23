# -*- coding: utf-8 -*-
"""
v11 전체 착순 모델 — 서울 전용 재학습
======================================
원본 `revised_v11_seoul_bugyeong_full_rank_20260822` 는 서울과 부경을 함께 썼습니다
(피처 목록에 `oh_meet_부경` 이 들어 있는 것이 증거입니다).

이 스크립트는 같은 설계를 **서울 데이터만으로** 다시 학습합니다.
모델은 우리가 채택하기로 한 세 가지만 돌립니다.

    1. Random Forest       500 trees · max_depth=14 · min_samples_leaf=8 · seed=42
    2. LightGBM            LambdaRank
    3. CatBoost            YetiRankPairwise

원본 패키지에는 학습 코드가 없어 `model_manifest.json` 의 피처 목록·타깃·설정을
역추적해 재구성했습니다. 재현 결과는 원본과 다를 수 있습니다 — 서울만 쓰므로
표본이 58% 로 줄고 부경 정보가 사라지기 때문입니다.

실행
    python src/6_서울전용_재학습/01_seoul_only_rank.py
    python src/6_서울전용_재학습/01_seoul_only_rank.py --quick    # 나무 수 축소
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# ── 경로 ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve()
for _ in range(5):
    if (_ROOT / "data" / "race_entries.csv.gz").exists():
        break
    _ROOT = _ROOT.parent
RACE_ENTRIES = _ROOT / "data" / "race_entries.csv.gz"
OUT = _ROOT / "final_report_and_models_20260823" / "models" / \
      "revised_v11_seoul_only_full_rank_20260823"

SEED = 42
SPLIT = (0.6, 0.2, 0.2)
MAX_ORD = 16          # 데이터상 최대 착순

# ── 원본 141개 피처에서 oh_meet_부경 만 제외한 140개를 만든다 ──
BASE_NUM = [
    "age", "buga1", "chulNo", "ilsu", "rating", "rcDist", "wgBudam", "wg", "wg_diff",
    "n_run", "hr_starts", "hr_winrate", "hr_plcrate", "jk_starts", "jk_winrate",
    "jk_plcrate", "tr_starts", "tr_winrate", "tr_plcrate", "ow_starts", "ow_winrate",
    "ow_plcrate", "hr_rest_days", "hr_last_dist", "hr_dist_chg", "hr_dist_starts",
    "hr_dist_winrate", "waterRate", "spRating", "stRating", "ill_n", "clinic_30d",
    "train_days_14", "train_runs_14", "train_sec_14", "wg__z", "age__pr", "rating__z",
    "hr_winrate__z", "jk_winrate__z", "tr_winrate__z", "hr_rest_days__z",
    "train_runs_14__z", "hr_style", "hr_style_sd", "race_style_mean", "race_style_sd",
    "race_front_n", "race_front_ratio", "is_front", "pace_conflict", "style_vs_race",
    "jkhr_starts", "jkhr_winrate", "wgBudam_chg", "hr_last_finpct", "hr_prev_rating",
    "wg_diff__pr", "wgBudam__pr", "hr_winrate__pr", "jk_winrate__pr",
    "hr_rest_days__pr", "hr_style_n", "tool_n",
]
MISSING_FLAGS = [
    "hr_style", "hr_style_sd", "race_style_sd", "hr_last_finpct", "hr_prev_rating",
    "age__z", "rating__pr", "jk_winrate__pr", "tr_winrate__pr", "hr_rest_days__pr",
]
ONEHOT = ["sex", "weather", "rcDay", "budam", "born", "ageCond"]   # meet 제외
TARGET_ENC = ["jkName", "trName", "owName", "rank"]


# ══════════════════════════════════════════════════════════
# 1. 데이터 준비
# ══════════════════════════════════════════════════════════
def load_seoul() -> pd.DataFrame:
    df = pd.read_csv(RACE_ENTRIES, low_memory=False)
    n_all = len(df)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    n_seoul = len(df)

    # 착순이 0인 행(실격·기권)이 있는 경주는 통째로 제외한다.
    # 전체 착순을 타깃으로 쓰므로 순서가 온전하지 않으면 학습·평가가 왜곡된다.
    bad = df.loc[df["ord"] == 0, "race_id"].unique()
    df = df[~df["race_id"].isin(bad)].reset_index(drop=True)

    print(f"  원천        {n_all:,}행")
    print(f"  서울 필터   {n_seoul:,}행  (부경 {n_all - n_seoul:,}행 제외)")
    print(f"  착순 결함   {len(bad)}경주 제외 -> {len(df):,}행 · {df.race_id.nunique():,}경주")

    # 시간순 6:2:2 — 경주 단위, 같은 날짜는 한 fold 에만
    df = df.sort_values(["rcDate", "race_id"], kind="stable").reset_index(drop=True)
    cnt = df["rcDate"].value_counts().sort_index()
    cum = cnt.cumsum() / len(df)
    tr = set(cum[cum <= SPLIT[0]].index)
    va = set(cum[(cum > SPLIT[0]) & (cum <= SPLIT[0] + SPLIT[1])].index)
    df["fold"] = np.where(df["rcDate"].isin(tr), "train",
                 np.where(df["rcDate"].isin(va), "valid", "test"))
    return df


def build_features(df: pd.DataFrame):
    """원본 manifest 의 141개 피처를 서울 전용으로 재구성 (oh_meet_부경 제외)."""
    X = pd.DataFrame(index=df.index)

    # ① 수치형 — 원본에 있는 것만
    for c in BASE_NUM:
        if c in df.columns:
            X[c] = pd.to_numeric(df[c], errors="coerce")

    # ② 결측 지시자 — 값을 채우기 전에 만들어야 한다
    for c in MISSING_FLAGS:
        if c in df.columns:
            X[f"{c}__missing"] = df[c].isna().astype(int)

    # ③ 원-핫
    for c in ONEHOT:
        if c not in df.columns:
            continue
        d = pd.get_dummies(df[c].astype(str), prefix=f"oh_{c}" if c != "ageCond" else "ageCond")
        X = pd.concat([X, d.astype(int)], axis=1)

    # ④ 마구(tool_set) 원-핫 — 쉼표로 여러 개가 들어 있다
    ts = df.get("tool_set", pd.Series("-", index=df.index)).fillna("-").astype(str)
    items = {}
    for row in ts:
        for t in row.split(","):
            t = t.strip()
            if t and t != "-":
                items[t] = items.get(t, 0) + 1
    keep = [k for k, v in sorted(items.items(), key=lambda x: -x[1]) if v >= 30][:34]
    for t in keep:
        X["tool_" + t.replace(" ", "_").replace("(", "").replace(")", "")] = \
            ts.str.contains(t, regex=False).astype(int)

    # ⑤ 타깃 인코딩 — train 에서만 적합 (누수 방지)
    m_tr = (df["fold"] == "train").values
    y_norm = df["y_norm"].values
    prior = y_norm[m_tr].mean()
    for c in TARGET_ENC:
        if c not in df.columns:
            continue
        g = pd.DataFrame({"k": df[c].astype(str), "y": y_norm})
        stat = g[m_tr].groupby("k")["y"].agg(["mean", "count"])
        k = 20.0                                      # 스무딩 강도
        sm = (stat["mean"] * stat["count"] + prior * k) / (stat["count"] + k)
        X[f"te_{c}"] = g["k"].map(sm).fillna(prior).values

    # ⑥ 남은 결측은 train 중앙값
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X[m_tr].median()).fillna(0)
    return X


# ══════════════════════════════════════════════════════════
# 2. 평가 지표 — 원본 metrics.json 과 같은 항목
# ══════════════════════════════════════════════════════════
def _dcg(rel):
    """선형 gain. 원본 v11 metrics 와 눈금을 맞추기 위해 2^rel-1 이 아니라 rel 을 쓴다.

    지수 gain(2^rel-1)을 쓰면 1착을 놓쳤을 때 벌점이 훨씬 커져
    같은 모델인데도 NDCG@1 이 0.73 -> 0.40 수준으로 내려간다.
    Spearman·Top-1 같은 정의 무관 지표가 원본과 근접하는 것으로 선형 gain 을 확인했다.
    """
    return np.sum(np.asarray(rel, float) / np.log2(np.arange(2, len(rel) + 2)))


def evaluate(df_eval: pd.DataFrame) -> dict:
    """race_id · ord(실제 착순) · score(예측 점수, 클수록 1착 예상) 가 필요."""
    n_races = 0
    ndcg = {1: [], 3: [], 5: [], "all": []}
    sp, kd, mae, pw, top1, top3r = [], [], [], [], [], []
    winner_rank = []

    for _, g in df_eval.groupby("race_id", sort=False):
        n = len(g)
        if n < 2:
            continue
        n_races += 1
        true_ord = g["ord"].values
        pred_ord = pd.Series(-g["score"].values).rank(method="first").values

        rel_true = np.maximum(0, MAX_ORD - true_ord)          # 등급 relevance
        order_pred = np.argsort(-g["score"].values)
        order_true = np.argsort(true_ord)
        for k in (1, 3, 5, "all"):
            kk = n if k == "all" else min(k, n)
            d = _dcg(rel_true[order_pred][:kk])
            i = _dcg(rel_true[order_true][:kk])
            ndcg[k].append(d / i if i > 0 else 0.0)

        if len(set(true_ord)) > 1 and len(set(pred_ord)) > 1:
            sp.append(spearmanr(true_ord, pred_ord).statistic)
            kd.append(kendalltau(true_ord, pred_ord).statistic)
        mae.append(np.mean(np.abs(true_ord - pred_ord)))

        # 쌍별 정확도
        ok = tot = 0
        for a in range(n):
            for b in range(a + 1, n):
                if true_ord[a] == true_ord[b]:
                    continue
                tot += 1
                ok += ((true_ord[a] < true_ord[b]) == (pred_ord[a] < pred_ord[b]))
        if tot:
            pw.append(ok / tot)

        top1.append(int(true_ord[order_pred[0]] == 1))
        k3 = min(3, n)
        top3r.append(len(set(np.where(true_ord <= 3)[0]) & set(order_pred[:k3])) / k3)
        winner_rank.append(float(pred_ord[np.argmin(true_ord)]))

    f = lambda x: float(np.mean(x)) if len(x) else None
    return {
        "evaluated_races": n_races,
        "evaluated_rows": int(len(df_eval)),
        "ndcg_at_1": f(ndcg[1]), "ndcg_at_3": f(ndcg[3]),
        "ndcg_at_5": f(ndcg[5]), "ndcg_all": f(ndcg["all"]),
        "mean_spearman": f(sp), "mean_kendall": f(kd),
        "mean_absolute_rank_error": f(mae),
        "pairwise_accuracy": f(pw),
        "top1_hit_rate": f(top1), "top3_recall": f(top3r),
        "mean_winner_predicted_rank": f(winner_rank),
    }


# ══════════════════════════════════════════════════════════
# 3. 모델 3종
# ══════════════════════════════════════════════════════════
def run_rf(X, df, m_tr, quick):
    """원본 설정: 500 trees, max_depth=14, min_samples_leaf=8, seed=42
       정규화된 착순 위치를 회귀. 예측값이 작을수록 상위."""
    mdl = RandomForestRegressor(
        n_estimators=150 if quick else 500, max_depth=14, min_samples_leaf=8,
        random_state=SEED, n_jobs=-1)
    mdl.fit(X[m_tr], df.loc[m_tr, "y_norm"])
    return mdl, -mdl.predict(X)          # 부호를 뒤집어 "클수록 상위"로 통일


def run_lgbm(X, df, m_tr, m_va, quick):
    """원본 설정: LambdaRank, 전체 착순에서 유도한 등급 relevance."""
    from lightgbm import LGBMRanker, early_stopping, log_evaluation
    rel = np.maximum(0, MAX_ORD - df["ord"].values).astype(int)
    grp = lambda m: df.loc[m].groupby("race_id", sort=False).size().values
    mdl = LGBMRanker(
        objective="lambdarank", n_estimators=100 if quick else 1000,
        learning_rate=0.05, num_leaves=63, min_child_samples=20,
        label_gain=[float(2 ** i - 1) for i in range(MAX_ORD + 1)],
        random_state=SEED, n_jobs=-1, verbose=-1)
    mdl.fit(X[m_tr], rel[m_tr], group=grp(m_tr),
            eval_set=[(X[m_va], rel[m_va])], eval_group=[grp(m_va)],
            eval_at=[1, 3, 5],
            callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
    return mdl, mdl.predict(X)


def run_catboost(X, df, m_tr, m_va, quick):
    """원본 설정: YetiRankPairwise, 전체 착순에서 유도한 등급 relevance."""
    from catboost import CatBoostRanker, Pool
    rel = np.maximum(0, MAX_ORD - df["ord"].values).astype(float)
    gid = df["race_id"].astype("category").cat.codes.values
    o_tr, o_va = np.argsort(gid[m_tr], kind="stable"), np.argsort(gid[m_va], kind="stable")
    ptr = Pool(X[m_tr].values[o_tr], rel[m_tr][o_tr], group_id=gid[m_tr][o_tr])
    pva = Pool(X[m_va].values[o_va], rel[m_va][o_va], group_id=gid[m_va][o_va])
    mdl = CatBoostRanker(
        loss_function="YetiRankPairwise", iterations=200 if quick else 1200,
        learning_rate=0.05, depth=6, random_seed=SEED,
        early_stopping_rounds=100, verbose=0)
    mdl.fit(ptr, eval_set=pva)
    return mdl, mdl.predict(X.values)


# ══════════════════════════════════════════════════════════
def main(quick=False):
    t0 = time.time()
    print("=" * 74)
    print("v11 전체 착순 모델 — 서울 전용 재학습")
    print("=" * 74)

    df = load_seoul()
    # 타깃: 경주 내 정규화 착순 (0 = 1착, 1 = 꼴찌)
    df["y_norm"] = df.groupby("race_id")["ord"].transform(
        lambda s: (s - 1) / max(1, len(s) - 1))

    X = build_features(df)
    m_tr = (df["fold"] == "train").values
    m_va = (df["fold"] == "valid").values
    m_te = (df["fold"] == "test").values

    print(f"\n  피처        {X.shape[1]}개   (원본 141개에서 oh_meet_부경 제외)")
    print(f"  train       {m_tr.sum():,}행 · {df.loc[m_tr,'race_id'].nunique():,}경주")
    print(f"  valid       {m_va.sum():,}행 · {df.loc[m_va,'race_id'].nunique():,}경주")
    print(f"  test        {m_te.sum():,}행 · {df.loc[m_te,'race_id'].nunique():,}경주")

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, fn in [
        ("random_forest_rank", lambda: run_rf(X, df, m_tr, quick)),
        ("lightgbm_rank", lambda: run_lgbm(X, df, m_tr, m_va, quick)),
        ("catboost_rank", lambda: run_catboost(X, df, m_tr, m_va, quick)),
    ]:
        print("\n" + "=" * 74)
        print(f"[{name}]")
        print("=" * 74)
        t = time.time()
        try:
            mdl, score = fn()
        except ImportError as e:
            print(f"  [건너뜀] {e}")
            continue
        df["score"] = score

        met = {}
        for split, m in [("valid", m_va), ("test", m_te)]:
            met[split] = evaluate(df.loc[m, ["race_id", "ord", "score"]])
            r = met[split]
            print(f"  {split:<6} NDCG@1 {r['ndcg_at_1']:.4f}  NDCG@3 {r['ndcg_at_3']:.4f}  "
                  f"NDCG@all {r['ndcg_all']:.4f}  Top-1 {r['top1_hit_rate']:.4f}  "
                  f"Spearman {r['mean_spearman']:.4f}")

        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        json.dump(met, open(d / "metrics.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        for split, m in [("valid", m_va), ("test", m_te)]:
            df.loc[m, ["race_id", "hrNo", "ord", "score"]].to_csv(
                d / f"{split}_rank_predictions.csv", index=False, encoding="utf-8-sig")
        imp = getattr(mdl, "feature_importances_", None)
        if imp is not None:
            pd.DataFrame({"feature": X.columns, "importance": imp}) \
              .sort_values("importance", ascending=False) \
              .to_csv(d / "feature_importance_full.csv", index=False, encoding="utf-8-sig")
        json.dump({
            "model": name, "scope": "서울 전용 (부경 제외)",
            "source_design": "revised_v11_seoul_bugyeong_full_rank_20260822",
            "feature_count": int(X.shape[1]),
            "features": list(X.columns),
            "target": "complete finish order from ord",
            "fit_rows": int(m_tr.sum()),
            "excluded_invalid_order_races": int(df.loc[df['ord'] == 0, 'race_id'].nunique()),
            "seed": SEED,
        }, open(d / "model_manifest.json", "w", encoding="utf-8"),
            ensure_ascii=False, indent=1)
        results[name] = met
        print(f"  소요 {time.time()-t:.0f}초 · 저장 {d.name}/")

    # ── 원본(서울+부경) 과 나란히 ──────────────────────────
    print("\n" + "=" * 74)
    print("원본(서울+부경) 대비")
    print("=" * 74)
    src = OUT.parent / "revised_v11_seoul_bugyeong_full_rank_20260822"
    print(f"  {'모델':<20}{'':>4}{'NDCG@1':>10}{'NDCG@all':>11}{'Top-1':>9}{'Spearman':>11}")
    for name in results:
        for label, met in [("서울+부경", None), ("서울만", results[name])]:
            if met is None:
                p = src / name / "metrics.json"
                if not p.exists():
                    continue
                met = json.load(open(p, encoding="utf-8"))
            v = met["valid"]
            print(f"  {name if label=='서울+부경' else '':<20}{label:>9}"
                  f"{v['ndcg_at_1']:>10.4f}{v['ndcg_all']:>11.4f}"
                  f"{v['top1_hit_rate']:>9.4f}{v['mean_spearman']:>11.4f}")
        print()

    json.dump(results, open(OUT / "comparison_summary.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"완료 — {time.time()-t0:.0f}초 · {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(**vars(ap.parse_args()))
