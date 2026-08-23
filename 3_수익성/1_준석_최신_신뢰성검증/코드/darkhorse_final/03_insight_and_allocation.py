"""
03_insight_and_allocation.py — 배당률 제외 다크호스 모델: 인사이트 + 조합전략 + 배분전략

기준 전략은 "필터 없음 + 상위10%"(홀드아웃에서 유의성 확인된 유일한 전략).
14_upset_insight_and_allocation.py(C모델 계보)와 동일한 구성을 q 없는
다크호스 모델에 적용한다.

Part A. 다크호스 vs 정상 피처 비교(Cohen's d) + K-means 유형 분류 + 의사결정나무 규칙
Part B. 인기마 붕괴(bust) 모델과의 조합 — 다크호스 후보 중 "붕괴 위험이 낮은
        경주"만 남기면 적중률이 개선되는지 검증 (개선실험 6단계와 동일 아이디어)
Part C. 베팅 배분 전략 비교 (Flat / EV비례 / 확률비례 / Half-Kelly)

실행:
    python src/darkhorse_final/03_insight_and_allocation.py

출력:
    results/darkhorse_final/feature_comparison.csv
    results/darkhorse_final/cluster_profiles.csv
    results/darkhorse_final/decision_tree_rules.txt
    results/darkhorse_final/combination_with_bust.csv
    results/darkhorse_final/allocation_comparison.csv
"""

import logging
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = Path("results/darkhorse_final")
MODEL_DIR = Path("results/final_validation")
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])
TOP_PCT = 0.10


def load_full():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)
    return df


def prep(df):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "fold"]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    train_mask = df["fold"] == "train"
    medians = df.loc[train_mask, num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)
    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        df[col] = LabelEncoder().fit(df[col].unique()).transform(df[col])
    return df, feature_cols, num_cols, cat_cols


def part_a_insight(dh_test_raw, dh_test_encoded, feature_cols, num_cols):
    logger.info("=" * 60)
    logger.info("[Part A] 다크호스 인사이트 — 피처 비교 · 군집화 · 의사결정나무")

    is_upset = dh_test_encoded["upset_B"] == 1
    upset_df = dh_test_encoded[is_upset]
    normal_df = dh_test_encoded[~is_upset]
    logger.info(f"  다크호스 {len(upset_df)}건 vs 정상 {len(normal_df)}건")

    rows = []
    for feat in num_cols:
        u = upset_df[feat].astype(float)
        n = normal_df[feat].astype(float)
        pooled_std = np.sqrt((u.var() + n.var()) / 2)
        d = (u.mean() - n.mean()) / pooled_std if pooled_std > 0 else np.nan
        rows.append({"feature": feat, "upset_mean": u.mean(), "normal_mean": n.mean(),
                     "cohens_d": d, "abs_d": abs(d)})
    fc = pd.DataFrame(rows).sort_values("abs_d", ascending=False)
    fc.to_csv(OUTPUT_DIR / "feature_comparison.csv", index=False, encoding="utf-8-sig")
    logger.info("  상위 8개 차이 피처:")
    for _, r in fc.head(8).iterrows():
        logger.info(f"    {r['feature']:<22s} d={r['cohens_d']:+.3f}")

    # K-means (다크호스 케이스만)
    X = StandardScaler().fit_transform(upset_df[num_cols].astype(float))
    best_k, best_score = 2, -1
    for k in range(2, 5):
        if len(upset_df) <= k:
            continue
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X)
    upset_df = upset_df.copy()
    upset_df["cluster"] = km.labels_
    profiles = upset_df.groupby("cluster")[num_cols].mean()
    profiles["n"] = upset_df["cluster"].value_counts().sort_index()
    profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv", encoding="utf-8-sig")
    logger.info(f"  K-means: k={best_k} (실루엣 {best_score:.3f}), 군집 크기: "
                f"{upset_df['cluster'].value_counts().sort_index().tolist()}")

    # 의사결정나무 (해석용, depth=4)
    y = dh_test_encoded["upset_B"].values
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=RANDOM_STATE)
    dt.fit(dh_test_encoded[feature_cols], y)
    rules = export_text(dt, feature_names=feature_cols, max_depth=4)
    with open(OUTPUT_DIR / "decision_tree_rules.txt", "w", encoding="utf-8") as f:
        f.write(rules)
    logger.info("  의사결정나무 규칙 저장 완료")

    return fc, profiles


def part_b_combination(dh_pred, bust_pred):
    logger.info("=" * 60)
    logger.info("[Part B] 인기마 붕괴 모델과의 조합 — 붕괴위험 낮은 경주만 필터링")

    # rcDate 단위로 그 날짜의 '평균 붕괴위험'을 계산해 다크호스 후보에 붙임
    bust_risk_by_date = bust_pred.groupby("rcDate")["proba"].mean().rename("bust_risk_mean")
    dh = dh_pred.merge(bust_risk_by_date, on="rcDate", how="left")
    dh["bust_risk_mean"] = dh["bust_risk_mean"].fillna(dh["bust_risk_mean"].median())

    k_all = max(1, int(len(dh) * TOP_PCT))
    base_strategy = dh.nlargest(k_all, "proba")
    base_roi = (np.where(base_strategy["upset_B"] == 1, base_strategy["odds"], 0.0) - 1.0).mean() * 100
    base_hits = int(base_strategy["upset_B"].sum())

    rows = [{"variant": "기준 (필터 없음, 상위10%)", "n_bets": k_all, "n_hits": base_hits, "roi_pct": base_roi}]

    for pct in [30, 50, 70]:
        cutoff = np.percentile(dh["bust_risk_mean"], pct)
        pool = dh[dh["bust_risk_mean"] <= cutoff]
        if len(pool) < 20:
            continue
        k = max(1, int(len(pool) * TOP_PCT))
        strat = pool.nlargest(k, "proba")
        hit = strat["upset_B"].values
        odds = strat["odds"].values
        roi = (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100
        rows.append({"variant": f"붕괴위험 하위{pct}% 경주만", "n_bets": k,
                    "n_hits": int(hit.sum()), "roi_pct": roi})
        logger.info(f"  붕괴위험 하위{pct}%만: 베팅 {k}건, 적중 {int(hit.sum())}건, ROI {roi:+.1f}%")

    comb = pd.DataFrame(rows)
    comb.to_csv(OUTPUT_DIR / "combination_with_bust.csv", index=False, encoding="utf-8-sig")
    logger.info(f"  기준 대비 조합 전략 개선 여부: "
                f"{'개선' if comb['roi_pct'].iloc[1:].max() > base_roi else '개선 없음'}")
    return comb


def part_c_allocation(dh_pred):
    logger.info("=" * 60)
    logger.info("[Part C] 베팅 배분 전략 비교")

    k = max(1, int(len(dh_pred) * TOP_PCT))
    strat = dh_pred.nlargest(k, "proba").copy()
    hit = strat["upset_B"].values
    odds = strat["odds"].values
    proba = strat["proba"].values

    def summarize(weights, label):
        weights = weights / weights.sum() * len(weights)  # 평균 가중치=1로 정규화
        ret = (np.where(hit == 1, odds, 0.0) * weights).sum()
        cost = weights.sum()
        roi = (ret - cost) / cost * 100
        return {"strategy": label, "roi_pct": roi,
                "max_weight": weights.max(), "min_weight": weights.min()}

    rows = []
    rows.append(summarize(np.ones(k), "Flat (균등)"))

    ev = proba * odds
    ev_w = np.clip(ev, 0.1, None)
    rows.append(summarize(ev_w, "EV 비례"))

    rows.append(summarize(np.clip(proba, 0.01, None), "확률 비례"))

    b = odds - 1
    f_star = (proba * b - (1 - proba)) / b
    f_star = np.clip(f_star, 0, None)
    half_kelly = np.clip(f_star / 2, 0.01, None)
    rows.append(summarize(half_kelly, "Half-Kelly"))

    alloc = pd.DataFrame(rows)
    alloc.to_csv(OUTPUT_DIR / "allocation_comparison.csv", index=False, encoding="utf-8-sig")
    logger.info(alloc.to_string(index=False))
    return alloc


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_full()
    dh_full, feature_cols, num_cols, cat_cols = prep(df.query("pop_pct >= 0.50").reset_index(drop=True).copy())
    dh_test_encoded = dh_full[dh_full["fold"] == "test"]

    part_a_insight(None, dh_test_encoded, feature_cols, num_cols)

    dh_pred = pd.read_csv(MODEL_DIR / "darkhorse_test_predictions.csv")
    bust_pred = pd.read_csv(MODEL_DIR / "bust_test_predictions.csv")
    part_b_combination(dh_pred, bust_pred)
    part_c_allocation(dh_pred)

    logger.info("=" * 60)
    logger.info("완료: results/darkhorse_final/ 에 5개 파일 저장")


if __name__ == "__main__":
    main()
