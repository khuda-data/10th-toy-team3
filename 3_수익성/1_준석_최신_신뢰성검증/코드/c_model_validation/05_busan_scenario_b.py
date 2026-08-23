"""
05_busan_scenario_b.py — C모델을 부경 데이터로 처음부터 재학습

부경 자체를 시간순 6:2:2로 분할해 C모델과 동일 스펙으로 재학습하고,
동일한 최종 전략(winOdds 10~50배 + 상위10%)을 부경 test에 적용한다.

실행:
    python src/c_model_validation/05_busan_scenario_b.py

출력:
    results/c_model_validation/busan_scenario_b.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import ID_COLS, OUTCOME_COLS, TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE, assign_time_split, SPLIT_RATIOS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/c_model_validation")
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
FINAL_FILTER = (10, 50)
FINAL_TOP_PCT = 0.10

ODDS_DROP = [
    "winOdds", "plcOdds", "p_raw", "logit_q", "log_q",
    "pop_rank", "is_fav", "book_sum", "takeout",
    "pl_harville", "pl_disc", "q_plc",
]
HIGH_CORR_DROP = [
    "chaksun2", "chaksun3", "chaksun4", "chaksun5",
    "buga2", "buga3", "dusu", "hr_style_n", "hr_prev_rating",
    "hr_last_finpct", "age__z", "train_runs_14__pr", "hr_last_wg",
    "wg__pr", "wg_diff__pr", "wgBudam__pr",
    "hr_winrate__pr", "hr_resid__pr",
    "jk_winrate__pr", "tr_winrate__pr",
    "hr_rest_days__pr", "bleed__pr", "rating__pr",
]
STRUCTURAL_MISSING = [
    "rating", "rating__z", "hr_winrate", "hr_plcrate", "hr_resid",
    "hr_starts", "hr_winrate__z", "hr_resid__z",
    "hr_last_ord", "hr_last_poppct", "hr_last_resid",
    "hr_last_dist", "hr_dist_chg", "hr_rest_days", "hr_rest_days__z",
    "hr_dist_winrate", "hr_dist_starts",
    "hr_style", "hr_style_sd", "style_vs_race",
    "race_style_mean", "race_style_sd", "race_front_ratio",
    "jkhr_winrate", "wgBudam_chg",
]
FULL_EXCLUDE = set(
    ID_COLS + OUTCOME_COLS + [TARGET_COL, "upset", "fold", "pop_pct"]
    + ODDS_DROP
    + ["winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"]
    + ["gap_h", "gap_d"]
)


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def main():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "부경"].reset_index(drop=True)
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)
    df["_winOdds"] = df["winOdds"].copy()
    df = df[df["pop_pct"] >= 0.5].reset_index(drop=True)

    drop_cols = [c for c in HIGH_CORR_DROP + ODDS_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, "rcDate", SPLIT_RATIOS)

    for col in STRUCTURAL_MISSING:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    c_features = [c for c in df.columns if c not in FULL_EXCLUDE and c != "_winOdds"]
    train_mask = df["fold"] == "train"
    cat_cols = [c for c in c_features if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in c_features if c not in cat_cols]

    remaining = [c for c in num_cols if df[c].isnull().any()]
    if remaining:
        medians = df.loc[train_mask, remaining].median()
        df[remaining] = df[remaining].fillna(medians)

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        df[col] = LabelEncoder().fit(df[col].unique()).transform(df[col])

    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    train = df[df["fold"] == "train"]
    test = df[df["fold"] == "test"]
    logger.info(f"부경 train {len(train):,} | test {len(test):,} | 기저율(test) {test['upset'].mean():.4f}")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=20,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(train[c_features].values.astype(np.float32), train["upset"].values)

    proba = model.predict_proba(test[c_features].values.astype(np.float32))[:, 1]
    y_test = test["upset"].values
    auc = roc_auc_score(y_test, proba)
    logger.info(f"test AUC {auc:.4f}")

    out = test[["rcDate", "upset", "_winOdds"]].copy()
    out["proba"] = proba

    lo, hi = FINAL_FILTER
    seg = out[(out["_winOdds"] >= lo) & (out["_winOdds"] < hi)]
    k = max(1, int(len(seg) * FINAL_TOP_PCT))
    strategy = seg.nlargest(k, "proba")
    hit = strategy["upset"].values
    odds = strategy["_winOdds"].values
    point_roi = roi_of(hit, odds)

    boot_rois = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = RNG.integers(0, k, size=k)
        boot_rois[i] = roi_of(hit[idx], odds[idx])
    ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])

    logger.info(f"최종 전략(pool {len(seg)}, 베팅 {k}, 적중 {int(hit.sum())}) ROI {point_roi:+.1f}% | "
                f"95% CI [{ci_low:+.1f}%, {ci_high:+.1f}%] | 0 포함: {'예' if ci_low<=0<=ci_high else '아니오'}")

    pd.DataFrame([{
        "scenario": "B_retrain_busan", "n_train": len(train), "n_test": len(test),
        "base_rate": y_test.mean(), "auc": auc,
        "strategy_pool": len(seg), "strategy_bets": k, "strategy_hits": int(hit.sum()),
        "roi_pct": point_roi, "roi_ci_low": ci_low, "roi_ci_high": ci_high,
        "roi_ci_includes_zero": ci_low <= 0 <= ci_high,
    }]).to_csv(OUTPUT_DIR / "busan_scenario_b.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/c_model_validation/busan_scenario_b.csv")


if __name__ == "__main__":
    main()
