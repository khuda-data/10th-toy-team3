"""
02_strategy_true_holdout.py — 01번에서 고른 필터 전략의 진짜 홀드아웃 검증

01번은 7개 배당구간 x 4개 top%를 test셋에서 그리드서치해서 "plcOdds>=10배+
상위10%"를 골랐다. 이건 12_final_strategy.py(C모델)에서 지적했던 것과 똑같은
"test셋에서 직접 그리드서치" 문제라, 같은 잣대로 재검증한다.

기존 test 마지막 8주를 홀드아웃으로 떼고, 그 앞부분만으로 모델을 재학습한 뒤
(하이퍼파라미터는 고정, 홀드아웃을 보고 다시 고르지 않음) 01번이 고른 필터를
그대로 홀드아웃에 적용해 1회 평가한다.

실행:
    python src/darkhorse_final/02_strategy_true_holdout.py

출력:
    results/darkhorse_final/strategy_true_holdout.csv
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import (
    ID_COLS, MARKET_COLS, DUAL_MARKET_COLS, OUTCOME_COLS,
    TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE, assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/darkhorse_final")
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
HOLDOUT_WEEKS = 8

# 01번에서 채택된 전략
FILTER_LO, FILTER_HI = 10, 99999
TOP_PCT = 0.10


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def main():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)

    test_dates = df.loc[df["fold"] == "test", "rcDate"]
    cutoff = int((pd.to_datetime(str(test_dates.max())) - pd.Timedelta(weeks=HOLDOUT_WEEKS))
                 .strftime("%Y%m%d"))
    df["is_holdout"] = (df["fold"] == "test") & (df["rcDate"] >= cutoff)
    logger.info(f"홀드아웃 컷오프 rcDate>={cutoff}")

    sub = df.query("pop_pct >= 0.50").reset_index(drop=True)
    feature_cols = [c for c in sub.columns if c not in EXCLUDE_COLS and c not in ("fold", "is_holdout")]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or sub[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    refit_mask = ~sub["is_holdout"]
    medians = sub.loc[refit_mask, num_cols].median()
    sub[num_cols] = sub[num_cols].fillna(medians)
    for col in cat_cols:
        sub[col] = sub[col].fillna("MISSING").astype(str)
        sub[col] = LabelEncoder().fit(sub[col].unique()).transform(sub[col])

    refit = sub[refit_mask]
    holdout = sub[sub["is_holdout"]]
    logger.info(f"재학습 {len(refit):,}건 | 홀드아웃 {len(holdout):,}건")

    model = RandomForestClassifier(
        n_estimators=600, max_depth=12, min_samples_leaf=50,
        class_weight=None, random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(refit[feature_cols], refit["upset_B"])
    proba = model.predict_proba(holdout[feature_cols])[:, 1]

    hold_df = holdout[["rcDate", "upset_B", "plcOdds"]].copy()
    hold_df["proba"] = proba
    hold_df = hold_df.rename(columns={"plcOdds": "odds"})

    seg = hold_df[(hold_df["odds"] >= FILTER_LO) & (hold_df["odds"] < FILTER_HI)]
    if len(seg) < 5:
        logger.warning(f"홀드아웃 내 필터 통과 표본이 {len(seg)}건뿐 — 해석에 극히 주의")
    k = max(1, int(len(seg) * TOP_PCT))
    strategy = seg.nlargest(k, "proba")
    hit = strategy["upset_B"].values
    odds = strategy["odds"].values
    point_roi = roi_of(hit, odds)

    boot_rois = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = RNG.integers(0, k, size=k)
        boot_rois[i] = roi_of(hit[idx], odds[idx])
    ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])

    logger.info(f"홀드아웃 전략 검증: pool {len(seg)}건 중 베팅 {k}건, 적중 {int(hit.sum())}건")
    logger.info(f"ROI {point_roi:+.1f}% | 95% CI [{ci_low:+.1f}%, {ci_high:+.1f}%] | "
                f"0 포함: {'예' if ci_low <= 0 <= ci_high else '아니오'}")
    if k <= 43:
        logger.info(f"참고: 원래 test 기준(43건 베팅)보다 홀드아웃 pool이 작아 표본이 더 적을 수 있음")

    pd.DataFrame([{
        "filter": f"plcOdds>={FILTER_LO}", "top_pct": TOP_PCT, "cutoff_rcDate": cutoff,
        "n_refit": len(refit), "n_holdout": len(holdout), "strategy_pool": len(seg),
        "strategy_bets": k, "strategy_hits": int(hit.sum()),
        "roi_pct": point_roi, "roi_ci_low": ci_low, "roi_ci_high": ci_high,
        "roi_ci_includes_zero": ci_low <= 0 <= ci_high,
    }]).to_csv(OUTPUT_DIR / "strategy_true_holdout.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/darkhorse_final/strategy_true_holdout.csv")


if __name__ == "__main__":
    main()
