"""
03_true_holdout.py — 진짜 최종 홀드아웃 검증

기존 test(2025-12-28~2026-08-09)는 이미 01_모델선정.docx와 02번 스크립트의
Lift/ROI 계산에 그대로 쓰인 구간이라 "미사용 데이터"가 아니다. 그런데
final.csv에는 이 이후의 새 데이터가 없고 KRA API 키도 설정되어 있지 않아
새로 수집할 수도 없다.

그래서 기존 test 구간의 마지막 ~8주를 별도로 떼어내 "진짜 홀드아웃"으로
쓴다. 모델은 그 앞부분(train + valid + test 앞부분)만으로 다시 학습하고,
하이퍼파라미터는 01_build_models.py와 동일하게 고정한다(홀드아웃을 보고
다시 고르지 않음). 홀드아웃은 한 번만 평가한다.

실행:
    python src/model_selection_validation/03_true_holdout.py

출력:
    results/final_validation/true_holdout_summary.csv
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
    TARGET_COL, CATEGORICAL_COLS, RANDOM_STATE,
    assign_time_split, SPLIT_RATIOS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/final_validation")
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])
RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000
TOP_PCT = 0.10
HOLDOUT_WEEKS = 8

MODEL_SPECS = {
    "darkhorse": {"subset_query": "pop_pct >= 0.50", "target": "upset_B", "max_depth": 12, "odds_col": "plcOdds"},
    "bust": {"subset_query": "pop_pct <= 0.25", "target": "upset_A", "max_depth": 8, "odds_col": None},
}


def load_data():
    df = pd.read_csv("final.csv", low_memory=False)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)
    return df


def prep_features(df):
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS and c != "fold" and c != "is_holdout"]
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or df[c].dtype == "object"]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    refit_mask = ~df["is_holdout"]
    medians = df.loc[refit_mask, num_cols].median()
    df[num_cols] = df[num_cols].fillna(medians)

    for col in cat_cols:
        df[col] = df[col].fillna("MISSING").astype(str)
        df[col] = LabelEncoder().fit(df[col].unique()).transform(df[col])

    return df, feature_cols


def roi_of(hit, odds):
    return (np.where(hit == 1, odds, 0.0) - 1.0).mean() * 100


def main():
    df = load_data()

    test_dates = df.loc[df["fold"] == "test", "rcDate"]
    cutoff = int((pd.to_datetime(str(test_dates.max())) - pd.Timedelta(weeks=HOLDOUT_WEEKS))
                 .strftime("%Y%m%d"))
    df["is_holdout"] = (df["fold"] == "test") & (df["rcDate"] >= cutoff)

    logger.info(f"test 구간: {test_dates.min()} ~ {test_dates.max()}")
    logger.info(f"홀드아웃 컷오프: rcDate >= {cutoff} (최근 {HOLDOUT_WEEKS}주)")
    logger.info(f"재학습 pool(=train+valid+test 앞부분) 행수: {(~df['is_holdout']).sum():,} | "
                f"홀드아웃 행수: {df['is_holdout'].sum():,}")

    summary_rows = []

    for name, spec in MODEL_SPECS.items():
        logger.info("=" * 60)
        logger.info(f"[{name}] 홀드아웃 검증")

        sub = df.query(spec["subset_query"]).reset_index(drop=True)
        sub, feature_cols = prep_features(sub.copy())

        refit = sub[~sub["is_holdout"]]
        holdout = sub[sub["is_holdout"]]

        if len(holdout) < 30:
            logger.warning(f"  홀드아웃 표본이 {len(holdout)}건뿐 — 결과 해석에 극히 주의 필요")

        X_refit, y_refit = refit[feature_cols], refit[spec["target"]]
        X_hold, y_hold = holdout[feature_cols], holdout[spec["target"]].values

        model = RandomForestClassifier(
            n_estimators=600, max_depth=spec["max_depth"], min_samples_leaf=50,
            class_weight=None, random_state=RANDOM_STATE, n_jobs=-1,
        )
        model.fit(X_refit, y_refit)
        proba = model.predict_proba(X_hold)[:, 1]

        auc = roc_auc_score(y_hold, proba) if y_hold.sum() > 0 else np.nan
        k = max(1, int(len(holdout) * TOP_PCT))
        order = np.argsort(-proba)
        top_idx = order[:k]
        top_rate = y_hold[top_idx].mean()
        base_rate = y_hold.mean()
        lift = top_rate / base_rate if base_rate > 0 else np.nan

        logger.info(f"  홀드아웃 {len(holdout)}건 | 기저율 {base_rate:.4f} | AUC {auc:.4f} | "
                    f"상위{TOP_PCT:.0%}({k}건) Lift {lift:.2f}")

        row = {
            "model": name, "cutoff_rcDate": cutoff, "n_refit": len(refit), "n_holdout": len(holdout),
            "holdout_base_rate": base_rate, "holdout_auc": auc, "holdout_k": k, "holdout_lift_at_10pct": lift,
        }

        if spec["odds_col"]:
            odds = df.loc[holdout.index, spec["odds_col"]].values[top_idx]
            hit = y_hold[top_idx]
            point_roi = roi_of(hit, odds)

            boot_rois = np.empty(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                idx = RNG.integers(0, k, size=k)
                boot_rois[i] = roi_of(hit[idx], odds[idx])
            ci_low, ci_high = np.percentile(boot_rois, [2.5, 97.5])

            logger.info(f"  홀드아웃 ROI 점추정 {point_roi:+.1f}% | 부트스트랩 95% CI "
                        f"[{ci_low:+.1f}%, {ci_high:+.1f}%] | 0 포함: {'예' if ci_low <= 0 <= ci_high else '아니오'}")
            row.update({
                "holdout_roi_pct": point_roi, "holdout_roi_ci_low": ci_low, "holdout_roi_ci_high": ci_high,
                "holdout_roi_ci_includes_zero": ci_low <= 0 <= ci_high,
            })
        else:
            # 인기마 붕괴: ROI 대신 Lift@10% 자체를 부트스트랩 (원본과 동일한 방식으로 비교 가능하게)
            def lift_boot(y_arr, p_arr, kk):
                order_ = np.argsort(-p_arr)
                top_rate = y_arr[order_[:kk]].mean()
                base_rate = y_arr.mean()
                return top_rate / base_rate if base_rate > 0 else np.nan

            n_hold = len(holdout)
            boot_lifts = np.empty(N_BOOTSTRAP)
            for i in range(N_BOOTSTRAP):
                idx = RNG.integers(0, n_hold, size=n_hold)
                boot_lifts[i] = lift_boot(y_hold[idx], proba[idx], k)
            ci_low, ci_high = np.percentile(boot_lifts, [2.5, 97.5])
            includes_one = ci_low <= 1.0 <= ci_high

            logger.info(f"  홀드아웃 Lift 부트스트랩 95% CI [{ci_low:.2f}, {ci_high:.2f}] | "
                        f"1.0 포함: {'예' if includes_one else '아니오'}")
            row.update({
                "holdout_lift_ci_low": ci_low, "holdout_lift_ci_high": ci_high,
                "holdout_lift_ci_includes_random_baseline": includes_one,
            })

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "true_holdout_summary.csv", index=False, encoding="utf-8-sig")
    logger.info("완료: results/final_validation/true_holdout_summary.csv")


if __name__ == "__main__":
    main()
