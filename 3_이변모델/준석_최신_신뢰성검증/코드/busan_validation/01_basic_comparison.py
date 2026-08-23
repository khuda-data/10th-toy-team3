"""
01_basic_comparison.py — 서울 vs 부경 기초 비교

부경(부산경남) 데이터를 서울과 나란히 비교해, 두 모집단이 통계적으로
얼마나 다른지 확인한다. 여기서 너무 이질적으로 나오면 2~4단계(모델 적용·
재학습·비교)를 진행하기 전에 먼저 팀과 방향을 다시 정한다.

실행:
    python src/busan_validation/01_basic_comparison.py

출력:
    results/busan_validation/01_basic_comparison.csv
    results/busan_validation/01_feature_distribution.csv
    results/busan_validation/01_summary.md
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/busan_validation")

KEY_FEATURES = ["jk_winrate", "tr_winrate", "wg_diff", "hr_winrate", "rating", "age", "train_runs_14"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("final.csv", low_memory=False)
    df["upset"] = ((df["pop_pct"] >= 0.5) & (df["win"] == 1)).astype(int)

    seoul = df[df["meet"] == "서울"]
    busan = df[df["meet"] == "부경"]

    # ------------------------------------------------------------------
    # 1. 기초 통계 비교
    # ------------------------------------------------------------------
    rows = []

    def add(metric, seoul_val, busan_val, note=""):
        rows.append({"metric": metric, "seoul": seoul_val, "busan": busan_val,
                      "diff": busan_val - seoul_val if isinstance(seoul_val, (int, float)) else None,
                      "note": note})

    add("행 수", len(seoul), len(busan))
    add("경주 수(고유 rcDate+rcNo)", seoul.groupby(["rcDate", "rcNo"]).ngroups,
        busan.groupby(["rcDate", "rcNo"]).ngroups)
    add("경주당 평균 출전두수", seoul["n_run"].mean(), busan["n_run"].mean())
    add("이변 비율 (pop_pct>=0.5 & win==1)", seoul["upset"].mean(), busan["upset"].mean())
    add("rating 결측률", seoul["rating"].isna().mean(), busan["rating"].isna().mean())
    add("데뷔전 비율(is_debut)", seoul["is_debut"].mean(), busan["is_debut"].mean())
    add("평균 배당(winOdds)", seoul["winOdds"].mean(), busan["winOdds"].mean())

    # rating 결측 패턴 — 등급별
    for rank in ["국6등급", "국5등급"]:
        s_rate = seoul.loc[seoul["rank"] == rank, "rating"].isna().mean()
        b_rate = busan.loc[busan["rank"] == rank, "rating"].isna().mean()
        add(f"rating 결측률 (rank={rank})", s_rate, b_rate)

    basic_df = pd.DataFrame(rows)
    basic_df.to_csv(OUTPUT_DIR / "01_basic_comparison.csv", index=False, encoding="utf-8-sig")

    logger.info("=== 기초 비교 ===")
    for _, r in basic_df.iterrows():
        logger.info(f"  {r['metric']}: 서울 {r['seoul']} vs 부경 {r['busan']}")

    # ------------------------------------------------------------------
    # 2. 주요 피처 분포 비교 (평균, 표준편차, Welch t-test)
    # ------------------------------------------------------------------
    feat_rows = []
    for feat in KEY_FEATURES:
        if feat not in df.columns:
            continue
        s = seoul[feat].dropna()
        b = busan[feat].dropna()
        t_stat, p_val = stats.ttest_ind(s, b, equal_var=False)
        pooled_std = np.sqrt((s.var() + b.var()) / 2)
        cohens_d = (b.mean() - s.mean()) / pooled_std if pooled_std > 0 else np.nan
        feat_rows.append({
            "feature": feat,
            "seoul_mean": s.mean(), "seoul_std": s.std(),
            "busan_mean": b.mean(), "busan_std": b.std(),
            "t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d,
            "significant_diff": p_val < 0.05,
        })

    feat_df = pd.DataFrame(feat_rows)
    feat_df.to_csv(OUTPUT_DIR / "01_feature_distribution.csv", index=False, encoding="utf-8-sig")

    logger.info("=== 주요 피처 분포 비교 (Welch t-test) ===")
    for _, r in feat_df.iterrows():
        logger.info(f"  {r['feature']}: 서울 {r['seoul_mean']:.3f}±{r['seoul_std']:.3f} vs "
                    f"부경 {r['busan_mean']:.3f}±{r['busan_std']:.3f} | "
                    f"Cohen's d={r['cohens_d']:.3f} | p={r['p_value']:.4f} "
                    f"({'유의차' if r['significant_diff'] else '유의차 없음'})")

    # ------------------------------------------------------------------
    # 3. 요약 문단 자동 생성
    # ------------------------------------------------------------------
    n_sig = feat_df["significant_diff"].sum()
    n_total = len(feat_df)
    large_d = feat_df[feat_df["cohens_d"].abs() >= 0.3]

    run_diff = busan["n_run"].mean() - seoul["n_run"].mean()
    upset_diff = busan["upset"].mean() - seoul["upset"].mean()
    rating_na_diff = busan["rating"].isna().mean() - seoul["rating"].isna().mean()

    summary_lines = [
        "# 서울 vs 부경 기초 비교 요약\n",
        f"- 행 수: 서울 {len(seoul):,}건 vs 부경 {len(busan):,}건\n",
        f"- 경주당 평균 출전두수: 서울 {seoul['n_run'].mean():.2f}두 vs 부경 {busan['n_run'].mean():.2f}두 "
        f"(차이 {run_diff:+.2f}두) — "
        + ("배경에서 가정한 '부경이 더 적다'는 통념과 반대로, 실제로는 부경이 서울보다 출전두수가 약간 많다.\n"
           if run_diff > 0 else "배경 가정과 일치.\n"),
        f"- 이변 비율: 서울 {seoul['upset'].mean():.4f} vs 부경 {busan['upset'].mean():.4f} (차이 {upset_diff:+.4f})\n",
        f"- rating 결측률: 서울 {seoul['rating'].isna().mean():.4f} vs 부경 {busan['rating'].isna().mean():.4f} "
        f"(차이 {rating_na_diff:+.4f})\n",
        f"- 주요 피처 {n_total}개 중 {n_sig}개에서 통계적으로 유의한 분포 차이(p<0.05) 확인. "
        f"그중 효과크기(|Cohen's d|)가 0.3 이상으로 실질적 차이인 피처: "
        f"{', '.join(large_d['feature'].tolist()) if len(large_d) else '없음'}\n",
    ]

    with open(OUTPUT_DIR / "01_summary.md", "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    logger.info("완료: results/busan_validation/01_*.csv, 01_summary.md")


if __name__ == "__main__":
    main()
