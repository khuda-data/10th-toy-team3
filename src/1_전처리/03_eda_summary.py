"""
03_eda_summary.py — 종합 요약 리포트 자동 생성 (5단계)

1~4단계의 결과 CSV를 읽어서 summary.md를 자동 작성한다.

실행:
    python src/1_전처리/03_eda_summary.py
"""

import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/eda")

# 피처명 → 한국어 설명 매핑
FEATURE_DESC = {
    "hr_style": "각질 (주행 스타일)",
    "hr_style_sd": "각질 일관성",
    "jkhr_winrate": "기수-마필 조합 승률",
    "race_style_mean": "경주 내 평균 각질",
    "race_style_sd": "경주 내 각질 분산",
    "race_front_ratio": "경주 내 선행마 비율",
    "style_vs_race": "본인 각질 - 경주 평균",
    "hr_prev_rating": "직전 경주 레이팅",
    "hr_winrate__z": "마필 승률 (경주 내 z점수)",
    "hr_resid__z": "마필 잔차 (경주 내 z점수)",
    "rating": "마필 레이팅",
    "rating__z": "레이팅 (경주 내 z점수)",
    "rating__pr": "레이팅 (경주 내 백분위)",
    "hr_winrate": "마필 통산 승률",
    "hr_plcrate": "마필 통산 입상률",
    "hr_resid": "마필 평균 잔차",
    "hr_starts": "마필 통산 출주 횟수",
    "hr_last_finpct": "직전 경주 착순 백분위",
    "hr_last_poppct": "직전 경주 인기 백분위",
    "hr_last_ord": "직전 경주 착순",
    "hr_last_resid": "직전 경주 잔차",
    "hr_rest_days": "직전 경주 후 휴양일수",
    "hr_dist_winrate": "같은 거리 승률",
    "wgBudam_chg": "직전 대비 부담중량 증감",
    "hr_winrate__pr": "마필 승률 (경주 내 백분위)",
    "hr_resid__pr": "마필 잔차 (경주 내 백분위)",
    "age__z": "연령 (경주 내 z점수)",
    "age__pr": "연령 (경주 내 백분위)",
    "wg__z": "마체중 (경주 내 z점수)",
    "wg__pr": "마체중 (경주 내 백분위)",
    "chaksun1": "1착 상금",
    "chaksun2": "2착 상금",
}


def get_desc(col: str) -> str:
    return FEATURE_DESC.get(col, col)


def build_summary() -> str:
    """결과 파일들을 읽어 summary.md 본문을 생성."""
    lines = []
    lines.append("# EDA Summary Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n---\n")

    # --- 결측치 ---
    lines.append("## 1. Missing Values — Top 10 & Treatment Suggestions")
    lines.append("")

    missing_path = OUTPUT_DIR / "missing_report.csv"
    pattern_path = OUTPUT_DIR / "missing_pattern_report.csv"

    if missing_path.exists():
        missing = pd.read_csv(missing_path)
        top10 = missing[missing["missing_pct"] > 0].head(10)

        # 패턴 분류 로드
        pattern_map = {}
        if pattern_path.exists():
            pattern = pd.read_csv(pattern_path)
            pattern_map = dict(zip(pattern["column"], pattern["classification"]))

        lines.append("| # | Column | Missing % | Classification | Suggested Treatment |")
        lines.append("|---|--------|-----------|---------------|---------------------|")

        for i, (_, row) in enumerate(top10.iterrows(), 1):
            col = row["column"]
            pct = row["missing_pct"]
            cls = pattern_map.get(col, "unknown")

            if cls == "structural":
                treatment = "Flag indicator + fill with 0 (or low value)"
            elif cls == "random":
                treatment = "Group median imputation"
            else:
                treatment = "Investigate further"

            desc = get_desc(col)
            lines.append(f"| {i} | {col} ({desc}) | {pct:.1f}% | {cls} | {treatment} |")

        lines.append("")
        lines.append("**Key findings:**")
        lines.append("- Structural missing: Caused by system design (e.g., rating not assigned to Grade 6 / 2yo horses)")
        lines.append("- Random missing: No specific group concentration, impute with group median")
    else:
        lines.append("*(missing_report.csv not found — run 01_missing_check.py first)*")

    lines.append("\n---\n")

    # --- 고상관 쌍 ---
    lines.append("## 2. High Correlation Pairs (|r| >= 0.8)")
    lines.append("")

    pairs_path = OUTPUT_DIR / "high_correlation_pairs.csv"
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)

        lines.append(f"Total pairs with |r| >= 0.8: **{len(pairs)}**")
        lines.append("")
        lines.append("| # | Feature 1 | Feature 2 | r | Drop Candidate | Reason |")
        lines.append("|---|-----------|-----------|---|----------------|--------|")

        for i, (_, row) in enumerate(pairs.head(20).iterrows(), 1):
            f1, f2 = row["feature_1"], row["feature_2"]
            r = row["correlation"]
            more_missing = row["more_missing_side"]

            # 제거 후보 결정 로직
            if more_missing != "same":
                drop = more_missing
                reason = "More missing values"
            elif "__z" in f2 or "__pr" in f2:
                drop = f2
                reason = "Derived (z-score/percentile)"
            elif "__z" in f1 or "__pr" in f1:
                drop = f1
                reason = "Derived (z-score/percentile)"
            else:
                drop = f2
                reason = "Redundant information"

            lines.append(f"| {i} | {f1} | {f2} | {r:.4f} | **{drop}** | {reason} |")

        if len(pairs) > 20:
            lines.append(f"\n*... and {len(pairs) - 20} more pairs (see high_correlation_pairs.csv)*")

        lines.append("")
        lines.append("**Recommendations:**")
        lines.append("- Consider removing one feature from each highly correlated pair")
        lines.append("- Prefer keeping the feature with less missing and more interpretability")
        lines.append("- `__z` and `__pr` suffixed columns are standardized versions — keep the original if redundant")
    else:
        lines.append("*(high_correlation_pairs.csv not found — run 02_correlation_check.py first)*")

    lines.append("\n---\n")

    # --- VIF (선택) ---
    vif_path = OUTPUT_DIR / "vif_report.csv"
    if vif_path.exists():
        lines.append("## 3. VIF Analysis (Supplementary — Beyond Textbook Scope)")
        lines.append("")
        lines.append("> **Note:** VIF is NOT covered in the textbook. This is a supplementary reference.")
        lines.append("")

        vif = pd.read_csv(vif_path)
        high_vif = vif[vif["VIF"] >= 10]

        if len(high_vif) > 0:
            lines.append(f"Features with VIF >= 10: **{len(high_vif)}**")
            lines.append("")
            lines.append("| # | Feature | VIF |")
            lines.append("|---|---------|-----|")
            for i, (_, row) in enumerate(high_vif.head(15).iterrows(), 1):
                lines.append(f"| {i} | {row['feature']} | {row['VIF']:.1f} |")
        else:
            lines.append("No features with VIF >= 10.")

        lines.append("")

    lines.append("\n---\n")
    lines.append("## Summary of Actions")
    lines.append("")
    lines.append("1. **Missing values**: Apply structural/random classification to decide imputation strategy")
    lines.append("2. **High correlation**: Drop one from each pair (preferably the derived or more-missing one)")
    lines.append("3. **Before modeling**: Re-check after dropping columns to confirm multicollinearity is resolved")
    lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[5 Stage] Summary Report Generation")
    logger.info("=" * 60)

    content = build_summary()

    output_path = OUTPUT_DIR / "summary.md"
    output_path.write_text(content, encoding="utf-8")
    logger.info(f"  Saved: {output_path}")

    logger.info("\n" + "=" * 60)
    logger.info("03_eda_summary.py Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
