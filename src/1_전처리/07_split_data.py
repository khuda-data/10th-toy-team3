"""
07_split_data.py — 버전별 CSV를 train/valid/test로 물리적 분리

data/versions/의 각 버전 CSV를 fold 컬럼 기준으로 분리하여
data/전처리_데이터셋/<version>/ 아래에 train.csv, valid.csv, test.csv로 저장한다.

fold 컬럼은 이미 시간순으로 3분할되어 있으므로 섞지 않고 그대로 사용한다.

실행:
    python src/1_전처리/07_split_data.py
    python src/1_전처리/07_split_data.py --version v2_standard   # 특정 버전만

출력:
    data/전처리_데이터셋/v1_base/{train,valid,test}.csv
    data/전처리_데이터셋/v2_standard/{train,valid,test}.csv
    data/전처리_데이터셋/v3_minmax/{train,valid,test}.csv
    data/전처리_데이터셋/v4_robust/{train,valid,test}.csv
    data/전처리_데이터셋/분할안내.md
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import VERSIONS_DIR, SPLITS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


VERSIONS = [
    "v1_base", "v2_standard", "v3_minmax", "v4_robust",
    "v5_base_no_outlier", "v6_standard_no_outlier",
    "v7_minmax_no_outlier", "v8_robust_no_outlier",
]


def split_version(version: str) -> dict:
    """한 버전 CSV를 fold 기준으로 분리하여 저장. 통계 dict 반환."""
    src = VERSIONS_DIR / f"{version}.csv"
    if not src.exists():
        logger.warning(f"  {src} not found. Skipping.")
        return None

    df = pd.read_csv(src, dtype={"hrNo": str, "jkNo": str, "owNo": str, "trNo": str})

    out_dir = SPLITS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"version": version, "total": len(df), "cols": len(df.columns)}

    for fold_name in ["train", "valid", "test"]:
        subset = df[df["fold"] == fold_name].reset_index(drop=True)
        out_path = out_dir / f"{fold_name}.csv"
        subset.to_csv(out_path, index=False, encoding="utf-8-sig")

        win_rate = subset["win"].mean() * 100 if "win" in subset.columns else 0
        date_min = subset["rcDate"].min() if "rcDate" in subset.columns else "-"
        date_max = subset["rcDate"].max() if "rcDate" in subset.columns else "-"

        stats[fold_name] = {
            "rows": len(subset),
            "win_rate": round(win_rate, 2),
            "date_min": date_min,
            "date_max": date_max,
        }

        logger.info(
            f"    {fold_name:5s}: {len(subset):>6,} rows | "
            f"win {win_rate:5.2f}% | {date_min}~{date_max}"
        )

    return stats


def write_split_info(all_stats: list):
    """분할 정보를 마크다운으로 저장."""
    lines = []
    lines.append("# Train/Valid/Test Split Info")
    lines.append(f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("\n---\n")

    lines.append("## Split Method")
    lines.append("")
    lines.append("- **Time-based split at 60 : 20 : 20** (train : valid : test)")
    lines.append("- Based on `rcDate` order — NOT shuffled")
    lines.append("- Split boundaries fall on race-date edges, so a single race never spans two folds")
    lines.append("- Prevents data leakage from future races into training")
    lines.append("")

    # 첫 번째 버전 기준으로 분할 통계 (모든 버전 동일)
    if all_stats:
        s = all_stats[0]
        lines.append("## Split Sizes (identical across all versions)")
        lines.append("")
        lines.append("| Fold | Rows | Ratio | Win Rate | Date Range |")
        lines.append("|------|------|-------|----------|------------|")
        total = s["total"]
        for fold in ["train", "valid", "test"]:
            f = s[fold]
            ratio = f["rows"] / total * 100
            lines.append(
                f"| {fold} | {f['rows']:,} | {ratio:.1f}% | "
                f"{f['win_rate']:.2f}% | {f['date_min']} ~ {f['date_max']} |"
            )
        lines.append(f"| **Total** | **{total:,}** | 100% | - | - |")
        lines.append("")

    lines.append("## Directory Structure")
    lines.append("")
    lines.append("```")
    lines.append("data/전처리_데이터셋/")
    for v in VERSIONS:
        lines.append(f"├── {v}/")
        lines.append("│   ├── train.csv")
        lines.append("│   ├── valid.csv")
        lines.append("│   └── test.csv")
    lines.append("└── 분할안내.md")
    lines.append("```")
    lines.append("")

    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("import pandas as pd")
    lines.append("")
    lines.append("# Choose your version")
    lines.append('VERSION = "v2_standard"  # or v1_base, v3_minmax, v4_robust')
    lines.append("")
    lines.append('train = pd.read_csv(f"data/전처리_데이터셋/{VERSION}/train.csv", encoding="utf-8-sig")')
    lines.append('valid = pd.read_csv(f"data/전처리_데이터셋/{VERSION}/valid.csv", encoding="utf-8-sig")')
    lines.append('test  = pd.read_csv(f"data/전처리_데이터셋/{VERSION}/test.csv",  encoding="utf-8-sig")')
    lines.append("")
    lines.append("# Separate X and y")
    lines.append('y_train = train["win"]')
    lines.append("# Drop ID/market/outcome columns for X (see src/1_전처리/config.py)")
    lines.append("```")
    lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Each split file still contains the `fold` column (redundant but harmless)")
    lines.append("- Encoding is `utf-8-sig` so Excel opens Korean text correctly")
    lines.append("- **valid** is for hyperparameter/threshold tuning")
    lines.append("- **test** is for final evaluation only — do not tune on it")
    lines.append("")

    (SPLITS_DIR / "분할안내.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"  Saved: {SPLITS_DIR / '분할안내.md'}")


def main():
    parser = argparse.ArgumentParser(description="Split versions into train/valid/test")
    parser.add_argument("--version", default=None, help="Specific version only (e.g. v2_standard)")
    args = parser.parse_args()

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Train/Valid/Test Split")
    logger.info("=" * 60)

    targets = [args.version] if args.version else VERSIONS
    all_stats = []

    for version in targets:
        logger.info(f"\n  [{version}]")
        stats = split_version(version)
        if stats:
            all_stats.append(stats)

    if all_stats:
        write_split_info(all_stats)

    logger.info("\n" + "=" * 60)
    logger.info(f"Split done! {len(all_stats)} version(s) processed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
