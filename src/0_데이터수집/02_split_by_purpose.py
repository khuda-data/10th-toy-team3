"""
race_entries.csv 전처리 스크립트

팀원이 생성한 race_entries.csv(156컬럼, 56,648행)를 정리하여
모델 학습용·배당률 분석용·사후 결과용 세 파일로 분리한다.

사용법:
    python src/02_split_by_purpose.py --input race_entries.csv
    python src/02_split_by_purpose.py  # 기본: 프로젝트 루트의 race_entries.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "pipeline"))
from config import assign_time_split, SPLIT_RATIOS

# ============================================================
# 로깅 설정
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# 컬럼 정의
# ============================================================

# 공통 키 컬럼 (세 파일 모두 포함, JOIN 키)
KEY_COLS = [
    "entry_id", "race_id", "rcDate", "rcNo",
    "meet", "hrNo", "hrName", "chulNo",
]

# 배당률 파생 컬럼 14개 → market_odds.csv
ODDS_COLS = [
    "winOdds", "plcOdds", "p_raw", "q", "logit_q", "log_q",
    "book_sum", "takeout", "pop_rank", "pop_pct", "is_fav",
    "pl_harville", "pl_disc", "q_plc",
]

# 사후 결과 컬럼 8개 → race_outcome.csv
OUTCOME_COLS = [
    "fin_rank", "fin_pct", "place", "ord",
    "resid", "upset", "upset_A", "upset_B",
]

# model_features에서 제외할 컬럼 집합 (배당률 + 사후결과)
# 단, 'win'은 타겟이므로 제거하지 않음
EXCLUDE_FROM_FEATURES = set(ODDS_COLS + OUTCOME_COLS)


# ============================================================
# STEP 1: 서울 데이터 필터링
# ============================================================
def filter_seoul(df: pd.DataFrame) -> pd.DataFrame:
    """meet == '서울' 행만 남긴다."""
    before = len(df)
    df = df[df["meet"] == "서울"].reset_index(drop=True)
    after = len(df)
    logger.info(f"[STEP 1] 서울 필터링: {before:,}행 → {after:,}행 (제거: {before - after:,}행)")
    return df


# ============================================================
# STEP 2: rating 결측치 처리
# ============================================================
def handle_rating_missing(df: pd.DataFrame) -> pd.DataFrame:
    """rating NaN을 0으로 채우고, rating_na 플래그를 검증/생성한다."""
    logger.info("[STEP 2] rating 결측치 처리")

    # rating_na 플래그 처리
    if "rating_na" in df.columns:
        # 정합성 검증: rating NaN인 곳에서만 rating_na == 1이어야 함
        expected = df["rating"].isna().astype(int)
        actual = df["rating_na"].astype(int)
        mismatch = (expected != actual).sum()

        if mismatch > 0:
            logger.warning(
                f"  rating_na 불일치 {mismatch:,}건 발견 → 실제 결측 기준으로 재생성"
            )
            df["rating_na"] = expected
        else:
            logger.info("  rating_na 플래그 검증 통과 (기존 컬럼 정확함)")
    else:
        # 새로 생성
        df["rating_na"] = df["rating"].isna().astype(int)
        logger.info("  rating_na 플래그 새로 생성")

    # rating NaN → 0
    nan_count = df["rating"].isna().sum()
    df["rating"] = df["rating"].fillna(0)
    logger.info(f"  rating 결측 {nan_count:,}건 → 0으로 채움")

    return df


# ============================================================
# STEP 2-b: fold 재배정 (시간순 6:2:2)
# ============================================================
def reassign_fold(df: pd.DataFrame) -> pd.DataFrame:
    """rcDate 기준 시간순 6:2:2로 fold를 재배정한다.

    원본 race_entries.csv의 fold는 약 33:33:33이라 학습 데이터가 너무 적으므로,
    학습 60% / 검증 20% / 평가 20%로 다시 나눈다.
    경계는 경주일 단위로 맞춰 한 경주가 두 fold로 쪼개지지 않게 한다.
    """
    logger.info("[STEP 2-b] fold 재배정 (시간순 6:2:2)")

    if "fold" in df.columns:
        old = df["fold"].value_counts().to_dict()
        logger.info(f"  기존 fold: {old}")

    df = df.sort_values("rcDate").reset_index(drop=True)
    df["fold"] = assign_time_split(df, date_col="rcDate", ratios=SPLIT_RATIOS)

    for f in ["train", "valid", "test"]:
        sub = df[df["fold"] == f]
        logger.info(
            f"  {f:5s}: {len(sub):>6,}행 ({len(sub)/len(df):5.1%}) | "
            f"{sub['rcDate'].min()}~{sub['rcDate'].max()} | "
            f"win {sub['win'].mean()*100:.2f}%"
        )

    # 경주가 fold를 넘나들지 않는지 확인
    spanning = (df.groupby("race_id")["fold"].nunique() > 1).sum()
    if spanning > 0:
        logger.error(f"  경주 {spanning}건이 여러 fold에 걸쳐 있습니다!")
        sys.exit(1)
    logger.info("  경주 단위 무결성 확인 완료 (fold 경계 위반 0건)")

    return df


# ============================================================
# STEP 3: 컬럼 분리
# ============================================================
def split_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """DataFrame을 3개 그룹으로 분리한다."""
    logger.info("[STEP 3] 컬럼 3그룹 분리")

    # model_features: 전체 컬럼 중 배당률·사후결과 제외
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]
    df_features = df[feature_cols].copy()

    # market_odds: 공통키 + 배당률 파생 14개
    odds_cols_present = [c for c in ODDS_COLS if c in df.columns]
    df_odds = df[KEY_COLS + odds_cols_present].copy()

    # race_outcome: 공통키 + 사후결과 8개
    outcome_cols_present = [c for c in OUTCOME_COLS if c in df.columns]
    df_outcome = df[KEY_COLS + outcome_cols_present].copy()

    # 로그 출력
    logger.info(f"  model_features: {df_features.shape} — 컬럼 수 {len(df_features.columns)}개")
    logger.info(f"    컬럼: {list(df_features.columns)}")
    logger.info(f"  market_odds:    {df_odds.shape} — 컬럼 수 {len(df_odds.columns)}개")
    logger.info(f"    컬럼: {list(df_odds.columns)}")
    logger.info(f"  race_outcome:   {df_outcome.shape} — 컬럼 수 {len(df_outcome.columns)}개")
    logger.info(f"    컬럼: {list(df_outcome.columns)}")

    return df_features, df_odds, df_outcome


# ============================================================
# STEP 4: 검증
# ============================================================
def validate(
    df_features: pd.DataFrame,
    df_odds: pd.DataFrame,
    df_outcome: pd.DataFrame,
) -> None:
    """7개 검증 조건을 확인한다. 하나라도 실패하면 에러 후 중단."""
    logger.info("[STEP 4] 검증")

    checks = [
        (
            set(ODDS_COLS).isdisjoint(set(df_features.columns)),
            "model_features에 배당률 파생 컬럼이 포함되어 있습니다",
        ),
        (
            set(OUTCOME_COLS).isdisjoint(set(df_features.columns)),
            "model_features에 사후 결과 컬럼이 포함되어 있습니다",
        ),
        (
            all(c in df_odds.columns for c in ODDS_COLS),
            f"market_odds에 배당률 파생 컬럼 누락: "
            f"{[c for c in ODDS_COLS if c not in df_odds.columns]}",
        ),
        (
            all(c in df_outcome.columns for c in OUTCOME_COLS),
            f"race_outcome에 사후 결과 컬럼 누락: "
            f"{[c for c in OUTCOME_COLS if c not in df_outcome.columns]}",
        ),
        (
            sorted(df_features["meet"].unique().tolist()) == ["서울"],
            f"model_features의 meet 고유값이 ['서울']이 아닙니다: "
            f"{df_features['meet'].unique().tolist()}",
        ),
        (
            df_features["rating"].isna().sum() == 0,
            f"model_features의 rating에 NaN {df_features['rating'].isna().sum()}건 남아있습니다",
        ),
        (
            len(df_features) == len(df_odds) == len(df_outcome),
            f"세 파일 행 수 불일치: features={len(df_features)}, "
            f"odds={len(df_odds)}, outcome={len(df_outcome)}",
        ),
        (
            sorted(df_features["fold"].unique().tolist()) == ["test", "train", "valid"],
            f"fold 값이 train/valid/test가 아닙니다: "
            f"{df_features['fold'].unique().tolist()}",
        ),
        (
            abs((df_features["fold"] == "train").mean() - 0.6) < 0.03,
            f"train 비율이 60%에서 3%p 이상 벗어남: "
            f"{(df_features['fold'] == 'train').mean():.1%}",
        ),
    ]

    total = len(checks)
    all_passed = True
    for i, (condition, error_msg) in enumerate(checks, 1):
        if condition:
            logger.info(f"  [PASS] 검증 {i}/{total}")
        else:
            logger.error(f"  [FAIL] 검증 {i}/{total}: {error_msg}")
            all_passed = False

    if not all_passed:
        logger.error("검증 실패! 출력 파일을 확인하세요.")
        sys.exit(1)

    logger.info("  모든 검증 통과!")


# ============================================================
# 저장
# ============================================================
def save_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    """DataFrame을 CSV로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logger.info(f"  {label} 저장 완료: {path} ({len(df):,}행)")


# ============================================================
# CLI & Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="race_entries.csv 전처리 — 모델 학습용 데이터셋 분리",
    )
    parser.add_argument(
        "--input",
        default="race_entries.csv",
        help="입력 CSV 경로 (기본: race_entries.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="출력 디렉토리 (기본: data/processed/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("전처리 시작")
    logger.info("=" * 60)

    # 입력 파일 확인
    if not input_path.exists():
        logger.error(f"입력 파일을 찾을 수 없습니다: {input_path}")
        sys.exit(1)

    # 데이터 로드
    df = pd.read_csv(input_path)
    logger.info(f"입력: {input_path} ({len(df):,}행 × {len(df.columns)}컬럼)")

    # 키 컬럼 존재 확인
    missing_keys = [c for c in KEY_COLS if c not in df.columns]
    if missing_keys:
        logger.error(f"공통 키 컬럼 누락: {missing_keys}")
        sys.exit(1)

    # STEP 1: 서울 필터링
    df = filter_seoul(df)

    # STEP 2: rating 결측치 처리
    df = handle_rating_missing(df)

    # STEP 2-b: fold 재배정 (6:2:2)
    df = reassign_fold(df)

    # STEP 3: 컬럼 분리
    df_features, df_odds, df_outcome = split_columns(df)

    # STEP 4: 검증
    validate(df_features, df_odds, df_outcome)

    # 저장
    logger.info("--- 파일 저장 ---")
    save_csv(df_features, output_dir / "model_features.csv", "model_features")
    save_csv(df_odds, output_dir / "market_odds.csv", "market_odds")
    save_csv(df_outcome, output_dir / "race_outcome.csv", "race_outcome")

    logger.info("=" * 60)
    logger.info("전처리 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
