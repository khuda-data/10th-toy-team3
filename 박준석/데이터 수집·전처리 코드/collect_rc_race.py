"""
한국마사회 RC경마경주정보 데이터 수집 스크립트

서울경마장 경주 데이터(배당률, 마체중, 트랙상태, 날씨, 착순 등)를
공공데이터포털 API를 통해 수집하고 CSV로 저장한다.

사용법:
    python src/collect_rc_race.py --from 20240101 --to 20240331
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from kra_client import KRAApiClient, KRAApiError

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 기본 설정
DEFAULT_OUTPUT_PATH = "data/raw/rc_race_info.csv"
ENDPOINT = "/SeoulRace_1"
MAX_DAYS_PER_CHUNK = 31  # 월 단위 분할 기준


def split_date_range(start_str: str, end_str: str) -> list[tuple[str, str]]:
    """기간을 월 단위로 분할한다.

    31일을 초과하는 기간은 월 단위로 나눠서 반환한다.

    Args:
        start_str: 시작일 (YYYYMMDD)
        end_str: 종료일 (YYYYMMDD)

    Returns:
        [(시작일, 종료일), ...] 형태의 분할된 기간 리스트
    """
    start = date(int(start_str[:4]), int(start_str[4:6]), int(start_str[6:8]))
    end = date(int(end_str[:4]), int(end_str[4:6]), int(end_str[6:8]))

    if start > end:
        raise ValueError(f"시작일({start_str})이 종료일({end_str})보다 큽니다.")

    # 31일 이하면 분할 불필요
    if (end - start).days <= MAX_DAYS_PER_CHUNK:
        return [(start_str, end_str)]

    chunks: list[tuple[str, str]] = []
    current = start

    while current <= end:
        # 현재 월의 마지막 날 계산
        if current.month == 12:
            month_end = date(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = date(current.year, current.month + 1, 1) - timedelta(days=1)

        # 전체 범위의 종료일과 월 말일 중 작은 값
        chunk_end = min(month_end, end)

        chunks.append((current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))

        # 다음 월 1일로 이동
        current = chunk_end + timedelta(days=1)

    return chunks


def collect_rc_race(
    date_from: str,
    date_to: str,
    num_of_rows: int = 100,
) -> pd.DataFrame:
    """RC경마경주정보를 수집하여 DataFrame으로 반환한다.

    Args:
        date_from: 수집 시작일 (YYYYMMDD)
        date_to: 수집 종료일 (YYYYMMDD)
        num_of_rows: 페이지당 요청 건수

    Returns:
        수집된 전체 데이터의 DataFrame
    """
    # .env에서 서비스키 로드
    load_dotenv()
    service_key = os.getenv("KRA_SERVICE_KEY")

    if not service_key:
        logger.error(
            "KRA_SERVICE_KEY가 설정되지 않았습니다. "
            ".env 파일에 KRA_SERVICE_KEY=<서비스키>를 추가하세요."
        )
        sys.exit(1)

    client = KRAApiClient(
        service_key=service_key,
        num_of_rows=num_of_rows,
    )

    # 기간 분할
    chunks = split_date_range(date_from, date_to)
    logger.info(
        f"수집 기간: {date_from} ~ {date_to} "
        f"({len(chunks)}개 구간으로 분할)"
    )

    all_items: list[dict] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, 1):
        logger.info(f"--- 구간 {i}/{len(chunks)}: {chunk_start} ~ {chunk_end} ---")

        params = {
            "rc_date_fr": chunk_start,
            "rc_date_to": chunk_end,
        }

        try:
            items = client.fetch_all(endpoint=ENDPOINT, params=params)
            all_items.extend(items)
            logger.info(
                f"구간 {i} 완료: {len(items)}건 수집 | 누적 총 {len(all_items)}건"
            )
        except KRAApiError as e:
            logger.error(f"구간 {i} 수집 실패: {e}")
            raise

    if not all_items:
        logger.warning("수집된 데이터가 없습니다.")
        return pd.DataFrame()

    df = pd.DataFrame(all_items)
    logger.info(f"전체 수집 완료: {len(df)}건, {len(df.columns)}개 컬럼")
    return df


def save_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """DataFrame을 CSV로 저장한다.

    Args:
        df: 저장할 DataFrame
        output_path: 저장 파일 경로
    """
    if df.empty:
        logger.warning("저장할 데이터가 없습니다.")
        return

    # 디렉토리 자동 생성
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"CSV 저장 완료: {output_path} ({len(df)}건)")


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="한국마사회 RC경마경주정보 데이터 수집 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python src/collect_rc_race.py --from 20240101 --to 20240131
  python src/collect_rc_race.py --from 20240101 --to 20240630 --output data/raw/2024_first_half.csv
  python src/collect_rc_race.py --from 20240301 --to 20240301 --rows 50
        """,
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        required=True,
        help="수집 시작일 (YYYYMMDD 형식, 예: 20240101)",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        required=True,
        help="수집 종료일 (YYYYMMDD 형식, 예: 20240131)",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"저장 파일 경로 (기본: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="페이지당 요청 건수 (기본: 100)",
    )
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()

    # 날짜 형식 검증
    for label, val in [("시작일", args.date_from), ("종료일", args.date_to)]:
        if len(val) != 8 or not val.isdigit():
            logger.error(f"{label} 형식이 올바르지 않습니다: {val} (YYYYMMDD 필요)")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("한국마사회 RC경마경주정보 데이터 수집 시작")
    logger.info("=" * 60)

    try:
        df = collect_rc_race(
            date_from=args.date_from,
            date_to=args.date_to,
            num_of_rows=args.rows,
        )
        save_to_csv(df, args.output)
    except KRAApiError as e:
        logger.error(f"수집 중단: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("사용자에 의해 중단되었습니다.")
        sys.exit(130)

    logger.info("=" * 60)
    logger.info("수집 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
