"""
파이프라인 공통 설정 — 컬럼 분류, 유틸리티 함수

모든 파이프라인 스크립트가 이 모듈을 import하여 일관된 설정을 공유한다.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# 경로 설정
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

# ============================================================
# 식별자 — 피처로 절대 사용 금지 (17개)
# ============================================================
ID_COLS = [
    "entry_id", "race_id", "rcDate", "rcDay", "rcNo",
    "meet", "meet_cd", "hrNo", "hrName",
    "jkName", "jkNo", "owName", "owNo",
    "trName", "trNo", "chulNo", "fold",
]

# ============================================================
# 시장 컬럼 — 배당률 및 발매금액 파생 (16개)
# ============================================================
MARKET_COLS = [
    "winOdds", "plcOdds", "p_raw", "q", "logit_q", "log_q",
    "book_sum", "takeout", "pop_rank", "pop_pct", "is_fav",
    "winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse",
]

# ============================================================
# 두시장 컬럼 — 단승 vs 연승 비교 (5개)
# ============================================================
DUAL_MARKET_COLS = [
    "pl_harville", "pl_disc", "q_plc", "gap_h", "gap_d",
]

# ============================================================
# 결과/라벨 — 사후 정보 (8개, win은 별도 타겟)
# ============================================================
OUTCOME_COLS = [
    "ord", "fin_rank", "fin_pct", "place",
    "resid", "upset_A", "upset_B", "upset",
]

# 타겟 컬럼
TARGET_COL = "win"

# ============================================================
# 제외 컬럼 합집합
# ============================================================
EXCLUDE_COLS = set(ID_COLS + MARKET_COLS + DUAL_MARKET_COLS + OUTCOME_COLS + [TARGET_COL])

# ============================================================
# 범주형 컬럼 (dtype=object, LabelEncoder 대상)
# ============================================================
CATEGORICAL_COLS = [
    "ageCond", "budam", "prizeCond", "rank", "rcName",
    "track", "weather", "name", "sex", "wgBudamBigo",
    "born", "tool_set", "rating_na",
]

# ============================================================
# 조절변수 매핑 (8단계 FLB 분석용)
# ============================================================
MODERATORS = {
    "경주등급": "rank",
    "거리": "rcDist",
    "트랙상태": "track",
    "출주두수": "n_run",
    "매출규모": "totalAmt",
}

# 배당 구간 정의
ODDS_BINS = [0, 2, 3, 5, 10, float("inf")]
ODDS_LABELS = ["<2x", "2-3x", "3-5x", "5-10x", "10x+"]

# 고정 시드
RANDOM_STATE = 42


# ============================================================
# 유틸리티 함수
# ============================================================

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """DataFrame 컬럼에서 EXCLUDE_COLS를 제거하고 피처 컬럼 목록을 반환한다."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


# ============================================================
# Train/Valid/Test 시간순 분할 (6:2:2 고정)
# ============================================================
SPLIT_RATIOS = (0.6, 0.2, 0.2)  # train : valid : test


def assign_time_split(
    df: pd.DataFrame,
    date_col: str = "rcDate",
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
) -> pd.Series:
    """rcDate 기준 시간순으로 train/valid/test를 6:2:2로 배정한다.

    같은 경주일(rcDate)의 행은 절대 서로 다른 fold로 갈라지지 않는다.
    (경주 단위 정보 누수 방지)

    Args:
        df: 대상 DataFrame
        date_col: 날짜 컬럼명 (YYYYMMDD 정수/문자)
        ratios: (train, valid, test) 비율

    Returns:
        'train'/'valid'/'test' 값을 가진 Series (df와 동일 인덱스)
    """
    train_r, valid_r, _ = ratios

    # 날짜별 행 수 → 누적 비율
    date_counts = df[date_col].value_counts().sort_index()
    cum_ratio = date_counts.cumsum() / len(df)

    # 누적 비율이 경계를 넘는 첫 날짜를 cutoff로 사용
    train_dates = cum_ratio[cum_ratio <= train_r].index
    valid_dates = cum_ratio[(cum_ratio > train_r) & (cum_ratio <= train_r + valid_r)].index

    train_set = set(train_dates)
    valid_set = set(valid_dates)

    def _assign(d):
        if d in train_set:
            return "train"
        if d in valid_set:
            return "valid"
        return "test"

    return df[date_col].map(_assign)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """파이프라인 공통 로깅 설정."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def setup_plot_style():
    """matplotlib 스타일 설정. 그래프 레이블은 영어로 작성.

    OS별 폰트 차이로 인한 경고를 피하기 위해 matplotlib 내장 폰트
    (DejaVu Sans)를 사용한다. Windows/macOS/Linux 모두 동일하게 동작.
    """
    import matplotlib
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.figsize"] = (10, 6)
    matplotlib.rcParams["figure.dpi"] = 150

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass


# 모듈 import 시 바로 설정 적용
setup_plot_style()


# ============================================================
# 피처명 → 한국어 설명 매핑 (보고서용)
# ============================================================
FEATURE_NAME_MAP = {
    "hr_last_finpct": "직전 경주 착순 백분위",
    "hr_last_poppct": "직전 경주 인기 백분위",
    "age__z": "연령 (경주 내 z점수)",
    "age__pr": "연령 (경주 내 백분위)",
    "hr_last_ord": "직전 경주 착순",
    "jk_plcrate": "기수 통산 입상률",
    "jk_winrate__z": "기수 승률 (경주 내 z점수)",
    "hr_plcrate": "마필 통산 입상률",
    "jk_winrate__pr": "기수 승률 (경주 내 백분위)",
    "train_runs_14__z": "14일 훈련 횟수 (경주 내 z점수)",
    "hr_winrate": "마필 통산 승률",
    "hr_winrate__z": "마필 승률 (경주 내 z점수)",
    "hr_resid": "마필 평균 잔차 (기대 대비 실적)",
    "hr_resid__z": "마필 잔차 (경주 내 z점수)",
    "tr_winrate": "조교사 통산 승률",
    "tr_winrate__z": "조교사 승률 (경주 내 z점수)",
    "jk_resid": "기수 평균 잔차",
    "hr_rest_days": "직전 경주 후 휴양일수",
    "hr_rest_days__z": "휴양일수 (경주 내 z점수)",
    "train_runs_14": "14일 훈련 주행 횟수",
    "train_sec_14": "14일 훈련 시간 합계",
    "train_days_14": "14일 훈련 일수",
    "wg": "마체중 (kg)",
    "wg__z": "마체중 (경주 내 z점수)",
    "wg_diff": "직전 대비 체중 증감",
    "wg_diff__z": "체중 증감 (경주 내 z점수)",
    "wgBudam": "부담중량 (kg)",
    "wgBudam__z": "부담중량 (경주 내 z점수)",
    "rating": "마필 레이팅",
    "rating__z": "레이팅 (경주 내 z점수)",
    "hr_starts": "마필 통산 출주 횟수",
    "hr_dist_winrate": "같은 거리 승률",
    "hr_dist_starts": "같은 거리 출주 횟수",
    "jk_starts": "기수 통산 기승 수",
    "tr_starts": "조교사 통산 출전 수",
    "ow_winrate": "마주 통산 승률",
    "rcDist": "경주 거리 (m)",
    "n_run": "출주두수",
    "chaksun1": "1착 상금 (원)",
    "hr_style": "각질 (0=선행 1=추입)",
    "is_debut": "첫 출주 여부",
    "clinic_30d": "30일 내 진료 건수",
    "start_delay": "발주 지연 이력",
    "model_prob": "모델 예측확률",
    "market_prob": "시장(배당률) 확률",
    "q": "단승 시장 확률",
    "pop_pct": "인기 백분위",
    "is_fav": "인기 1위 여부",
    "gap": "괴리 (모델-시장)",
    "gap_abs": "괴리 절대값",
    "prob_ratio": "확률 비율 (모델/시장)",
}


def translate_feature_name(name: str) -> str:
    """피처 영문명을 한국어 설명으로 변환."""
    return FEATURE_NAME_MAP.get(name, name)


def ensure_dirs():
    """결과 저장 디렉토리를 생성한다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
