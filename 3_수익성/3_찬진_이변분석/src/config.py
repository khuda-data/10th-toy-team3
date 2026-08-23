from __future__ import annotations

from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = PROJECT_ROOT / "전처리 데이터셋" / "v1_base"
FINAL_CSV = PROJECT_ROOT / "final.csv.gz"

OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
PREDICTION_DIR = OUTPUT_ROOT / "predictions"
MODEL_DIR = OUTPUT_ROOT / "models"
CONFIG_DIR = ANALYSIS_ROOT / "configs"
REPORT_DIR = ANALYSIS_ROOT / "reports"

FOLDS = ("train", "valid", "test")
EXPECTED_ROWS = {"train": 19_637, "valid": 6_591, "test": 6_660}
EXPECTED_SUBSETS = {
    "darkhorse": {"train": 10_374, "valid": 3_490, "test": 3_528},
    "favorite_bust": {"train": 5_597, "valid": 1_881, "test": 1_902},
}

MERGE_COLUMNS = [
    "entry_id",
    "pop_pct",
    "fin_pct",
    "place",
    "ord",
    "upset",
    "upset_A",
    "upset_B",
    "winOdds",
    "plcOdds",
]

CURRENT_MARKET_COLUMNS = {
    "winOdds", "plcOdds", "winAmt", "plcAmt", "totalAmt", "log_winAmt",
    "liq_per_horse", "q", "p_raw", "log_q", "logit_q", "q_plc",
    "pop_rank", "pop_pct", "is_fav", "book_sum", "takeout",
    "pl_harville", "pl_disc", "gap_h", "gap_d",
}

OUTCOME_COLUMNS = {
    "ord", "fin_rank", "fin_pct", "place", "win", "resid",
    "upset", "upset_A", "upset_B", "fold",
}

IDENTIFIER_COLUMNS = {
    "race_id", "entry_id", "hrNo", "jkNo", "trNo", "owNo",
    "hrName", "jkName", "trName", "owName", "name",
}

TIME_CONTEXT_COLUMNS = {"rcDate", "rcNo", "rcDay", "meet_cd"}

HISTORICAL_MARKET_COLUMNS = {
    "hr_last_poppct", "hr_resid", "hr_last_resid", "jk_resid", "tr_resid",
    "ow_resid", "hr_resid__z",
}

LEAKAGE_COLUMNS = (
    CURRENT_MARKET_COLUMNS | OUTCOME_COLUMNS | IDENTIFIER_COLUMNS | TIME_CONTEXT_COLUMNS
)

PERCENTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00)
BOOTSTRAP_REPS = 5_000
RANDOM_SEED = 42
STABILITY_SEEDS = (11, 22, 33, 44, 55)

TARGETS = {
    "darkhorse": {
        "subset_column": "pop_pct",
        "subset_op": "ge",
        "subset_value": 0.50,
        "target_column": "place",
        "stored_label": "upset_B",
    },
    "favorite_bust": {
        "subset_column": "pop_pct",
        "subset_op": "le",
        "subset_value": 0.25,
        "target_column": "fin_pct",
        "target_op": "ge",
        "target_value": 0.50,
        "stored_label": "upset_A",
    },
}


def ensure_output_dirs() -> None:
    for path in (
        OUTPUT_ROOT, TABLE_DIR, FIGURE_DIR, PREDICTION_DIR, MODEL_DIR,
        CONFIG_DIR, REPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
