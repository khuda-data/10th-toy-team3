"""Stage 27: compare constrained candidates and freeze one future challenger."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file
from src.evaluation.race_metrics import race_metrics
from src.evaluation.ranking_metrics import rank_entries, ranking_metrics
from src.models.common import utc_now


PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
MANIFEST_DIR = PROJECT_ROOT / "data" / "manifests"
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"

M2_PATH = PREDICTION_DIR / "m2_xgboost_calibration.csv.gz"
R3_PATH = PREDICTION_DIR / "m2_xgboost_calibration_final.csv.gz"
R2_PATH = PREDICTION_DIR / "r2_xgb_ranker_calibration_probability.csv.gz"
R4_PATH = PREDICTION_DIR / "r4_gated_calibration.csv.gz"
POLICY_PATH = MANIFEST_DIR / "top1_challenger_freeze.json"
REPORT_PATH = REPORT_DIR / "stage_27_candidate_freeze.json"
SUMMARY_PATH = REPORT_DIR / "stage_27_summary.md"
TABLE_PATH = ANALYSIS_DIR / "stage_27_candidate_comparison.csv"

EXPECTED_ROWS = 6582
EXPECTED_RACES = 641
MAX_CALIBRATION_DATE = 20251227
TOLERANCE = 1e-12

CANDIDATE_COMPLEXITY = {
    "R0_market": 0,
    "R1_existing_m2_standalone": 1,
    "R2_ranker_probability": 2,
    "R3_fixed_m2_market_blend": 2,
    "R4_gated_adaptive_blend": 4,
}


def _validate_base(frame: pd.DataFrame) -> None:
    required = {"race_id", "entry_id", "rcDate", "win", "q_market"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")
    if len(frame) != EXPECTED_ROWS or frame["race_id"].nunique() != EXPECTED_RACES:
        raise ValueError("Stage 27 must use the fixed 6,582-row, 641-race Calibration fold")
    if frame["entry_id"].duplicated().any():
        raise ValueError("Stage 27 requires unique entry_id values")
    if not frame.groupby("race_id", sort=False)["win"].sum().eq(1).all():
        raise ValueError("Every comparison race must have exactly one winner")
    if int(frame["rcDate"].max()) > MAX_CALIBRATION_DATE:
        raise ValueError("Stage 27 cannot load the opened Final Test period")


def _aligned_probabilities(
    base: pd.DataFrame,
    path: Path,
    probability_column: str,
) -> np.ndarray:
    other = pd.read_csv(path)
    required = {"entry_id", "race_id", "rcDate", "win", "q_market", probability_column}
    missing = sorted(required - set(other.columns))
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    if other["entry_id"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate entry_id values")
    aligned = base[["entry_id", "race_id", "rcDate", "win", "q_market"]].merge(
        other[["entry_id", "race_id", "rcDate", "win", "q_market", probability_column]],
        on="entry_id",
        how="left",
        suffixes=("_base", "_candidate"),
        validate="one_to_one",
    )
    if aligned[probability_column].isna().any():
        raise ValueError(f"{path.name} does not cover every Calibration entry")
    for column in ("race_id", "rcDate", "win"):
        if not aligned[f"{column}_base"].equals(aligned[f"{column}_candidate"]):
            raise ValueError(f"{path.name} disagrees on {column}")
    if not np.allclose(
        aligned["q_market_base"].to_numpy(float),
        aligned["q_market_candidate"].to_numpy(float),
        atol=1e-12,
    ):
        raise ValueError(f"{path.name} disagrees on q_market")
    return aligned[probability_column].to_numpy(float)


def evaluate_candidate(
    frame: pd.DataFrame,
    probabilities,
    *,
    name: str,
    market_metrics: dict[str, float | int] | None = None,
) -> dict[str, object]:
    """Evaluate one candidate with the frozen unique Top-1 tie-break contract."""
    values = np.asarray(probabilities, dtype=float)
    if len(values) != len(frame):
        raise ValueError("Candidate probability length does not match frame")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Candidate probabilities must be finite and non-negative")
    sums = pd.Series(values).groupby(frame["race_id"].to_numpy(), sort=False).sum()
    if not np.allclose(sums.to_numpy(), 1.0, atol=1e-9):
        raise ValueError("Candidate probabilities must sum to one within every race")

    probability_metrics = race_metrics(frame, values)
    strict_ranking = ranking_metrics(frame, values)
    probability_metrics.update(
        {
            "top1_correct": strict_ranking["top1_correct"],
            "top1_accuracy": strict_ranking["top1_accuracy"],
            "mean_reciprocal_rank": strict_ranking["mean_reciprocal_rank"],
            "winner_mean_rank": strict_ranking["winner_mean_rank"],
        }
    )
    market_top = rank_entries(frame, frame["q_market"].to_numpy(), score_name="score")
    candidate_top = rank_entries(frame, values, score_name="score")
    market_ids = market_top.loc[market_top["rank_position"].eq(1), "entry_id"].to_numpy()
    candidate_ids = candidate_top.loc[
        candidate_top["rank_position"].eq(1), "entry_id"
    ].to_numpy()
    result: dict[str, object] = {
        "candidate": name,
        **probability_metrics,
        "actual_top1_override_races": int(np.sum(market_ids != candidate_ids)),
        "complexity_rank": CANDIDATE_COMPLEXITY[name],
    }
    if market_metrics is None:
        result.update(
            {
                "delta_logloss": 0.0,
                "delta_brier": 0.0,
                "delta_top1": 0.0,
                "delta_top1_correct": 0,
                "eligible_probability_guardrail": True,
            }
        )
    else:
        delta_logloss = float(
            market_metrics["race_log_loss"] - result["race_log_loss"]
        )
        delta_brier = float(market_metrics["race_brier"] - result["race_brier"])
        result.update(
            {
                "delta_logloss": delta_logloss,
                "delta_brier": delta_brier,
                "delta_top1": float(
                    result["top1_accuracy"] - market_metrics["top1_accuracy"]
                ),
                "delta_top1_correct": int(
                    result["top1_correct"] - market_metrics["top1_correct"]
                ),
                "eligible_probability_guardrail": bool(
                    delta_logloss >= -TOLERANCE and delta_brier >= -TOLERANCE
                ),
            }
        )
    return result


def select_constrained_candidate(candidates: list[dict[str, object]]) -> dict[str, object]:
    """Apply the predeclared loss guardrails before maximizing unique Top-1."""
    eligible = [row for row in candidates if row["eligible_probability_guardrail"]]
    if not eligible:
        raise ValueError("No candidate passed the probability guardrails")
    return min(
        eligible,
        key=lambda row: (
            -int(row["top1_correct"]),
            int(row["actual_top1_override_races"]),
            int(row["complexity_rank"]),
            str(row["candidate"]),
        ),
    )


def _frozen_components() -> list[dict[str, str]]:
    paths = [
        "artifacts/models/r2_xgb_ranker.joblib",
        "artifacts/models/r4_market_gate_logistic.joblib",
        "data/manifests/ranker_temperature_policy.json",
        "data/manifests/market_gate_policy.json",
        "data/manifests/top1_research_policy.json",
        "src/evaluation/race_metrics.py",
        "src/evaluation/ranking_metrics.py",
        "src/models/market_gate.py",
        "src/models/freeze_top1_candidate.py",
        "FUTURE_HOLDOUT_VALIDATION.md",
    ]
    return [{"path": path, "sha256": sha256_file(PROJECT_ROOT / path)} for path in paths]


def _summary(report: dict[str, object]) -> str:
    selected = report["selected_challenger"]
    rows = [
        "# 개발 27단계 결과 — Top-1 후보 비교 및 사전 동결",
        "",
        f"생성 시점: {report['created_at']}",
        "",
        "## 데이터 정책",
        "",
        "- 후보 선택: Calibration 6,582행·641경주만 사용",
        "- 기존 Final Test: 로드·재평가·선택에 사용하지 않음",
        "- 모든 후보: 동일 entry_id, 고유 Top-1 동률 처리, 경주별 확률합 1 검증",
        "",
        "## 동일 기준 비교",
        "",
        "| 후보 | Log Loss | Brier | Top-1 | 시장 대비 적중 | 실제 교체 | 제약 통과 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in report["candidates"]:
        rows.append(
            f"| {candidate['candidate']} | {candidate['race_log_loss']:.6f} | "
            f"{candidate['race_brier']:.6f} | {candidate['top1_correct']}/{candidate['races']} "
            f"({candidate['top1_accuracy']:.2%}) | {candidate['delta_top1_correct']:+d} | "
            f"{candidate['actual_top1_override_races']} | "
            f"{'통과' if candidate['eligible_probability_guardrail'] else '탈락'} |"
        )
    rows.extend(
        [
            "",
            "## 동결 결과",
            "",
            f"- challenger: `{report['challenger_id']}`",
            f"- 선택 후보: `{selected['candidate']}`",
            f"- gate threshold: `{report['frozen_parameters']['gate_threshold']}`",
            f"- lambda_switch: `{report['frozen_parameters']['lambda_switch']}`",
            f"- rank temperature: `{report['frozen_parameters']['rank_temperature']}`",
            f"- 시장 대비: Log Loss `{selected['delta_logloss']:+.6f}`, Brier `{selected['delta_brier']:+.6f}`, Top-1 `{selected['delta_top1_correct']:+d}`경주",
            "- 상태: `frozen_pending_future_holdout`",
            "",
            "R4는 확률 비열화 제약을 통과한 후보 중 Top-1 적중이 가장 높아 선택됐다. 이 결과는 Calibration 선택 결과이며 공식 시장 우위가 아니다. 기존 챔피언과 `no_bet` 정책은 유지한다.",
            "",
            "새로운 공식 검증은 `FUTURE_HOLDOUT_VALIDATION.md`에 사전등록된 절차와 2026-08-09 이후 첫 500개 적격 경주를 사용한다.",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> int:
    base = pd.read_csv(M2_PATH)
    _validate_base(base)
    probabilities = {
        "R0_market": base["q_market"].to_numpy(float),
        "R1_existing_m2_standalone": base["p_model_race"].to_numpy(float),
        "R2_ranker_probability": _aligned_probabilities(base, R2_PATH, "p_ranker_race"),
        "R3_fixed_m2_market_blend": _aligned_probabilities(base, R3_PATH, "p_final"),
        "R4_gated_adaptive_blend": _aligned_probabilities(base, R4_PATH, "p_final"),
    }
    market = evaluate_candidate(base, probabilities["R0_market"], name="R0_market")
    candidates = [market]
    for name in CANDIDATE_COMPLEXITY:
        if name == "R0_market":
            continue
        candidates.append(
            evaluate_candidate(
                base,
                probabilities[name],
                name=name,
                market_metrics=market,
            )
        )
    selected = select_constrained_candidate(candidates)
    created_at = utc_now()
    report: dict[str, object] = {
        "experiment": "stage_27_constrained_candidate_freeze",
        "created_at": created_at,
        "data_policy": {
            "selection_fold": "Calibration only",
            "rows": int(len(base)),
            "races": int(base["race_id"].nunique()),
            "date_min": int(base["rcDate"].min()),
            "date_max": int(base["rcDate"].max()),
            "opened_final_test": "not_loaded_not_evaluated",
        },
        "selection_contract": [
            "validate_same_entries_and_probability_simplex",
            "require_delta_logloss_gte_zero",
            "require_delta_brier_gte_zero",
            "maximize_unique_top1_correct",
            "minimize_actual_top1_overrides",
            "minimize_complexity",
        ],
        "candidates": candidates,
        "selected_challenger": selected,
        "challenger_id": "r4_gate_ranker_t065_gate065_l030_v1",
        "frozen_parameters": {
            "rank_temperature": 0.65,
            "gate_threshold": 0.65,
            "lambda_switch": 0.3,
            "random_seed": 42,
            "default_action": "keep_market",
            "top1_tie_break": [
                "p_final_desc",
                "q_market_desc",
                "entry_id_asc",
            ],
        },
        "frozen_components": _frozen_components(),
        "status": "frozen_pending_future_holdout",
        "existing_champion_action": "retain_existing_champion_and_no_bet",
        "official_claim": "none_until_new_future_holdout",
        "future_holdout": {
            "anchor_date_exclusive": 20260809,
            "target_eligible_races": 500,
            "status": "pending_data",
            "protocol": "FUTURE_HOLDOUT_VALIDATION.md",
            "single_open_after_target": True,
            "no_interim_model_changes": True,
        },
    }

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidates).to_csv(TABLE_PATH, index=False, encoding="utf-8")
    report["output_sha256"] = {
        "candidate_comparison": sha256_file(TABLE_PATH),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    SUMMARY_PATH.write_text(_summary(report), encoding="utf-8")

    policy = {
        "policy_version": 1,
        "stage": 27,
        "created_at": created_at,
        "selection_source": "reports/experiments/stage_27_candidate_freeze.json",
        "selection_fold": "calibration",
        "challenger_id": report["challenger_id"],
        "selected_candidate": selected["candidate"],
        "selected_metrics": {
            key: selected[key]
            for key in (
                "race_log_loss",
                "race_brier",
                "top1_correct",
                "top1_accuracy",
                "delta_logloss",
                "delta_brier",
                "delta_top1",
                "delta_top1_correct",
                "actual_top1_override_races",
            )
        },
        "frozen_parameters": report["frozen_parameters"],
        "frozen_components": report["frozen_components"],
        "future_holdout": report["future_holdout"],
        "status": report["status"],
        "existing_champion_action": report["existing_champion_action"],
        "opened_final_test_evaluated": False,
        "outputs": [
            {
                "path": "data/analysis/stage_27_candidate_comparison.csv",
                "sha256": report["output_sha256"]["candidate_comparison"],
            }
        ],
    }
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
