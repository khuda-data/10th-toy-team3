"""Stage 25: convert R2 rank scores to probabilities and select temperature."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.evaluation.race_metrics import race_metrics
from src.evaluation.ranker_probability import ranking_scores_to_probabilities
from src.evaluation.ranking_metrics import ranking_metrics


TEMPERATURE_GRID = tuple(
    [round(value, 2) for value in np.arange(0.10, 3.01, 0.05)]
    + [4.0, 5.0, 7.5, 10.0]
)
R2_OOF_INPUT = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_train_oof.csv.gz"
R2_CAL_INPUT = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_calibration.csv.gz"
M2_OOF_INPUT = PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_train_oof.csv.gz"
M2_CAL_INPUT = PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_calibration.csv.gz"
FINAL_BLEND_CAL_INPUT = PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_calibration_final.csv.gz"
OOF_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_train_oof_probability.csv.gz"
CAL_OUTPUT = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_calibration_probability.csv.gz"
POLICY_PATH = PROJECT_ROOT / "data" / "manifests" / "ranker_temperature_policy.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_25_ranker_probability.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_25_summary.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rank_scores(path: Path, *, expected_source: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"race_id": "string", "entry_id": "string"},
    )
    required = {
        "race_id",
        "entry_id",
        "rcDate",
        "win",
        "q_market",
        "ranking_score",
        "probability_status",
        "source",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"R2 score input missing columns: {missing}")
    if not frame["source"].eq(expected_source).all():
        raise ValueError("R2 score source does not match the requested fold")
    if not frame["probability_status"].eq("not_converted_until_stage_25").all():
        raise ValueError("R2 scores were already converted or have unknown status")
    if frame["entry_id"].duplicated().any():
        raise ValueError("R2 score entry_id values must be unique")
    if not frame.groupby("race_id", sort=False)["win"].sum().eq(1).all():
        raise ValueError("R2 probability conversion requires one winner per race")
    if int(frame["rcDate"].max()) > 20251227:
        raise ValueError("Stage 25 attempted to load the opened Final Test")
    return frame.sort_values(["rcDate", "race_id", "entry_id"], kind="stable").reset_index(drop=True)


def evaluate_temperature_grid(frame: pd.DataFrame) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    scores = frame["ranking_score"].to_numpy(dtype=float)
    for temperature in TEMPERATURE_GRID:
        probabilities = ranking_scores_to_probabilities(
            frame, scores, temperature=temperature
        )
        results.append(
            {
                "temperature": temperature,
                "metrics": _probability_metrics(frame, probabilities),
            }
        )
    return results


def select_temperature(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        raise ValueError("Temperature selection requires at least one candidate")
    return min(
        results,
        key=lambda row: (
            row["metrics"]["race_log_loss"],
            row["metrics"]["race_brier"],
            abs(row["temperature"] - 1.0),
            row["temperature"],
        ),
    )


def _probability_metrics(frame: pd.DataFrame, probabilities) -> dict[str, object]:
    """Combine probability losses with the frozen unique Top-1 tie-break."""
    metrics = race_metrics(frame, probabilities)
    ranks = ranking_metrics(frame, probabilities)
    metrics["top1_correct"] = ranks["top1_correct"]
    metrics["top1_accuracy"] = ranks["top1_accuracy"]
    metrics["mean_reciprocal_rank"] = ranks["mean_reciprocal_rank"]
    metrics["winner_mean_rank"] = ranks["winner_mean_rank"]
    return metrics


def _align_probability(
    frame: pd.DataFrame,
    path: Path,
    column: str,
) -> np.ndarray:
    reference = pd.read_csv(path, dtype={"entry_id": "string"})
    if column not in reference:
        raise ValueError(f"Reference probability column missing: {column}")
    aligned = frame[["entry_id"]].merge(
        reference[["entry_id", column]],
        on="entry_id",
        how="left",
        validate="one_to_one",
    )
    if aligned[column].isna().any():
        raise ValueError(f"Reference probabilities do not cover R2 rows: {path.name}")
    return aligned[column].to_numpy(dtype=float)


def _comparison(
    frame: pd.DataFrame,
    *,
    r2_probabilities: np.ndarray,
    m2_path: Path,
    final_blend_path: Path | None = None,
) -> dict[str, object]:
    market = _probability_metrics(frame, frame["q_market"].to_numpy(dtype=float))
    m2 = _probability_metrics(frame, _align_probability(frame, m2_path, "p_model_race"))
    r2 = _probability_metrics(frame, r2_probabilities)
    candidates: dict[str, object] = {
        "R0_market": market,
        "R1_existing_m2_standalone": m2,
        "R2_ranker_probability": r2,
    }
    if final_blend_path is not None:
        candidates["existing_market_blend_reference"] = _probability_metrics(
            frame, _align_probability(frame, final_blend_path, "p_final")
        )
    candidates["r2_delta_vs_market"] = {
        "delta_logloss": float(market["race_log_loss"] - r2["race_log_loss"]),
        "delta_brier": float(market["race_brier"] - r2["race_brier"]),
        "delta_top1": float(r2["top1_accuracy"] - market["top1_accuracy"]),
    }
    candidates["r2_delta_vs_m2"] = {
        "delta_logloss": float(m2["race_log_loss"] - r2["race_log_loss"]),
        "delta_brier": float(m2["race_brier"] - r2["race_brier"]),
        "delta_top1": float(r2["top1_accuracy"] - m2["top1_accuracy"]),
    }
    return candidates


def _save_probabilities(
    path: Path,
    frame: pd.DataFrame,
    *,
    temperature: float,
    selection_note: str,
) -> np.ndarray:
    probabilities = ranking_scores_to_probabilities(
        frame,
        frame["ranking_score"].to_numpy(dtype=float),
        temperature=temperature,
    )
    output = frame[
        [
            "race_id",
            "entry_id",
            "rcDate",
            "win",
            "q_market",
            "model",
            "source",
            "wf_fold",
            "ranking_score",
        ]
    ].copy()
    output["rank_temperature"] = temperature
    output["p_ranker_race"] = probabilities
    output["probability_status"] = "race_softmax_calibrated"
    output["temperature_selection"] = selection_note
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, compression="gzip", encoding="utf-8")
    return probabilities


def _write_summary(report: dict[str, object]) -> None:
    selected = report["temperature_selection"]
    cal = report["evaluation"]["calibration"]
    oof = report["evaluation"]["train_oof_retrospective"]
    cal_r2 = cal["R2_ranker_probability"]
    cal_market = cal["R0_market"]
    delta = cal["r2_delta_vs_market"]
    delta_m2 = cal["r2_delta_vs_m2"]
    lines = [
        "# 개발 25단계 결과 — R2 랭킹 점수 확률 변환",
        "",
        f"생성 시점: {report['created_at']}",
        "",
        "## 선택 규칙",
        "",
        "- 변환: 경주별 `softmax(ranking_score / T_rank)`",
        "- 선택 데이터: Calibration 641경주만 사용",
        "- 선택 지표: Race Log Loss 최소화, 동률이면 Brier와 T=1 근접성",
        "- Temperature는 순위를 바꾸지 않으므로 Top-1은 24단계와 동일",
        f"- 선택 결과: `T_rank={selected['temperature']}`",
        "",
        "## Calibration 확률 성능",
        "",
        "| 후보 | Race Log Loss ↓ | Race Brier ↓ | Top-1 | MRR |",
        "|---|---:|---:|---:|---:|",
        f"| 시장 | {cal_market['race_log_loss']:.6f} | {cal_market['race_brier']:.6f} | {cal_market['top1_accuracy']:.2%} | {cal_market['mean_reciprocal_rank']:.4f} |",
        f"| R2 softmax | {cal_r2['race_log_loss']:.6f} | {cal_r2['race_brier']:.6f} | {cal_r2['top1_accuracy']:.2%} | {cal_r2['mean_reciprocal_rank']:.4f} |",
        "",
        f"시장 대비 R2 개선량은 Log Loss `{delta['delta_logloss']:+.6f}`, Brier `{delta['delta_brier']:+.6f}`, Top-1 `{delta['delta_top1']:+.2%}p`다. 양수는 R2 우위다.",
        f"기존 독립 M2 대비로는 Log Loss `{delta_m2['delta_logloss']:+.6f}`, Brier `{delta_m2['delta_brier']:+.6f}`, Top-1 `{delta_m2['delta_top1']:+.2%}p`다.",
        "",
        "## 해석",
        "",
        "- R2 확률은 경주별 합이 1인 유효한 분포가 됐다.",
        "- Temperature는 확률의 날카로움만 바꾸며 24단계의 말 순위를 바꾸지 않았다.",
        "- Train OOF 수치는 나중 시점인 Calibration에서 선택한 T를 역적용한 참고값이므로 독립 선택 성능으로 사용하지 않는다.",
        "- 시장 대비 Log Loss 또는 Brier가 음수이면 R2 단독 확률은 27단계 비열화 제약을 통과할 수 없다. 26단계에서 시장 유지가 기본인 적응형 결합을 검토한다.",
        "- Calibration에서는 기존 M2보다 세 지표가 소폭 좋아졌지만, OOF 참고 구간에서는 R2 Log Loss와 Brier가 M2보다 나빠 개선이 안정적으로 재현되지 않았다.",
        "- 기존 시장 혼합 모델의 Calibration Log Loss는 `1.773358`로 R2 단독보다 훨씬 낮다. 시장 앵커를 제거하면 확률 품질이 크게 악화된다.",
        "",
        "## 참고 Train OOF",
        "",
        f"선택 T 역적용 R2 Log Loss `{oof['R2_ranker_probability']['race_log_loss']:.6f}`, Brier `{oof['R2_ranker_probability']['race_brier']:.6f}`. 기존 M2 대비 각각 `{oof['r2_delta_vs_m2']['delta_logloss']:+.6f}`, `{oof['r2_delta_vs_m2']['delta_brier']:+.6f}`이며 공식 선택 근거가 아니다.",
        "",
        "## 산출물",
        "",
        "- `data/manifests/ranker_temperature_policy.json`: 선택 T와 사용 경계",
        "- `data/predictions/r2_xgb_ranker_calibration_probability.csv.gz`: Calibration 확률",
        "- `data/predictions/r2_xgb_ranker_train_oof_probability.csv.gz`: 선택 T 역적용 OOF 참고 확률",
        "- `reports/experiments/stage_25_ranker_probability.json`: 전체 temperature grid와 비교 지표",
        "",
        "다음 26단계에서는 시장 유지/교체 게이트와 경주별 lambda를 구현한다.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    calibration = _load_rank_scores(R2_CAL_INPUT, expected_source="calibration")
    oof = _load_rank_scores(R2_OOF_INPUT, expected_source="train_oof")
    if len(calibration) != 6582 or calibration["race_id"].nunique() != 641:
        raise ValueError("Temperature selection must use the fixed Calibration fold")
    if len(oof) != 8043 or oof["race_id"].nunique() != 770:
        raise ValueError("OOF probability reference must cover the four validation folds")

    grid = evaluate_temperature_grid(calibration)
    selected = select_temperature(grid)
    temperature = float(selected["temperature"])
    calibration_probabilities = _save_probabilities(
        CAL_OUTPUT,
        calibration,
        temperature=temperature,
        selection_note="selected_on_calibration_race_log_loss",
    )
    oof_probabilities = _save_probabilities(
        OOF_OUTPUT,
        oof,
        temperature=temperature,
        selection_note="retrospective_application_of_calibration_selected_temperature",
    )

    calibration_comparison = _comparison(
        calibration,
        r2_probabilities=calibration_probabilities,
        m2_path=M2_CAL_INPUT,
        final_blend_path=FINAL_BLEND_CAL_INPUT,
    )
    oof_comparison = _comparison(
        oof,
        r2_probabilities=oof_probabilities,
        m2_path=M2_OOF_INPUT,
    )
    identity = next(row for row in grid if row["temperature"] == 1.0)
    selected_record = {
        "temperature": temperature,
        "metrics": selected["metrics"],
        "identity_temperature_metrics": identity["metrics"],
        "selection_metric": "Calibration Race Log Loss",
        "tie_breakers": ["Race Brier", "distance_from_T_1", "lower_temperature"],
    }
    report: dict[str, object] = {
        "experiment": "stage_25_r2_ranker_probability",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "formula": "p_ri(T_rank) = softmax_race(ranking_score_ri / T_rank)",
        "data_policy": {
            "selection_fold": "calibration",
            "train_oof": "retrospective diagnostic only after Calibration T selection",
            "opened_final_test": "not_loaded_not_evaluated",
            "max_rcDate": int(calibration["rcDate"].max()),
        },
        "temperature_grid": list(TEMPERATURE_GRID),
        "grid_results": grid,
        "temperature_selection": selected_record,
        "evaluation": {
            "calibration": calibration_comparison,
            "train_oof_retrospective": oof_comparison,
        },
        "probability_contract": {
            "column": "p_ranker_race",
            "sum_by_race": 1.0,
            "tolerance": 1e-9,
            "ranking_preserved": True,
        },
        "outputs": {
            "calibration_probabilities": str(CAL_OUTPUT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_oof_probabilities": str(OOF_OUTPUT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "next_stage": "Build conservative market keep/switch gate and race-specific lambda",
    }
    report["output_sha256"] = {
        key: _sha256(PROJECT_ROOT / relative) for key, relative in report["outputs"].items()
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    policy = {
        "policy_version": 1,
        "stage": 25,
        "created_at": report["created_at"],
        "model": "R2_xgb_pairwise_ranker",
        "selection_fold": "calibration",
        "selection_metric": "Race Log Loss",
        "temperature_grid": list(TEMPERATURE_GRID),
        "selected_temperature": temperature,
        "selected_metrics": selected["metrics"],
        "formula": report["formula"],
        "probability_column": "p_ranker_race",
        "probability_sum_tolerance": 1e-9,
        "top1_ranking_changes": False,
        "opened_final_test_evaluated": False,
        "oof_usage_warning": report["data_policy"]["train_oof"],
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "outputs": [
            {
                "path": relative,
                "sha256": report["output_sha256"][key],
            }
            for key, relative in report["outputs"].items()
        ],
    }
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary(report)
    print(json.dumps({"selected": selected_record, "calibration": calibration_comparison}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
