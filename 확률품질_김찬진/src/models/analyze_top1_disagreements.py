"""Stage 22: analyze market/M2 Top-1 disagreements without opened-test reuse."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries, make_walk_forward_folds
from src.evaluation.top1_disagreement import (
    build_race_disagreement_table,
    build_segment_summary,
)
from src.features.preprocess import infer_feature_schema, model_frame
from src.models.common import evaluate_estimator, utc_now
from src.models.train_m2_xgboost import make_estimator


ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
PREDICTION_DIR = PROJECT_ROOT / "data" / "predictions"
REPORT_DIR = PROJECT_ROOT / "reports" / "experiments"
OOF_PATH = PREDICTION_DIR / "m2_xgboost_train_oof.csv.gz"
CALIBRATION_PATH = PREDICTION_DIR / "m2_xgboost_calibration.csv.gz"
RACE_PATH = ANALYSIS_DIR / "stage_22_race_disagreements.csv.gz"
SEGMENT_PATH = ANALYSIS_DIR / "stage_22_segment_summary.csv"
REPORT_PATH = REPORT_DIR / "stage_22_top1_disagreement.json"
SUMMARY_PATH = REPORT_DIR / "stage_22_summary.md"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_train_oof_predictions(train: pd.DataFrame) -> pd.DataFrame:
    schema = infer_feature_schema(train)
    outputs: list[pd.DataFrame] = []
    for fold in make_walk_forward_folds(train):
        fit_frame = train.loc[fold["train_index"]]
        valid_frame = train.loc[fold["valid_index"]]
        estimator = make_estimator(schema)
        estimator.fit(model_frame(fit_frame, schema), fit_frame["win"].to_numpy())
        _, raw, normalized = evaluate_estimator(estimator, valid_frame, schema)
        output = valid_frame[
            ["race_id", "entry_id", "rcDate", "win", "q_market"]
        ].copy()
        output["model"] = "M2_xgboost_oof"
        output["wf_fold"] = int(fold["fold"])
        output["p_model_raw"] = raw
        output["p_model_race"] = normalized
        outputs.append(output)
    return pd.concat(outputs, ignore_index=True).sort_values(
        ["rcDate", "race_id", "entry_id"], kind="stable"
    ).reset_index(drop=True)


def _merge_predictions(
    entries: pd.DataFrame, predictions: pd.DataFrame, *, require_fold: bool
) -> pd.DataFrame:
    prediction_columns = ["entry_id", "p_model_raw", "p_model_race"]
    if require_fold:
        prediction_columns.append("wf_fold")
    merged = entries.merge(
        predictions[prediction_columns],
        on="entry_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(predictions):
        raise ValueError("Prediction rows do not match their declared data fold")
    if set(merged["race_id"]) != set(predictions["race_id"]):
        raise ValueError("Prediction race_ids do not match their declared data fold")
    return merged.sort_values(["rcDate", "race_id", "entry_id"], kind="stable")


def _overall_records(segment_summary: pd.DataFrame) -> dict[str, dict[str, object]]:
    overall = segment_summary.loc[
        segment_summary["segment_name"].eq("all")
        & segment_summary["segment_value"].eq("all")
    ]
    return {
        str(row["source"]): {
            key: (None if pd.isna(value) else value.item() if hasattr(value, "item") else value)
            for key, value in row.items()
            if key not in {"source", "segment_name", "segment_value"}
        }
        for _, row in overall.iterrows()
    }


def _write_summary(report: dict[str, object]) -> None:
    overall = report["overall"]
    combined = overall["combined"]
    oof = overall["train_oof"]
    calibration = overall["calibration"]
    lines = [
        "# 개발 22단계 결과 — 시장/M2 Top-1 불일치 분석",
        "",
        f"생성 시점: {report['created_at']}",
        "",
        "## 데이터 경계",
        "",
        "- Train: 시간순 4-fold OOF 검증행만 사용",
        "- Calibration: 전체 Train으로 학습한 기존 M2의 저장 예측 사용",
        "- 기존 Final Test: 읽거나 집계하지 않음",
        "",
        "## 전체 결과",
        "",
        "| 구간 | 경주 | 불일치 | 불일치율 | 시장 적중 | M2 적중 | M2 순증감 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in (("Train OOF", oof), ("Calibration", calibration), ("합계", combined)):
        lines.append(
            f"| {label} | {row['races']} | {row['disagreement_races']} | "
            f"{row['disagreement_rate']:.2%} | {row['market_correct']} | "
            f"{row['model_correct']} | {row['model_minus_market_correct']:+d} |"
        )
    lines.extend(
        [
            "",
            "## 합계 정오 조합",
            "",
            f"- 둘 다 적중: {combined['both_correct']}경주",
            f"- 시장만 적중: {combined['market_only_correct']}경주",
            f"- M2만 적중: {combined['model_only_correct']}경주",
            f"- 둘 다 실패: {combined['both_wrong']}경주",
            "",
            "시장과 M2가 다른 말을 1위로 고른 경주에서만 시장 순위를 뒤집을 기회와 위험이 생긴다. "
            "`model_only_correct`는 뒤집기의 이익, `market_only_correct`는 뒤집기의 손실이며, "
            "둘의 차이가 게이트가 확보해야 할 순적중 개선량이다.",
            "",
            "## 세그먼트 해석",
            "",
            "- 시장 1·2위 확률 차이가 `<=0.02`인 127경주는 합계 기준 M2가 `+2`경주였지만, Train OOF `+4`와 Calibration `-2`로 방향이 재현되지 않았다.",
            "- 출전두수 `<=8`인 115경주는 합계 `+3`경주, Train OOF `+3`, Calibration `0`이었다. 표본이 작아 게이트 규칙으로 확정하지 않는다.",
            "- M2 1·2위 확률 차이가 `>0.10`이어도 합계 `-20`경주였다. M2 자체 확신만으로 시장을 뒤집을 수 없다.",
            "- 단일 세그먼트들은 서로 겹치며 결과를 보고 비교한 탐색 결과다. 22단계에서는 후보 규칙을 선택하지 않고 26단계 게이트의 입력 후보로만 보존한다.",
            "",
            "## 산출물",
            "",
            "- `data/predictions/m2_xgboost_train_oof.csv.gz`: Train 시간순 OOF 예측",
            "- `data/analysis/stage_22_race_disagreements.csv.gz`: 경주별 후보·격차·엔트로피·정오 조합",
            "- `data/analysis/stage_22_segment_summary.csv`: 고정 세그먼트별 뒤집기 성과",
            "- `reports/experiments/stage_22_top1_disagreement.json`: 재현 정보와 전체 집계",
            "",
            "다음 23단계에서는 같은 데이터 경계를 유지한 채 경주 그룹 랭킹 데이터셋과 무결성 검사를 구현한다.",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    policy = json.loads(
        (PROJECT_ROOT / "data" / "manifests" / "top1_research_policy.json").read_text(
            encoding="utf-8"
        )
    )
    train = load_model_entries(("train",))
    calibration = load_model_entries(("calibration",))
    schema = infer_feature_schema(train)

    oof_predictions = generate_train_oof_predictions(train)
    calibration_predictions = pd.read_csv(CALIBRATION_PATH, dtype={"entry_id": "string"})
    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)
    oof_predictions.to_csv(OOF_PATH, index=False, compression="gzip", encoding="utf-8")

    train_oof_entries = train.loc[train["entry_id"].isin(oof_predictions["entry_id"])].copy()
    train_oof = _merge_predictions(train_oof_entries, oof_predictions, require_fold=True)
    calibration_merged = _merge_predictions(
        calibration, calibration_predictions, require_fold=False
    )

    oof_races = build_race_disagreement_table(
        train_oof,
        source="train_oof",
        feature_columns=schema.features,
    )
    calibration_races = build_race_disagreement_table(
        calibration_merged,
        source="calibration",
        feature_columns=schema.features,
    )
    races = pd.concat([oof_races, calibration_races], ignore_index=True)
    if int(races["rcDate"].max()) > policy["data_boundaries"]["calibration"]["date_max"]:
        raise ValueError("Stage 22 attempted to read beyond Calibration")

    segments = build_segment_summary(races)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    races.to_csv(RACE_PATH, index=False, compression="gzip", encoding="utf-8")
    segments.to_csv(SEGMENT_PATH, index=False, encoding="utf-8")

    report: dict[str, object] = {
        "experiment": "stage_22_top1_disagreement",
        "created_at": utc_now(),
        "policy_version": policy["policy_version"],
        "data_policy": {
            "train": "chronological four-fold OOF validation rows only",
            "calibration": "stored M2 predictions fitted on full Train",
            "opened_final_test": "not_loaded_not_aggregated_forbidden_for_selection",
            "max_rcDate": int(races["rcDate"].max()),
        },
        "analysis_tie_break_contract": {
            "model": ["p_model_race_desc", "q_market_desc", "entry_id_asc"],
            "market": policy["market_top1_tie_break"],
        },
        "feature_missingness_count": len(schema.features),
        "overall": _overall_records(segments),
        "outputs": {
            "train_oof_predictions": str(OOF_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "race_analysis": str(RACE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "segment_summary": str(SEGMENT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report["output_sha256"] = {
        "train_oof_predictions": _sha256(OOF_PATH),
        "race_analysis": _sha256(RACE_PATH),
        "segment_summary": _sha256(SEGMENT_PATH),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary(report)
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
