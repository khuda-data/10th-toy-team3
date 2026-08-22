"""Stage 26: train a conservative market keep/switch gate and adaptive lambda."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries
from src.evaluation.race_metrics import race_metrics
from src.evaluation.ranking_metrics import rank_entries, ranking_metrics
from src.features.registry import select_premarket_features
from src.models.market_gate import (
    GATE_CATEGORICAL_FEATURES,
    GATE_FEATURES,
    GATE_NUMERIC_FEATURES,
    adaptive_geometric_blend,
    build_gate_race_table,
)


THRESHOLD_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99)
LAMBDA_GRID = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
OOF_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_train_oof_probability.csv.gz"
CAL_PRED_PATH = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_calibration_probability.csv.gz"
TRAIN_GATE_PATH = PROJECT_ROOT / "data" / "analysis" / "stage_26_gate_train_oof.csv.gz"
CAL_GATE_PATH = PROJECT_ROOT / "data" / "analysis" / "stage_26_gate_calibration.csv.gz"
GRID_PATH = PROJECT_ROOT / "data" / "analysis" / "stage_26_gate_grid.csv"
PREDICTION_PATH = PROJECT_ROOT / "data" / "predictions" / "r4_gated_calibration.csv.gz"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "models" / "r4_market_gate_logistic.joblib"
POLICY_PATH = PROJECT_ROOT / "data" / "manifests" / "market_gate_policy.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_26_market_gate.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_26_summary.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probability_metrics(frame: pd.DataFrame, probabilities) -> dict[str, object]:
    metrics = race_metrics(frame, probabilities)
    ranks = ranking_metrics(frame, probabilities)
    metrics["top1_correct"] = ranks["top1_correct"]
    metrics["top1_accuracy"] = ranks["top1_accuracy"]
    metrics["mean_reciprocal_rank"] = ranks["mean_reciprocal_rank"]
    metrics["winner_mean_rank"] = ranks["winner_mean_rank"]
    return metrics


def _load_and_merge_entries(fold: str, path: Path) -> pd.DataFrame:
    entries = load_model_entries((fold,))
    predictions = pd.read_csv(
        path, dtype={"race_id": "string", "entry_id": "string"}
    )
    required = {"entry_id", "ranking_score", "p_ranker_race", "rank_temperature"}
    if not required <= set(predictions.columns):
        raise ValueError(f"Gate prediction columns missing: {sorted(required - set(predictions.columns))}")
    if fold == "train":
        entries = entries.loc[entries["entry_id"].isin(predictions["entry_id"])].copy()
    merged = entries.merge(
        predictions[["entry_id", "ranking_score", "p_ranker_race", "rank_temperature"]],
        on="entry_id",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(predictions):
        raise ValueError("Gate predictions do not align with their declared fold")
    if int(merged["rcDate"].max()) > 20251227:
        raise ValueError("Stage 26 attempted to load the opened Final Test")
    return merged.sort_values(["rcDate", "race_id", "entry_id"], kind="stable").reset_index(drop=True)


def make_gate_estimator() -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(GATE_NUMERIC_FEATURES),
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                list(GATE_CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def _candidate_record(
    entries: pd.DataFrame,
    race_table: pd.DataFrame,
    *,
    threshold: float | None,
    lambda_switch: float,
    baseline: dict[str, object],
) -> tuple[dict[str, object], np.ndarray, pd.Series]:
    if threshold is None:
        actions = pd.Series(False, index=race_table.index)
    else:
        actions = race_table["top1_disagreement"] & race_table["gate_score"].ge(threshold)
    race_lambdas = pd.Series(
        np.where(actions, lambda_switch, 0.0),
        index=race_table["race_id"].astype(str),
        dtype=float,
    )
    probabilities = adaptive_geometric_blend(
        entries,
        entries["q_market"].to_numpy(dtype=float),
        entries["p_ranker_race"].to_numpy(dtype=float),
        race_lambdas,
    )
    metrics = _probability_metrics(entries, probabilities)
    market_top = rank_entries(entries, entries["q_market"].to_numpy()).loc[
        lambda frame: frame["rank_position"].eq(1), ["race_id", "entry_id"]
    ].set_index("race_id")["entry_id"]
    final_top = rank_entries(entries, probabilities).loc[
        lambda frame: frame["rank_position"].eq(1), ["race_id", "entry_id"]
    ].set_index("race_id")["entry_id"]
    overrides = market_top.ne(final_top)
    selected_races = set(overrides.index[overrides])
    overridden = race_table.loc[race_table["race_id"].isin(selected_races)]
    delta_logloss = float(baseline["race_log_loss"] - metrics["race_log_loss"])
    delta_brier = float(baseline["race_brier"] - metrics["race_brier"])
    delta_top1 = float(metrics["top1_accuracy"] - baseline["top1_accuracy"])
    record = {
        "threshold": threshold,
        "lambda_switch": lambda_switch,
        "gate_action_races": int(actions.sum()),
        "actual_top1_override_races": int(overrides.sum()),
        "beneficial_overrides": int(overridden["switch_beneficial"].sum()),
        "harmful_overrides": int(overridden["switch_harmful"].sum()),
        "both_wrong_overrides": int(overridden["both_wrong"].sum()),
        **metrics,
        "delta_logloss": delta_logloss,
        "delta_brier": delta_brier,
        "delta_top1": delta_top1,
        "eligible_probability_guardrail": bool(delta_logloss >= -1e-12 and delta_brier >= -1e-12),
    }
    return record, probabilities, race_lambdas


def _select_candidate(grid: list[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in grid if row["eligible_probability_guardrail"]]
    if not eligible:
        raise ValueError("Market baseline must keep the candidate set non-empty")
    return min(
        eligible,
        key=lambda row: (
            -row["top1_correct"],
            row["actual_top1_override_races"],
            row["gate_action_races"],
            row["lambda_switch"],
            -(row["threshold"] if row["threshold"] is not None else 2.0),
        ),
    )


def _write_summary(report: dict[str, object]) -> None:
    train = report["gate_training"]
    cal = report["gate_calibration"]
    selected = report["selected_candidate"]
    accuracy_only = report["accuracy_only_best"]
    baseline = report["market_baseline"]
    lines = [
        "# 개발 26단계 결과 — 시장 유지/교체 게이트",
        "",
        f"생성 시점: {report['created_at']}",
        "",
        "## 구현",
        "",
        "- 게이트 학습: Train OOF에서 시장/R2 1위가 다른 경주만 사용",
        "- 목표 1: R2만 적중해 시장을 뒤집는 것이 유리한 경주",
        "- 목표 0: 시장만 적중하거나 둘 다 실패한 경주",
        "- Calibration 역할: gate threshold와 `lambda_switch` 선택",
        "- 기본 행동: `lambda_race=0`, 시장 유지",
        "- 선택 제약: 시장 대비 Race Log Loss와 Brier가 모두 악화되지 않아야 함",
        "",
        "## 게이트 데이터",
        "",
        f"- Train OOF 불일치: {train['disagreement_races']}경주, 유리한 교체 {train['positive_races']}경주",
        f"- Calibration 불일치: {cal['disagreement_races']}경주, 유리한 교체 {cal['positive_races']}경주",
        f"- Calibration gate ROC-AUC `{cal['roc_auc']:.4f}`, Average Precision `{cal['average_precision']:.4f}`",
        "",
        "## 선택 결과",
        "",
        f"- 정책 상태: `{report['policy_action']}`",
        f"- threshold: `{selected['threshold']}`",
        f"- lambda_switch: `{selected['lambda_switch']}`",
        f"- gate action: {selected['gate_action_races']}경주",
        f"- 실제 시장 Top-1 교체: {selected['actual_top1_override_races']}경주",
        f"- 시장 Top-1 `{baseline['top1_correct']}/{baseline['races']}` → 후보 `{selected['top1_correct']}/{selected['races']}`",
        f"- Delta Log Loss `{selected['delta_logloss']:+.6f}`, Delta Brier `{selected['delta_brier']:+.6f}`, Delta Top-1 `{selected['delta_top1']:+.2%}p`",
        "",
        "## 제약의 효과",
        "",
        f"정확도만 최대화하면 threshold `{accuracy_only['threshold']}`, lambda `{accuracy_only['lambda_switch']}`에서 `{accuracy_only['top1_correct']}/{accuracy_only['races']}`로 시장보다 `{accuracy_only['top1_correct'] - baseline['top1_correct']:+d}`경주 높다. 그러나 Delta Log Loss `{accuracy_only['delta_logloss']:+.6f}`, Delta Brier `{accuracy_only['delta_brier']:+.6f}`로 두 확률 제약을 통과하지 못해 탈락했다.",
        "",
        "시장 유지 후보도 grid에 포함했다. 제약을 지키면서 적중을 늘리지 못하면 `no_change`가 정식 결과이며, 억지로 시장 순위를 뒤집지 않는다.",
        "현재 선택값은 Calibration 결과를 보고 정한 예비 후보다. 시장 우위는 새 Future Holdout에서 검증하기 전까지 주장하지 않는다.",
        "",
        "## 산출물",
        "",
        "- `artifacts/models/r4_market_gate_logistic.joblib`: Train OOF 게이트 모델",
        "- `data/analysis/stage_26_gate_grid.csv`: threshold/lambda 전체 후보",
        "- `data/predictions/r4_gated_calibration.csv.gz`: 선택 정책 확률과 경주별 lambda",
        "- `data/manifests/market_gate_policy.json`: 선택 정책과 해시",
        "- `reports/experiments/stage_26_market_gate.json`: 학습·Calibration 상세 결과",
        "",
        "다음 27단계에서는 시장, 기존 고정 혼합, R2, R4를 동일한 비열화 제약으로 최종 비교하고 새 미래 holdout 전에 후보를 동결한다.",
        "",
    ]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    feature_columns = tuple(select_premarket_features())
    train_entries = _load_and_merge_entries("train", OOF_PRED_PATH)
    calibration_entries = _load_and_merge_entries("calibration", CAL_PRED_PATH)
    train_races = build_gate_race_table(train_entries, feature_columns=feature_columns)
    calibration_races = build_gate_race_table(calibration_entries, feature_columns=feature_columns)

    train_disagreement = train_races.loc[train_races["top1_disagreement"]].copy()
    calibration_disagreement = calibration_races.loc[
        calibration_races["top1_disagreement"]
    ].copy()
    if train_disagreement["switch_beneficial"].nunique() != 2:
        raise ValueError("Gate training target requires both classes")
    gate = make_gate_estimator()
    gate.fit(
        train_disagreement.loc[:, GATE_FEATURES],
        train_disagreement["switch_beneficial"].astype(int),
    )
    train_races["gate_score"] = 0.0
    calibration_races["gate_score"] = 0.0
    train_races.loc[train_races["top1_disagreement"], "gate_score"] = gate.predict_proba(
        train_disagreement.loc[:, GATE_FEATURES]
    )[:, 1]
    calibration_races.loc[
        calibration_races["top1_disagreement"], "gate_score"
    ] = gate.predict_proba(calibration_disagreement.loc[:, GATE_FEATURES])[:, 1]

    cal_y = calibration_disagreement["switch_beneficial"].astype(int).to_numpy()
    cal_score = calibration_races.loc[
        calibration_races["top1_disagreement"], "gate_score"
    ].to_numpy()
    baseline = _probability_metrics(
        calibration_entries, calibration_entries["q_market"].to_numpy(dtype=float)
    )
    grid: list[dict[str, object]] = []
    baseline_record, _, _ = _candidate_record(
        calibration_entries,
        calibration_races,
        threshold=None,
        lambda_switch=0.0,
        baseline=baseline,
    )
    grid.append(baseline_record)
    for threshold in THRESHOLD_GRID:
        for lambda_switch in LAMBDA_GRID:
            record, _, _ = _candidate_record(
                calibration_entries,
                calibration_races,
                threshold=threshold,
                lambda_switch=lambda_switch,
                baseline=baseline,
            )
            grid.append(record)
    selected = _select_candidate(grid)
    accuracy_only_best = min(
        grid,
        key=lambda row: (
            -row["top1_correct"],
            -row["delta_logloss"],
            row["actual_top1_override_races"],
        ),
    )
    selected_record, selected_probabilities, selected_lambdas = _candidate_record(
        calibration_entries,
        calibration_races,
        threshold=selected["threshold"],
        lambda_switch=float(selected["lambda_switch"]),
        baseline=baseline,
    )
    if selected_record != selected:
        raise AssertionError("Selected gate candidate is not reproducible")
    policy_action = (
        "no_change"
        if selected["actual_top1_override_races"] == 0
        else "candidate_pending_stage_27_and_future_holdout"
    )

    TRAIN_GATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_races.to_csv(TRAIN_GATE_PATH, index=False, compression="gzip", encoding="utf-8")
    calibration_races.to_csv(CAL_GATE_PATH, index=False, compression="gzip", encoding="utf-8")
    pd.DataFrame(grid).to_csv(GRID_PATH, index=False, encoding="utf-8")

    row_lambda = calibration_entries["race_id"].astype(str).map(selected_lambdas).to_numpy()
    gate_score_by_race = calibration_races.set_index("race_id")["gate_score"]
    prediction = calibration_entries[
        ["race_id", "entry_id", "rcDate", "win", "q_market", "ranking_score", "p_ranker_race"]
    ].copy()
    prediction["gate_score"] = prediction["race_id"].map(gate_score_by_race)
    prediction["gate_action"] = np.where(row_lambda > 0, "trust_ranker", "keep_market")
    prediction["lambda_race"] = row_lambda
    prediction["p_final"] = selected_probabilities
    prediction.to_csv(PREDICTION_PATH, index=False, compression="gzip", encoding="utf-8")
    joblib.dump(
        {
            "estimator": gate,
            "features": GATE_FEATURES,
            "numeric_features": GATE_NUMERIC_FEATURES,
            "categorical_features": GATE_CATEGORICAL_FEATURES,
            "threshold": selected["threshold"],
            "lambda_switch": selected["lambda_switch"],
            "policy_action": policy_action,
        },
        ARTIFACT_PATH,
    )

    created_at = datetime.now(timezone.utc).isoformat()
    report: dict[str, object] = {
        "experiment": "stage_26_market_keep_switch_gate",
        "created_at": created_at,
        "data_policy": {
            "gate_fit": "Train chronological OOF disagreement races only",
            "threshold_lambda_selection": "Calibration only",
            "opened_final_test": "not_loaded_not_evaluated",
            "max_rcDate": int(calibration_entries["rcDate"].max()),
        },
        "gate_model": {
            "type": "LogisticRegression",
            "C": 0.1,
            "class_weight": "balanced",
            "random_state": 42,
            "features": list(GATE_FEATURES),
            "default_action": "keep_market",
        },
        "gate_training": {
            "races": int(len(train_races)),
            "disagreement_races": int(len(train_disagreement)),
            "positive_races": int(train_disagreement["switch_beneficial"].sum()),
            "negative_races": int((~train_disagreement["switch_beneficial"]).sum()),
        },
        "gate_calibration": {
            "races": int(len(calibration_races)),
            "disagreement_races": int(len(calibration_disagreement)),
            "positive_races": int(calibration_disagreement["switch_beneficial"].sum()),
            "negative_races": int((~calibration_disagreement["switch_beneficial"]).sum()),
            "roc_auc": float(roc_auc_score(cal_y, cal_score)),
            "average_precision": float(average_precision_score(cal_y, cal_score)),
        },
        "selection_grid": {
            "thresholds": list(THRESHOLD_GRID),
            "lambda_switches": list(LAMBDA_GRID),
            "candidate_count_including_market_baseline": len(grid),
            "selection_order": [
                "require_delta_logloss_gte_zero",
                "require_delta_brier_gte_zero",
                "maximize_top1_correct",
                "minimize_actual_overrides",
                "minimize_gate_actions",
                "lower_lambda",
            ],
        },
        "market_baseline": baseline,
        "selected_candidate": selected,
        "accuracy_only_best": accuracy_only_best,
        "policy_action": policy_action,
        "outputs": {
            "artifact": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_gate_races": str(TRAIN_GATE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "calibration_gate_races": str(CAL_GATE_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "candidate_grid": str(GRID_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "calibration_predictions": str(PREDICTION_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "next_stage": "Compare and freeze the constrained candidate family before future holdout",
    }
    report["output_sha256"] = {
        key: _sha256(PROJECT_ROOT / relative) for key, relative in report["outputs"].items()
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    policy = {
        "policy_version": 1,
        "stage": 26,
        "created_at": created_at,
        "selection_source": str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "gate_fit_fold": "train_oof_disagreements",
        "selection_fold": "calibration",
        "selected_threshold": selected["threshold"],
        "selected_lambda_switch": selected["lambda_switch"],
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
                "gate_action_races",
                "actual_top1_override_races",
            )
        },
        "policy_action": policy_action,
        "default_action": "keep_market",
        "opened_final_test_evaluated": False,
        "outputs": [
            {"path": relative, "sha256": report["output_sha256"][key]}
            for key, relative in report["outputs"].items()
        ],
    }
    POLICY_PATH.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary(report)
    print(json.dumps({"gate_calibration": report["gate_calibration"], "selected": selected, "policy_action": policy_action}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
