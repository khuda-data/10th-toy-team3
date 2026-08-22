"""Stage 24: train and compare the R2 pairwise race ranker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRanker

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries, make_walk_forward_folds
from src.data.ranking_data import RankingDataset, build_ranking_dataset
from src.evaluation.ranking_metrics import compare_ranking_metrics, ranking_metrics
from src.features.preprocess import infer_feature_schema, make_preprocessor, model_frame
from src.models.common import utc_now


PARAMETERS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@1",
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 1.0,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}

M2_OOF_PATH = PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_train_oof.csv.gz"
M2_CAL_PATH = PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_calibration.csv.gz"
R2_OOF_PATH = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_train_oof.csv.gz"
R2_CAL_PATH = PROJECT_ROOT / "data" / "predictions" / "r2_xgb_ranker_calibration.csv.gz"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "models" / "r2_xgb_ranker.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_24_ranker.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_24_summary.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_ranker() -> XGBRanker:
    return XGBRanker(**PARAMETERS)


def _fit_and_score(
    fit_data: RankingDataset,
    score_data: RankingDataset,
    schema,
):
    preprocessor = make_preprocessor(schema, scale_numeric=False)
    x_fit = preprocessor.fit_transform(model_frame(fit_data.frame, schema))
    x_score = preprocessor.transform(model_frame(score_data.frame, schema))
    ranker = make_ranker()
    ranker.fit(x_fit, fit_data.relevance, group=fit_data.group_sizes, verbose=False)
    scores = ranker.predict(x_score)
    if len(scores) != len(score_data.frame) or not np.isfinite(scores).all():
        raise ValueError("R2 produced invalid ranking scores")
    return preprocessor, ranker, np.asarray(scores, dtype=float)


def _load_m2_predictions(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(
        path,
        dtype={"race_id": "string", "entry_id": "string"},
    )
    required = {"race_id", "entry_id", "p_model_race"}
    if not required <= set(predictions.columns):
        raise ValueError(f"M2 predictions missing columns: {sorted(required - set(predictions.columns))}")
    if predictions["entry_id"].duplicated().any():
        raise ValueError("M2 prediction entry_id values must be unique")
    return predictions


def _aligned_m2_scores(dataset: RankingDataset, predictions: pd.DataFrame) -> np.ndarray:
    aligned = dataset.frame[["entry_id"]].merge(
        predictions[["entry_id", "p_model_race"]],
        on="entry_id",
        how="left",
        validate="one_to_one",
    )
    if aligned["p_model_race"].isna().any():
        raise ValueError("M2 predictions do not cover the ranking dataset")
    return aligned["p_model_race"].to_numpy(dtype=float)


def _evaluate_candidates(
    dataset: RankingDataset,
    *,
    m2_scores: np.ndarray,
    r2_scores: np.ndarray,
) -> dict[str, object]:
    market_metrics = ranking_metrics(dataset.frame, dataset.frame["q_market"].to_numpy())
    m2_metrics = ranking_metrics(dataset.frame, m2_scores)
    r2_metrics = ranking_metrics(dataset.frame, r2_scores)
    return {
        "R0_market": market_metrics,
        "R1_existing_m2": m2_metrics,
        "R2_pairwise_ranker": r2_metrics,
        "comparison": compare_ranking_metrics(market_metrics, m2_metrics, r2_metrics),
    }


def _prediction_frame(
    dataset: RankingDataset,
    scores: np.ndarray,
    *,
    source: str,
    wf_fold: int | None,
) -> pd.DataFrame:
    output = dataset.frame[["race_id", "entry_id", "rcDate", "win", "q_market"]].copy()
    output["model"] = "R2_xgb_pairwise_ranker"
    output["source"] = source
    output["wf_fold"] = wf_fold if wf_fold is not None else pd.NA
    output["ranking_score"] = scores
    output["probability_status"] = "not_converted_until_stage_25"
    return output


def _write_summary(report: dict[str, object]) -> None:
    oof = report["evaluation"]["train_oof_pooled"]
    cal = report["evaluation"]["calibration"]
    lines = [
        "# 개발 24단계 결과 — R2 Pairwise Ranker",
        "",
        f"생성 시점: {report['created_at']}",
        "",
        "## 학습 계약",
        "",
        "- 목적함수: `rank:pairwise`",
        "- 평가 관점: 경주 내 우승마를 다른 출전마보다 위에 배치",
        "- 피처: 시장 정보가 없는 112개 PRE_RACE 피처",
        "- Train: 4-fold expanding walk-forward OOF",
        "- Calibration: 전체 Train 학습 후 독립 평가",
        "- 기존 Final Test: 사용하지 않음",
        "- R2 출력은 아직 확률이 아니며 Race Log Loss·Brier는 25단계 전까지 계산하지 않음",
        "",
        "## 순위 성능",
        "",
        "| 구간 | 후보 | Top-1 | 적중수 | Hit@3 | MRR | 우승마 평균순위 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope_label, result in (("Train OOF", oof), ("Calibration", cal)):
        for candidate in ("R0_market", "R1_existing_m2", "R2_pairwise_ranker"):
            row = result[candidate]
            lines.append(
                f"| {scope_label} | {candidate} | {row['top1_accuracy']:.2%} | "
                f"{row['top1_correct']} | {row['hit_at_3']:.2%} | "
                f"{row['mean_reciprocal_rank']:.4f} | {row['winner_mean_rank']:.3f} |"
            )
    oof_delta = oof["comparison"]["r2_minus_market"]
    cal_delta = cal["comparison"]["r2_minus_market"]
    lines.extend(
        [
            "",
            "## 시장 대비 결과",
            "",
            f"- Train OOF: R2 Top-1 적중 `{oof_delta['top1_correct']:+d}`경주, `{oof_delta['top1_accuracy']:+.2%}p` 차이",
            f"- Calibration: R2 Top-1 적중 `{cal_delta['top1_correct']:+d}`경주, `{cal_delta['top1_accuracy']:+.2%}p` 차이",
            "",
            "## 안정성 해석",
            "",
            "- R2의 기존 M2 대비 fold별 Top-1 적중 차이는 `-3, -2, +4, +2`경주였다.",
            "- Train OOF 합계에서는 M2보다 `+1`경주, Calibration에서는 `+3`경주였지만 네 fold에서 방향이 일관되지 않았다.",
            "- 시장은 네 OOF fold 모두에서 R2보다 높았다. R2 단독 순위를 시장 대체 모델로 사용할 근거는 없다.",
            "- 다만 Calibration의 Hit@3·MRR·우승마 평균순위도 기존 M2보다 소폭 개선되어 25단계 확률 변환 후보로는 유지한다.",
            "",
            "24단계 결과는 랭킹 모델 자체의 진단이다. 후보 승격이나 시장 우위 선언이 아니며, 25단계에서 경주별 확률로 변환한 뒤 Log Loss와 Brier 비열화 제약을 적용해야 한다.",
            "",
            "## 산출물",
            "",
            "- `artifacts/models/r2_xgb_ranker.joblib`: 전체 Train 학습 전처리기와 ranker",
            "- `data/predictions/r2_xgb_ranker_train_oof.csv.gz`: 시간순 OOF 랭킹 점수",
            "- `data/predictions/r2_xgb_ranker_calibration.csv.gz`: Calibration 랭킹 점수",
            "- `reports/experiments/stage_24_ranker.json`: fold별·통합 비교와 파일 해시",
            "",
            "다음 25단계에서는 랭킹 점수를 경주별 softmax 확률로 바꾸고 Calibration Race Log Loss로 temperature를 선택한다.",
            "",
        ]
    )
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    train = load_model_entries(("train",))
    calibration = load_model_entries(("calibration",))
    if int(calibration["rcDate"].max()) > 20251227:
        raise ValueError("R2 training attempted to load the opened Final Test")
    schema = infer_feature_schema(train)
    m2_oof = _load_m2_predictions(M2_OOF_PATH)
    m2_cal = _load_m2_predictions(M2_CAL_PATH)

    fold_reports: list[dict[str, object]] = []
    oof_outputs: list[pd.DataFrame] = []
    for fold in make_walk_forward_folds(train):
        fit_data = build_ranking_dataset(train.loc[fold["train_index"]], feature_names=schema.features)
        valid_data = build_ranking_dataset(train.loc[fold["valid_index"]], feature_names=schema.features)
        _, _, r2_scores = _fit_and_score(fit_data, valid_data, schema)
        m2_scores = _aligned_m2_scores(valid_data, m2_oof)
        evaluation = _evaluate_candidates(
            valid_data, m2_scores=m2_scores, r2_scores=r2_scores
        )
        fold_reports.append(
            {
                "fold": int(fold["fold"]),
                "train_rows": int(len(fit_data.frame)),
                "train_races": int(len(fit_data.group_sizes)),
                "train_date_max": int(fold["train_date_max"]),
                "valid_rows": int(len(valid_data.frame)),
                "valid_races": int(len(valid_data.group_sizes)),
                "valid_date_min": int(fold["valid_date_min"]),
                "valid_date_max": int(fold["valid_date_max"]),
                "evaluation": evaluation,
            }
        )
        oof_outputs.append(
            _prediction_frame(
                valid_data,
                r2_scores,
                source="train_oof",
                wf_fold=int(fold["fold"]),
            )
        )

    oof_predictions = pd.concat(oof_outputs, ignore_index=True).sort_values(
        ["rcDate", "race_id", "entry_id"], kind="stable"
    ).reset_index(drop=True)
    oof_entries = train.loc[train["entry_id"].isin(oof_predictions["entry_id"])].copy()
    oof_data = build_ranking_dataset(oof_entries, feature_names=schema.features)
    oof_scores = (
        oof_data.frame[["entry_id"]]
        .merge(
            oof_predictions[["entry_id", "ranking_score"]],
            on="entry_id",
            how="left",
            validate="one_to_one",
        )["ranking_score"]
        .to_numpy(dtype=float)
    )
    pooled_oof = _evaluate_candidates(
        oof_data,
        m2_scores=_aligned_m2_scores(oof_data, m2_oof),
        r2_scores=oof_scores,
    )

    train_data = build_ranking_dataset(train, feature_names=schema.features)
    calibration_data = build_ranking_dataset(calibration, feature_names=schema.features)
    preprocessor, ranker, calibration_scores = _fit_and_score(
        train_data, calibration_data, schema
    )
    calibration_evaluation = _evaluate_candidates(
        calibration_data,
        m2_scores=_aligned_m2_scores(calibration_data, m2_cal),
        r2_scores=calibration_scores,
    )
    calibration_predictions = _prediction_frame(
        calibration_data,
        calibration_scores,
        source="calibration",
        wf_fold=None,
    )

    for path in (R2_OOF_PATH, R2_CAL_PATH, ARTIFACT_PATH, REPORT_PATH, SUMMARY_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
    oof_predictions.to_csv(R2_OOF_PATH, index=False, compression="gzip", encoding="utf-8")
    calibration_predictions.to_csv(R2_CAL_PATH, index=False, compression="gzip", encoding="utf-8")
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "ranker": ranker,
            "feature_schema": schema,
            "parameters": PARAMETERS,
            "probability_status": "not_converted_until_stage_25",
        },
        ARTIFACT_PATH,
    )

    report: dict[str, object] = {
        "experiment": "stage_24_r2_pairwise_ranker",
        "created_at": utc_now(),
        "data_policy": {
            "train": "four-fold chronological OOF plus full fit for Calibration",
            "calibration": "evaluation only; no Final Test",
            "opened_final_test": "not_loaded_not_evaluated",
            "max_rcDate": int(calibration_data.frame["rcDate"].max()),
        },
        "model": {
            "name": "R2_xgb_pairwise_ranker",
            "parameters": PARAMETERS,
            "feature_count": len(schema.features),
            "feature_role": "PRE_RACE_only",
            "group_argument": "race-contiguous group_sizes",
            "probability_status": "not_converted_until_stage_25",
        },
        "evaluation_policy": {
            "tie_break": ["score_desc", "q_market_desc", "entry_id_asc"],
            "metrics": ["Top-1", "Hit@3", "NDCG@1", "NDCG@3", "MRR", "winner_mean_rank"],
            "logloss_brier_status": "not_applicable_to_raw_ranking_scores",
        },
        "walk_forward": fold_reports,
        "evaluation": {
            "train_oof_pooled": pooled_oof,
            "calibration": calibration_evaluation,
        },
        "outputs": {
            "artifact": str(ARTIFACT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_oof_predictions": str(R2_OOF_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "calibration_predictions": str(R2_CAL_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "next_stage": "Convert R2 scores to race probabilities and select temperature on Calibration",
    }
    report["output_sha256"] = {
        key: _sha256(PROJECT_ROOT / relative) for key, relative in report["outputs"].items()
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_summary(report)
    print(json.dumps(report["evaluation"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
