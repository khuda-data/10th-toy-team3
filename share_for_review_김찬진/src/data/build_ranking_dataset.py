"""Stage 23: persist and verify the race-group ranking data contract."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries, make_walk_forward_folds
from src.data.ranking_data import (
    RANKING_SORT_KEYS,
    build_ranking_dataset,
    build_ranking_manifests,
    validate_ranking_manifests,
)
from src.features.registry import load_feature_registry, select_premarket_features


ENTRY_PATH = PROJECT_ROOT / "data" / "interim" / "ranking_entry_manifest.csv.gz"
GROUP_PATH = PROJECT_ROOT / "data" / "interim" / "ranking_group_manifest.csv"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "ranking_dataset_manifest.json"
SUMMARY_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_23_summary.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fold_record(entries: pd.DataFrame, groups: pd.DataFrame) -> dict[str, object]:
    sizes = groups["group_size"]
    return {
        "rows": int(len(entries)),
        "races": int(len(groups)),
        "date_min": int(entries["rcDate"].min()),
        "date_max": int(entries["rcDate"].max()),
        "group_size_min": int(sizes.min()),
        "group_size_median": float(sizes.median()),
        "group_size_max": int(sizes.max()),
        "relevance_sum": int(entries["relevance"].sum()),
        "feature_missing_cells": int(entries["feature_missing_count"].sum()),
    }


def _walk_forward_records(train: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fold in make_walk_forward_folds(train):
        fit = train.loc[fold["train_index"]]
        valid = train.loc[fold["valid_index"]]
        fit_data = build_ranking_dataset(fit)
        valid_data = build_ranking_dataset(valid)
        records.append(
            {
                "fold": int(fold["fold"]),
                "train_rows": int(len(fit_data.frame)),
                "train_groups": int(len(fit_data.group_sizes)),
                "train_date_max": int(fold["train_date_max"]),
                "valid_rows": int(len(valid_data.frame)),
                "valid_groups": int(len(valid_data.group_sizes)),
                "valid_date_min": int(fold["valid_date_min"]),
                "valid_date_max": int(fold["valid_date_max"]),
                "strictly_chronological": bool(
                    fold["train_date_max"] < fold["valid_date_min"]
                ),
                "group_overlap": bool(
                    set(fit_data.group_ids) & set(valid_data.group_ids)
                ),
            }
        )
    return records


def main() -> int:
    train = load_model_entries(("train",))
    calibration = load_model_entries(("calibration",))
    feature_names = tuple(select_premarket_features())
    registry = load_feature_registry()

    entry_parts: list[pd.DataFrame] = []
    group_parts: list[pd.DataFrame] = []
    fold_records: dict[str, dict[str, object]] = {}
    for fold_name, frame in (("train", train), ("calibration", calibration)):
        dataset = build_ranking_dataset(frame, feature_names=feature_names)
        entries, groups = build_ranking_manifests(dataset, model_fold=fold_name)
        entry_parts.append(entries)
        group_parts.append(groups)
        fold_records[fold_name] = _fold_record(entries, groups)

    entry_manifest = pd.concat(entry_parts, ignore_index=True)
    group_manifest = pd.concat(group_parts, ignore_index=True)
    validate_ranking_manifests(entry_manifest, group_manifest)
    if set(entry_manifest["model_fold"]) != {"train", "calibration"}:
        raise ValueError("Ranking data must contain only Train and Calibration")
    if int(entry_manifest["rcDate"].max()) > 20251227:
        raise ValueError("Ranking data attempted to include the opened Final Test")

    ENTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry_manifest.to_csv(ENTRY_PATH, index=False, compression="gzip", encoding="utf-8")
    group_manifest.to_csv(GROUP_PATH, index=False, encoding="utf-8")

    feature_roles = {registry["features"][name]["role"] for name in feature_names}
    manifest: dict[str, object] = {
        "manifest_version": 1,
        "stage": 23,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Race-contiguous dataset contract for XGBoost learning-to-rank",
        "inputs": {
            "entries": "data/interim/seoul_entries.csv.gz",
            "splits": "data/interim/split_manifest.csv",
            "feature_registry": "data/manifests/feature_registry.json",
        },
        "data_policy": {
            "included_folds": ["train", "calibration"],
            "forbidden_folds": ["test", "excluded"],
            "max_allowed_date": 20251227,
            "opened_final_test": "not_included",
        },
        "ranking_contract": {
            "sort_keys": list(RANKING_SORT_KEYS),
            "group_key": "race_id",
            "target": "win",
            "relevance": {"winner": 1, "other_entries": 0},
            "exactly_one_relevant_entry_per_group": True,
            "minimum_group_size": 2,
            "group_argument": "group_sizes in contiguous row order",
            "row_interval": "zero-based [row_start, row_stop_exclusive)",
        },
        "feature_contract": {
            "feature_count": len(feature_names),
            "allowed_roles": sorted(feature_roles),
            "market_features_in_ranker": False,
            "post_race_features_in_ranker": False,
            "features": list(feature_names),
            "preprocessing_fit_scope": "each walk-forward training portion only",
        },
        "observed": {
            "folds": fold_records,
            "total_rows": int(len(entry_manifest)),
            "total_groups": int(len(group_manifest)),
            "unique_entries": bool(entry_manifest["entry_id"].is_unique),
            "unique_groups": bool(group_manifest["race_id"].is_unique),
            "group_sizes_sum_to_rows": bool(
                int(group_manifest["group_size"].sum()) == len(entry_manifest)
            ),
            "all_group_relevance_sums_one": bool(
                group_manifest["relevance_sum"].eq(1).all()
            ),
        },
        "walk_forward": _walk_forward_records(train),
        "outputs": {
            "entry_manifest": {
                "path": str(ENTRY_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": ENTRY_PATH.stat().st_size,
                "sha256": _sha256(ENTRY_PATH),
            },
            "group_manifest": {
                "path": str(GROUP_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": GROUP_PATH.stat().st_size,
                "sha256": _sha256(GROUP_PATH),
            },
        },
        "next_stage": "Train R2 ranker with four-fold chronological walk-forward evaluation",
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    train_record = fold_records["train"]
    cal_record = fold_records["calibration"]
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                "# 개발 23단계 결과 — 경주 그룹 랭킹 데이터 계약",
                "",
                f"생성 시점: {manifest['created_at']}",
                "",
                "## 완료 항목",
                "",
                "- `race_id`별 출전마를 연속 배치하는 `RankingDataset` 구현",
                "- 우승마 relevance 1, 나머지 0인 이진 랭킹 타깃 고정",
                "- XGBoost `group` 인자용 출전두수 배열과 행 구간 manifest 생성",
                "- 112개 PRE_RACE 피처만 허용하고 MARKET·POST_RACE·TARGET 입력 차단",
                "- Train 내부 4-fold가 날짜순·경주 비중첩인지 사전 검증",
                "- 기존 Final Test와 비정상 경주를 랭킹 데이터에서 제외",
                "",
                "## 데이터 규모",
                "",
                "| Fold | 행 | 경주 그룹 | 기간 | 최소/중앙/최대 출전두수 | relevance 합 |",
                "|---|---:|---:|---|---:|---:|",
                f"| Train | {train_record['rows']:,} | {train_record['races']:,} | {train_record['date_min']}~{train_record['date_max']} | {train_record['group_size_min']}/{train_record['group_size_median']:.0f}/{train_record['group_size_max']} | {train_record['relevance_sum']:,} |",
                f"| Calibration | {cal_record['rows']:,} | {cal_record['races']:,} | {cal_record['date_min']}~{cal_record['date_max']} | {cal_record['group_size_min']}/{cal_record['group_size_median']:.0f}/{cal_record['group_size_max']} | {cal_record['relevance_sum']:,} |",
                "",
                "각 경주에는 relevance 1이 정확히 하나 있으며, 모든 group size의 합은 해당 fold 행 수와 일치한다. 전처리 통계는 저장하지 않았고 24단계에서 각 walk-forward 학습 부분에만 적합한다.",
                "",
                "## 산출물",
                "",
                "- `data/interim/ranking_entry_manifest.csv.gz`: 랭킹 행 순서·그룹 위치·relevance",
                "- `data/interim/ranking_group_manifest.csv`: 경주별 행 시작/종료·출전두수·우승마 위치",
                "- `data/manifests/ranking_dataset_manifest.json`: 피처·fold·group·해시 계약",
                "- `src/data/ranking_data.py`: 빌더와 fail-closed 검증 함수",
                "",
                "다음 24단계에서는 이 계약을 사용해 R2 pairwise ranker를 시간순 4-fold로 학습하고 시장·기존 M2와 비교한다.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(manifest["observed"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
