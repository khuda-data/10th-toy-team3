"""Enforce the approved dataset policy for new model pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.load_raw import PROJECT_ROOT


DEFAULT_POLICY_PATH = PROJECT_ROOT / "data" / "manifests" / "dataset_policy.json"


def load_dataset_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def dataset_record(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()

    policy = load_dataset_policy()
    matches: list[tuple[int, dict[str, Any]]] = []
    for record in policy["datasets"]:
        target = (PROJECT_ROOT / record["path"]).resolve()
        if candidate == target or target in candidate.parents:
            matches.append((len(target.parts), record))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def assert_dataset_allowed(path: str | Path) -> None:
    record = dataset_record(path)
    if record is None:
        raise ValueError(f"Dataset is not registered in dataset_policy.json: {path}")
    if record["status"] != "canonical":
        raise ValueError(
            f"Dataset is not allowed for new modeling: {record['path']} "
            f"(status={record['status']}; {record['usage']})"
        )
