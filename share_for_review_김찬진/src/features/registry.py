"""Read and enforce the static reviewed feature registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.data.load_raw import PROJECT_ROOT


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "data" / "manifests" / "feature_registry.json"
FORBIDDEN_MODEL_ROLES = {"ID", "SPLIT", "MARKET", "POST_RACE", "TARGET", "LEGACY", "CONTROL"}


def load_feature_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_registry_columns(columns: Iterable[str]) -> None:
    registry = load_feature_registry()
    registered = set(registry["features"])
    actual = set(columns)
    unknown = sorted(actual - registered)
    missing = sorted(registered - actual)
    if unknown or missing:
        raise ValueError(f"Feature registry mismatch: unknown={unknown}, missing={missing}")


def select_premarket_features(columns: Iterable[str] | None = None) -> list[str]:
    registry = load_feature_registry()
    selected = [
        name
        for name, meta in registry["features"].items()
        if meta["role"] == "PRE_RACE"
    ]
    if columns is not None:
        validate_registry_columns(columns)
    return selected


def assert_feature_list(features: Iterable[str], *, model_kind: str = "premarket") -> None:
    registry = load_feature_registry()
    unknown = sorted(set(features) - set(registry["features"]))
    if unknown:
        raise ValueError(f"Unregistered feature columns: {unknown}")
    forbidden = {
        name: registry["features"][name]["role"]
        for name in features
        if registry["features"][name]["role"] in FORBIDDEN_MODEL_ROLES
    }
    if forbidden:
        raise ValueError(f"Forbidden {model_kind} model features: {forbidden}")
