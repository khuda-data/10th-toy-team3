"""Build leakage-safe revised_v5 features from revised_v3."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.common.tabular_data import load_splits, select_features


IN_DIR = Path("data/revised_v3")
OUT_DIR = Path("data/revised_v5")
SIMPLE_CATEGORICAL = ("sex", "weather", "rcDay", "budam", "born")
TARGET_ENCODED = ("jkName", "trName", "owName", "rank")
DROP_COLUMNS = ("meet", "fold", "wgBudamBigo", "track")
SMOOTHING = 20.0


def clean(value: object) -> str:
    value = "<NA>" if pd.isna(value) else str(value).strip()
    return value or "<NA>"


def safe(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "_", value).strip("_") or "NA"


def tools(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() in {"", "-"}:
        return []
    return sorted({x.strip() for x in str(value).split(",") if x.strip() and x.strip() != "-"})


def target_encode(train: pd.DataFrame, others: list[pd.DataFrame], column: str) -> None:
    """Race-wise chronological encoding: no row can use its own or future result."""
    prior = float(train["win"].mean())
    counts: dict[str, int] = {}; totals: dict[str, float] = {}
    result = pd.Series(index=train.index, dtype=float)
    ordered = train.sort_values(["rcDate", "race_id"], kind="stable")
    for _, race in ordered.groupby("race_id", sort=False):
        values = race[column].map(clean)
        for index, key in values.items():
            result.loc[index] = (totals.get(key, 0.0) + SMOOTHING * prior) / (counts.get(key, 0) + SMOOTHING)
        for key, y in zip(values, race["win"]):
            counts[key] = counts.get(key, 0) + 1; totals[key] = totals.get(key, 0.0) + float(y)
    train[f"te_{column}"] = result.fillna(prior)
    mapping = {key: (totals[key] + SMOOTHING * prior) / (counts[key] + SMOOTHING) for key in counts}
    for frame in others:
        frame[f"te_{column}"] = frame[column].map(clean).map(mapping).fillna(prior)


def transform(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict:
    frames = [train, valid, test]
    info: dict = {"one_hot": {}, "target_encoded": {}, "tool_columns": []}
    for column in SIMPLE_CATEGORICAL:
        levels = sorted(train[column].map(clean).unique())
        output = []
        for level in levels:
            name = f"oh_{column}_{safe(level)}"
            for frame in frames:
                frame[name] = (frame[column].map(clean) == level).astype("int8")
            output.append(name)
        for frame in frames:
            frame.drop(columns=column, inplace=True)
        info["one_hot"][column] = output
    for column in TARGET_ENCODED:
        target_encode(train, [valid, test], column)
        for frame in frames:
            frame.drop(columns=column, inplace=True)
        info["target_encoded"][column] = f"te_{column}"
    vocabulary = sorted({item for value in train["tool_set"] for item in tools(value)})
    for item in vocabulary:
        name = f"tool_{safe(item)}"
        for frame in frames:
            frame[name] = frame["tool_set"].map(lambda value: int(item in tools(value))).astype("int8")
        info["tool_columns"].append(name)
    for frame in frames:
        frame.drop(columns=["tool_set", *DROP_COLUMNS], errors="ignore", inplace=True)
    info["dropped"] = ["tool_set", *DROP_COLUMNS]
    return info


def main() -> None:
    train, valid, test = load_splits(str(IN_DIR / "train_revised_v3.csv"), str(IN_DIR / "valid_revised_v3.csv"), str(IN_DIR / "test_revised_v3.csv"))
    for frame in (train, valid, test):
        frame.drop(columns="_split", inplace=True)
    info = transform(train, valid, test)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for frame, name in ((train, "train_revised_v5.csv"), (valid, "valid_revised_v5.csv"), (test, "test_revised_v5.csv")):
        frame.to_csv(OUT_DIR / name, index=False, encoding="utf-8-sig")
    numeric, categorical = select_features(train, valid, test)
    report = {"dataset": "revised_v5", "source": "revised_v3", "fit_scope": "train only", "target_encoding": "chronological prior-race, smoothing=20", "string_transformations": info, "columns": len(train.columns), "model_features": len(numeric) + len(categorical)}
    (OUT_DIR / "preparation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
