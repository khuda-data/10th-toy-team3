from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import HISTORICAL_MARKET_COLUMNS, LEAKAGE_COLUMNS


def select_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in {"core", "history_plus"}:
        raise ValueError(f"Unknown feature set: {feature_set}")
    drops = set(LEAKAGE_COLUMNS)
    if feature_set == "core":
        drops |= HISTORICAL_MARKET_COLUMNS
    keep = [column for column in df.columns if column not in drops]
    X = df.loc[:, keep].copy()
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns
    X.loc[:, numeric] = X.loc[:, numeric].replace([np.inf, -np.inf], np.nan)
    categorical = X.select_dtypes(exclude=[np.number, "bool"]).columns
    for column in categorical:
        X[column] = X[column].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return X


def assert_no_leakage(X: pd.DataFrame, feature_set: str) -> None:
    forbidden = set(LEAKAGE_COLUMNS)
    if feature_set == "core":
        forbidden |= HISTORICAL_MARKET_COLUMNS
    remaining = sorted(forbidden & set(X.columns))
    if remaining:
        raise AssertionError(f"Leakage columns remain: {remaining}")


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]

    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, categorical)],
        remainder="drop",
        sparse_threshold=0.3,
    )


def feature_manifest(X: pd.DataFrame, feature_set: str) -> dict:
    categorical = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return {
        "feature_set": feature_set,
        "raw_feature_count": len(X.columns),
        "numeric_count": len(numeric),
        "categorical_count": len(categorical),
        "numeric": numeric,
        "categorical": categorical,
    }
