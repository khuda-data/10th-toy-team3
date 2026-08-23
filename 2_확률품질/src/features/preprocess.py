"""Train-only preprocessing for reviewed pre-race model features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.registry import assert_feature_list, select_premarket_features


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric columns using quantiles learned during fit only."""

    def __init__(self, lower: float = 0.005, upper: float = 0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


@dataclass(frozen=True)
class FeatureSchema:
    features: tuple[str, ...]
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_count": len(self.features),
            "numeric_count": len(self.numeric),
            "categorical_count": len(self.categorical),
            "features": list(self.features),
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
        }


def infer_feature_schema(frame: pd.DataFrame) -> FeatureSchema:
    features = tuple(select_premarket_features())
    assert_feature_list(features)
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing registered pre-race features: {missing}")
    numeric = tuple(
        name
        for name in features
        if is_numeric_dtype(frame[name]) or is_bool_dtype(frame[name])
    )
    categorical = tuple(name for name in features if name not in numeric)
    return FeatureSchema(features, numeric, categorical)


def model_frame(frame: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """Select the reviewed columns and normalize dtypes without fitting statistics."""
    missing = sorted(set(schema.features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    result = frame.loc[:, schema.features].copy()
    for column in schema.numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in schema.categorical:
        result[column] = result[column].astype("string").fillna("__MISSING__")
    return result


def make_preprocessor(schema: FeatureSchema, *, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(schema.numeric)),
            ("categorical", categorical_pipeline, list(schema.categorical)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )
