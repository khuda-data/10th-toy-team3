"""Validate the model matched to each preprocessing version on all non-market predictors."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


EXCLUDE = {
    # post-race results / labels
    "win", "ord", "fin_rank", "fin_pct", "place", "resid", "upset_A", "upset_B", "upset",
    # current-market amounts and market-derived variables
    "winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse", "gap_h", "gap_d",
    # technical identifiers rather than predictors
    "race_id", "entry_id", "fold",
}
FORCE_CATEGORICAL = {"hrNo", "jkNo", "trNo", "owNo"}


def read(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def is_number(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def make_matrices(train, valid, features):
    categoricals = []
    for feature in features:
        values = [row[feature] for row in train if row[feature] != ""]
        if feature in FORCE_CATEGORICAL or (values and sum(is_number(value) for value in values) / len(values) < 0.95):
            categoricals.append(feature)
    numerics = [feature for feature in features if feature not in categoricals]
    medians = {}
    for feature in numerics:
        values = [float(row[feature]) for row in train if is_number(row[feature])]
        medians[feature] = float(np.median(values)) if values else 0.0

    def numeric_matrix(rows):
        result = np.empty((len(rows), len(numerics)), dtype=np.float32)
        for i, row in enumerate(rows):
            for j, feature in enumerate(numerics):
                try:
                    value = float(row[feature])
                    result[i, j] = value if math.isfinite(value) else medians[feature]
                except ValueError:
                    result[i, j] = medians[feature]
        return result

    def cat_matrix(rows):
        return np.asarray([[row[feature] or "__MISSING__" for feature in categoricals] for row in rows], dtype=object)

    x_num_train, x_num_valid = numeric_matrix(train), numeric_matrix(valid)
    if categoricals:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        x_cat_train = encoder.fit_transform(cat_matrix(train))
        x_cat_valid = encoder.transform(cat_matrix(valid))
        return sparse.hstack([sparse.csr_matrix(x_num_train), x_cat_train]).tocsr(), sparse.hstack([sparse.csr_matrix(x_num_valid), x_cat_valid]).tocsr(), len(categoricals)
    return sparse.csr_matrix(x_num_train), sparse.csr_matrix(x_num_valid), 0


def locate(version: int, prefix: str):
    files = sorted(Path(f"v{version}").glob(f"{prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"v{version}: {prefix} CSV not found")
    return files[0]


def main():
    specs = {
        2: ("Logistic Regression", "logistic"),
        3: ("KNN (SVD-reduced one-hot features)", "knn"),
        4: ("Logistic Regression", "logistic"),
        5: ("XGBoost", "xgboost"),
        6: ("Logistic Regression", "logistic"),
        7: ("KNN (SVD-reduced one-hot features)", "knn"),
        8: ("Logistic Regression", "logistic"),
    }
    results = []
    for version, (label, kind) in specs.items():
        train, valid = read(locate(version, "train")), read(locate(version, "valid"))
        features = [column for column in train[0] if column in valid[0] and column not in EXCLUDE]
        x_train, x_valid, category_count = make_matrices(train, valid, features)
        y_train = np.asarray([int(float(row["win"])) for row in train])
        y_valid = np.asarray([int(float(row["win"])) for row in valid])
        if kind == "logistic":
            model = LogisticRegression(C=0.2, max_iter=2000, solver="lbfgs", n_jobs=4)
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_valid)[:, 1]
        elif kind == "knn":
            components = min(100, x_train.shape[1] - 1)
            svd = TruncatedSVD(n_components=components, random_state=42)
            z_train, z_valid = svd.fit_transform(x_train), svd.transform(x_valid)
            model = KNeighborsClassifier(n_neighbors=75, weights="distance", n_jobs=4)
            model.fit(z_train, y_train)
            probability = model.predict_proba(z_valid)[:, 1]
        else:
            model = XGBClassifier(
                n_estimators=700, max_depth=4, learning_rate=0.035, min_child_weight=8,
                subsample=0.85, colsample_bytree=0.9, reg_lambda=8.0, reg_alpha=0.1,
                objective="binary:logistic", eval_metric="auc", random_state=42, n_jobs=4, tree_method="hist",
            )
            # XGBoost's histogram implementation accepts CSR matrices directly.
            model.fit(x_train, y_train)
            probability = model.predict_proba(x_valid)[:, 1]
        auc = float(roc_auc_score(y_valid, probability))
        item = {"version": f"v{version}", "model": label, "feature_count": len(features), "categorical_count": category_count, "valid_roc_auc": round(auc, 6)}
        results.append(item)
        print(json.dumps(item, ensure_ascii=False))
    Path("v2_to_v8_nonmarket_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
