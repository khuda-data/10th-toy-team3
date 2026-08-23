"""revised_v5를 누수 없는 완전 숫자형 모델 행렬로 전처리한다."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.common.column_policy import ID_COLS, LEAK_COLS


SPLITS = ("train", "valid", "test")
SOURCE = Path("data/revised_v5")
OUTPUT = Path("data/revised_v5_preprocessed")
TARGET = "win"
KEY_COLUMNS = ["rcDate", "race_id", "entry_id", "hrName", TARGET]


def global_codes(frames: dict[str, pd.DataFrame], column: str) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    values = pd.concat([frames[split][column].astype(str) for split in SPLITS], ignore_index=True)
    mapping = {value: index for index, value in enumerate(sorted(values.unique()))}
    return mapping, {
        split: frames[split][column].astype(str).map(mapping).to_numpy(dtype=np.int64)
        for split in SPLITS
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    source_dtypes = {}
    for split in SPLITS:
        frame = pd.read_csv(SOURCE / f"{split}_revised_v5.csv", low_memory=False)
        if frame.columns.duplicated().any():
            raise ValueError(f"{split}: 중복 열 이름")
        if frame["entry_id"].astype(str).duplicated().any():
            raise ValueError(f"{split}: entry_id 중복")
        if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{split}: 타깃이 이진값이 아님")
        frame = frame.sort_values(["rcDate", "race_id", "entry_id"], kind="stable").reset_index(drop=True)
        frames[split] = frame
        source_dtypes[split] = {column: str(dtype) for column, dtype in frame.dtypes.items()}

    if any(list(frames[split].columns) != list(frames["train"].columns) for split in SPLITS):
        raise ValueError("train/valid/test 열 순서 또는 스키마가 다릅니다.")

    banned = LEAK_COLS | ID_COLS | {TARGET}
    candidates = [column for column in frames["train"].columns if column not in banned]
    combined = pd.concat([frames[split][candidates] for split in SPLITS], ignore_index=True)
    numeric_columns, categorical_columns, conversion_failures = [], [], {}
    for column in candidates:
        converted = pd.to_numeric(combined[column], errors="coerce")
        failures = int((combined[column].notna() & converted.isna()).sum())
        if failures:
            categorical_columns.append(column)
            conversion_failures[column] = failures
        else:
            numeric_columns.append(column)
    constants = [
        column for column in numeric_columns
        if pd.to_numeric(frames["train"][column], errors="coerce").nunique(dropna=True) <= 1
    ]
    if constants:
        numeric_columns = [column for column in numeric_columns if column not in constants]

    prepared = {}
    for split in SPLITS:
        block = frames[split][numeric_columns + categorical_columns].copy()
        for column in numeric_columns:
            original = block[column]
            numeric = pd.to_numeric(original, errors="coerce")
            failed = int((original.notna() & numeric.isna()).sum())
            if failed:
                raise ValueError(f"{split}/{column}: 숫자 변환 실패 {failed}건")
            block[column] = numeric.replace([np.inf, -np.inf], np.nan)
        for column in categorical_columns:
            block[column] = block[column].astype("string")
        prepared[split] = block

    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)),
        ]), categorical_columns),
    ], remainder="drop", verbose_feature_names_out=False)
    transformer.fit(prepared["train"])
    output_features = transformer.get_feature_names_out().tolist()
    joblib.dump(transformer, OUTPUT / "revised_v5_preprocessor.joblib")

    id_mappings, id_codes = {}, {}
    for column in ("race_id", "entry_id", "hrName"):
        mapping, codes = global_codes(frames, column)
        id_mappings[column] = mapping
        id_codes[column] = codes
    (OUTPUT / "identifier_mappings.json").write_text(
        json.dumps(id_mappings, ensure_ascii=False), encoding="utf-8"
    )

    results = {}
    for split in SPLITS:
        matrix = transformer.transform(prepared[split])
        features = pd.DataFrame(matrix, columns=output_features)
        result = pd.DataFrame({
            "row_id": np.arange(len(frames[split]), dtype=np.int64),
            "rcDate": frames[split]["rcDate"].to_numpy(dtype=np.int64),
            "race_id_code": id_codes["race_id"][split],
            "entry_id_code": id_codes["entry_id"][split],
            "hrName_code": id_codes["hrName"][split],
            TARGET: frames[split][TARGET].to_numpy(dtype=np.int8),
        })
        result = pd.concat([result, features], axis=1)
        result.to_csv(OUTPUT / f"{split}_revised_v5_numeric_scaled.csv.gz", index=False, encoding="utf-8-sig")
        frames[split][KEY_COLUMNS].to_csv(
            OUTPUT / f"{split}_metadata.csv.gz", index=False, encoding="utf-8-sig"
        )
        results[split] = result

    numeric_train = results["train"][numeric_columns]
    validation = {
        "status": "PASS",
        "source_shapes": {split: list(frames[split].shape) for split in SPLITS},
        "output_shapes": {split: list(results[split].shape) for split in SPLITS},
        "numeric_source_features": len(numeric_columns),
        "categorical_source_features": categorical_columns,
        "one_hot_output_features": len(output_features) - len(numeric_columns),
        "total_model_features": len(output_features),
        "removed_constant_features": constants,
        "source_dtype_mismatches": {
            column: {split: source_dtypes[split][column] for split in SPLITS}
            for column in frames["train"].columns
            if len({source_dtypes[split][column] for split in SPLITS}) > 1
        },
        "all_columns_numeric": {}, "nan_count": {}, "infinite_count": {},
        "rows_preserved": {}, "unique_entry_codes": {},
        "train_numeric_max_abs_mean": float(numeric_train.mean().abs().max()),
        "train_numeric_max_abs_std_error": float((numeric_train.std(ddof=0) - 1).abs().max()),
    }
    for split in SPLITS:
        result = results[split]
        array = result.to_numpy(dtype=float)
        validation["all_columns_numeric"][split] = all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)
        validation["nan_count"][split] = int(np.isnan(array).sum())
        validation["infinite_count"][split] = int(np.isinf(array).sum())
        validation["rows_preserved"][split] = len(result) == len(frames[split])
        validation["unique_entry_codes"][split] = result["entry_id_code"].nunique() == len(result)
    checks = [
        *validation["all_columns_numeric"].values(),
        *(value == 0 for value in validation["nan_count"].values()),
        *(value == 0 for value in validation["infinite_count"].values()),
        *validation["rows_preserved"].values(), *validation["unique_entry_codes"].values(),
        validation["train_numeric_max_abs_mean"] < 1e-10,
        validation["train_numeric_max_abs_std_error"] < 1e-10,
    ]
    if not all(checks):
        validation["status"] = "FAIL"
    (OUTPUT / "preprocessing_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "fit_scope": "train only",
        "numeric_pipeline": "strict numeric conversion -> median imputation -> StandardScaler",
        "categorical_pipeline": "most-frequent imputation -> OneHotEncoder(handle_unknown=ignore)",
        "model_features": output_features,
        "numeric_source_features": numeric_columns,
        "categorical_source_features": categorical_columns,
        "excluded_leak_or_identifier_columns": sorted(column for column in frames["train"].columns if column in banned),
        "excluded_output_columns": ["row_id", "rcDate", "race_id_code", "entry_id_code", "hrName_code", TARGET],
        "validation": str(OUTPUT / "preprocessing_validation.json"),
    }
    (OUTPUT / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if validation["status"] != "PASS":
        raise RuntimeError("revised_v5 전처리 사후 검증 실패")
    print(f"PASS: revised_v5 -> 숫자 피처 {len(output_features)}개")
    print(f"numeric={len(numeric_columns)}, categorical={categorical_columns}")
    print(f"max|train mean|={validation['train_numeric_max_abs_mean']:.3e}")
    print(f"max|train std-1|={validation['train_numeric_max_abs_std_error']:.3e}")


if __name__ == "__main__":
    main()
