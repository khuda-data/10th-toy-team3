from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.config import (  # noqa: E402
    CONFIG_DIR,
    MODEL_DIR,
    RANDOM_SEED,
    STABILITY_SEEDS,
    TABLE_DIR,
    ensure_output_dirs,
)
from src.data import load_fold, subset_and_target  # noqa: E402
from src.features import (  # noqa: E402
    assert_no_leakage,
    build_preprocessor,
    feature_manifest,
    select_features,
)
from src.metrics import classification_metrics  # noqa: E402
from src.modeling import Candidate, candidates, make_estimator, predict_scores  # noqa: E402


def choose_candidate(frame: pd.DataFrame) -> pd.Series:
    best_lift = frame["lift_10pct"].max()
    eligible = frame[frame["lift_10pct"].ge(best_lift - 0.02)].copy()
    family_order = {"logit": 0, "rf": 1, "xgb": 2}
    eligible["family_order"] = eligible["family"].map(family_order)
    return eligible.sort_values(
        ["pr_auc", "family_order"], ascending=[False, True]
    ).iloc[0]


def main() -> None:
    ensure_output_dirs()
    train_all = load_fold("train", include_outcomes=True)
    valid_all = load_fold("valid", include_outcomes=True)
    all_records: list[dict] = []
    draft: dict = {"primary_feature_set": "core", "selections": {}}

    for target_name in ("darkhorse", "favorite_bust"):
        train, y_train = subset_and_target(train_all, target_name)
        valid, y_valid = subset_and_target(valid_all, target_name)
        draft["selections"][target_name] = {}

        for feature_set in ("core", "history_plus"):
            X_train = select_features(train, feature_set)
            X_valid = select_features(valid, feature_set)
            X_valid = X_valid.reindex(columns=X_train.columns)
            assert_no_leakage(X_train, feature_set)

            manifest = feature_manifest(X_train, feature_set)
            (TABLE_DIR / f"features_{target_name}_{feature_set}.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            matrices: dict[str, tuple[object, object, object]] = {}
            for scale_key, scale_numeric in (("scaled", True), ("tree", False)):
                preprocessor = build_preprocessor(X_train, scale_numeric=scale_numeric)
                train_matrix = preprocessor.fit_transform(X_train)
                valid_matrix = preprocessor.transform(X_valid)
                matrices[scale_key] = (preprocessor, train_matrix, valid_matrix)

            fitted: dict[str, tuple[object, object, np.ndarray]] = {}
            for candidate in candidates():
                matrix_key = "scaled" if candidate.family == "logit" else "tree"
                preprocessor, train_matrix, valid_matrix = matrices[matrix_key]
                started = time.time()
                model = make_estimator(candidate, RANDOM_SEED)
                model.fit(train_matrix, y_train)
                scores = predict_scores(model, valid_matrix)
                metrics = classification_metrics(y_valid.to_numpy(), scores)
                record = {
                    "target": target_name,
                    "feature_set": feature_set,
                    "candidate": candidate.name,
                    "family": candidate.family,
                    "params": json.dumps(candidate.params, ensure_ascii=False),
                    "elapsed_seconds": time.time() - started,
                    **metrics,
                }
                all_records.append(record)
                fitted[candidate.name] = (preprocessor, model, scores)
                print(
                    f"[{target_name}/{feature_set}] {candidate.name}: "
                    f"Lift10={metrics['lift_10pct']:.3f}, PR-AUC={metrics['pr_auc']:.4f}, "
                    f"ROC-AUC={metrics['roc_auc']:.4f}",
                    flush=True,
                )

            group = pd.DataFrame(
                [r for r in all_records if r["target"] == target_name and r["feature_set"] == feature_set]
            )
            selected = choose_candidate(group)
            selected_name = str(selected["candidate"])
            selected_candidate = next(c for c in candidates() if c.name == selected_name)
            preprocessor, model, selected_scores = fitted[selected_name]

            # Screen the RF grid with 200 trees, then refit only the winner with 600.
            final_candidate = selected_candidate
            if selected_candidate.family == "rf":
                final_params = dict(selected_candidate.params)
                final_params["n_estimators"] = 600
                final_candidate = Candidate(selected_candidate.name, "rf", final_params)
                matrix_key = "tree"
                preprocessor, train_matrix, valid_matrix = matrices[matrix_key]
                model = make_estimator(final_candidate, RANDOM_SEED)
                model.fit(train_matrix, y_train)
                selected_scores = predict_scores(model, valid_matrix)
            final_metrics = classification_metrics(y_valid.to_numpy(), selected_scores)

            stability = []
            matrix_key = "scaled" if final_candidate.family == "logit" else "tree"
            _, train_matrix, valid_matrix = matrices[matrix_key]
            for seed in STABILITY_SEEDS:
                seeded_model = make_estimator(final_candidate, seed)
                seeded_model.fit(train_matrix, y_train)
                seeded_scores = predict_scores(seeded_model, valid_matrix)
                seeded_metrics = classification_metrics(y_valid.to_numpy(), seeded_scores)
                stability.append(
                    {
                        "seed": seed,
                        "lift_10pct": seeded_metrics["lift_10pct"],
                        "pr_auc": seeded_metrics["pr_auc"],
                        "roc_auc": seeded_metrics["roc_auc"],
                    }
                )

            threshold = float(np.quantile(selected_scores, 0.90))
            bundle_path = MODEL_DIR / f"{target_name}_{feature_set}_valid_bundle.joblib"
            joblib.dump(
                {
                    "preprocessor": preprocessor,
                    "model": model,
                    "candidate": final_candidate,
                    "feature_columns": X_train.columns.tolist(),
                    "valid_scores": selected_scores,
                    "valid_target": y_valid.to_numpy(),
                },
                bundle_path,
            )
            draft["selections"][target_name][feature_set] = {
                "candidate": selected_name,
                "family": final_candidate.family,
                "params": final_candidate.params,
                "search_n_estimators": selected_candidate.params.get("n_estimators"),
                "valid_metrics": {
                    key: float(final_metrics[key])
                    for key in ("roc_auc", "pr_auc", "brier", "lift_5pct", "lift_10pct", "lift_20pct")
                },
                "score_threshold_valid_top10": threshold,
                "stability": stability,
                "stability_lift10_mean": float(np.mean([x["lift_10pct"] for x in stability])),
                "stability_lift10_std": float(np.std([x["lift_10pct"] for x in stability], ddof=1)),
                "bundle": str(bundle_path),
            }

    result_frame = pd.DataFrame(all_records).sort_values(
        ["target", "feature_set", "lift_10pct", "pr_auc"],
        ascending=[True, True, False, False],
    )
    result_frame.to_csv(TABLE_DIR / "validation_model_comparison.csv", index=False, encoding="utf-8-sig")
    (CONFIG_DIR / "selection_draft.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OK] validation complete: {CONFIG_DIR / 'selection_draft.json'}", flush=True)


if __name__ == "__main__":
    main()
