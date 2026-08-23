"""Independent audit and supplemental reporting for the upset-feature experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.training.build_full_rank_all_bets_report import daily_block_roi_ci, markdown_table
from src.training.evaluate_bet_type_edge import BET_TYPES, combination_key, event_probability
from src.training.train_full_rank_models import load_data
from src.training.train_market_anchor_same_test import attach_market, probability_from_margin, sha256


POOL_FEATURES = {"winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse"}
FRACTIONS = (0.10, 0.20, 0.30, 0.40)
MODEL_FILES = {
    "full_upset_base_margin": "full_upset_base_margin.joblib",
    "selected_upset_base_margin": "selected_upset_base_margin.joblib",
}


def check(condition: bool, label: str, checks: list[dict]) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL"})
    if not condition:
        raise AssertionError(label)


def file_evidence(path: Path) -> dict:
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256(path),
    }


def attach_upset(meta: dict[str, pd.DataFrame], market_path: Path) -> dict[str, pd.DataFrame]:
    extra = pd.read_csv(
        market_path, compression="infer", usecols=["entry_id", "pop_pct", "upset_B"], low_memory=False
    )
    extra["entry_id"] = extra["entry_id"].astype(str)
    if extra["entry_id"].duplicated().any():
        raise AssertionError("market reference entry_id duplicate")
    result = {}
    for split, frame in meta.items():
        joined = frame.copy()
        joined["entry_id"] = joined["entry_id"].astype(str)
        joined = joined.merge(extra, on="entry_id", how="left", validate="one_to_one")
        joined["top3_upset"] = (
            joined["pop_pct"].ge(0.5) & joined["finish_order"].le(3)
        ).astype(int)
        result[split] = joined
    return result


def independently_clean(
    data: dict[str, pd.DataFrame], meta: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict]:
    clean_data, clean_meta, audit = {}, {}, {}
    for split in ("train", "valid", "test"):
        keep = []
        reasons = {"incomplete_order": 0, "invalid_win_odds": 0, "invalid_place_market": 0}
        for race_id, positions_raw in meta[split].groupby("race_id", sort=False).indices.items():
            positions = np.asarray(positions_raw, dtype=int)
            race = meta[split].iloc[positions]
            if not race["rank_group_status"].eq("complete_1_to_n").all():
                reasons["incomplete_order"] += 1
            elif not (
                np.isfinite(race["winOdds"]).all()
                and race["winOdds"].between(1.0, 9999.0, inclusive="neither").all()
            ):
                reasons["invalid_win_odds"] += 1
            elif not (
                np.isfinite(race["plcOdds"]).all()
                and race["plcOdds"].gt(0).all()
                and np.isfinite(race["q_plc"]).all()
            ):
                reasons["invalid_place_market"] += 1
            else:
                keep.append(str(race_id))
        mask = meta[split]["race_id"].astype(str).isin(keep).to_numpy()
        clean_data[split] = data[split].loc[mask].reset_index(drop=True)
        clean_meta[split] = meta[split].loc[mask].reset_index(drop=True)
        audit[split] = {
            "rows": int(mask.sum()), "races": int(clean_meta[split]["race_id"].nunique()),
            "excluded_races": reasons,
        }
    return clean_data, clean_meta, audit


def place_probabilities(meta: pd.DataFrame, p_win: np.ndarray) -> np.ndarray:
    result = np.zeros(len(meta), dtype=float)
    for positions_raw in meta.groupby("race_id", sort=False).indices.values():
        positions = np.asarray(positions_raw, dtype=int)
        slots = int(meta.iloc[positions]["place"].sum())
        for local, global_position in enumerate(positions):
            result[global_position] = event_probability(p_win[positions], (local,), "연승", slots)
    return result


def bh_q_values(probability_positive: pd.Series, bootstraps: int = 5000) -> np.ndarray:
    # Add-one smoothing prevents an impossible exact zero p-value from finite resampling.
    successes = np.rint(probability_positive.to_numpy(dtype=float) * bootstraps)
    p_values = (bootstraps - successes + 1.0) / (bootstraps + 1.0)
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def edge_accuracy(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["split", "model_key", "fraction", "fraction_label", "bet_type"]
    for values, group in bets.groupby(keys, sort=False):
        split, model_key, fraction, fraction_label, bet_type = values
        y = group["model_hit"].to_numpy(dtype=float)
        p = group["model_probability"].to_numpy(dtype=float)
        q = group["market_probability"].to_numpy(dtype=float)
        positive = group[group["predicted_edge"].gt(0) & group["actual_odds_available_for_selection"]]
        rows.append({
            "split": split, "model_key": model_key, "fraction": float(fraction),
            "fraction_label": fraction_label, "bet_type": bet_type, "tickets": int(len(group)),
            "observed_hit_rate": float(y.mean()), "mean_model_probability": float(p.mean()),
            "mean_market_probability": float(q.mean()), "mean_predicted_edge": float((p - q).mean()),
            "realized_market_residual": float((y - q).mean()),
            "model_brier": float(np.mean((p - y) ** 2)),
            "market_brier": float(np.mean((q - y) ** 2)),
            "brier_improvement_vs_market": float(np.mean((q - y) ** 2) - np.mean((p - y) ** 2)),
            "model_probability_mae": float(np.mean(np.abs(p - y))),
            "market_probability_mae": float(np.mean(np.abs(q - y))),
            "model_calibration_bias": float(np.mean(p - y)),
            "market_calibration_bias": float(np.mean(q - y)),
            "edge_hit_spearman": float(pd.Series(p - q).corr(pd.Series(y), method="spearman"))
                if len(np.unique(y)) > 1 and len(np.unique(p - q)) > 1 else None,
            "positive_ticket_edge_bets": int(len(positive)),
            "positive_ticket_edge_hit_rate": float(positive["model_hit"].mean()) if len(positive) else None,
            "positive_ticket_edge_roi": float(positive["realized_return"].mean()) if len(positive) else None,
            "market_probability_scope": (
                "direct final-odds probability" if bet_type in {"단승", "연승"}
                else "q-derived PL/Harville proxy; not combination-pool probability"
            ),
        })
    return pd.DataFrame(rows)


def locked_threshold_sensitivity(
    candidates: pd.DataFrame, cutoffs: pd.DataFrame, bets: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for model_key in MODEL_FILES:
        test_candidates = candidates[
            candidates["split"].eq("test") & candidates["model_key"].eq(model_key)
        ]
        top40 = bets[
            bets["split"].eq("test") & bets["model_key"].eq(model_key)
            & bets["fraction_label"].eq("top_40pct")
        ]
        for fraction in FRACTIONS:
            validation_cutoff = float(cutoffs[
                cutoffs["split"].eq("valid") & cutoffs["model_key"].eq(model_key)
                & np.isclose(cutoffs["fraction"], fraction)
            ]["score_threshold"].iloc[0])
            expected_entries = set(test_candidates.loc[
                test_candidates["upset_score"].ge(validation_cutoff), "entry_id"
            ].astype(str))
            for bet_type in BET_TYPES:
                selected = top40[
                    top40["bet_type"].eq(bet_type) & top40["entry_id"].astype(str).isin(expected_entries)
                    & top40["actual_odds_available_for_selection"]
                ].sort_values("race_id", kind="stable")
                if len(selected) != len(expected_entries):
                    raise AssertionError(
                        f"locked threshold selection not contained in exact top40: {model_key}/{fraction}/{bet_type}"
                    )
                low, high, positive = daily_block_roi_ci(
                    selected,
                    int(hashlib.sha256(f"locked|{model_key}|{fraction}|{bet_type}".encode()).hexdigest()[:8], 16),
                ) if len(selected) else (None, None, None)
                rows.append({
                    "model_key": model_key, "fraction": fraction,
                    "fraction_label": f"top_{int(fraction * 100)}pct_validation_cutoff",
                    "bet_type": bet_type, "validation_score_threshold": validation_cutoff,
                    "test_candidate_pool": int(len(test_candidates)), "test_selected": int(len(selected)),
                    "test_actual_coverage": float(len(selected) / len(test_candidates)),
                    "wins": int(selected["model_hit"].sum()),
                    "hit_rate": float(selected["model_hit"].mean()) if len(selected) else None,
                    "roi": float(selected["realized_return"].mean()) if len(selected) else None,
                    "roi_ci_low_daily_block": low, "roi_ci_high_daily_block": high,
                    "bootstrap_probability_positive": positive,
                })
    result = pd.DataFrame(rows)
    result["roi_fdr_q_value"] = bh_q_values(result["bootstrap_probability_positive"])
    result["mechanical_fdr_pass"] = (
        result["roi_ci_low_daily_block"].gt(0) & result["roi_fdr_q_value"].le(0.05)
    )
    return result


def format_stability(summary: pd.DataFrame, model_key: str) -> pd.DataFrame:
    frame = summary[summary["split"].eq("test") & summary["model_key"].eq(model_key)][[
        "fraction_label", "bet_type", "bets", "wins", "hit_rate", "roi",
        "roi_ci_low_daily_block", "roi_ci_high_daily_block", "flat_1pct_mdd",
        "per_bet_sharpe", "average_selected_odds", "average_winning_dividend",
        "roi_without_largest_winning_return", "roi_fdr_q_value",
    ]].copy()
    for column in ["hit_rate", "roi", "roi_ci_low_daily_block", "roi_ci_high_daily_block",
                   "flat_1pct_mdd", "roi_without_largest_winning_return", "roi_fdr_q_value"]:
        frame[column] = frame[column].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.2%}")
    for column in ["average_selected_odds", "average_winning_dividend", "per_bet_sharpe"]:
        frame[column] = frame[column].map(lambda x: "N/A" if pd.isna(x) else f"{x:.3f}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-reference", type=Path, required=True)
    parser.add_argument("--official-dividends", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    market_path = args.market_reference.resolve()
    official_path = args.official_dividends.resolve()
    model_root = args.model_root.resolve()
    out = args.report_root.resolve()
    checks: list[dict] = []

    manifest_path = data_root / "preprocessing_manifest.json"
    preprocessing_validation_path = data_root / "preprocessing_validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preprocessing_validation = json.loads(preprocessing_validation_path.read_text(encoding="utf-8"))
    source_features, data, metadata = load_data(data_root)
    meta = attach_upset(attach_market(metadata, market_path), market_path)
    data, meta, clean_audit = independently_clean(data, meta)
    features = [feature for feature in source_features if feature not in POOL_FEATURES]

    split_template = manifest["split_file_template"]
    input_files = {
        "preprocessing_manifest": file_evidence(manifest_path),
        "preprocessing_validation": file_evidence(preprocessing_validation_path),
        "market_reference": file_evidence(market_path),
        "official_dividends": file_evidence(official_path),
    }
    for split in ("train", "valid", "test"):
        input_files[f"{split}_data"] = file_evidence(data_root / split_template.format(split=split))
        input_files[f"{split}_metadata"] = file_evidence(data_root / f"{split}_metadata.csv")

    check(preprocessing_validation["status"] == "PASS", "upstream preprocessing validation is PASS", checks)
    check("train" in manifest["fit_scope"].lower(), "preprocessing fit scope is train-only", checks)
    check(not manifest["selection_used_validation_or_test_statistics"],
          "correlation/feature pruning did not use validation or test", checks)
    check(manifest["reference_sha256"] == input_files["market_reference"]["sha256"],
          "market reference SHA256 matches preprocessing manifest", checks)
    for split in ("train", "valid", "test"):
        expected = manifest["output_hashes"][split]
        check(input_files[f"{split}_data"]["sha256"] == expected["data_sha256"],
              f"{split} data SHA256 matches manifest", checks)
        check(input_files[f"{split}_metadata"]["sha256"] == expected["metadata_sha256"],
              f"{split} metadata SHA256 matches manifest", checks)
        matrix = data[split][features]
        check(all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes),
              f"{split} audited {len(features)}-feature matrix is numeric", checks)
        check(np.isfinite(matrix.to_numpy(dtype=float)).all(), f"{split} feature matrix is finite", checks)
        check(not meta[split]["entry_id"].astype(str).duplicated().any(), f"{split} entry_id unique", checks)
    experiment_manifest = json.loads((out / "experiment_manifest.json").read_text(encoding="utf-8"))
    expected_clean_counts = {
        split: int(experiment_manifest["clean_split_audit"][split]["clean_races"])
        for split in ("train", "valid", "test")
    }
    actual_clean_counts = {split: clean_audit[split]["races"] for split in clean_audit}
    check(actual_clean_counts == expected_clean_counts,
          f"independent clean race counts match primary run: {actual_clean_counts}", checks)
    check(max(meta["train"]["race_id"].astype(str).str[:8]) < min(meta["valid"]["race_id"].astype(str).str[:8]),
          "train strictly precedes validation", checks)
    check(max(meta["valid"]["race_id"].astype(str).str[:8]) < min(meta["test"]["race_id"].astype(str).str[:8]),
          "validation strictly precedes test", checks)
    check(not (set(features) & set(manifest["market_or_result_columns_excluded_from_features"])),
          "audited feature matrix excludes declared odds/results", checks)
    check(not (set(source_features) & POOL_FEATURES),
          "all five final-pool features are absent from the clean source matrix", checks)
    rebuilt = ((meta["test"]["pop_pct"] >= 0.5) & meta["test"]["place"].eq(1)).astype(int)
    check(rebuilt.eq(meta["test"]["upset_B"].astype(int)).all(), "test upset_B exactly reproduced", checks)

    corr = data["train"][features].corr(method="pearson").abs()
    np.fill_diagonal(corr.values, np.nan)
    maximum_correlation = float(np.nanmax(corr.to_numpy()))
    maximum_pair_index = np.unravel_index(np.nanargmax(corr.to_numpy()), corr.shape)
    maximum_pair = [str(corr.index[maximum_pair_index[0]]), str(corr.columns[maximum_pair_index[1]])]
    check(maximum_correlation <= float(manifest["correlation_threshold"]) + 1e-12,
          "post-pool-removal train absolute Pearson correlation <= 0.95", checks)

    predictions = pd.read_csv(out / "runner_probabilities.csv", dtype={"race_id": str, "entry_id": str})
    candidates = pd.read_csv(out / "upset_horse_candidates.csv", dtype={"race_id": str, "entry_id": str})
    cutoffs = pd.read_csv(out / "score_fraction_cutoffs.csv")
    bets = pd.read_csv(out / "upset_bet_details.csv", dtype={"race_id": str, "entry_id": str})
    summary = pd.read_csv(out / "upset_bet_summary.csv")
    importance = pd.read_csv(out / "upset_feature_importance.csv")
    selected_manifest = json.loads((out / "selected_upset_features.json").read_text(encoding="utf-8"))
    if bets["actual_odds_available_for_selection"].dtype != bool:
        bets["actual_odds_available_for_selection"] = bets["actual_odds_available_for_selection"].astype(str).eq("True")

    artifact_evidence = {}
    for model_key, filename in MODEL_FILES.items():
        artifact_path = model_root / filename
        artifact_evidence[filename] = file_evidence(artifact_path)
        artifact = joblib.load(artifact_path)
        artifact_features = list(artifact["features"])
        check(not (set(artifact_features) & POOL_FEATURES), f"{model_key} artifact excludes final-pool features", checks)
        check(not (set(artifact_features) & set(manifest["market_or_result_columns_excluded_from_features"])),
              f"{model_key} artifact excludes odds/results", checks)
        X = data["test"][artifact_features].to_numpy(dtype=float)
        lnq = np.log(np.clip(meta["test"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
        recomputed = probability_from_margin(meta["test"], lnq + artifact["model"].margin(X))
        saved = predictions[
            predictions["split"].eq("test") & predictions["model_key"].eq(model_key)
        ]["model_probability"].to_numpy(dtype=float)
        check(len(saved) == len(recomputed) and np.max(np.abs(saved - recomputed)) <= 1e-12,
              f"{model_key} artifact independently reproduces saved test probabilities", checks)
        maximum_sum_error = max(
            abs(float(recomputed[np.asarray(pos, dtype=int)].sum()) - 1.0)
            for pos in meta["test"].groupby("race_id", sort=False).indices.values()
        )
        check(maximum_sum_error <= 1e-12, f"{model_key} test race probabilities sum to one", checks)
        p_place = place_probabilities(meta["test"], recomputed)
        output_place = predictions[
            predictions["split"].eq("test") & predictions["model_key"].eq(model_key)
        ]["model_place_probability"].to_numpy(dtype=float)
        check(np.max(np.abs(p_place - output_place)) <= 1e-12,
              f"{model_key} place probabilities independently reproduced", checks)

    check((candidates["pop_pct"] >= 0.5).all(), "all saved upset horses are market-bottom-half", checks)
    check((candidates["predicted_edge"] > 0).all(), "all saved upset horses have positive place EDGE", checks)
    check(candidates.groupby(["split", "model_key", "race_id"]).size().max() == 1,
          "at most one upset horse per race", checks)
    check(set(cutoffs["fraction"].round(2)) == set(FRACTIONS), "all exact 10/20/30/40 fractions exist", checks)
    exact_count_ok = True
    ticket_count_ok = True
    ticket_contains_upset = True
    for cutoff in cutoffs.itertuples(index=False):
        expected = int(math.ceil(int(cutoff.candidate_pool) * float(cutoff.fraction)))
        exact_count_ok &= expected == int(cutoff.selected_target_count)
        for bet_type in BET_TYPES:
            group = bets[
                bets["split"].eq(cutoff.split) & bets["model_key"].eq(cutoff.model_key)
                & np.isclose(bets["fraction"], cutoff.fraction) & bets["bet_type"].eq(bet_type)
            ]
            ticket_count_ok &= len(group) == expected
            ticket_contains_upset &= all(
                str(int(row.chulNo)) in str(row.selection_chul_numbers).split("-")
                for row in group.itertuples(index=False)
            )
    check(exact_count_ok, "cutoff selected counts equal ceil(candidate pool*fraction)", checks)
    check(ticket_count_ok, "every model/fraction/bet has the exact selected horse count", checks)
    check(ticket_contains_upset, "every seven-bet ticket contains its selected upset horse", checks)

    official = pd.read_csv(official_path, dtype={"race_id": str, "winning_combination": str})
    official_lookup = {
        (str(row.race_id), row.bet_type,
         combination_key([int(x) for x in str(row.winning_combination).split("-")], row.bet_type)):
            float(row.dividend)
        for row in official.itertuples(index=False)
    }
    official_winner_ok = True
    return_formula_ok = True
    for row in bets.itertuples(index=False):
        expected_return = float(row.model_hit * row.actual_odds - 1.0)
        return_formula_ok &= abs(expected_return - float(row.realized_return)) <= 1e-12
        if row.model_hit and row.bet_type not in {"단승", "연승"}:
            combo = combination_key([int(x) for x in str(row.selection_chul_numbers).split("-")], row.bet_type)
            official_winner_ok &= abs(
                official_lookup[(str(row.race_id), row.bet_type, combo)] - float(row.actual_odds)
            ) <= 1e-12
    check(return_formula_ok, "all realized returns independently equal hit*official-odds minus one", checks)
    check(official_winner_ok, "all winning combination dividends match official KRA file", checks)
    summary_ok = True
    for key, group in bets[bets["actual_odds_available_for_selection"]].groupby(
        ["split", "model_key", "fraction", "bet_type"], sort=False
    ):
        split, model_key, fraction, bet_type = key
        row = summary[
            summary["split"].eq(split) & summary["model_key"].eq(model_key)
            & np.isclose(summary["fraction"], fraction) & summary["bet_type"].eq(bet_type)
        ].iloc[0]
        summary_ok &= int(row["bets"]) == len(group)
        summary_ok &= int(row["wins"]) == int(group["model_hit"].sum())
        summary_ok &= abs(float(row["roi"]) - float(group["realized_return"].mean())) <= 1e-12
    check(summary_ok, "all saved bet counts, wins, and ROI independently recomputed", checks)
    expected_roi_cells = len(MODEL_FILES) * len(FRACTIONS) * len(BET_TYPES)
    check(len(summary[summary["split"].eq("test")]) == expected_roi_cells,
          f"test contains {expected_roi_cells} model/fraction/bet-type ROI cells", checks)
    test_summary_flags = summary[summary["split"].eq("test")]

    def as_bool(series: pd.Series) -> pd.Series:
        if pd.api.types.is_bool_dtype(series):
            return series
        return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})

    check(as_bool(test_summary_flags["profit_confirmed_fdr_5pct"]).eq(
        as_bool(test_summary_flags["profit_confirmed_95pct"])
        & test_summary_flags["roi_fdr_q_value"].le(0.05)
    ).all(), "saved FDR profit flags exactly follow CI and q-value rule", checks)

    selected_features = selected_manifest["selected_features"]
    calculated_selected = importance.loc[
        importance["positive_importance_weight"].ge(selected_manifest["positive_importance_weight_threshold"])
        & importance["mean_upset_tail_realized_edge_drop"].gt(0), "feature"
    ].tolist()
    check(selected_features == calculated_selected, "selected feature list exactly follows reported 0.5% rule", checks)
    check(len(selected_features) == int(selected_manifest["selected_feature_count"]),
          f"reported operational threshold selects {len(selected_features)} features", checks)
    statistically_positive = importance[importance["ci_low_daily_block"].gt(0)]["feature"].tolist()

    # Correct a known floating-point display artifact: a constant return series
    # has zero variance, so its Sharpe-like statistic is undefined rather than
    # a huge signed number caused by a residual ~1e-18 standard deviation.
    summary_verified = summary.copy()
    for index, row in summary_verified.iterrows():
        group = bets[
            bets["split"].eq(row["split"]) & bets["model_key"].eq(row["model_key"])
            & np.isclose(bets["fraction"], row["fraction"]) & bets["bet_type"].eq(row["bet_type"])
            & bets["actual_odds_available_for_selection"]
        ]
        returns = group["realized_return"].to_numpy(dtype=float)
        if len(returns) <= 1 or float(np.ptp(returns)) <= 1e-12:
            summary_verified.loc[index, "per_bet_sharpe"] = np.nan
    summary = summary_verified
    summary.to_csv(out / "upset_bet_summary_independently_verified.csv", index=False, encoding="utf-8-sig")

    edge = edge_accuracy(bets)
    edge.to_csv(out / "edge_accuracy_summary.csv", index=False, encoding="utf-8-sig")
    locked = locked_threshold_sensitivity(candidates, cutoffs, bets)
    locked.to_csv(out / "validation_locked_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")

    summary_test = summary[summary["split"].eq("test")].copy()
    high_roi = summary_test[summary_test["roi"].ge(0.30)].sort_values("roi", ascending=False)
    best = summary_test.sort_values("roi", ascending=False).head(12)[[
        "model", "fraction_label", "bet_type", "bets", "wins", "roi",
        "roi_ci_low_daily_block", "roi_ci_high_daily_block",
        "roi_without_largest_winning_return", "roi_fdr_q_value",
    ]]
    model_metrics = pd.read_csv(out / "model_delta_ll_metrics.csv")
    metrics_test = model_metrics[model_metrics["split"].eq("test")][[
        "model", "races", "delta_ll_per_race", "delta_ll_ci_low_daily_block",
        "delta_ll_ci_high_daily_block", "model_top1_hit_rate", "market_top1_hit_rate",
    ]]
    selected_importance = importance[importance["selected_for_retraining"]][[
        "rank", "feature", "positive_importance_weight", "mean_upset_tail_realized_edge_drop",
        "ci_low_daily_block", "ci_high_daily_block", "positive_repeat_rate",
    ]]
    direct_edge = edge[
        edge["split"].eq("test") & edge["bet_type"].isin(["단승", "연승"])
    ][[
        "model_key", "fraction_label", "bet_type", "tickets", "observed_hit_rate",
        "mean_model_probability", "mean_market_probability", "mean_predicted_edge",
        "brier_improvement_vs_market", "positive_ticket_edge_bets", "positive_ticket_edge_roi",
    ]]
    locked_overview = locked.groupby(["model_key", "fraction_label"], sort=False).first().reset_index()[[
        "model_key", "fraction_label", "validation_score_threshold", "test_candidate_pool",
        "test_selected", "test_actual_coverage",
    ]]

    validation = {
        "status": "PASS", "passed_checks": len(checks), "checks": checks,
        "input_files": input_files, "model_artifacts": artifact_evidence,
        "clean_split_audit": clean_audit,
        "post_pruning_max_abs_pearson": maximum_correlation,
        "post_pruning_max_abs_pearson_pair": maximum_pair,
        "selected_features_operational_0p5pct": selected_features,
        "importance_model_trees": int(selected_manifest["importance_split"]["n_trees"]),
        "importance_model_tree_used_feature_count": int(importance["tree_used_feature"].sum()),
        "importance_ci_positive_features": statistically_positive,
        "test_roi_cells": expected_roi_cells,
        "test_unadjusted_ci_positive_cells": int(summary_test["profit_confirmed_95pct"].sum()),
        "test_fdr_positive_cells": int(summary_test["profit_confirmed_fdr_5pct"].sum()),
        "test_roi_ge_30pct_cells": int(len(high_roi)),
        "warnings": [
            "The 0.5% importance cutoff is an operational normalized-weight rule, not a p-value.",
            (
                "No selected feature has a daily-block 95% importance CI lower bound above zero."
                if not statistically_positive else
                f"{len(statistically_positive)} feature(s) have a daily-block 95% importance CI lower bound above zero."
            ),
            "The test period was previously inspected; results are exploratory reuse, not fresh confirmation.",
            "Exact within-test top fractions are batch rankings, not a causal real-time threshold policy.",
            "Combination market probabilities are q-derived proxies because losing-ticket pool odds are absent.",
            "Final odds are not available at live bet placement time.",
        ],
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    (out / "independent_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    glossary = f"""## 지표 개념과 산식

- **이변 점수**: 시장 인기 하위 절반 말의 `P_model(연승권)-q_plc`.
- **순열 중요도**: 피처 교란으로 상위 이변 말들의 실현 `place-q_plc`가 감소한 양.
- **0.5% 가중치**: 양의 중요도 총합 중 피처 비중을 뜻하는 운영상 선별선이며 p-value가 아니다.
- **ΔLL/경주**: 모델과 시장의 승자 로그확률 차이. 양수면 모델 확률분포가 시장보다 정보량이 많다.
- **ROI**: `(환급-원금)/원금`. 공식 최종 환급배당에는 공제가 반영되어 20%·27%를 재차 빼지 않았다.
- **MDD**: 1% 고정 지분 누적자산의 최대 고점 대비 하락률.
- **Sharpe-like**: 베팅별 평균수익/표준편차×√베팅수. 연율화 Sharpe가 아니라 표본 안정성 통계다.
- **FDR**: {expected_roi_cells}개 ROI 셀의 다중비교를 보정한다. 현재 test 재사용 때문에 통과해도 독립 확정은 아니다.
"""
    report = f"""# 이변 실험 독립 검증·EDGE·7승식 상세 보고서

{glossary}
## 독립 검증 결론

- 독립 재계산 검사 **{len(checks)}개가 전부 PASS**했다.
- exact top 10·20·30·40%의 test {expected_roi_cells}개 ROI 셀 중 보정 전 CI 하한 양수는 **{int(summary_test['profit_confirmed_95pct'].sum())}개**, FDR까지 통과한 셀은 **{int(summary_test['profit_confirmed_fdr_5pct'].sum())}개**다.
- ROI 30% 이상 점추정 셀은 **{len(high_roi)}개**지만 모두 독립 수익확정이 아니다.
- 0.5% 운영 가중치로 선택된 피처는 **{len(selected_features)}개**다: {', '.join(selected_features)}. 중요도용 조기중단 모델은 **{int(selected_manifest['importance_split']['n_trees'])}개 트리**였고 실제 분기에 사용된 피처는 **{int(importance['tree_used_feature'].sum())}개**다. 중요도 CI 하한이 0보다 큰 피처는 **{len(statistically_positive)}개**이며, 나머지 선택 피처를 통계적으로 유의하다고 표현하지 않는다.

## 데이터 무결성

- 완전착순·유효 단승/연승시장 공통 표본: train **{clean_audit['train']['races']}**, validation **{clean_audit['valid']['races']}**, test **{clean_audit['test']['races']}**경주.
- {len(features)}개 입력은 모두 수치·유한값이고 결과/배당 열과 최종 풀 5개가 모델 행렬에 없다.
- train 최대 절대 Pearson 상관은 `{maximum_pair[0]}`-`{maximum_pair[1]}`의 **{maximum_correlation:.6f}**로 사전 기준 0.95 이하이다.
- 입력 절대경로·bytes·mtime·SHA-256은 `independent_validation.json`에 저장했다.

## 시장 대비 확률정보

{markdown_table(metrics_test)}

## 0.5% 운영 가중치 통과 피처

{markdown_table(selected_importance)}

## 점추정 ROI 상위 셀과 고배당 1건 제거 민감도

{markdown_table(best)}

CI와 FDR이 모두 통과한 셀은 없다. `roi_without_largest_winning_return`이 크게 하락하거나 음수로 바뀌는 셀은 한 번의 고환급 적중 의존성이 높다.

## 전체 {len(features)}피처 모델: 7승식 × 상위비율 안정성

{markdown_table(format_stability(summary, 'full_upset_base_margin'))}

## {len(selected_features)}개 선별피처 재학습 모델: 7승식 × 상위비율 안정성

{markdown_table(format_stability(summary, 'selected_upset_base_margin'))}

복합 승식의 `average_selected_odds`는 패배 조합의 실제 사전 배당이 없어 N/A다. `average_winning_dividend`만 공식 적중 환급에서 계산했다. 같은 이유로 복합 Kelly는 계산하지 않았다.

## 단승·연승 EDGE 정확도와 시장 비교

{markdown_table(direct_edge)}

전체 7승식 결과는 `edge_accuracy_summary.csv`에 있다. 단승·연승은 실제 최종배당 기반 시장확률과 직접 비교했으며, 나머지는 q 기반 PL/Harville proxy라 실제 조합시장 EDGE가 아니다.

## validation 절대 임계값을 test에 적용한 배치 민감도

{markdown_table(locked_overview)}

요청한 exact 비율 분석은 test 기간 전체의 점수 순위를 쓰므로 실시간 정책이 아니다. 위 표는 validation 절대 점수선을 test에 그대로 적용했을 때의 실제 test 적용률이며, 상세 ROI는 `validation_locked_threshold_sensitivity.csv`에 있다. 이 민감도에서도 FDR 통과 셀은 **{int(locked['mechanical_fdr_pass'].sum())}개**다.

## 판정과 다음 검증

이 실행은 모델 파일 재로딩, 확률합, 공식환급, 티켓 구성, ROI를 재계산해 기계적 무결성을 확인했다. 그러나 기존 test를 다시 사용했고 최종배당을 사용했으므로 실전 수익성은 확인되지 않았다. 다음 신규 기간에서 사전에 잠근 validation 절대 임계값과 실제 베팅 시점 배당 스냅샷으로만 최종 확인해야 한다.
"""
    (out / "03_independent_validation_and_edge_report.md").write_text(report, encoding="utf-8")
    print(f"INDEPENDENT UPSET VALIDATION PASS: checks={len(checks)}")
    print(json.dumps({
        "clean_races": {key: value["races"] for key, value in clean_audit.items()},
        "selected_features": selected_features,
        "importance_ci_positive_features": statistically_positive,
        "test_fdr_positive_cells": int(summary_test["profit_confirmed_fdr_5pct"].sum()),
        "locked_fdr_positive_cells": int(locked["mechanical_fdr_pass"].sum()),
        "roi_ge_30pct_cells": int(len(high_roi)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
