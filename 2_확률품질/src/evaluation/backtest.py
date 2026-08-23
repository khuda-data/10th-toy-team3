"""Closing-odds economic backtest with Calibration-only threshold selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries
from src.data.validate_schema import sha256_file


THRESHOLDS = (0.05, 0.10, 0.15)
MIN_BETS = 100
N_BOOTSTRAP = 5000
RANDOM_SEED = 42
INITIAL_BANKROLL = 100.0
KELLY_FRACTION = 0.10
MAX_SINGLE_BET_FRACTION = 0.01

CALIBRATION_PREDICTIONS = (
    PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_calibration_final.csv.gz"
)
TEST_PREDICTIONS = (
    PROJECT_ROOT / "data" / "predictions" / "m2_xgboost_test_final.csv.gz"
)
FINAL_TEST_MANIFEST = (
    PROJECT_ROOT / "data" / "manifests" / "final_test_evaluation.json"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "experiments" / "stage_17_backtest.json"
POLICY_PATH = PROJECT_ROOT / "data" / "manifests" / "betting_policy.json"
SELECTIONS_PATH = PROJECT_ROOT / "data" / "analysis" / "stage_17_bet_selections.csv.gz"


def attach_closing_odds(predictions: pd.DataFrame, fold: str) -> pd.DataFrame:
    entries = load_model_entries((fold,))
    odds = entries[["entry_id", "race_id", "rcDate", "win", "winOdds"]].copy()
    if odds["entry_id"].duplicated().any():
        raise ValueError("entry_id must be unique before odds merge")
    merged = predictions.merge(
        odds,
        on=["entry_id", "race_id", "rcDate"],
        how="left",
        validate="one_to_one",
    )
    if merged["winOdds"].isna().any() or (merged["winOdds"] <= 0).any():
        raise ValueError("Closing win odds must be present and positive")
    if not np.array_equal(merged["win_x"].to_numpy(), merged["win_y"].to_numpy()):
        raise ValueError("Prediction and canonical outcome columns disagree")
    merged = merged.drop(columns="win_y").rename(columns={"win_x": "win"})
    merged["break_even_prob"] = 1.0 / merged["winOdds"]
    merged["expected_edge"] = merged["p_final"] * merged["winOdds"] - 1.0
    return merged.sort_values(["rcDate", "race_id", "entry_id"], kind="stable")


def selected_bets(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = frame.loc[frame["expected_edge"].ge(threshold)].copy()
    selected["stake"] = 1.0
    selected["return"] = np.where(selected["win"].eq(1), selected["winOdds"], 0.0)
    selected["profit"] = selected["return"] - selected["stake"]
    return selected


def maximum_drawdown_by_race(selected: pd.DataFrame) -> dict[str, float]:
    if selected.empty:
        return {"stake_units": 0.0, "percent_of_100_unit_bankroll": 0.0}
    race_profit = (
        selected.groupby(["rcDate", "race_id"], sort=True)["profit"].sum().to_numpy()
    )
    equity = INITIAL_BANKROLL + np.concatenate([[0.0], np.cumsum(race_profit)])
    peaks = np.maximum.accumulate(equity)
    drawdown = peaks - equity
    maximum = float(drawdown.max())
    return {
        "stake_units": maximum,
        "percent_of_100_unit_bankroll": maximum / INITIAL_BANKROLL,
    }


def equal_stake_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    selected = selected_bets(frame, threshold)
    stakes = float(selected["stake"].sum())
    returns = float(selected["return"].sum())
    profit = returns - stakes
    return {
        "threshold": threshold,
        "bets": int(len(selected)),
        "races_bet": int(selected["race_id"].nunique()),
        "wins": int(selected["win"].sum()),
        "hit_rate": float(selected["win"].mean()) if len(selected) else None,
        "mean_closing_odds": float(selected["winOdds"].mean()) if len(selected) else None,
        "median_closing_odds": float(selected["winOdds"].median()) if len(selected) else None,
        "total_stake": stakes,
        "total_return": returns,
        "net_profit": profit,
        "roi": profit / stakes if stakes else None,
        "maximum_drawdown": maximum_drawdown_by_race(selected),
    }


def roi_bootstrap(frame: pd.DataFrame, threshold: float, *, seed: int) -> dict[str, object]:
    selected = selected_bets(frame, threshold)
    all_races = pd.Index(sorted(frame["race_id"].unique()))
    by_race = selected.groupby("race_id").agg(stake=("stake", "sum"), profit=("profit", "sum"))
    stake = by_race["stake"].reindex(all_races, fill_value=0.0).to_numpy(dtype=float)
    profit = by_race["profit"].reindex(all_races, fill_value=0.0).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(all_races), size=(N_BOOTSTRAP, len(all_races)))
    sampled_stake = stake[indices].sum(axis=1)
    sampled_profit = profit[indices].sum(axis=1)
    valid = sampled_stake > 0
    roi = sampled_profit[valid] / sampled_stake[valid]
    if not valid.any():
        return {
            "unit": "race_id",
            "n_bootstrap": N_BOOTSTRAP,
            "random_seed": seed,
            "valid_replicates": 0,
            "ci_95_percentile": {"lower": None, "upper": None},
            "probability_positive_roi": None,
            "ci_lower_above_zero": False,
        }
    lower, upper = np.quantile(roi, [0.025, 0.975])
    return {
        "unit": "race_id",
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": seed,
        "valid_replicates": int(valid.sum()),
        "ci_95_percentile": {"lower": float(lower), "upper": float(upper)},
        "probability_positive_roi": float(np.mean(roi > 0.0)),
        "ci_lower_above_zero": bool(lower > 0.0),
    }


def fractional_kelly_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, object]:
    selected = selected_bets(frame, threshold)
    bankroll = INITIAL_BANKROLL
    peak = bankroll
    max_drawdown = 0.0
    total_stake = 0.0
    for (_, _), race in selected.groupby(["rcDate", "race_id"], sort=True):
        starting = bankroll
        odds = race["winOdds"].to_numpy(dtype=float)
        probability = race["p_final"].to_numpy(dtype=float)
        denominator = np.maximum(odds - 1.0, 1e-12)
        full_kelly = np.maximum(0.0, (probability * odds - 1.0) / denominator)
        fractions = np.minimum(full_kelly * KELLY_FRACTION, MAX_SINGLE_BET_FRACTION)
        stakes = starting * fractions
        returns = np.where(race["win"].to_numpy(dtype=int) == 1, stakes * odds, 0.0)
        bankroll += float(returns.sum() - stakes.sum())
        total_stake += float(stakes.sum())
        peak = max(peak, bankroll)
        max_drawdown = max(max_drawdown, peak - bankroll)
    return {
        "initial_bankroll": INITIAL_BANKROLL,
        "ending_bankroll": bankroll,
        "net_profit": bankroll - INITIAL_BANKROLL,
        "total_stake": total_stake,
        "profit_over_total_stake": (
            (bankroll - INITIAL_BANKROLL) / total_stake if total_stake else None
        ),
        "maximum_drawdown_units": max_drawdown,
        "kelly_multiplier": KELLY_FRACTION,
        "max_single_bet_fraction": MAX_SINGLE_BET_FRACTION,
    }


def evaluate_thresholds(frame: pd.DataFrame, fold_seed_offset: int) -> list[dict[str, object]]:
    results = []
    for index, threshold in enumerate(THRESHOLDS):
        metrics = equal_stake_metrics(frame, threshold)
        metrics["roi_bootstrap"] = roi_bootstrap(
            frame, threshold, seed=RANDOM_SEED + fold_seed_offset + index
        )
        metrics["fractional_kelly_secondary"] = fractional_kelly_metrics(frame, threshold)
        results.append(metrics)
    return results


def select_calibration_policy(results: list[dict[str, object]]) -> dict[str, object]:
    sufficiently_sampled = [row for row in results if row["bets"] >= MIN_BETS]
    eligible = [
        row
        for row in sufficiently_sampled
        if row["roi"] is not None
        and row["roi"] > 0
        and row["roi_bootstrap"]["ci_lower_above_zero"]
    ]
    exploratory = (
        max(sufficiently_sampled, key=lambda row: (row["roi"], row["threshold"]))
        if sufficiently_sampled
        else None
    )
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["roi_bootstrap"]["ci_95_percentile"]["lower"],
                row["roi"],
                row["threshold"],
            ),
        )
        deployment = {
            "action": "bet",
            "threshold": selected["threshold"],
            "reason": "Positive Calibration ROI with positive 95% race-bootstrap lower bound",
        }
    else:
        deployment = {
            "action": "no_bet",
            "threshold": None,
            "reason": "No threshold met sample-size and positive ROI confidence-bound requirements",
        }
    return {
        "deployment_policy": deployment,
        "exploratory_threshold_not_for_deployment": (
            exploratory["threshold"] if exploratory else None
        ),
        "eligible_thresholds": [row["threshold"] for row in eligible],
    }


def main() -> int:
    if REPORT_PATH.exists() or POLICY_PATH.exists() or SELECTIONS_PATH.exists():
        raise FileExistsError("Backtest outputs already exist and will not be overwritten")
    final_manifest = json.loads(FINAL_TEST_MANIFEST.read_text(encoding="utf-8"))
    if final_manifest["evaluation_count"] != 1:
        raise ValueError("Backtest requires the locked single Final Test evaluation")
    for item in final_manifest["outputs"]:
        path = PROJECT_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Final Test output checksum mismatch: {item['path']}")

    calibration = attach_closing_odds(pd.read_csv(CALIBRATION_PREDICTIONS), "calibration")
    calibration_results = evaluate_thresholds(calibration, 0)
    selection = select_calibration_policy(calibration_results)

    # Test results are descriptive only; no policy selection occurs below this line.
    test = attach_closing_odds(pd.read_csv(TEST_PREDICTIONS), "test")
    test_results = evaluate_thresholds(test, 100)
    exploratory_threshold = selection["exploratory_threshold_not_for_deployment"]
    selected_test_diagnostic = next(
        (row for row in test_results if row["threshold"] == exploratory_threshold), None
    )

    selections = []
    for fold, frame in (("calibration", calibration), ("test", test)):
        for threshold in THRESHOLDS:
            part = selected_bets(frame, threshold)
            part = part[
                [
                    "race_id",
                    "entry_id",
                    "rcDate",
                    "win",
                    "winOdds",
                    "q_market",
                    "p_final",
                    "break_even_prob",
                    "expected_edge",
                    "profit",
                ]
            ].copy()
            part.insert(0, "fold", fold)
            part.insert(1, "threshold", threshold)
            selections.append(part)
    selection_frame = pd.concat(selections, ignore_index=True)
    SELECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    selection_frame.to_csv(
        SELECTIONS_PATH, index=False, compression="gzip", encoding="utf-8"
    )

    report = {
        "experiment": "stage_17_closing_odds_economic_backtest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_warning": "Uses final closing odds and is retrospective research, not an executable betting strategy or financial advice.",
        "selection_rule": {
            "fold": "calibration only",
            "thresholds": list(THRESHOLDS),
            "minimum_bets": MIN_BETS,
            "activation": "ROI > 0 and race-bootstrap 95% ROI CI lower > 0",
            "primary_staking": "one unit per selected entry",
        },
        "calibration": {"threshold_results": calibration_results, **selection},
        "test_descriptive_only": {
            "threshold_results": test_results,
            "exploratory_calibration_threshold_result": selected_test_diagnostic,
        },
        "final_policy": selection["deployment_policy"],
        "post_test_policy": "Test ROI did not select or modify any threshold, model, or probability policy.",
        "limitations": [
            "Closing odds are unavailable at realistic decision time.",
            "No transaction latency, odds movement, rejection, pool impact, or minimum stake is modeled.",
            "ROI estimates are sensitive to rare high-odds winners and multiple comparisons.",
        ],
        "selections_path": str(SELECTIONS_PATH.relative_to(PROJECT_ROOT)).replace(
            "\\", "/"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    policy = {
        "policy_version": 1,
        "created_at": report["created_at"],
        "selection_source": "reports/experiments/stage_17_backtest.json",
        "selection_fold": "calibration",
        "threshold_grid": list(THRESHOLDS),
        "minimum_bets": MIN_BETS,
        "deployment_policy": selection["deployment_policy"],
        "exploratory_threshold_not_for_deployment": exploratory_threshold,
        "uses_closing_odds": True,
        "executable_live_strategy": False,
        "final_test_model_changed": False,
        "outputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (REPORT_PATH, SELECTIONS_PATH)
        ],
    }
    POLICY_PATH.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(policy, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
