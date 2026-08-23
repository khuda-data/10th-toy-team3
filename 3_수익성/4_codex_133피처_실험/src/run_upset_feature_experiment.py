"""Rank-locked upset-horse discovery, profitability, and feature-subset retraining.

An upset candidate follows the project's verified ``upset_B`` universe: a
bottom-half market-popularity runner.  The candidate with the largest positive
place-market anomaly score in its race is used::

    score = P_model(top-K place) - q_plc

The model is a Base Margin booster with log(q) as a fixed offset.  Odds are not
part of the fundamental feature matrix.  Feature importance is learned on a
chronological train-only holdout.  The 10/20/30/40 percent policies are fixed
in advance, and each evaluation split is ranked without using its outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.conditional_logit import group_softmax
from src.training.build_full_rank_all_bets_report import (
    daily_block_roi_ci,
    equity_statistics,
    fractional_kelly_statistics,
    markdown_table,
)
from src.training.core.market_base_margin_boosting import BaseMarginBooster
from src.training.evaluate_bet_type_edge import (
    BET_TYPES,
    combination_key,
    event_hit,
    event_probability,
)
from src.training.train_full_rank_models import SPLITS, load_data
from src.training.train_market_anchor_same_test import (
    attach_market,
    probability_from_margin,
    race_groups,
    sha256,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_KEYS = ("full_upset_base_margin", "selected_upset_base_margin")
MODEL_NAMES = {
    "full_upset_base_margin": "감사 후 전체 피처 이변 Base Margin",
    "selected_upset_base_margin": "상위 이변 피처 Base Margin",
}
FRACTIONS = (0.10, 0.20, 0.30, 0.40)
IMPORTANCE_WEIGHT_THRESHOLD = 0.005
PERMUTATION_REPEATS = 20
BOOTSTRAPS = 5000
FINAL_POOL_FEATURES = ("winAmt", "plcAmt", "totalAmt", "log_winAmt", "liq_per_horse")


def check(condition: bool, label: str, checks: list[dict]) -> None:
    checks.append({"check": label, "status": "PASS" if condition else "FAIL"})
    if not condition:
        raise AssertionError(label)


def attach_upset_market_columns(meta: dict[str, pd.DataFrame], market_path: Path) -> dict[str, pd.DataFrame]:
    extra = pd.read_csv(
        market_path, compression="infer", usecols=["entry_id", "pop_pct", "upset_B"], low_memory=False
    )
    extra["entry_id"] = extra["entry_id"].astype(str)
    if extra["entry_id"].duplicated().any():
        raise ValueError("upset market reference entry_id is not unique")
    result = {}
    for split, frame in meta.items():
        joined = frame.copy()
        joined["entry_id"] = joined["entry_id"].astype(str)
        joined = joined.merge(extra, on="entry_id", how="left", validate="one_to_one")
        if joined[["pop_pct", "upset_B"]].isna().any().any():
            raise ValueError(f"{split}: pop_pct/upset_B join failed")
        rebuilt = ((joined["pop_pct"] >= 0.5) & joined["place"].eq(1)).astype(int)
        if not rebuilt.eq(joined["upset_B"].astype(int)).all():
            raise ValueError(f"{split}: stored upset_B definition mismatch")
        joined["top3_upset"] = ((joined["pop_pct"] >= 0.5) & joined["finish_order"].le(3)).astype(int)
        result[split] = joined
    return result


def clean_splits(
    data: dict[str, pd.DataFrame], meta: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict]]:
    clean_data, clean_meta, audit = {}, {}, {}
    for split in ("train", "valid", "test"):
        keep_races = []
        reasons = {"incomplete_order": 0, "invalid_win_odds": 0, "invalid_place_market": 0}
        for race_id, positions_raw in meta[split].groupby("race_id", sort=False).indices.items():
            positions = np.asarray(positions_raw, dtype=int)
            race = meta[split].iloc[positions]
            if not race["rank_group_status"].eq("complete_1_to_n").all():
                reasons["incomplete_order"] += 1
                continue
            if not (np.isfinite(race["winOdds"]).all() and race["winOdds"].between(
                    1.0, 9999.0, inclusive="neither").all()):
                reasons["invalid_win_odds"] += 1
                continue
            if not (np.isfinite(race["plcOdds"]).all() and (race["plcOdds"] > 0).all()
                    and np.isfinite(race["q_plc"]).all()):
                reasons["invalid_place_market"] += 1
                continue
            keep_races.append(str(race_id))
        mask = meta[split]["race_id"].astype(str).isin(keep_races).to_numpy()
        clean_data[split] = data[split].loc[mask].reset_index(drop=True)
        clean_meta[split] = meta[split].loc[mask].reset_index(drop=True)
        audit[split] = {
            "input_rows": int(len(meta[split])), "input_races": int(meta[split]["race_id"].nunique()),
            "clean_rows": int(len(clean_meta[split])), "clean_races": int(clean_meta[split]["race_id"].nunique()),
            "excluded_races": reasons,
        }
    return clean_data, clean_meta, audit


def model_place_probability(meta: pd.DataFrame, win_probability: np.ndarray) -> np.ndarray:
    """Exact PL top-2/top-3 probability without repeated permutation enumeration."""
    result = np.zeros(len(meta), dtype=float)
    for positions_raw in meta.groupby("race_id", sort=False).indices.values():
        positions = np.asarray(positions_raw, dtype=int)
        slots = int(meta.iloc[positions]["place"].sum())
        if slots not in {2, 3}:
            raise ValueError(f"unsupported place slots={slots}")
        probability = np.asarray(win_probability[positions], dtype=float)
        probability = probability / probability.sum()
        one_minus = np.clip(1.0 - probability, 1e-15, None)
        odds_after_first = probability / one_minus
        second = probability * (odds_after_first.sum() - odds_after_first)
        local_place = probability + second
        if slots == 3:
            # pair_term[j,k] is the first-two PL factor divided by the
            # remaining denominator needed before horse i finishes third.
            denom = 1.0 - probability[:, None] - probability[None, :]
            pair_term = (
                probability[:, None] * probability[None, :]
                / one_minus[:, None]
                / np.clip(denom, 1e-15, None)
            )
            np.fill_diagonal(pair_term, 0.0)
            total = float(pair_term.sum())
            third = probability * (total - pair_term.sum(axis=1) - pair_term.sum(axis=0))
            local_place = local_place + third
        result[positions] = np.clip(local_place, 0.0, 1.0)
    return result


def fit_base_margin(
    features: list[str], data: dict[str, pd.DataFrame], meta: dict[str, pd.DataFrame]
) -> tuple[BaseMarginBooster, BaseMarginBooster, dict[str, np.ndarray], int]:
    X_train = data["train"][features].to_numpy(dtype=float)
    X_valid = data["valid"][features].to_numpy(dtype=float)
    X_test = data["test"][features].to_numpy(dtype=float)
    g_train = race_groups(meta["train"])
    g_valid = race_groups(meta["valid"])
    q_train = np.log(np.clip(meta["train"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
    q_valid = np.log(np.clip(meta["valid"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
    q_test = np.log(np.clip(meta["test"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
    valid_model = BaseMarginBooster().fit(
        X_train, g_train, meta["train"]["win"].to_numpy(dtype=float), q_train,
        X_valid, g_valid, q_valid,
    )
    n_trees = max(int(valid_model.n_used), 1)
    p_valid = probability_from_margin(meta["valid"], q_valid + valid_model.margin(X_valid))

    X_both = np.vstack([X_train, X_valid])
    both_meta = pd.concat([meta["train"], meta["valid"]], ignore_index=True)
    q_both = np.log(np.clip(both_meta["q"].to_numpy(dtype=float), 1e-15, 1.0))
    g_both = race_groups(both_meta)
    final_model = BaseMarginBooster(max_trees=n_trees).fit(
        X_both, g_both, both_meta["win"].to_numpy(dtype=float), q_both
    )
    final_model.n_used = n_trees
    p_test = probability_from_margin(meta["test"], q_test + final_model.margin(X_test))
    return valid_model, final_model, {"valid": p_valid, "test": p_test}, n_trees


def day_block_mean_ci(values: np.ndarray, race_dates: np.ndarray, seed: int) -> tuple[float, float, float]:
    frame = pd.DataFrame({"date": race_dates.astype(str), "value": values})
    daily = frame.groupby("date", sort=True)["value"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(daily), size=(BOOTSTRAPS, len(daily)))
    means = daily["sum"].to_numpy()[sampled].sum(axis=1) / daily["count"].to_numpy()[sampled].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)), float(np.mean(means > 0))


def split_train_for_importance(
    data: pd.DataFrame, meta: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    dates = np.sort(meta["race_id"].astype(str).str[:8].unique())
    fit_cut = dates[int(len(dates) * 0.70)]
    tune_cut = dates[int(len(dates) * 0.80)]
    date_values = meta["race_id"].astype(str).str[:8]
    masks = {
        "fit": date_values.lt(fit_cut).to_numpy(),
        "tune": date_values.ge(fit_cut).to_numpy() & date_values.lt(tune_cut).to_numpy(),
        "importance": date_values.ge(tune_cut).to_numpy(),
    }
    return (
        {name: data.loc[mask].reset_index(drop=True) for name, mask in masks.items()},
        {name: meta.loc[mask].reset_index(drop=True) for name, mask in masks.items()},
    )


def fit_importance_model(
    features: list[str], data: pd.DataFrame, meta: pd.DataFrame
) -> tuple[BaseMarginBooster, pd.DataFrame, pd.DataFrame, dict]:
    parts_data, parts_meta = split_train_for_importance(data, meta)
    X_fit = parts_data["fit"][features].to_numpy(dtype=float)
    X_tune = parts_data["tune"][features].to_numpy(dtype=float)
    g_fit, g_tune = race_groups(parts_meta["fit"]), race_groups(parts_meta["tune"])
    q_fit = np.log(np.clip(parts_meta["fit"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
    q_tune = np.log(np.clip(parts_meta["tune"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
    tuned = BaseMarginBooster().fit(
        X_fit, g_fit, parts_meta["fit"]["win"].to_numpy(dtype=float), q_fit,
        X_tune, g_tune, q_tune,
    )
    n_trees = max(int(tuned.n_used), 1)
    combined_data = pd.concat([parts_data["fit"], parts_data["tune"]], ignore_index=True)
    combined_meta = pd.concat([parts_meta["fit"], parts_meta["tune"]], ignore_index=True)
    X_combined = combined_data[features].to_numpy(dtype=float)
    q_combined = np.log(np.clip(combined_meta["q"].to_numpy(dtype=float), 1e-15, 1.0))
    model = BaseMarginBooster(max_trees=n_trees).fit(
        X_combined, race_groups(combined_meta), combined_meta["win"].to_numpy(dtype=float), q_combined
    )
    model.n_used = n_trees
    audit = {
        "fit_dates": [str(parts_meta["fit"]["race_id"].astype(str).str[:8].min()),
                      str(parts_meta["fit"]["race_id"].astype(str).str[:8].max())],
        "tune_dates": [str(parts_meta["tune"]["race_id"].astype(str).str[:8].min()),
                       str(parts_meta["tune"]["race_id"].astype(str).str[:8].max())],
        "importance_dates": [str(parts_meta["importance"]["race_id"].astype(str).str[:8].min()),
                            str(parts_meta["importance"]["race_id"].astype(str).str[:8].max())],
        "fit_races": int(parts_meta["fit"]["race_id"].nunique()),
        "tune_races": int(parts_meta["tune"]["race_id"].nunique()),
        "importance_races": int(parts_meta["importance"]["race_id"].nunique()),
        "n_trees": n_trees,
    }
    return model, parts_data["importance"], parts_meta["importance"], audit


def permute_feature(
    X: np.ndarray, meta: pd.DataFrame, feature_index: int, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    permuted = X.copy()
    groups = [np.asarray(pos, dtype=int) for pos in meta.groupby("race_id", sort=False).indices.values()]
    varying = np.mean([np.unique(X[pos, feature_index]).size > 1 for pos in groups])
    if varying >= 0.05:
        for positions in groups:
            permuted[positions, feature_index] = rng.permutation(permuted[positions, feature_index])
        return permuted, "within_race"
    # Race-constant features need a block permutation.  Restrict exchange to
    # races with the same field size so the grouped choice structure is kept.
    by_size: dict[int, list[np.ndarray]] = {}
    for positions in groups:
        by_size.setdefault(len(positions), []).append(positions)
    for same_size in by_size.values():
        order = rng.permutation(len(same_size))
        source_values = [X[positions[0], feature_index] for positions in same_size]
        for target_index, positions in enumerate(same_size):
            permuted[positions, feature_index] = source_values[int(order[target_index])]
    return permuted, "race_block_same_field_size"


def tail_quality_contribution(
    candidates: pd.DataFrame, all_race_ids: np.ndarray, fractions: tuple[float, ...]
) -> np.ndarray:
    race_index = {str(race_id): index for index, race_id in enumerate(all_race_ids)}
    contribution = np.zeros(len(all_race_ids), dtype=float)
    ordered = candidates.sort_values(["upset_score", "race_id"], ascending=[False, True], kind="stable")
    for fraction in fractions:
        count = max(1, int(math.ceil(len(ordered) * fraction)))
        selected = ordered.head(count)
        for row in selected.itertuples(index=False):
            contribution[race_index[str(row.race_id)]] += float(row.realized_upset_edge) / count
    return contribution / len(fractions)


def permutation_importance(
    features: list[str], data: pd.DataFrame, meta: pd.DataFrame, model: BaseMarginBooster,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-only holdout importance for top-tail realized upset edge."""
    X = data[features].to_numpy(dtype=float)
    lnq = np.log(np.clip(meta["q"].to_numpy(dtype=float), 1e-15, 1.0))
    baseline_win = probability_from_margin(meta, lnq + model.margin(X))
    baseline_candidates = find_upset_candidates("train_importance", "importance_model", meta, baseline_win)
    race_ids = meta.iloc[race_groups(meta).offsets]["race_id"].astype(str).to_numpy()
    race_dates = np.asarray([race_id[:8] for race_id in race_ids])
    baseline_contribution = tail_quality_contribution(baseline_candidates, race_ids, FRACTIONS)
    used_indices = {
        int(index) for tree in model.trees[:model.n_used] for index in tree.tree_.feature if int(index) >= 0
    }
    stored, raw_means, positive_rates, methods = {}, {}, {}, {}
    positive_parts = []
    for feature_index, feature in enumerate(features):
        repeated = np.zeros((PERMUTATION_REPEATS, len(race_ids)), dtype=float)
        method = "unused_by_model"
        if feature_index in used_indices:
            for repeat in range(PERMUTATION_REPEATS):
                rng = np.random.default_rng(2026082200 + feature_index * 100 + repeat)
                permuted, method = permute_feature(X, meta, feature_index, rng)
                p_win = probability_from_margin(meta, lnq + model.margin(permuted))
                permuted_candidates = find_upset_candidates(
                    "train_importance", "importance_model", meta, p_win
                )
                permuted_contribution = tail_quality_contribution(permuted_candidates, race_ids, FRACTIONS)
                # Multiplication by race count turns the mean per-race vector
                # back into the average top-tail quality difference.
                repeated[repeat] = (baseline_contribution - permuted_contribution) * len(race_ids)
        averaged = repeated.mean(axis=0)
        raw = float(averaged.mean())
        stored[feature] = averaged
        raw_means[feature] = raw
        positive_rates[feature] = float(np.mean(repeated.sum(axis=1) > 0))
        methods[feature] = method
        positive_parts.append(max(raw, 0.0))
        if (feature_index + 1) % 10 == 0 or feature_index + 1 == len(features):
            print(f"UPSET-TAIL PERMUTATION IMPORTANCE: {feature_index + 1}/{len(features)}")
    denominator = float(np.sum(positive_parts))
    rows, contribution_rows = [], []
    for feature_index, feature in enumerate(features):
        values = stored[feature]
        low, high, probability_positive = day_block_mean_ci(values, race_dates, 20260822 + feature_index)
        weight = max(raw_means[feature], 0.0) / denominator if denominator > 0 else 0.0
        selected = bool(weight >= IMPORTANCE_WEIGHT_THRESHOLD and raw_means[feature] > 0)
        rows.append({
            "feature": feature,
            "mean_upset_tail_realized_edge_drop": raw_means[feature],
            "ci_low_daily_block": low, "ci_high_daily_block": high,
            "bootstrap_probability_positive": probability_positive,
            "positive_repeat_rate": positive_rates[feature],
            "positive_importance_weight": weight,
            "weight_threshold": IMPORTANCE_WEIGHT_THRESHOLD,
            "selected_for_retraining": selected,
            "ci_confirmed_positive": bool(low > 0),
            "permutation_method": methods[feature],
            "tree_used_feature": bool(feature_index in used_indices),
        })
        for race_id, value in zip(race_ids, values):
            contribution_rows.append({"feature": feature, "race_id": race_id,
                                      "upset_tail_realized_edge_drop": float(value)})
    importance = pd.DataFrame(rows).sort_values(
        ["positive_importance_weight", "mean_upset_tail_realized_edge_drop", "feature"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))
    return importance, pd.DataFrame(contribution_rows)


def probability_metrics(
    split: str, model_key: str, meta: pd.DataFrame, probability: np.ndarray
) -> tuple[dict, pd.DataFrame]:
    q = meta["q"].to_numpy(dtype=float)
    g = race_groups(meta)
    winners = g.winner_rows
    model_logp = np.log(np.clip(probability[winners], 1e-15, 1.0))
    market_logp = np.log(np.clip(q[winners], 1e-15, 1.0))
    delta = model_logp - market_logp
    race_ids = meta.iloc[g.offsets]["race_id"].astype(str).to_numpy()
    dates = np.asarray([race_id[:8] for race_id in race_ids])
    low, high, positive = day_block_mean_ci(
        delta, dates, int(hashlib.sha256(f"{split}|{model_key}|dll".encode()).hexdigest()[:8], 16)
    )
    max_sum_error = 0.0
    model_hits, market_hits = [], []
    outcome = meta["win"].to_numpy(dtype=int)
    for positions in meta.groupby("race_id", sort=False).indices.values():
        pos = np.asarray(positions, dtype=int)
        max_sum_error = max(max_sum_error, abs(float(probability[pos].sum()) - 1.0))
        model_hits.append(int(outcome[pos[np.argmax(probability[pos])]] == 1))
        market_hits.append(int(outcome[pos[np.argmax(q[pos])]] == 1))
    row = {
        "split": split, "model_key": model_key,
        "model": MODEL_NAMES.get(model_key, "train-internal importance model"), "races": g.n_races,
        "model_log_loss_per_race": float(-model_logp.mean()),
        "market_log_loss_per_race": float(-market_logp.mean()),
        "delta_ll_per_race": float(delta.mean()),
        "delta_ll_ci_low_daily_block": low, "delta_ll_ci_high_daily_block": high,
        "delta_ll_bootstrap_probability_positive": positive,
        "delta_ll_confirmed_positive_95pct": bool(low > 0),
        "model_top1_hit_rate": float(np.mean(model_hits)),
        "market_top1_hit_rate": float(np.mean(market_hits)),
        "maximum_abs_probability_sum_error": max_sum_error,
    }
    contributions = pd.DataFrame({"split": split, "model_key": model_key,
                                  "race_id": race_ids, "race_date": dates, "delta_ll": delta})
    return row, contributions


def find_upset_candidates(
    split: str, model_key: str, meta: pd.DataFrame, probability: np.ndarray
) -> pd.DataFrame:
    place_probability = model_place_probability(meta, probability)
    rows = []
    for positions_raw in meta.groupby("race_id", sort=False).indices.values():
        positions = np.asarray(positions_raw, dtype=int)
        race = meta.iloc[positions]
        if not race["rank_group_status"].eq("complete_1_to_n").all():
            continue
        q = race["q"].to_numpy(dtype=float)
        p = probability[positions]
        p_place = place_probability[positions]
        q_plc = race["q_plc"].to_numpy(dtype=float)
        eligible = race["pop_pct"].to_numpy(dtype=float) >= 0.5
        score = p_place - q_plc
        score[~eligible] = -np.inf
        candidate_local = int(np.argmax(score))
        if not np.isfinite(score[candidate_local]) or score[candidate_local] <= 0:
            continue
        candidate_global = int(positions[candidate_local])
        market_rank = int(np.where(np.argsort(-q, kind="stable") == candidate_local)[0][0] + 1)
        row = race.iloc[candidate_local]
        rows.append({
            "split": split, "model_key": model_key,
            "model": MODEL_NAMES.get(model_key, "train-internal importance model"),
            "race_id": str(row["race_id"]), "race_date": str(row["race_id"])[:8],
            "row_position": candidate_global, "local_position": candidate_local,
            "entry_id": str(row["entry_id"]), "hrName": str(row.get("hrName", "")),
            "chulNo": int(row["chulNo"]), "market_rank": market_rank,
            "market_probability": float(q[candidate_local]), "model_probability": float(p[candidate_local]),
            "model_place_probability": float(p_place[candidate_local]),
            "market_place_probability": float(q_plc[candidate_local]),
            "predicted_edge": float(p_place[candidate_local] - q_plc[candidate_local]),
            "upset_score": float(score[candidate_local]), "winOdds": float(row["winOdds"]),
            "plcOdds": float(row["plcOdds"]), "win": int(row["win"]), "place": int(row["place"]),
            "finish_order": int(row["finish_order"]), "pop_pct": float(row["pop_pct"]),
            "upset_B": int(row["upset_B"]), "top3_upset": int(row["top3_upset"]),
            "realized_upset_edge": float(int(row["place"]) - q_plc[candidate_local]),
        })
    return pd.DataFrame(rows).sort_values(["race_id"], kind="stable").reset_index(drop=True)


def score_fraction_cutoffs(candidates: pd.DataFrame) -> pd.DataFrame:
    """Record deterministic exact top-fraction cutoffs without using outcomes."""
    rows = []
    for (split, model_key), group in candidates.groupby(["split", "model_key"], sort=False):
        ordered = group.sort_values(
            ["upset_score", "race_id"], ascending=[False, True], kind="stable"
        ).reset_index(drop=True)
        for fraction in FRACTIONS:
            count = max(1, int(math.ceil(len(ordered) * fraction)))
            threshold = float(ordered.iloc[count - 1]["upset_score"])
            rows.append({
                "split": split, "model_key": model_key, "model": MODEL_NAMES[model_key],
                "fraction": fraction, "fraction_label": f"top_{int(fraction * 100)}pct",
                "score_threshold": threshold, "candidate_pool": int(len(ordered)),
                "selected_target_count": count,
                "actual_selected_fraction": float(count / len(ordered)),
                "selection_scope": "within-split score rank; outcomes unused",
            })
    return pd.DataFrame(rows)


def centered_selection(
    p: np.ndarray, upset_local: int, bet_type: str, place_slots: int
) -> tuple[int, ...]:
    indices = list(range(len(p)))
    others = [index for index in indices if index != upset_local]
    if bet_type in {"단승", "연승"}:
        return (upset_local,)
    if bet_type in {"복승", "복연승"}:
        choices = [(upset_local, other) for other in others]
        return max(choices, key=lambda choice: event_probability(p, choice, bet_type, place_slots))
    if bet_type == "쌍승":
        choices = [order for other in others for order in ((upset_local, other), (other, upset_local))]
        return max(choices, key=lambda choice: event_probability(p, choice, bet_type, place_slots))
    if bet_type == "삼복승":
        choices = [(upset_local, *pair) for pair in itertools.combinations(others, 2)]
        return max(choices, key=lambda choice: event_probability(p, choice, bet_type, place_slots))
    if bet_type == "삼쌍승":
        choices = [order for pair in itertools.combinations(others, 2)
                   for order in itertools.permutations((upset_local, *pair), 3)]
        return max(choices, key=lambda choice: event_probability(p, choice, bet_type, place_slots))
    raise KeyError(bet_type)


def official_lookup(path: Path):
    official = pd.read_csv(path, dtype={"race_id": str, "winning_combination": str})
    if official.duplicated(["race_id", "bet_type", "winning_combination"]).any():
        raise ValueError("official dividend key is not unique")
    lookup = {
        (str(row.race_id), row.bet_type,
         combination_key([int(value) for value in str(row.winning_combination).split("-")], row.bet_type)):
            float(row.dividend)
        for row in official.itertuples(index=False)
    }
    coverage = set(zip(official["race_id"].astype(str), official["bet_type"].astype(str)))
    return lookup, coverage


def build_bets(
    candidates: pd.DataFrame, cutoffs: pd.DataFrame, meta: dict[str, pd.DataFrame],
    probabilities: dict[str, dict[str, np.ndarray]], official_path: Path,
) -> pd.DataFrame:
    lookup, coverage = official_lookup(official_path)
    rows = []
    for cutoff in cutoffs.itertuples(index=False):
        model_key = cutoff.model_key
        split = cutoff.split
        selected_candidates = candidates[
            candidates["model_key"].eq(model_key) & candidates["split"].eq(split)
        ].sort_values(["upset_score", "race_id"], ascending=[False, True], kind="stable").head(
            int(cutoff.selected_target_count)
        )
        split_meta = meta[split]
        p_all = probabilities[model_key][split]
        for candidate in selected_candidates.itertuples(index=False):
                positions = np.asarray(split_meta.groupby("race_id", sort=False).indices[str(candidate.race_id)], dtype=int)
                race = split_meta.iloc[positions].reset_index(drop=True)
                local = int(np.flatnonzero(positions == int(candidate.row_position))[0])
                p = p_all[positions]
                q = race["q"].to_numpy(dtype=float)
                q = q / q.sum()
                q_plc = race["q_plc"].to_numpy(dtype=float)
                actual_order = tuple(int(value) for value in np.argsort(
                    race["finish_order"].to_numpy(dtype=int), kind="stable"
                ))
                place_set = set(np.flatnonzero(race["place"].to_numpy(dtype=int) == 1).tolist())
                place_slots = len(place_set)
                chul = race["chulNo"].astype(int).tolist()
                for bet_type in BET_TYPES:
                    if bet_type == "연승" and not np.isfinite(q_plc[local]):
                        continue
                    selection = centered_selection(p, local, bet_type, place_slots)
                    model_event_probability = event_probability(p, selection, bet_type, place_slots)
                    if bet_type == "연승":
                        market_event_probability = float(q_plc[local])
                        odds = float(race.iloc[local]["plcOdds"])
                        market_source = "direct q_plc"
                    else:
                        market_event_probability = event_probability(q, selection, bet_type, place_slots)
                        odds = float(race.iloc[local]["winOdds"]) if bet_type == "단승" else np.nan
                        market_source = "direct normalized q" if bet_type == "단승" else "q-derived PL/Harville proxy"
                    hit = event_hit(selection, actual_order, place_set, bet_type)
                    race_id = str(candidate.race_id)
                    if bet_type not in {"단승", "연승"}:
                        combo = combination_key([chul[index] for index in selection], bet_type)
                        available = (race_id, bet_type) in coverage
                        if available:
                            odds = lookup.get((race_id, bet_type, combo), np.nan) if hit else 0.0
                            if hit and not np.isfinite(odds):
                                raise ValueError(f"winning official dividend missing: {race_id}/{bet_type}/{combo}")
                    else:
                        available = bool(np.isfinite(odds) and 1.0 < odds < 9999.0)
                    rows.append({
                        "split": split, "model_key": model_key, "model": MODEL_NAMES[model_key],
                        "fraction": float(cutoff.fraction), "fraction_label": cutoff.fraction_label,
                        "score_threshold": float(cutoff.score_threshold),
                        "race_id": race_id, "race_date": race_id[:8], "entry_id": str(candidate.entry_id),
                        "hrName": str(candidate.hrName), "chulNo": int(candidate.chulNo),
                        "market_rank": int(candidate.market_rank), "upset_score": float(candidate.upset_score),
                        "bet_type": bet_type,
                        "selection_chul_numbers": "-".join(str(chul[index]) for index in selection),
                        "model_probability": float(model_event_probability),
                        "market_probability": float(market_event_probability),
                        "market_probability_source": market_source,
                        "predicted_edge": float(model_event_probability - market_event_probability),
                        "model_hit": int(hit), "actual_odds_available_for_selection": bool(available),
                        "actual_odds": float(odds) if available else np.nan,
                        "realized_return": float(hit * odds - 1.0) if available else np.nan,
                    })
    return pd.DataFrame(rows)


def summarize_bets(bets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in bets.groupby(["split", "model_key", "fraction", "fraction_label", "bet_type"], sort=False):
        split, model_key, fraction, fraction_label, bet_type = keys
        selected = group[group["actual_odds_available_for_selection"]].sort_values("race_id", kind="stable")
        returns = selected["realized_return"].to_numpy(dtype=float)
        low, high, positive = daily_block_roi_ci(
            selected,
            int(hashlib.sha256(f"{split}|{model_key}|{fraction_label}|{bet_type}".encode()).hexdigest()[:8], 16),
        ) if len(selected) else (None, None, None)
        direct = bet_type in {"단승", "연승"}
        kelly = fractional_kelly_statistics(selected) if direct else {
            "kelly_status": "UNAVAILABLE_FOR_LOSING_COMBINATIONS", "kelly_bets": 0,
            "kelly_total_return": None, "kelly_mdd": None, "kelly_sharpe": None,
            "mean_kelly_fraction": None,
        }
        winning = selected[selected["model_hit"].eq(1)]
        if len(winning):
            largest_index = winning["actual_odds"].idxmax()
            largest_row = selected.loc[largest_index]
            without_largest = selected.drop(index=largest_index)
            roi_without_largest = (
                float(without_largest["realized_return"].mean()) if len(without_largest) else None
            )
            largest_dividend = float(largest_row["actual_odds"])
            largest_race_id = str(largest_row["race_id"])
            largest_gross_share = float(largest_dividend / winning["actual_odds"].sum())
        else:
            roi_without_largest = None
            largest_dividend = None
            largest_race_id = None
            largest_gross_share = None
        odds_cap_30 = selected[selected["actual_odds"].le(30.0)] if direct else selected.iloc[0:0]
        positive_model_ev = (
            selected[selected["model_probability"] * selected["actual_odds"] > 1.0]
            if direct else selected.iloc[0:0]
        )
        stability = equity_statistics(returns)
        if len(returns) <= 1 or float(np.ptp(returns)) <= 1e-12:
            stability["per_bet_sharpe"] = None
        rows.append({
            "split": split, "model_key": model_key, "model": MODEL_NAMES[model_key],
            "fraction": float(fraction), "fraction_label": fraction_label, "bet_type": bet_type,
            "bets": int(len(selected)), "wins": int(selected["model_hit"].sum()),
            "hit_rate": float(selected["model_hit"].mean()) if len(selected) else None,
            "unit_profit": float(returns.sum()) if len(returns) else None,
            "roi": float(returns.mean()) if len(returns) else None,
            "roi_ci_low_daily_block": low, "roi_ci_high_daily_block": high,
            "bootstrap_probability_positive": positive,
            "profit_confirmed_95pct": bool(low is not None and low > 0),
            "average_market_rank": float(selected["market_rank"].mean()) if len(selected) else None,
            "average_upset_score": float(selected["upset_score"].mean()) if len(selected) else None,
            "average_selected_odds": float(selected["actual_odds"].mean()) if direct and len(selected) else None,
            "average_winning_dividend": float(winning["actual_odds"].mean()) if len(winning) else None,
            "largest_winning_dividend": largest_dividend,
            "largest_winner_race_id": largest_race_id,
            "largest_winner_share_of_winning_gross": largest_gross_share,
            "roi_without_largest_winning_return": roi_without_largest,
            "odds_le_30_bets": int(len(odds_cap_30)) if direct else None,
            "odds_le_30_roi": float(odds_cap_30["realized_return"].mean()) if len(odds_cap_30) else None,
            "positive_model_ev_bets": int(len(positive_model_ev)) if direct else None,
            "positive_model_ev_roi": (
                float(positive_model_ev["realized_return"].mean()) if len(positive_model_ev) else None
            ),
            **stability, **kelly,
        })
    order = {name: index for index, name in enumerate(BET_TYPES)}
    return pd.DataFrame(rows).sort_values(
        ["split", "model_key", "fraction", "bet_type"],
        key=lambda values: values.map(order) if values.name == "bet_type" else values,
        kind="stable",
    ).reset_index(drop=True)


def add_roi_fdr(summary: pd.DataFrame) -> pd.DataFrame:
    """Benjamini-Hochberg correction across all reported ROI cells per split."""
    result = summary.copy()
    result["roi_one_sided_bootstrap_p"] = np.nan
    result["roi_fdr_q_value"] = np.nan
    for _split, group in result.groupby("split", sort=False):
        usable = group[group["bootstrap_probability_positive"].notna()]
        if usable.empty:
            continue
        indices = usable.index.to_numpy()
        p_values = np.clip(
            1.0 - usable["bootstrap_probability_positive"].to_numpy(dtype=float), 0.0, 1.0
        )
        order = np.argsort(p_values, kind="stable")
        ranked = p_values[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        q_values = np.empty_like(adjusted)
        q_values[order] = np.clip(adjusted, 0.0, 1.0)
        result.loc[indices, "roi_one_sided_bootstrap_p"] = p_values
        result.loc[indices, "roi_fdr_q_value"] = q_values
    result["profit_confirmed_fdr_5pct"] = (
        result["profit_confirmed_95pct"] & result["roi_fdr_q_value"].le(0.05)
    )
    return result


def candidate_distribution(candidates: pd.DataFrame, cutoffs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cutoff in cutoffs.itertuples(index=False):
        selected = candidates[
            candidates["split"].eq(cutoff.split) & candidates["model_key"].eq(cutoff.model_key)
        ].sort_values(["upset_score", "race_id"], ascending=[False, True], kind="stable").head(
            int(cutoff.selected_target_count)
        )
        rows.append({
            "split": cutoff.split, "model_key": cutoff.model_key, "model": cutoff.model,
            "fraction": float(cutoff.fraction), "fraction_label": cutoff.fraction_label,
            "candidate_pool": int(cutoff.candidate_pool), "selected_horses": int(len(selected)),
            "actual_selected_fraction": float(len(selected) / cutoff.candidate_pool),
            "mean_market_rank": float(selected["market_rank"].mean()),
            "mean_win_odds": float(selected["winOdds"].mean()),
            "actual_win_rate": float(selected["win"].mean()),
            "actual_place_rate": float(selected["place"].mean()),
            "stored_upset_B_rate": float(selected["upset_B"].mean()),
            "exact_top3_upset_rate": float(selected["top3_upset"].mean()),
            "market_rank_2_3": int(selected["market_rank"].between(2, 3).sum()),
            "market_rank_4_6": int(selected["market_rank"].between(4, 6).sum()),
            "market_rank_7_plus": int(selected["market_rank"].ge(7).sum()),
            "win_odds_lt_10": int(selected["winOdds"].lt(10).sum()),
            "win_odds_10_20": int(selected["winOdds"].between(10, 20, inclusive="left").sum()),
            "win_odds_20_30": int(selected["winOdds"].between(20, 30, inclusive="left").sum()),
            "win_odds_30_plus": int(selected["winOdds"].ge(30).sum()),
        })
    return pd.DataFrame(rows)


def local_reasons(
    candidates: pd.DataFrame, features_by_model: dict[str, list[str]], data: dict[str, pd.DataFrame],
    meta: dict[str, pd.DataFrame], models: dict[str, BaseMarginBooster], top_n: int = 30,
) -> pd.DataFrame:
    rows = []
    test_groups = {str(key): np.asarray(value, dtype=int)
                   for key, value in meta["test"].groupby("race_id", sort=False).indices.items()}
    for model_key, group in candidates[candidates["split"].eq("test")].groupby("model_key", sort=False):
        features = features_by_model[model_key]
        X = data["test"][features].to_numpy(dtype=float)
        model = models[model_key]
        for candidate in group.nlargest(top_n, "upset_score").itertuples(index=False):
            position = int(candidate.row_position)
            race_positions = test_groups[str(candidate.race_id)]
            original = X[position:position + 1].copy()
            base_margin = float(model.margin(original)[0])
            effects = []
            for feature_index, feature in enumerate(features):
                counterfactual = original.copy()
                counterfactual[0, feature_index] = float(np.median(X[race_positions, feature_index]))
                effect = base_margin - float(model.margin(counterfactual)[0])
                effects.append((feature, effect))
            positive = sorted((item for item in effects if item[1] > 0), key=lambda item: -item[1])[:5]
            rows.append({
                "model_key": model_key, "race_id": str(candidate.race_id), "entry_id": str(candidate.entry_id),
                "hrName": str(candidate.hrName), "market_rank": int(candidate.market_rank),
                "winOdds": float(candidate.winOdds), "upset_score": float(candidate.upset_score),
                "actual_win": int(candidate.win),
                "top_positive_local_reasons": " | ".join(f"{name}:{value:+.4f}" for name, value in positive),
                "local_reason_method": "candidate margin minus margin after replacing one feature by race median",
            })
    return pd.DataFrame(rows)


def fmt_pct(value) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):+.2%}"


def roi_matrix(summary: pd.DataFrame, model_key: str) -> pd.DataFrame:
    test = summary[summary["split"].eq("test") & summary["model_key"].eq(model_key)].copy()
    matrix = test.pivot(index="bet_type", columns="fraction_label", values="roi").reindex(BET_TYPES)
    return matrix.rename_axis(None, axis=0).rename_axis(None, axis=1).map(fmt_pct).reset_index(names="승식")


def detailed_direct_table(summary: pd.DataFrame, model_key: str) -> pd.DataFrame:
    frame = summary[
        summary["split"].eq("test") & summary["model_key"].eq(model_key)
        & summary["bet_type"].isin(["단승", "연승"])
    ][["bet_type", "fraction_label", "bets", "wins", "hit_rate", "average_selected_odds", "roi",
       "roi_ci_low_daily_block", "roi_ci_high_daily_block", "flat_1pct_mdd", "per_bet_sharpe",
       "kelly_total_return", "kelly_mdd", "roi_without_largest_winning_return",
       "odds_le_30_roi", "positive_model_ev_bets", "positive_model_ev_roi",
       "roi_fdr_q_value", "profit_confirmed_fdr_5pct"]].copy()
    for column in ["hit_rate", "roi", "roi_ci_low_daily_block", "roi_ci_high_daily_block",
                   "flat_1pct_mdd", "kelly_total_return", "kelly_mdd",
                   "roi_without_largest_winning_return", "odds_le_30_roi",
                   "positive_model_ev_roi", "roi_fdr_q_value"]:
        frame[column] = frame[column].map(fmt_pct)
    frame["average_selected_odds"] = frame["average_selected_odds"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.3f}"
    )
    frame["per_bet_sharpe"] = frame["per_bet_sharpe"].map(
        lambda value: "N/A" if pd.isna(value) else f"{value:.3f}"
    )
    return frame


def build_reports(
    out: Path, importance: pd.DataFrame, selected_features: list[str], metrics: pd.DataFrame,
    cutoffs: pd.DataFrame, summary: pd.DataFrame, candidates: pd.DataFrame,
    distribution: pd.DataFrame, reasons: pd.DataFrame, config: dict, validation: dict,
) -> None:
    top_importance = importance.head(30)[[
        "rank", "feature", "positive_importance_weight", "mean_upset_tail_realized_edge_drop",
        "ci_low_daily_block", "ci_high_daily_block", "selected_for_retraining", "ci_confirmed_positive",
    ]].copy()
    top_importance["positive_importance_weight"] = top_importance["positive_importance_weight"].map(
        lambda value: f"{value:.3%}"
    )
    for column in ["mean_upset_tail_realized_edge_drop", "ci_low_daily_block", "ci_high_daily_block"]:
        top_importance[column] = top_importance[column].map(lambda value: f"{value:+.6f}")
    test_metrics = metrics[metrics["split"].eq("test")][[
        "model", "races", "delta_ll_per_race", "delta_ll_ci_low_daily_block",
        "delta_ll_ci_high_daily_block", "delta_ll_confirmed_positive_95pct",
        "model_top1_hit_rate", "market_top1_hit_rate",
    ]].copy()
    for column in ["delta_ll_per_race", "delta_ll_ci_low_daily_block", "delta_ll_ci_high_daily_block"]:
        test_metrics[column] = test_metrics[column].map(lambda value: f"{value:+.6f}")
    for column in ["model_top1_hit_rate", "market_top1_hit_rate"]:
        test_metrics[column] = test_metrics[column].map(lambda value: f"{value:.2%}")
    cutoff_view = cutoffs.copy()
    cutoff_view["fraction"] = cutoff_view["fraction"].map(lambda value: f"{value:.0%}")
    cutoff_view["actual_selected_fraction"] = cutoff_view["actual_selected_fraction"].map(
        lambda value: f"{value:.2%}"
    )
    cutoff_view["score_threshold"] = cutoff_view["score_threshold"].map(
        lambda value: f"{value:.8f}"
    )
    distribution_view = distribution[distribution["split"].eq("test")].copy()
    for column in ["actual_selected_fraction", "actual_win_rate", "actual_place_rate",
                   "stored_upset_B_rate", "exact_top3_upset_rate"]:
        distribution_view[column] = distribution_view[column].map(lambda value: f"{value:.2%}")
    top_horses = candidates[candidates["split"].eq("test")].sort_values(
        ["model_key", "upset_score"], ascending=[True, False], kind="stable"
    ).groupby("model_key", sort=False).head(10)
    top_horses = top_horses.merge(
        reasons[["model_key", "race_id", "entry_id", "top_positive_local_reasons"]],
        on=["model_key", "race_id", "entry_id"], how="left", validate="one_to_one",
    )[["model", "race_id", "hrName", "market_rank", "winOdds", "model_place_probability",
       "market_place_probability", "predicted_edge", "upset_score", "win", "place",
       "top3_upset", "top_positive_local_reasons"]]
    for column in ["model_place_probability", "market_place_probability", "predicted_edge"]:
        top_horses[column] = top_horses[column].map(lambda value: f"{value:.3%}")
    top_horses["upset_score"] = top_horses["upset_score"].map(lambda value: f"{value:.6f}")
    top_horses["winOdds"] = top_horses["winOdds"].map(lambda value: f"{value:.1f}")

    confirmed = summary[summary["split"].eq("test") & summary["profit_confirmed_fdr_5pct"]]
    unadjusted_positive = summary[summary["split"].eq("test") & summary["profit_confirmed_95pct"]]
    roi_warnings = summary[summary["split"].eq("test") & summary["roi"].ge(0.30)]
    metric_glossary = """## 지표 개념과 산식

- **이변 점수**: `모델의 연승권 진입확률 - 시장 q_plc`. 시장 인기 하위 50% 말 중 이 값이 가장 큰 한 마리를 경주별 후보로 삼는다.
- **이변 피처 순열 중요도**: train 내부 마지막 20%에서 한 피처를 교란했을 때 상위 10·20·30·40% 이변 후보의 평균 실현 EDGE(`place-q_plc`)가 얼마나 감소하는지 나타낸다.
- **정규화 중요도 가중치**: 양의 순열 중요도를 합계 1로 정규화한 비중이다.
- **ΔLL/경주**: `(모델 LL-시장 LL)/경주 수`. 양수일수록 시장보다 확률분포가 정확하다.
- **EDGE**: 같은 사건에 대한 `모델확률-시장확률`이다.
- **ROI**: `(총 환급-총 베팅액)/총 베팅액`이다. 공식 최종 배당에는 공제가 반영돼 있어 20%·27%를 다시 차감하지 않는다.
- **Hit Rate**: 선택 베팅 중 적중 비율이다.
- **MDD**: 누적자산의 이전 최고점 대비 최대 하락률이다.
- **Sharpe**: 평균 수익을 수익 변동성으로 나눈 안정성 지표다.
- **95% CI**: 경주일 블록 5,000회 부트스트랩 구간이며, 하한이 0보다 클 때만 수익·정보 개선을 확인한다.
- **FDR q-value**: 56개 ROI 비교의 우연한 양성을 통제하는 Benjamini-Hochberg 보정값이다. `q<=0.05`와 CI 하한 양수를 함께 만족해야 기계적 통계 통과로 표시한다.
"""
    common_method = f"""## 사전 고정 설계

- 프로젝트 원본에서 정확히 재현된 `upset_B=(pop_pct>=0.5 AND place=1)`를 양성 이변 정의로 사용했다. `upset_B` 자체는 사후 라벨이므로 입력에는 넣지 않았다.
- 각 경주에서 시장 인기 하위 50% 중 양의 연승 EDGE를 가진 말 가운데 이변 점수가 가장 높은 한 마리만 후보로 삼았다.
- 상위 10·20·30·40% 비율 자체를 사전에 고정했다. 각 split의 이변 점수만 내림차순 정렬해 정확히 `ceil(후보수×비율)`개를 골랐으며 결과 라벨은 경계 계산에 쓰지 않았다.
- 피처 중요도는 train을 앞 70% 학습·다음 10% 트리수 조정·마지막 20% 중요도 평가로 나눠 산출했다. validation과 test 결과는 피처 선별에 사용하지 않았다.
- 피처 재학습 임계값은 **정규화된 양의 순열 중요도 {IMPORTANCE_WEIGHT_THRESHOLD:.1%} 이상**인 운영 기준이다. 이는 p-value나 통계적 유의성 기준이 아니다.
- 선택 피처는 {len(selected_features)}개이며 목록은 `selected_upset_features.json`에 저장했다.
- 배당은 `ln(q)` 고정 offset과 평가에만 사용했고 일반 피처행렬에는 넣지 않았다. 최종 발매풀 5개(`winAmt·plcAmt·totalAmt·log_winAmt·liq_per_horse`)도 제외해 {config['full_feature_count_after_pool_exclusion']}피처를 사용했다.
- 복합 승식은 이변 말 한 마리를 반드시 포함하는 모든 가능한 티켓 중 모델 사건확률이 최대인 티켓을 택했다. 복합 시장확률은 q 기반 PL/Harville proxy이며 실제 풀 확률은 아니다.
- 재학습 타깃은 승자(`win`)이고, 연승확률은 경주별 승리확률에서 Plackett-Luce로 파생했다. `upset_B` 자체를 직접 분류한 모델은 아니다.
"""

    full_report = f"""# 감사 후 전체 {config['full_feature_count_after_pool_exclusion']}피처 이변 말 탐색 및 수익성 보고서

{metric_glossary}
{common_method}
## 이변 피처 중요도 상위 30개

{markdown_table(top_importance)}

## split별 정확 상위 비율과 점수 경계

{markdown_table(cutoff_view[cutoff_view['model_key'].eq('full_upset_base_margin')])}

## test 이변 말 구성과 실제 적중

{markdown_table(distribution_view[distribution_view['model_key'].eq('full_upset_base_margin')])}

## 전체 피처 모델 test ΔLL

{markdown_table(test_metrics[test_metrics['model'].eq(MODEL_NAMES['full_upset_base_margin'])])}

## 전체 피처 모델 test 승식별 ROI

{markdown_table(roi_matrix(summary, 'full_upset_base_margin'))}

## 단승·연승 안정성 상세

{markdown_table(detailed_direct_table(summary, 'full_upset_base_margin'))}

## test 상위 이변 말 예시

{markdown_table(top_horses[top_horses['model'].eq(MODEL_NAMES['full_upset_base_margin'])])}

위 말별 근거는 후보 피처를 같은 경주 중앙값으로 한 번씩 치환했을 때 시장 보정 margin이 얼마나 감소하는지를 계산한 국소 반사실 설명이다. 인과효과가 아니다.
"""
    (out / "01_full_feature_upset_report.md").write_text(full_report, encoding="utf-8")

    subset_report = f"""# 상위 이변 피처 전용 재학습 보고서

{metric_glossary}
{common_method}
## 선택 피처와 모델 비교

감사 후 {config['full_feature_count_after_pool_exclusion']}개 중 {len(selected_features)}개를 선택했다. 임계값은 test 확인 전에 정한 0.5%다.

{markdown_table(top_importance[top_importance['selected_for_retraining']])}

## 전체 피처와 선택 피처 test ΔLL 비교

{markdown_table(test_metrics)}

## 선택 피처 모델 test 승식별 ROI

{markdown_table(roi_matrix(summary, 'selected_upset_base_margin'))}

## 선택 피처 모델 단승·연승 안정성

{markdown_table(detailed_direct_table(summary, 'selected_upset_base_margin'))}

## 선택 피처 모델 test 상위 이변 말 예시

{markdown_table(top_horses[top_horses['model'].eq(MODEL_NAMES['selected_upset_base_margin'])])}
"""
    (out / "02_selected_feature_retrain_report.md").write_text(subset_report, encoding="utf-8")

    final = f"""# 이변 피처·이변 말·상위 10~40% 수익성 최종 보고서

{metric_glossary}
## 한 줄 결론

재사용 test에서 CI 하한 양수와 FDR 5%를 함께 통과한 ROI 셀은 **{len(confirmed)}개**다. 다만 이 test는 과거 분석에서 이미 반복 열람됐으므로 통과하더라도 독립적인 수익 확정으로 부르지 않는다. 보정 전 CI 하한만 양수인 셀은 {len(unadjusted_positive)}개다.

{common_method}
## 시장오류 보정 피처 순위

{markdown_table(top_importance)}

## 전체 피처와 상위 피처 재학습 ΔLL

{markdown_table(test_metrics)}

## 감사 후 전체 {config['full_feature_count_after_pool_exclusion']}피처 이변 전략 ROI

{markdown_table(roi_matrix(summary, 'full_upset_base_margin'))}

## 상위 {len(selected_features)}피처 재학습 이변 전략 ROI

{markdown_table(roi_matrix(summary, 'selected_upset_base_margin'))}

## CI+FDR 기계적 통과 셀(독립 확정 아님)

{markdown_table(confirmed[['model', 'fraction_label', 'bet_type', 'bets', 'roi', 'roi_ci_low_daily_block', 'roi_ci_high_daily_block', 'roi_fdr_q_value']] if len(confirmed) else pd.DataFrame())}

## 30% 이상 ROI 누수·과적합 경고 대상

{markdown_table(roi_warnings[['model', 'fraction_label', 'bet_type', 'bets', 'wins', 'roi', 'roi_ci_low_daily_block', 'roi_ci_high_daily_block', 'roi_without_largest_winning_return', 'largest_winning_dividend']] if len(roi_warnings) else pd.DataFrame())}

30% 이상 점추정치는 자동으로 경고 대상으로 분류했다. CI, 적중 건수, 특정 고배당 한 건 의존성을 함께 확인해야 하며 점추정치만으로 채택하지 않는다.

## 검증 상태와 한계

- 1차 검증 {validation['passed_checks']}개가 PASS다. 별도 독립 검증 결과는 `independent_validation.json`에 기록한다.
- 비율은 사전에 고정했고 test 점수의 순위만 사용했으며 test 착순·환급 결과는 경계 산출에 쓰지 않았다. 이는 기간 전체를 순위화한 탐색 분석이라 실시간 배치 규칙과는 다르다.
- 이 test는 과거 작업에서 이미 여러 번 열람된 구간이다. 최종 수익 확인에는 이후 신규 기간 또는 nested walk-forward가 필요하다.
- 현재 배당은 최종배당이다. 실전 배치 판단에는 실제 주문 시점 배당 스냅샷이 필요하다.
- 복합 승식은 패배 조합의 선택별 사전 배당이 없어 Kelly를 산출하지 않았다.
- 상세 CI·MDD·Sharpe·평균배당·Kelly 값은 `upset_bet_summary.csv`에 모두 보존했다.
"""
    (out / "final_report.md").write_text(final, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--market-reference", type=Path, required=True)
    parser.add_argument("--official-dividends", type=Path, required=True)
    parser.add_argument("--prior-anchor-report", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    market_path = args.market_reference.resolve()
    official_path = args.official_dividends.resolve()
    prior_anchor = args.prior_anchor_report.resolve()
    model_root = args.model_root.resolve()
    out = args.report_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    source_features, data, metadata = load_data(data_root)
    meta = attach_upset_market_columns(attach_market(metadata, market_path), market_path)
    data, meta, clean_audit = clean_splits(data, meta)
    pool_features_present = [feature for feature in source_features if feature in FINAL_POOL_FEATURES]
    features = [feature for feature in source_features if feature not in FINAL_POOL_FEATURES]
    manifest = json.loads((data_root / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    forbidden = set(manifest["market_or_result_columns_excluded_from_features"])
    check(len(source_features) == int(manifest["model_feature_count"]),
          f"source feature count matches manifest ({len(source_features)})", checks)
    check(not pool_features_present, "all five final-pool columns already absent from clean dataset", checks)
    check(features == source_features, "upset model uses the complete audited clean feature order", checks)
    check(not (set(features) & forbidden), "odds/result columns absent from feature matrix", checks)
    check(all(np.isfinite(data[split][features].to_numpy(dtype=float)).all()
              for split in ("train", "valid", "test")), "feature matrices finite", checks)
    check(max(meta["train"]["race_id"].astype(str).str[:8].astype(int))
          < min(meta["valid"]["race_id"].astype(str).str[:8].astype(int)), "train precedes validation", checks)
    check(max(meta["valid"]["race_id"].astype(str).str[:8].astype(int))
          < min(meta["test"]["race_id"].astype(str).str[:8].astype(int)), "validation precedes test", checks)
    check(sha256(market_path) == manifest["reference_sha256"], "market reference hash matches dataset manifest", checks)
    clean_race_counts = {split: clean_audit[split]["clean_races"] for split in clean_audit}
    check(all(value > 0 for value in clean_race_counts.values()),
          f"complete-order valid-market races exist in every split: {clean_race_counts}", checks)
    check(all(set(meta[split]["meet_cd"].astype(int).unique()) == {1, 3} for split in SPLITS),
          "Seoul and Bugyeong remain in every clean split", checks)
    check((meta["train"]["upset_B"].astype(int)
           == ((meta["train"]["pop_pct"] >= 0.5) & meta["train"]["place"].eq(1)).astype(int)).all(),
          "stored upset_B exactly reproduces project definition", checks)

    print("TRAIN-INTERNAL IMPORTANCE MODEL (70% FIT / 10% TUNE / 20% IMPORTANCE)")
    importance_model, importance_data, importance_meta, importance_audit = fit_importance_model(
        features, data["train"], meta["train"]
    )
    importance, importance_contributions = permutation_importance(
        features, importance_data, importance_meta, importance_model
    )
    selected_features = importance.loc[importance["selected_for_retraining"], "feature"].tolist()
    check(len(selected_features) > 0, "importance threshold selects at least one feature", checks)
    check(all(feature in features for feature in selected_features), "selected features belong to audited clean matrix", checks)
    check(not (set(selected_features) & forbidden), "selected features exclude odds/results", checks)
    check(not (set(selected_features) & set(FINAL_POOL_FEATURES)), "selected features exclude final-pool columns", checks)
    importance.to_csv(out / "upset_feature_importance.csv", index=False, encoding="utf-8-sig")
    importance_contributions.to_csv(
        out / "upset_feature_importance_race_contributions.csv", index=False, encoding="utf-8-sig"
    )
    (out / "selected_upset_features.json").write_text(json.dumps({
        "selection_scope": "train-internal chronological holdout only",
        "importance_split": importance_audit,
        "importance_method": "drop in mean top-10/20/30/40% realized upset edge place-q_plc",
        "upset_universe": "pop_pct >= 0.5; stored upset_B target is place == 1",
        "positive_importance_weight_threshold": IMPORTANCE_WEIGHT_THRESHOLD,
        "permutation_repeats": PERMUTATION_REPEATS,
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FEATURE SELECTION COMPLETE: threshold={IMPORTANCE_WEIGHT_THRESHOLD:.3%}, selected={len(selected_features)}")

    print(f"TRAIN FULL {len(features)}-FEATURE UPSET BASE MARGIN")
    full_valid_model, full_final_model, full_probabilities, full_trees = fit_base_margin(features, data, meta)
    print(f"FULL MODEL COMPLETE: trees={full_trees}")
    print("TRAIN SELECTED-FEATURE UPSET BASE MARGIN")
    selected_valid_model, selected_final_model, selected_probabilities, selected_trees = fit_base_margin(
        selected_features, data, meta
    )
    print(f"SELECTED MODEL COMPLETE: features={len(selected_features)}, trees={selected_trees}")
    probabilities = {
        "full_upset_base_margin": full_probabilities,
        "selected_upset_base_margin": selected_probabilities,
    }
    models = {
        "full_upset_base_margin": full_final_model,
        "selected_upset_base_margin": selected_final_model,
    }
    features_by_model = {
        "full_upset_base_margin": features,
        "selected_upset_base_margin": selected_features,
    }
    joblib.dump({"model": full_final_model, "features": features, "n_trees": full_trees,
                 "offset": "log(q), fixed coefficient 1", "upset_score": "P_model(place)-q_plc"},
                model_root / "full_upset_base_margin.joblib")
    joblib.dump({"model": selected_final_model, "features": selected_features, "n_trees": selected_trees,
                 "offset": "log(q), fixed coefficient 1", "importance_weight_threshold": IMPORTANCE_WEIGHT_THRESHOLD,
                 "upset_score": "P_model(place)-q_plc"},
                model_root / "selected_upset_base_margin.joblib")

    metric_rows, contribution_frames, prediction_frames, candidate_frames = [], [], [], []
    for model_key in MODEL_KEYS:
        for split in ("valid", "test"):
            p = probabilities[model_key][split]
            row, contribution = probability_metrics(split, model_key, meta[split], p)
            metric_rows.append(row)
            contribution_frames.append(contribution)
            pred = meta[split][["race_id", "entry_id", "win", "q"]].copy()
            pred.insert(0, "model_key", model_key)
            pred.insert(0, "split", split)
            pred["model_probability"] = p
            pred["predicted_edge"] = p - pred["q"].to_numpy(dtype=float)
            pred["model_place_probability"] = model_place_probability(meta[split], p)
            pred["market_place_probability"] = meta[split]["q_plc"].to_numpy(dtype=float)
            pred["upset_place_edge"] = pred["model_place_probability"] - pred["market_place_probability"]
            prediction_frames.append(pred)
            candidate_frames.append(find_upset_candidates(split, model_key, meta[split], p))
    metrics = pd.DataFrame(metric_rows)
    contributions = pd.concat(contribution_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    cutoffs = score_fraction_cutoffs(candidates)
    bets = build_bets(candidates, cutoffs, meta, probabilities, official_path)
    summary = add_roi_fdr(summarize_bets(bets))
    distribution = candidate_distribution(candidates, cutoffs)
    reasons = local_reasons(candidates, features_by_model, data, meta, models)
    metrics.to_csv(out / "model_delta_ll_metrics.csv", index=False, encoding="utf-8-sig")
    contributions.to_csv(out / "model_delta_ll_race_contributions.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out / "runner_probabilities.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(out / "upset_horse_candidates.csv", index=False, encoding="utf-8-sig")
    cutoffs.to_csv(out / "score_fraction_cutoffs.csv", index=False, encoding="utf-8-sig")
    distribution.to_csv(out / "upset_candidate_distribution.csv", index=False, encoding="utf-8-sig")
    bets.to_csv(out / "upset_bet_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "upset_bet_summary.csv", index=False, encoding="utf-8-sig")
    reasons.to_csv(out / "top_upset_horse_local_reasons.csv", index=False, encoding="utf-8-sig")

    # Reload final artifacts and independently reproduce their test probabilities in-process.
    for model_key, artifact_name in (
        ("full_upset_base_margin", "full_upset_base_margin.joblib"),
        ("selected_upset_base_margin", "selected_upset_base_margin.joblib"),
    ):
        artifact = joblib.load(model_root / artifact_name)
        artifact_features = artifact["features"]
        X_test = data["test"][artifact_features].to_numpy(dtype=float)
        lnq_test = np.log(np.clip(meta["test"]["q"].to_numpy(dtype=float), 1e-15, 1.0))
        reloaded = probability_from_margin(meta["test"], lnq_test + artifact["model"].margin(X_test))
        check(float(np.max(np.abs(reloaded - probabilities[model_key]["test"]))) <= 1e-12,
              f"{model_key}: artifact reload reproduces test", checks)
    check(metrics["maximum_abs_probability_sum_error"].max() <= 1e-12,
          "all race probability sums equal one", checks)
    check(set(cutoffs["fraction"]) == set(FRACTIONS), "10/20/30/40 percent cutoffs present", checks)
    check(np.allclose(cutoffs["actual_selected_fraction"],
                      cutoffs["selected_target_count"] / cutoffs["candidate_pool"]),
          "reported selected fractions exactly match ceil candidate counts", checks)
    check(len(summary[summary["split"].eq("test")]) == len(MODEL_KEYS) * len(FRACTIONS) * len(BET_TYPES),
          "all model/fraction/seven-bet test cells present", checks)
    check((candidates["pop_pct"] >= 0.5).all(), "all upset candidates are in market bottom half", checks)
    check((candidates["predicted_edge"] > 0).all(), "all upset candidates have positive EDGE", checks)

    config = {
        "status": "PASS", "data_root": str(data_root), "market_reference": str(market_path),
        "market_sha256": sha256(market_path), "official_dividends": str(official_path),
        "official_dividends_sha256": sha256(official_path), "source_feature_count": len(source_features),
        "full_feature_count_after_pool_exclusion": len(features),
        "final_pool_features_excluded": list(FINAL_POOL_FEATURES),
        "selected_feature_count": len(selected_features),
        "positive_importance_weight_threshold": IMPORTANCE_WEIGHT_THRESHOLD,
        "permutation_repeats": PERMUTATION_REPEATS, "bootstrap_repeats": BOOTSTRAPS,
        "upset_definition": "project upset_B universe pop_pct>=0.5; per race highest positive P_model(place)-q_plc",
        "upset_outcome": "stored upset_B = (pop_pct>=0.5 and place==1), reproduced exactly",
        "exact_top3_sensitivity_label": "top3_upset = (pop_pct>=0.5 and finish_order<=3)",
        "fraction_selection_rule": "within each split take exact ceil(candidate_count*fraction) by score; no outcomes",
        "importance_fit_scope": importance_audit,
        "full_model_trees": full_trees, "selected_model_trees": selected_trees,
        "prior_anchor_report": str(prior_anchor), "prior_anchor_report_sha256": sha256(prior_anchor),
        "clean_split_audit": clean_audit,
        "feature_matrix_uses_odds": False,
        "takeout_handling": "official final dividends already net of 20/27 percent; no double deduction",
        "combination_probability_limit": "q-derived PL/Harville proxy, not actual combination-pool probability",
        "split_dates": {split: [str(meta[split]["race_id"].astype(str).str[:8].min()),
                                str(meta[split]["race_id"].astype(str).str[:8].max())]
                        for split in ("train", "valid", "test")},
    }
    (out / "experiment_manifest.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = {
        "status": "PASS", "passed_checks": len(checks), "checks": checks,
        "profit_confirmed_test_cells_after_fdr": summary.loc[
            summary["split"].eq("test") & summary["profit_confirmed_fdr_5pct"],
            ["model_key", "fraction_label", "bet_type", "bets", "roi",
             "roi_ci_low_daily_block", "roi_ci_high_daily_block", "roi_fdr_q_value"],
        ].to_dict("records"),
        "unadjusted_positive_ci_test_cells": summary.loc[
            summary["split"].eq("test") & summary["profit_confirmed_95pct"],
            ["model_key", "fraction_label", "bet_type", "bets", "roi",
             "roi_ci_low_daily_block", "roi_ci_high_daily_block", "roi_fdr_q_value"],
        ].to_dict("records"),
        "roi_30pct_warnings": summary.loc[
            summary["split"].eq("test") & summary["roi"].ge(0.30),
            ["model_key", "fraction_label", "bet_type", "bets", "wins", "roi",
             "roi_ci_low_daily_block", "roi_ci_high_daily_block"],
        ].to_dict("records"),
        "model_artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in model_root.glob("*.joblib")
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
    }
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    build_reports(out, importance, selected_features, metrics, cutoffs, summary, candidates,
                  distribution, reasons, config, validation)
    commands = (
        "python -m py_compile src/training/run_upset_feature_experiment.py\n"
        "python -u -m src.training.run_upset_feature_experiment "
        "--data-root data/revised_v7_rank_preprocessed "
        "--market-reference \"C:/Users/user/Downloads/final (2).csv.gz\" "
        "--official-dividends outputs/reports/revised_v7_full_rank_rerun_20260820/kra_official_dividends/kra_official_dividends.csv "
        "--prior-anchor-report outputs/reports/market_anchor_same_test_20260822/runner_probabilities.csv "
        "--model-root models/upset_feature_experiment_20260822 "
        "--report-root outputs/reports/upset_feature_experiment_20260822\n"
        "python -m py_compile src/training/validate_upset_feature_experiment.py\n"
        "python -u -m src.training.validate_upset_feature_experiment "
        "--data-root data/revised_v7_rank_preprocessed "
        "--market-reference \"C:/Users/user/Downloads/final (2).csv.gz\" "
        "--official-dividends outputs/reports/revised_v7_full_rank_rerun_20260820/kra_official_dividends/kra_official_dividends.csv "
        "--model-root models/upset_feature_experiment_20260822 "
        "--report-root outputs/reports/upset_feature_experiment_20260822\n"
    )
    (out / "reproduction_commands.txt").write_text(commands, encoding="utf-8")
    print(f"UPSET FEATURE EXPERIMENT PASS: checks={len(checks)}, selected_features={len(selected_features)}")
    print(metrics[metrics["split"].eq("test")][[
        "model_key", "delta_ll_per_race", "delta_ll_ci_low_daily_block", "delta_ll_ci_high_daily_block"
    ]].to_string(index=False))
    print(summary[summary["split"].eq("test") & summary["bet_type"].eq("단승")][[
        "model_key", "fraction_label", "bets", "wins", "roi", "roi_ci_low_daily_block",
        "roi_ci_high_daily_block", "roi_fdr_q_value", "profit_confirmed_fdr_5pct"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
