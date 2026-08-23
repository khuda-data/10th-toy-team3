"""검증된 EDGE 42피처 데이터로 최종 모델들을 개별 학습한다."""
from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from src.common.edge_evaluation import save_edge_result
from src.common.feature_report import write_bottom_70
from src.common.market_reference import market_metrics


DATA_ROOT = Path("data/revised_v5_preprocessed")
MODEL_ROOT = Path("models/revised_v5_preprocessed_full")
DATA_FILE_TOKEN = "revised_v5"
MODEL_CHOICES = ("random_forest", "xgboost", "lightgbm", "catboost", "deep", "plackett_luce")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_data() -> tuple[list[str], dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    manifest = json.loads((DATA_ROOT / "preprocessing_manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((DATA_ROOT / "preprocessing_validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS":
        raise RuntimeError("전처리 검증 상태가 PASS가 아닙니다.")
    features = manifest["model_features"]
    data, metadata = {}, {}
    for split in ("train", "valid", "test"):
        data[split] = pd.read_csv(DATA_ROOT / f"{split}_{DATA_FILE_TOKEN}_numeric_scaled.csv", low_memory=False)
        metadata[split] = pd.read_csv(DATA_ROOT / f"{split}_metadata.csv", low_memory=False)
        matrix = data[split][features]
        if len(data[split]) != len(metadata[split]):
            raise ValueError(f"{split}: 데이터/메타데이터 행 수 불일치")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
            raise TypeError(f"{split}: 비숫자 모델 피처 존재")
        if not np.isfinite(matrix.to_numpy(dtype=float)).all():
            raise ValueError(f"{split}: NaN/무한값 존재")
        if not np.array_equal(data[split]["win"].to_numpy(), metadata[split]["win"].to_numpy()):
            raise ValueError(f"{split}: 타깃 정렬 불일치")
    print(f"DATA PASS: features={len(features)}, rows="
          f"{len(data['train'])}/{len(data['valid'])}/{len(data['test'])}")
    return features, data, metadata


def race_groups(frame: pd.DataFrame) -> list[np.ndarray]:
    groups = []
    for positions in frame.groupby("race_id_code", sort=False).indices.values():
        positions = np.asarray(positions)
        if len(positions) >= 2 and frame.iloc[positions]["win"].sum() == 1:
            groups.append(positions)
    return groups


def race_normalize(frame: pd.DataFrame, probability: np.ndarray) -> np.ndarray:
    result = np.zeros(len(probability), dtype=float)
    for positions in frame.groupby("race_id_code", sort=False).indices.values():
        positions = np.asarray(positions)
        values = np.clip(probability[positions], 1e-12, None)
        result[positions] = values / values.sum()
    return result


def calibration_mae(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    quantiles = pd.qcut(probability, bins, labels=False, duplicates="drop")
    errors = [abs(target[quantiles == value].mean() - probability[quantiles == value].mean())
              for value in np.unique(quantiles)]
    return float(np.mean(errors))


def top1_hit(frame: pd.DataFrame, target: np.ndarray, probability: np.ndarray) -> float:
    hits = total = 0
    for positions in frame.groupby("race_id_code", sort=False).indices.values():
        positions = np.asarray(positions)
        if target[positions].sum() != 1:
            continue
        hits += int(target[positions[np.argmax(probability[positions])]] == 1)
        total += 1
    return hits / max(total, 1)


def evaluate(
    split: str,
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    raw_probability: np.ndarray,
    out_dir: Path,
) -> tuple[dict, np.ndarray]:
    raw_probability = np.asarray(raw_probability, dtype=float)
    if len(raw_probability) != len(frame) or not np.isfinite(raw_probability).all():
        raise ValueError(f"{split}: 예측 길이 또는 유한값 검사 실패")
    probability = race_normalize(frame, raw_probability)
    target = frame["win"].to_numpy(dtype=int)
    metrics = {
        "raw_roc_auc": float(roc_auc_score(target, raw_probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "pr_auc": float(average_precision_score(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(target, probability)),
        "cal_mae": calibration_mae(target, probability),
        "top1_hit_rate": top1_hit(frame, target, probability),
        "avg_winner_prob": float(probability[target == 1].mean()),
    }
    market = market_metrics(metadata, probability, name=split)
    if market is not None:
        metrics["market_roc_auc_ref"] = market["roc_auc"]
        metrics["market_pr_auc_ref"] = market["pr_auc"]
    if split == "test":
        metrics["edge_test"] = save_edge_result(metadata, probability, str(out_dir), name="Test")
    print(f"{split.upper()} PASS: auc={metrics['roc_auc']:.6f}, pr={metrics['pr_auc']:.6f}, "
          f"top1={metrics['top1_hit_rate']:.6f}")
    return metrics, probability


def save_common(out_dir: Path, model_name: str, features: list[str], metrics: dict, extra: dict) -> None:
    manifest = {
        "model": model_name,
        "data": str(DATA_ROOT),
        "preprocessing_validation": str(DATA_ROOT / "preprocessing_validation.json"),
        "feature_count": len(features),
        "features": features,
        **extra,
    }
    (out_dir / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def train_random_forest(features, data, metadata) -> None:
    from sklearn.ensemble import RandomForestClassifier

    out = MODEL_ROOT / "random_forest"
    out.mkdir(parents=True, exist_ok=True)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", n_jobs=1, random_state=42,
    )
    model.fit(data["train"][features], data["train"]["win"])
    metrics = {}
    for split in ("train", "valid", "test"):
        probability = model.predict_proba(data[split][features])[:, 1]
        metrics[split], _ = evaluate(split, data[split], metadata[split], probability, out)
    joblib.dump(model, out / "model.joblib")
    write_bottom_70(features, model.feature_importances_, str(out / "bottom_70_feature_importance.csv"), "gini_importance")
    save_common(out, "Random Forest", features, metrics, {
        "configuration": "300 trees, max_depth=8, min_samples_leaf=20, balanced, seed=42"
    })


def train_xgboost(features, data, metadata) -> None:
    from xgboost import XGBClassifier

    out = MODEL_ROOT / "xgboost"
    out.mkdir(parents=True, exist_ok=True)
    target = data["train"]["win"].to_numpy(dtype=int)
    positives = int(target.sum())
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", n_estimators=2000,
        learning_rate=0.03, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=(len(target) - positives) / positives,
        early_stopping_rounds=100, random_state=42, n_jobs=-1, tree_method="hist",
    )
    model.fit(
        data["train"][features], target,
        eval_set=[(data["valid"][features], data["valid"]["win"])], verbose=False,
    )
    metrics = {}
    for split in ("valid", "test"):
        probability = model.predict_proba(data[split][features])[:, 1]
        metrics[split], _ = evaluate(split, data[split], metadata[split], probability, out)
    joblib.dump(model, out / "model.joblib")
    write_bottom_70(features, model.feature_importances_, str(out / "bottom_70_feature_importance.csv"), "gain_importance")
    save_common(out, "XGBoost", features, metrics, {
        "best_iteration": int(model.best_iteration + 1),
        "configuration": "up to 2000 trees, lr=0.03, depth=6, early_stopping=100, seed=42",
    })


def train_catboost(features, data, metadata) -> None:
    from catboost import CatBoostClassifier

    out = MODEL_ROOT / "catboost"
    out.mkdir(parents=True, exist_ok=True)
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="AUC", iterations=3000,
        learning_rate=0.03, depth=7, l2_leaf_reg=5.0,
        random_seed=42, verbose=200, od_type="Iter", od_wait=150,
        auto_class_weights="Balanced", boosting_type="Ordered", allow_writing_files=False,
    )
    model.fit(
        data["train"][features], data["train"]["win"],
        eval_set=(data["valid"][features], data["valid"]["win"]),
        use_best_model=True, verbose=200,
    )
    metrics = {}
    for split in ("valid", "test"):
        probability = model.predict_proba(data[split][features])[:, 1]
        metrics[split], _ = evaluate(split, data[split], metadata[split], probability, out)
    model.save_model(str(out / "model.cbm"))
    write_bottom_70(features, model.get_feature_importance(), str(out / "bottom_70_feature_importance.csv"), "prediction_values_change")
    save_common(out, "CatBoost ordered", features, metrics, {
        "best_iteration": int(model.get_best_iteration()),
        "configuration": "up to 3000 iterations, lr=0.03, depth=7, ordered, early_stopping=150, seed=42",
    })


def train_lightgbm(features, data, metadata) -> None:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    out = MODEL_ROOT / "lightgbm"
    out.mkdir(parents=True, exist_ok=True)
    train, valid = data["train"], data["valid"]
    x_train, x_valid = train[features], valid[features]
    y_train = train["win"].to_numpy(dtype=int)
    y_valid = valid["win"].to_numpy(dtype=int)
    group_train = train.groupby("race_id_code", sort=False).size().to_numpy()
    group_valid = valid.groupby("race_id_code", sort=False).size().to_numpy()
    common = dict(
        learning_rate=0.03, num_leaves=127, max_depth=-1, min_data_in_leaf=40,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=1.0, verbosity=-1, seed=42,
    )
    rank_model = lgb.train(
        {**common, "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1],
         "lambdarank_truncation_level": 20},
        lgb.Dataset(x_train, label=y_train, group=group_train, free_raw_data=False),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(x_valid, label=y_valid, group=group_valid)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)],
    )
    binary_model = lgb.train(
        {**common, "objective": "binary", "metric": "auc",
         "scale_pos_weight": float((y_train == 0).sum() / (y_train == 1).sum())},
        lgb.Dataset(x_train, label=y_train, free_raw_data=False),
        num_boost_round=3000,
        valid_sets=[lgb.Dataset(x_valid, label=y_valid)],
        callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(200)],
    )

    def zscore(values):
        return (values - values.mean()) / (values.std() + 1e-12)

    rank_valid = zscore(rank_model.predict(x_valid))
    binary_valid = zscore(binary_model.predict(x_valid, raw_score=True))
    blend_weight, blend_auc = 0.0, -np.inf
    for weight in np.linspace(0, 1, 21):
        score = weight * rank_valid + (1 - weight) * binary_valid
        auc = roc_auc_score(y_valid, score)
        if auc > blend_auc:
            blend_weight, blend_auc = float(weight), float(auc)
    raw_valid = scores_to_probs(
        blend_weight * rank_valid + (1 - blend_weight) * binary_valid,
        race_groups(valid),
    )
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999)
    calibrator.fit(raw_valid, y_valid)

    def predict(frame):
        matrix = frame[features]
        rank_score = zscore(rank_model.predict(matrix))
        binary_score = zscore(binary_model.predict(matrix, raw_score=True))
        raw = scores_to_probs(
            blend_weight * rank_score + (1 - blend_weight) * binary_score,
            race_groups(frame),
        )
        return race_normalize(frame, calibrator.predict(raw))

    metrics = {}
    for split in ("valid", "test"):
        metrics[split], _ = evaluate(split, data[split], metadata[split], predict(data[split]), out)
    rank_model.save_model(str(out / "lgb_rank.txt"))
    binary_model.save_model(str(out / "lgb_binary.txt"))
    with (out / "calibrator.pkl").open("wb") as file:
        pickle.dump(calibrator, file)
    write_bottom_70(features, rank_model.feature_importance("gain"),
                    str(out / "bottom_70_feature_importance.csv"), "gain")
    save_common(out, "LightGBM rank+binary", features, metrics, {
        "blend_w_rank": blend_weight, "valid_blend_auc": blend_auc,
        "rank_best_iteration": rank_model.best_iteration,
        "binary_best_iteration": binary_model.best_iteration,
        "configuration": "LambdaRank + binary, up to 3000 rounds, lr=0.03, early_stopping=150, isotonic",
    })


def torch_components():
    import torch
    import torch.nn as nn

    class NumericHorseNet(nn.Module):
        def __init__(self, input_size: int):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_size, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.25),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.25),
            )
            self.head = nn.Linear(128, 1)

        def forward(self, values):
            return self.head(self.mlp(values)).squeeze(-1)

    return torch, nn, NumericHorseNet


def predict_scores(model, matrix: np.ndarray, device: str, batch_size: int = 1024) -> np.ndarray:
    torch, _, _ = torch_components()
    result = np.zeros(len(matrix), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(matrix), batch_size):
            values = torch.from_numpy(matrix[start:start + batch_size]).to(device)
            result[start:start + batch_size] = model(values).cpu().numpy()
    return result


def scores_to_probs(scores: np.ndarray, groups: list[np.ndarray], temperature: float = 1.0) -> np.ndarray:
    result = np.zeros(len(scores), dtype=float)
    for positions in groups:
        values = scores[positions] / temperature
        values = values - values.max()
        exp_values = np.exp(values)
        result[positions] = exp_values / exp_values.sum()
    return result


def input_weight_importance(model, features: list[str]) -> np.ndarray:
    first_layer = model.mlp[0].weight.detach().abs().cpu().numpy()
    return first_layer.mean(axis=0)


def fit_temperature(scores: np.ndarray, target: np.ndarray, groups: list[np.ndarray]) -> float:
    candidates = np.linspace(0.5, 3.0, 26)
    losses = []
    for temperature in candidates:
        probability = scores_to_probs(scores, groups, float(temperature))
        losses.append(log_loss(target, probability, labels=[0, 1]))
    return float(candidates[int(np.argmin(losses))])


def train_listwise_member(matrix, target, train_groups, valid_matrix, valid_target, valid_groups,
                          seed: int, epochs: int, lr: float, device: str, hybrid: bool = False):
    torch, nn, NumericHorseNet = torch_components()
    import torch.nn.functional as functional

    set_seed(seed)
    model = NumericHorseNet(matrix.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_auc, best_state, bad_epochs = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.permutation(len(train_groups))
        total_loss = 0.0
        for start in range(0, len(order), 64):
            batch_groups = [train_groups[index] for index in order[start:start + 64]]
            flat = np.concatenate(batch_groups)
            scores = model(torch.from_numpy(matrix[flat]).to(device))
            pointer = np.cumsum([0] + [len(group) for group in batch_groups])
            losses = []
            for group_index, group in enumerate(batch_groups):
                local = scores[pointer[group_index]:pointer[group_index + 1]]
                winner = int(np.argmax(target[group]))
                pl_loss = functional.cross_entropy(
                    local.unsqueeze(0), torch.tensor([winner], device=device)
                )
                if hybrid:
                    losers = torch.arange(len(group), device=device) != winner
                    bt_loss = functional.softplus(-(local[winner] - local[losers])).mean()
                    pl_loss = pl_loss + bt_loss
                losses.append(pl_loss)
            loss = torch.stack(losses).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_groups)
        valid_scores = predict_scores(model, valid_matrix, device)
        valid_probability = scores_to_probs(valid_scores, valid_groups)
        auc = float(roc_auc_score(valid_target, valid_probability))
        print(f"epoch={epoch:03d} loss={total_loss/len(train_groups):.6f} valid_auc={auc:.6f}")
        if auc > best_auc + 1e-4:
            best_auc = auc
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 8:
                print(f"EARLY STOP: best_valid_auc={best_auc:.6f}")
                break
    if best_state is None:
        raise RuntimeError("딥러닝 최적 가중치가 생성되지 않았습니다.")
    model.load_state_dict(best_state)
    return model, best_auc


def train_deep(features, data, metadata) -> None:
    torch, _, _ = torch_components()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out = MODEL_ROOT / "deep"
    out.mkdir(parents=True, exist_ok=True)
    matrices = {split: data[split][features].to_numpy(dtype=np.float32) for split in data}
    targets = {split: data[split]["win"].to_numpy(dtype=np.int64) for split in data}
    groups = {split: race_groups(data[split]) for split in data}
    models, member_auc = [], []
    for member in range(5):
        print(f"DEEP MEMBER {member + 1}/5")
        model, auc = train_listwise_member(
            matrices["train"], targets["train"], groups["train"],
            matrices["valid"], targets["valid"], groups["valid"],
            1234 + member, 100, 2e-3, device,
        )
        models.append(model)
        member_auc.append(auc)
        torch.save(model.state_dict(), out / f"model_seed{member}.pt")

    scores = {}
    for split in ("valid", "test"):
        scores[split] = np.mean(
            [predict_scores(model, matrices[split], device) for model in models], axis=0
        )
    temperature = fit_temperature(scores["valid"], targets["valid"], groups["valid"])
    metrics = {}
    for split in ("valid", "test"):
        probability = scores_to_probs(scores[split], groups[split], temperature)
        metrics[split], _ = evaluate(split, data[split], metadata[split], probability, out)
    importance = np.mean([input_weight_importance(model, features) for model in models], axis=0)
    write_bottom_70(features, importance, str(out / "bottom_70_feature_importance.csv"), "input_weight_importance")
    save_common(out, "Deep listwise ensemble", features, metrics, {
        "device": device, "seeds": 5, "member_valid_auc": member_auc,
        "temperature": temperature,
        "architecture": [512, 256, 128], "epochs_max": 100, "early_stopping_patience": 8,
    })


def train_plackett_luce(features, data, metadata) -> None:
    torch, _, _ = torch_components()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    out = MODEL_ROOT / "plackett_luce"
    out.mkdir(parents=True, exist_ok=True)
    matrices = {split: data[split][features].to_numpy(dtype=np.float32) for split in data}
    targets = {split: data[split]["win"].to_numpy(dtype=np.int64) for split in data}
    groups = {split: race_groups(data[split]) for split in data}
    model, valid_auc = train_listwise_member(
        matrices["train"], targets["train"], groups["train"],
        matrices["valid"], targets["valid"], groups["valid"],
        42, 50, 1e-3, device, hybrid=True,
    )
    metrics = {}
    for split in ("valid", "test"):
        probability = scores_to_probs(predict_scores(model, matrices[split], device), groups[split])
        metrics[split], _ = evaluate(split, data[split], metadata[split], probability, out)
    torch.save(model.state_dict(), out / "model.pt")
    write_bottom_70(features, input_weight_importance(model, features),
                    str(out / "bottom_70_feature_importance.csv"), "input_weight_importance")
    save_common(out, "Plackett-Luce hybrid", features, metrics, {
        "device": device, "valid_training_auc": valid_auc,
        "architecture": [512, 256, 128], "epochs_max": 50,
        "objective": "plackett_luce + bradley_terry", "seed": 42,
    })


def main() -> None:
    global DATA_ROOT, MODEL_ROOT, DATA_FILE_TOKEN
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES)
    parser.add_argument("--dataset", choices=["full", "edge_subset", "edge_selected"], default="full")
    args = parser.parse_args()
    if args.dataset == "full":
        DATA_ROOT = Path("data/revised_v5_preprocessed")
        MODEL_ROOT = Path("models/revised_v5_preprocessed_full")
        DATA_FILE_TOKEN = "revised_v5"
    elif args.dataset == "edge_subset":
        DATA_ROOT = Path("data/edge_top_005_preprocessed")
        MODEL_ROOT = Path("models/edge_top_005_preprocessed")
        DATA_FILE_TOKEN = "edge_top_005"
    else:
        DATA_ROOT = Path("data/edge_selected_validation_preprocessed")
        MODEL_ROOT = Path("models/edge_selected_validation_preprocessed")
        DATA_FILE_TOKEN = "edge_selected"
    set_seed(42)
    features, data, metadata = load_data()
    trainers = {
        "random_forest": train_random_forest,
        "xgboost": train_xgboost,
        "lightgbm": train_lightgbm,
        "catboost": train_catboost,
        "deep": train_deep,
        "plackett_luce": train_plackett_luce,
    }
    trainers[args.model](features, data, metadata)
    print(f"MODEL TRAINING COMPLETE: {args.model}")


if __name__ == "__main__":
    main()
