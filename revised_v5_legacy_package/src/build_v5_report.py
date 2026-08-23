"""Create the final v5 metrics table, charts, and Markdown report."""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("models/v5")
OUT = Path("outputs/reports")
SPECS = {
    "Random Forest": ROOT / "random_forest/metrics_random_forest.json",
    "XGBoost": ROOT / "xgboost/metrics_xgb.json",
    "LightGBM rank+binary": ROOT / "lightgbm/metrics_ml.json",
    "CatBoost ordered": ROOT / "catboost/metrics_catboost.json",
    "Deep listwise ensemble": ROOT / "deep/metrics.json",
    "Plackett-Luce hybrid": ROOT / "plackett_luce/metrics_plackett_luce.json",
}


def read_metric(name: str, path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    test = raw.get("test", {})
    valid = raw.get("valid", {})
    edge = test.get("edge_test") or raw.get("edge_test", {})
    roc = test.get("roc_auc", raw.get("test_roc_auc"))
    pr = test.get("pr_auc", raw.get("test_pr_auc"))
    valid_roc = valid.get("roc_auc", raw.get("valid_roc_auc"))
    return {"model": name, "valid_roc_auc": valid_roc, "test_roc_auc": roc, "test_pr_auc": pr,
            "top10pct_roi": edge.get("top10pct_roi"),
            "mean_predicted_edge": edge.get("top10pct_mean_predicted_edge"),
            "mean_realized_edge": edge.get("top10pct_mean_realized_edge"),
            "edge_realized_correlation": edge.get("edge_realized_correlation"),
            "bets": edge.get("top10pct_count"), "metric_file": str(path)}


def detailed_metrics() -> pd.DataFrame:
    """Collect every recorded, model-comparable metric without inventing gaps."""
    records = []
    for name, path in SPECS.items():
        raw = json.loads(path.read_text(encoding="utf-8"))
        train, valid, test = raw.get("train", {}), raw.get("valid", {}), raw.get("test", {})
        edge = test.get("edge_test") or raw.get("edge_test", {})
        records.append({
            "model": name,
            "train_roc_auc": train.get("roc_auc"),
            "valid_roc_auc": valid.get("roc_auc", raw.get("valid_roc_auc")),
            "valid_pr_auc": valid.get("pr_auc"),
            "valid_f1": valid.get("f1"),
            "test_roc_auc": test.get("roc_auc", raw.get("test_roc_auc")),
            "test_pr_auc": test.get("pr_auc", raw.get("test_pr_auc")),
            "test_f1": test.get("f1"),
            "test_cal_mae": test.get("cal_mae"),
            "test_top1_hit_rate": test.get("top1_hit_rate"),
            "test_avg_winner_prob": test.get("avg_winner_prob"),
            "best_iteration": raw.get("best_iteration"),
            "decision_threshold": raw.get("threshold"),
            "top10_bets": edge.get("top10pct_count"),
            "mean_predicted_edge": edge.get("top10pct_mean_predicted_edge"),
            "mean_realized_edge": edge.get("top10pct_mean_realized_edge"),
            "top10_roi": edge.get("top10pct_roi"),
            "positive_edge_races": edge.get("positive_edge_count"),
            "positive_edge_roi": edge.get("positive_edge_roi"),
            "edge_realized_correlation": edge.get("edge_realized_correlation"),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "v5_model_detail_metrics.csv", index=False, encoding="utf-8-sig")
    return frame


def format_value(value: object, kind: str = "number") -> str:
    if pd.isna(value):
        return "—"
    if kind == "pct":
        return f"{float(value):.1%}"
    if kind == "int":
        return f"{int(value):,}"
    return f"{float(value):.4f}"


def apply_note_style(text: str) -> str:
    """Keep a readable study-note tone in standard Korean declarative prose."""
    return text


def split_readable_sentences(text: str, limit: int = 260) -> list[str]:
    """Keep the note-style prose scannable by grouping only one or two sentences."""
    sentences = re.split(r"(?<=[가-힣A-Za-z])\.\s+(?=[가-힣A-Z<])", text)
    if len(sentences) <= 1:
        return [text]
    groups, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        plain_length = len(re.sub(r"<[^>]+>", "", candidate))
        if current and plain_length > limit:
            groups.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        groups.append(current)
    return groups


def split_html_paragraphs(text: str) -> str:
    """Split long HTML prose blocks without touching tables, figures, or values."""
    def replace(match: re.Match[str]) -> str:
        content = match.group(1)
        if len(re.sub(r"<[^>]+>", "", content)) <= 260:
            return match.group(0)
        return "".join(f"<p>{part}</p>" for part in split_readable_sentences(content))

    return re.sub(r"<p>(.*?)</p>", replace, text, flags=re.DOTALL)


def split_markdown_paragraphs(text: str) -> str:
    """Apply the same paragraph rhythm to the Markdown copy of the report."""
    blocks = text.split("\n\n")
    output = []
    for block in blocks:
        stripped = block.lstrip()
        if (len(block) <= 260 or not stripped or stripped.startswith(("#", "|", "- ", "!["))):
            output.append(block)
        else:
            output.extend(split_readable_sentences(block))
    return "\n\n".join(output)


def model_settings_table() -> pd.DataFrame:
    """Record the actual default configuration used for this rerun."""
    return pd.DataFrame([
        ["Random Forest", "Binary classifier", "300 trees; depth 8; min leaf 20; balanced class weight; seed 42"],
        ["XGBoost", "Binary logistic", "up to 2,000 rounds; lr 0.03; depth 6; early stopping 100; hist; seed 42"],
        ["LightGBM rank+binary", "LambdaRank + binary blend", "up to 3,000 rounds each; lr 0.03; early stopping 150; rank/binary blend; isotonic calibration"],
        ["CatBoost ordered", "Ordered boosting classifier", "up to 3,000 iterations; lr 0.03; depth 7; L2 5; early stopping 150; seed 42"],
        ["Deep listwise ensemble", "Race-wise softmax", "5 seeds; 100 epochs maximum; lr 0.002; AdamW; temperature calibration"],
        ["Plackett-Luce hybrid", "Listwise + Bradley-Terry", "50 epochs; lr 0.001; AdamW; hybrid objective; seed 42"],
    ], columns=["model", "learning_form", "rerun_configuration"])


def chart(table: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    roc = table.sort_values("test_roc_auc")
    axes[0].barh(roc.model, roc.test_roc_auc, color="#377eb8")
    axes[0].axvline(0.8121586921, color="#d62728", linestyle="--", label="Market (0.8122)")
    axes[0].set(xlabel="Test ROC-AUC", xlim=(0.70, 0.83), title="Predictive discrimination")
    axes[0].legend()
    roi = table.sort_values("top10pct_roi")
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in roi.top10pct_roi]
    axes[1].barh(roi.model, roi.top10pct_roi * 100, color=colors)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Top-10% EDGE ROI (%)", title="Betting backtest (64 bets)")
    for y, value in enumerate(roi.top10pct_roi * 100):
        axes[1].text(value + (1 if value >= 0 else -1), y, f"{value:.1f}%", va="center", ha="left" if value >= 0 else "right")
    fig.tight_layout()
    fig.savefig(OUT / "v5_model_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_walk_forward(folds: list[dict]) -> None:
    frame = pd.DataFrame(folds)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    series, offsets = (("Top 10%", "top10_roi"), ("Top 20%", "top20_roi"), ("Top 30%", "top30_roi")), (-.25, 0, .25)
    for (label, column), offset, color in zip(series, offsets, ("#377eb8", "#ff7f0e", "#2ca02c")):
        values = frame[column] * 100
        x = frame.fold.to_numpy() + offset
        ax.bar(x, values, width=.23, label=label, color=color)
        for xi, value in zip(x, values):
            ax.text(xi, value + (1 if value >= 0 else -1), f"{value:.1f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xlabel="Walk-forward fold", ylabel="Conservative ROI (%)", title="Market-residual walk-forward validation", xticks=frame.fold)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "v5_market_residual_walk_forward.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def feature_consensus() -> pd.DataFrame:
    """Aggregate rank-normalized importance from every final general model.

    Raw importance magnitudes are not comparable across Gini, gain, CatBoost
    prediction-change, and neural input-weight measures.  Within-model ranks
    are therefore converted to [0, 1] percentiles before averaging.
    """
    folders = {
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "CatBoost": "catboost",
        "Deep listwise": "deep",
        "Plackett-Luce": "plackett_luce",
    }
    ranked = []
    for model, folder in folders.items():
        data = pd.read_csv(ROOT / folder / "feature_importance_full.csv")
        # XGBoost reports transformed column names; align them to original v5 names.
        data["feature"] = data["feature"].str.replace(r"^(num|cat)__", "", regex=True)
        data = data.groupby("feature", as_index=False)["rank"].min()
        maximum = data["rank"].max()
        data[model] = 1 - (data["rank"] - 1) / max(1, maximum - 1)
        ranked.append(data[["feature", model]])
    merged = ranked[0]
    for data in ranked[1:]:
        merged = merged.merge(data, on="feature", how="outer")
    score_columns = list(folders)
    merged[score_columns] = merged[score_columns].fillna(0.0)
    merged["mean_rank_score"] = merged[score_columns].mean(axis=1)
    merged["rank_score_std"] = merged[score_columns].std(axis=1)
    merged["top20_models"] = (merged[score_columns] >= (1 - 19 / 120)).sum(axis=1)
    merged = merged.sort_values(["mean_rank_score", "top20_models"], ascending=False).reset_index(drop=True)
    merged.insert(0, "consensus_rank", merged.index + 1)
    merged.to_csv(OUT / "v5_feature_importance_consensus.csv", index=False, encoding="utf-8-sig")
    return merged


def chart_feature_importance(consensus: pd.DataFrame) -> None:
    """Plot agreement across all six final models on a common rank scale."""
    top = consensus.head(15).sort_values("mean_rank_score")
    model_columns = ["Random Forest", "XGBoost", "LightGBM", "CatBoost", "Deep listwise", "Plackett-Luce"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.1, 1]})
    axes[0].barh(top.feature, top.mean_rank_score, color="#377eb8")
    axes[0].set(xlim=(0, 1), xlabel="Mean within-model rank score (1 = most important)", title="Six-model consensus importance")
    for y, row in enumerate(top.itertuples()):
        axes[0].text(min(row.mean_rank_score + .015, .96), y, f"top-20 in {row.top20_models}/6", va="center", fontsize=8)
    heat = top[model_columns].to_numpy()
    image = axes[1].imshow(heat, aspect="auto", vmin=0, vmax=1, cmap="Blues", origin="lower")
    axes[1].set(yticks=range(len(top)), yticklabels=top.feature, xticks=range(len(model_columns)), xticklabels=model_columns,
                title="Per-model rank score")
    axes[1].tick_params(axis="x", rotation=35)
    fig.colorbar(image, ax=axes[1], label="Rank score")
    fig.suptitle("Feature importance uses ranks, not incompatible raw importance units")
    fig.tight_layout()
    fig.savefig(OUT / "v5_feature_importance_consensus.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_data_split() -> None:
    """Visual inventory of the locked chronological evaluation split."""
    split = pd.DataFrame({
        "split": ["Train", "Validation", "Test"],
        "rows": [19617, 6582, 6639],
        "races": [1891, 641, 635],
        "start": pd.to_datetime(["2023-08-05", "2025-05-17", "2025-12-28"]),
        "end": pd.to_datetime(["2025-05-11", "2025-12-27", "2026-08-09"]),
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = ["#377eb8", "#ff7f0e", "#2ca02c"]
    for row, color in zip(split.itertuples(), colors):
        axes[0].barh(row.split, (row.end - row.start).days + 1, left=row.start, color=color)
        axes[0].text(row.start + (row.end - row.start) / 2, row.split, f"{row.rows:,} rows / {row.races:,} races", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    axes[0].set(title="Chronological hold-out design", xlabel="Calendar date")
    axes[0].grid(axis="y", visible=False)
    axes[1].bar(split.split, split.rows, color=colors)
    axes[1].set(title="Rows per split", ylabel="Rows")
    for x, value in enumerate(split.rows):
        axes[1].text(x, value + 350, f"{value:,}", ha="center")
    fig.tight_layout()
    fig.savefig(OUT / "v5_data_split.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_edge_summary(table: pd.DataFrame) -> None:
    """Keep ROI and edge diagnostics separate so their units are not conflated."""
    ordered = table.sort_values("top10pct_roi")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fields = [
        ("top10pct_roi", "Top-10% ROI (%)", 100, "#377eb8"),
        ("mean_realized_edge", "Mean realized EDGE", 1, "#ff7f0e"),
        ("edge_realized_correlation", "EDGE-realized correlation", 1, "#2ca02c"),
    ]
    for axis, (field, title, scale, color) in zip(axes, fields):
        values = ordered[field] * scale
        axis.barh(ordered.model, values, color=[color if value >= 0 else "#d62728" for value in values])
        axis.axvline(0, color="black", linewidth=.8)
        axis.set(title=title)
        for y, value in enumerate(values):
            label = f"{value:+.1f}%" if scale == 100 else f"{value:+.3f}"
            axis.text(value + (.7 if value >= 0 and scale == 100 else -.7 if scale == 100 else .003 if value >= 0 else -.003), y, label,
                      va="center", ha="left" if value >= 0 else "right", fontsize=9)
    fig.suptitle("Test-set EDGE diagnostics: one highest-EDGE runner per race, top 10% (64 bets)")
    fig.tight_layout()
    fig.savefig(OUT / "v5_edge_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def general_model_thresholds() -> pd.DataFrame:
    """Evaluate 10/20/30% selection rules from every final model's test CSV."""
    folder_to_name = {
        "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM rank+binary",
        "catboost": "CatBoost ordered", "deep": "Deep listwise ensemble", "plackett_luce": "Plackett-Luce hybrid",
    }
    rows = []
    for folder, model in folder_to_name.items():
        data = pd.read_csv(ROOT / folder / "edge_test_predictions.csv")
        usable = data[data["market_winOdds"].between(1.0, 9999.0, inclusive="neither") & data["edge"].notna()]
        ranked = (usable.sort_values("edge", ascending=False).groupby("race_id", sort=False, as_index=False).head(1)
                  .sort_values("edge", ascending=False))
        for fraction in (.10, .20, .30):
            selected = ranked.head(int(__import__("math").ceil(len(ranked) * fraction)))
            rows.append({"model": model, "selection_pct": int(fraction * 100), "bets": len(selected),
                         "mean_predicted_edge": selected["edge"].mean(), "mean_realized_edge": selected["realized_edge"].mean(),
                         "roi": selected["realized_return"].mean(), "win_rate": selected["win"].mean()})
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "v5_general_model_threshold_comparison.csv", index=False, encoding="utf-8-sig")
    return result


def chart_general_model_thresholds(thresholds: pd.DataFrame) -> None:
    pivot = thresholds.pivot(index="model", columns="selection_pct", values="roi").loc[
        ["Random Forest", "XGBoost", "LightGBM rank+binary", "CatBoost ordered", "Deep listwise ensemble", "Plackett-Luce hybrid"]]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(pivot)); width = .24
    for offset, pct, color in zip((-.25, 0, .25), (10, 20, 30), ("#377eb8", "#ff7f0e", "#2ca02c")):
        values = pivot[pct].to_numpy() * 100
        bars = ax.bar(x + offset, values, width, label=f"Top {pct}%", color=color)
        ax.bar_label(bars, labels=[f"{v:+.1f}" for v in values], padding=2, fontsize=8, rotation=90)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xticks=x, xticklabels=pivot.index, ylabel="Raw test ROI (%)", title="General-model test ROI by EDGE selection range")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="One highest-EDGE runner per race")
    fig.tight_layout()
    fig.savefig(OUT / "v5_general_model_threshold_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def chart_kfold(summary: dict) -> None:
    """Show variation, not just a five-fold average."""
    folds = pd.DataFrame(summary["folds"])
    series = [("roc_auc", "ROC-AUC", "#377eb8"), ("pr_auc", "PR-AUC", "#ff7f0e"), ("top1_hit_rate", "Top-1 hit rate", "#2ca02c")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    for axis, (column, label, color) in zip(axes, series):
        values = folds[column]
        axis.bar(folds["fold"], values, color=color)
        axis.axhline(summary["mean"][column], color="black", linestyle="--", linewidth=1, label=f"mean {summary['mean'][column]:.4f}")
        axis.set(title=label, xlabel="Race-group fold", xticks=folds["fold"], ylim=(0, max(.36, values.max() + .06)))
        for fold, value in zip(folds["fold"], values):
            axis.text(fold, value + .008, f"{value:.3f}", ha="center", fontsize=9)
        axis.legend(fontsize=8, loc="lower right")
    fig.suptitle("Five-fold race-group validation: LightGBM rank+binary, train interval only")
    fig.tight_layout()
    fig.savefig(OUT / "v5_lightgbm_5fold_validation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_self_contained_html(table: pd.DataFrame, details: pd.DataFrame, settings: pd.DataFrame, general_thresholds: pd.DataFrame, kfold: dict) -> None:
    """Write a report that embeds every figure, avoiding Markdown image-path issues."""
    images = [
        ("Chronological data split", "v5_data_split.png", "이 그림은 자료가 시간 순서대로 Train·Validation·Test로 한 번만 분리됐음을 보여 준다. 파란 Train으로 과거 패턴을 학습하고, 주황 Validation으로 반복 횟수·혼합 비중을 고르며, 녹색 Test는 마지막까지 잠가 둔 최종 평가 구간이다. 세 막대가 겹치지 않으므로 미래 경주의 결과나 통계가 과거 학습에 직접 들어가는 누수를 줄인다. 오른쪽 행 수 막대는 Train이 더 크고 Validation·Test가 비슷한 규모임을 보여 준다. 다만 날짜 분할만으로 모든 피처의 시간 누수가 자동으로 사라지는 것은 아니므로, Target Encoding도 과거 경주만 사용하도록 별도로 처리했다."),
        ("Model comparison", "v5_model_comparison.png", "왼쪽은 모든 test 행을 대상으로 한 ROC-AUC, 오른쪽은 각 경주에서 최고 EDGE 말 한 두를 남긴 뒤 상위 10%인 64회만 선택했을 때의 원시 ROI다. ROC-AUC는 우승마를 전반적으로 더 높은 순위에 두는 능력이고, ROI는 특정 선택 규칙과 배당 결과가 결합된 수익률이라 단위와 목적이 다르다. 점선 시장 기준보다 일반 모델 ROC-AUC가 낮다는 사실과, 일부 모델의 양(+) ROI가 동시에 나타날 수 있다. 이는 작은 선택 표본이나 고배당 적중의 영향 때문일 수 있으므로, 오른쪽 양수 막대를 장기 수익 증명이나 모델의 절대 우위로 읽으면 안 된다."),
        ("Test-set EDGE diagnostics", "v5_edge_diagnostics.png", "세 패널은 같은 64회 선택 결과를 서로 다른 관점에서 분리해 보여 준다. ROI는 실제 배당을 포함한 수익률, 평균 실현 EDGE는 선택 말들이 시장 예상보다 실제로 더 자주 이겼는지, 상관은 큰 예측 EDGE를 준 말일수록 시장 예상보다 더 잘 뛰었는지를 뜻한다. 예를 들어 ROI가 높아도 실현 EDGE나 상관이 약하면 우연한 배당 적중에 민감할 수 있다. 반대로 실현 EDGE가 양수여도 수수료와 배당 구조 때문에 ROI가 음수일 수 있다. 따라서 세 막대를 하나의 점수로 합치지 말고, 방향성과 표본 수 64회를 함께 읽어야 한다."),
        ("General-model selection-range comparison", "v5_general_model_threshold_comparison.png", "여섯 일반 모델의 저장된 test 예측에서 경주당 최고 EDGE 말 한 두만 고른 뒤, 그 후보를 EDGE 순으로 상위 10%·20%·30%까지 넓혀 계산한 원시 ROI다. 선택 범위를 넓히면 표본은 64회에서 127회, 191회로 커지지만 평균 EDGE는 낮아진다. 따라서 상위 10%의 큰 양수 ROI가 20%·30%에서도 유지되는지 확인하면 특정 소수 경주에만 의존했는지 판단할 수 있다. 이 그림은 수수료를 차감하지 않은 단일 Test 기간의 결과이며, 시간 순서 반복 검증이나 실제 배팅 한도를 반영하지 않는다. 막대의 부호 변화는 모델 신호와 수익률이 선택 범위에 민감하다는 경고로 읽어야 한다."),
        ("Five-fold race-group validation", "v5_lightgbm_5fold_validation.png", "이 그림은 최종 Validation·Test를 건드리지 않고 원래 Train 기간의 1,891경주를 경주 단위로 다섯 묶음으로 나눈 LightGBM rank+binary 검증 결과다. 막대 하나는 한 묶음을 검증용으로 두고 나머지 네 묶음으로 새로 학습한 결과이며, 점선은 다섯 결과의 평균이다. ROC-AUC·PR-AUC·top-1 적중률이 fold마다 다르게 나타나므로 평균만으로 안정성을 과장하지 않도록 한다. 같은 경주의 말은 같은 fold에 묶여 경주 단위 누수는 막지만, GroupKFold가 시간 순서를 보장하지는 않는다. 또한 시장 배당률을 입력·평가에 쓰지 않았으므로 이 그림은 ROI나 미래 수익성 검증이 아니라 확률·순위 성능의 내부 변동성 점검이다."),
    ]
    figures = "\n".join(
        f'<figure><img src="{data_uri(OUT / filename)}" alt="{html.escape(title)}"><figcaption>{html.escape(title)}</figcaption></figure><p>{html.escape(explanation)}</p>'
        for title, filename, explanation in images
    )
    metric_table = table[["model", "valid_roc_auc", "test_roc_auc", "test_pr_auc", "top10pct_roi", "mean_realized_edge", "edge_realized_correlation", "bets"]].copy()
    metric_table.columns = ["Model", "Valid ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Top-10% ROI", "Mean realized EDGE", "EDGE-realized corr.", "Bets"]
    for column in ["Valid ROC-AUC", "Test ROC-AUC", "Test PR-AUC", "Mean realized EDGE", "EDGE-realized corr."]:
        metric_table[column] = metric_table[column].map(lambda value: "-" if pd.isna(value) else f"{value:.4f}")
    metric_table["Top-10% ROI"] = metric_table["Top-10% ROI"].map(lambda value: f"{value:+.1%}")
    metric_table["Bets"] = metric_table["Bets"].map(lambda value: f"{int(value):,}")
    table_html = metric_table.to_html(index=False, border=0, classes="metrics")
    detail_view = details[["model", "train_roc_auc", "valid_roc_auc", "valid_pr_auc", "valid_f1", "test_roc_auc", "test_pr_auc", "test_f1", "test_cal_mae", "test_top1_hit_rate", "test_avg_winner_prob", "best_iteration", "decision_threshold"]].copy()
    detail_view.columns = ["Model", "Train ROC", "Valid ROC", "Valid PR", "Valid F1", "Test ROC", "Test PR", "Test F1", "Test calibration MAE", "Test top-1 hit", "Avg winner probability", "Best iteration", "Decision threshold"]
    for column in detail_view.columns[1:-2]:
        detail_view[column] = detail_view[column].map(format_value)
    detail_view["Best iteration"] = detail_view["Best iteration"].map(lambda value: format_value(value, "int"))
    detail_view["Decision threshold"] = detail_view["Decision threshold"].map(format_value)
    detail_html = detail_view.to_html(index=False, border=0, classes="metrics")
    edge_view = details[["model", "top10_bets", "mean_predicted_edge", "mean_realized_edge", "top10_roi", "positive_edge_races", "positive_edge_roi", "edge_realized_correlation"]].copy()
    edge_view.columns = ["Model", "Top-10% bets", "Mean predicted EDGE", "Mean realized EDGE", "Top-10% ROI", "Positive-EDGE races", "Positive-EDGE ROI", "EDGE-realized corr."]
    for column in ["Mean predicted EDGE", "Mean realized EDGE", "Positive-EDGE ROI", "EDGE-realized corr."]:
        edge_view[column] = edge_view[column].map(format_value)
    edge_view["Top-10% ROI"] = edge_view["Top-10% ROI"].map(lambda value: format_value(value, "pct"))
    for column in ["Top-10% bets", "Positive-EDGE races"]:
        edge_view[column] = edge_view[column].map(lambda value: format_value(value, "int"))
    edge_html = edge_view.to_html(index=False, border=0, classes="metrics")
    settings_html = settings.rename(columns={"model": "Model", "learning_form": "Learning form", "rerun_configuration": "Rerun configuration"}).to_html(index=False, border=0, classes="metrics")
    threshold_view = general_thresholds.copy()
    threshold_view.columns = ["Model", "Selection range", "Bets", "Mean predicted EDGE", "Mean realized EDGE", "Raw test ROI", "Win rate"]
    threshold_view["Selection range"] = threshold_view["Selection range"].map(lambda value: f"Top {int(value)}%")
    threshold_view["Bets"] = threshold_view["Bets"].map(lambda value: format_value(value, "int"))
    for column in ["Mean predicted EDGE", "Mean realized EDGE"]:
        threshold_view[column] = threshold_view[column].map(format_value)
    for column in ["Raw test ROI", "Win rate"]:
        threshold_view[column] = threshold_view[column].map(lambda value: format_value(value, "pct"))
    threshold_html = threshold_view.to_html(index=False, border=0, classes="metrics")
    kfold_view = pd.DataFrame(kfold["folds"])[["fold", "train_rows", "valid_rows", "train_races", "valid_races", "rank_weight", "rank_best_iteration", "binary_best_iteration", "roc_auc", "pr_auc", "calibration_mae", "top1_hit_rate"]].copy()
    kfold_view.columns = ["Fold", "Train rows", "Valid rows", "Train races", "Valid races", "Rank weight", "Rank best iter.", "Binary best iter.", "ROC-AUC", "PR-AUC", "Calibration MAE", "Top-1 hit rate"]
    for column in ["Train rows", "Valid rows", "Train races", "Valid races", "Rank best iter.", "Binary best iter."]:
        kfold_view[column] = kfold_view[column].map(lambda value: format_value(value, "int"))
    for column in ["Rank weight", "ROC-AUC", "PR-AUC", "Calibration MAE", "Top-1 hit rate"]:
        kfold_view[column] = kfold_view[column].map(format_value)
    kfold_html = kfold_view.to_html(index=False, border=0, classes="metrics")
    page = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>revised_v5 전체 최종 보고서</title><style>
body{{font-family:Arial,'Malgun Gothic',sans-serif;line-height:1.65;max-width:1180px;margin:0 auto;padding:28px;color:#17202a;background:#fff}}
h1{{border-bottom:3px solid #377eb8;padding-bottom:12px}} h2{{margin-top:42px;color:#1b4f72}} h3{{margin-top:26px}} h4{{margin:22px 0 10px;color:#34495e}}
p{{max-width:940px;margin:0 0 18px;line-height:1.82}}
table{{border-collapse:collapse;width:100%;margin:15px 0 24px;font-size:14px}} th,td{{border:1px solid #d5d8dc;padding:8px;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf2f8}}
figure{{margin:28px 0 38px}} img{{display:block;max-width:100%;height:auto;border:1px solid #d5d8dc}}figcaption{{color:#566573;font-size:13px;margin-top:6px}}
.note{{background:#fef9e7;border-left:4px solid #f5b041;padding:12px 16px}} .pass{{color:#196f3d;font-weight:bold}} .fail{{color:#922b21;font-weight:bold}}
</style></head><body>
<h1>revised_v5 전체 최종 보고서</h1>
<p>생성일: 2026-08-18 · 고정된 시간 순서 분할 · 일반 모델의 입력에서 시장 확률/배당률 제외</p>
<h2>1. 데이터, 전처리 및 무결성</h2>
<ul><li>총 32,838행, 3,167경주. Train 19,617행/1,891경주, Validation 6,582행/641경주, Test 6,639행/635경주다.</li>
<li>세 분할은 동일 스키마이며 결측치와 중복 <code>entry_id</code>가 없고, 모든 경주에 우승마가 정확히 한 두 있다.</li>
<li>단순 범주형은 One-Hot, <code>tool_set</code>은 31개 장구류 Multi-Hot, 기수·조교사·마주·등급은 과거 경주만 이용한 스무딩 Target Encoding이다.</li>
<li><code>race_id</code>는 그룹 평가용, <code>entry_id</code>·<code>hrName</code>은 식별용으로만 보존했다. <code>meet</code>·<code>fold</code>·<code>wgBudamBigo</code>·중복 <code>track</code>은 제거했다.</li>
<li>전처리 후 146개 열, 일반 모델 학습 입력은 121개 열이다.</li></ul>
<h2>2. 시각적 데이터 점검</h2>{figures}
<h2>3. 일반 모델의 최종 테스트 결과</h2>
<p>시장 기준은 ROC-AUC 0.8122, PR-AUC 0.3303이다.</p>{table_html}
<p>ROC-AUC는 전체 순위 분별력, PR-AUC는 우승마 탐지 성능, EDGE는 모델 확률−시장 확률, ROI는 선택 베팅의 과거 실현수익률이다. 따라서 서로 다른 단위를 합산해 단일 순위로 만들지 않았다.</p>
<h3>모델별 상세 성능</h3><p>학습 로그와 지표 JSON에 실제로 기록된 값만 수록했다. 어떤 모델에 동일한 지표가 기록되지 않은 경우에는 “—”로 표시했으며, 이를 0 또는 성능 저하로 해석하면 안 된다.</p>{detail_html}
<h3>모델별 EDGE 백테스트 진단</h3><p>상위 10%는 test 635경주에서 각 경주의 최고 EDGE 말 1두만 남긴 뒤 64회를 선택한 규칙이다. Positive-EDGE ROI는 모든 양(+) EDGE 경주 선택의 ROI이므로 top-10% ROI와 표본 및 의미가 다르다.</p>{edge_html}
<h3>상위 10%·20%·30% 선택 범위 비교</h3><p>여섯 일반 모델의 저장된 test 예측을 같은 규칙으로 다시 계산했다. 이는 수수료를 차감하지 않은 단일 test의 원시 ROI이며, 수익성이 확인됐다는 결론이 아니다. 선택 범위를 넓힐수록 높은 EDGE를 가진 극소수만 골랐을 때의 결과가 유지되는지 확인한다.</p>{threshold_html}
<h3>재학습 설정</h3><p>아래는 재실행에 사용한 기본 설정이다. 조기 종료가 적용된 모델은 실제 최적 반복 횟수를 바로 위 상세 성능 표에 별도로 기록했다.</p>{settings_html}
<h2>4. 피처·성과 해석 및 검증 기준</h2><figure><img src="{data_uri(OUT / 'v5_feature_importance_consensus.png')}" alt="All-model feature-importance consensus"><figcaption>All-model feature-importance consensus</figcaption></figure><p>왼쪽은 여섯 모델의 평균 순위 점수, 오른쪽은 모델별 순위 점수다. 여러 열에서 진하게 나타나고 top-20 횟수가 많은 변수는 특정 알고리즘 하나에만 의존하지 않는 반복 신호다.</p><p>기존처럼 Random Forest와 LightGBM의 원시 중요도만 나란히 두면, 두 모델의 편향과 서로 다른 단위(Gini, gain)에 영향을 받는다. 이 보고서는 최종 일반 모델 6개 모두의 <code>feature_importance_full.csv</code>를 사용한다. 각 모델에서 1위=1, 최하위=0이 되도록 <b>순위 점수</b>로 정규화하고 평균을 냈다. 따라서 이 그림은 서로 다른 중요도 단위를 합산한 것이 아니라 <b>모델 간 반복성</b>을 보는 보수적 요약이다. 신경망의 입력 가중치 중요도는 예측을 바꿨을 때의 기여도가 아니므로, 인과효과나 절대적 영향도로 해석하면 안 된다.</p>
<h2>5. 최종 결과 정리와 인사이트</h2>
<h3>핵심 결과</h3><ul><li><b>승률·순위 예측:</b> Deep listwise ensemble이 일반 모델 중 가장 높은 Test ROC-AUC(0.7710)와 경주당 1위 적중률(33.1%)을 기록했다. 다만 시장 기준 ROC-AUC 0.8122보다 낮으므로, 시장 확률을 단독으로 대체할 근거는 부족하다.</li><li><b>시장 대비 신호:</b> Plackett-Luce hybrid는 평균 실현 EDGE(+0.0424)와 EDGE-실현 상관(+0.0471)이 가장 높았다. 모델이 시장보다 높게 평가한 말들이 평균적으로 시장 예상보다 조금 더 잘 뛰는 방향성은 확인했지만, 상관값 자체는 작다.</li><li><b>상위 10% 원시 ROI:</b> Random Forest와 XGBoost가 각각 +55.0%, LightGBM이 +45.0%를 기록했다. 모두 경주당 1두, 총 64회 선택, 수수료 미차감이라는 같은 규칙에서 산출한 단일 Test 기간의 값이다.</li></ul>
<h3>결과에서 읽을 수 있는 점</h3><p>상위 10% ROI가 높다는 사실만으로 모델이 안정적인 수익 신호를 냈다고 보기는 어렵다. Random Forest는 선택 범위를 10%에서 20%·30%로 넓히자 +55.0%에서 −17.9%·−6.2%로, XGBoost는 +55.0%에서 −3.7%·−23.5%로 바뀌었다. 소수의 높은 EDGE 사례 또는 고배당 적중에 결과가 크게 좌우될 수 있다는 뜻이다.</p><p>반면 Plackett-Luce는 상위 10% ROI가 +23.3%, 평균 실현 EDGE가 양수이고 EDGE-실현 상관도 양수여서 시장 대비 방향성을 살펴볼 후보로 볼 수 있다. 그러나 이 역시 64회 선택의 단일 Test 결과이므로 운영 성과로 확정하지 않는다. 다음 단계의 판단은 수수료·배팅 한도·유효 배당률을 적용한 여러 시간 구간 검증이 추가된 뒤에만 가능하다.</p><p>LightGBM의 5-fold 경주 그룹 검증은 평균 ROC-AUC 0.7457 ± 0.0115로 경주 묶음에 따른 순위 성능 변동 범위를 보여 준다. 이 값은 확률·순위 모델의 내부 안정성을 점검한 결과이며, ROI나 미래 수익성을 검증한 결과는 아니다. 따라서 이 보고서에서는 예측 성능, 시장 대비 EDGE, 선택 규칙별 ROI를 서로 다른 근거로 분리해 해석한다.</p>
<h3>EDGE·ROI의 해석과 검증 기준</h3><p>EDGE = 모델의 경주 내 정규화 확률 − 시장의 정규화 확률이다. 실현 EDGE = <code>win − 시장 확률</code>이다. 기본 ROI = <code>win × 단승 배당률 − 1</code>의 평균이며, 수수료를 반영하면 <code>win × 단승 배당률 × (1−수수료율) − 1</code>이다.</p>
<h4>EDGE 지표 해석 기준</h4><ul><li><b>평균 실현 EDGE</b>는 선택한 말들이 실제로 시장 예상보다 더 자주 이겼는지를 나타낸다. 양수일수록 선택 집합 전체가 시장 예상보다 좋았다는 의미다.</li><li><b>EDGE–실현 상관</b>은 모델이 더 큰 EDGE를 준 말일수록 실제로 시장 예상보다 더 잘 뛰었는지를 나타낸다. 양수라도 0에 가까우면 신호의 구별력은 약하다.</li><li><b>Positive-EDGE ROI</b>는 예측 EDGE가 0보다 큰 모든 경주 후보를 선택했을 때의 ROI다. 상위 10% ROI와 달리 많은 후보를 포함하므로, 두 값의 차이는 선택 규칙이 결과에 얼마나 민감한지 보여 준다.</li><li>높은 예측 EDGE는 모델의 주장일 뿐이다. 확률이 과대추정되었거나 배당률·수수료를 이길 만큼의 차이가 아니라면 높은 EDGE도 손실로 이어질 수 있다.</li></ul>
<h4>상위 10% 선택 규칙의 표본 한계</h4><p>일반 모델 test에서 Random Forest·XGBoost의 상위 10% ROI는 각각 +55.0%, LightGBM은 +45.0%였지만, 모두 64회 선택의 단일 구간 결과다. 선택 범위를 20%·30%로 넓힌 일반 모델 비교는 3번 표와 그래프에 수록했으며, 예를 들어 Random Forest는 +55.0% → −17.9% → −6.2%, XGBoost는 +55.0% → −3.7% → −23.5%로 변한다. 따라서 상위 10%는 추가 검토가 필요한 후보 규칙일 뿐, 자동으로 높은 수익률을 보장하지 않는다.</p>
<h3>5-fold 경주 그룹 검증</h3><p>최종 Test는 그대로 잠가 둔 채, 원래 Train 기간의 1,891경주만 5개 묶음으로 나눴다. 매 fold마다 4개 묶음(약 1,512~1,513경주)으로 LightGBM rank+binary를 새로 학습하고, 남은 1개 묶음(약 378~379경주)에서 평가했다. 한 경주의 말들은 항상 같은 fold에 있어 경주 단위 정보가 학습·검증에 나뉘지 않는다. 범주형 코드는 매 fold의 학습 데이터로만 다시 만들었고, 시장 확률·배당률·결과 누수 열은 입력에서 제외했다.</p>
<p>이 검증은 <b>분류·랭킹 성능의 안정성</b>을 확인하는 절차라서 ROI를 계산하지 않는다. ROI에는 같은 시점에 알 수 있는 시장 배당률과 배팅 규칙이 추가로 필요하다. 또한 GroupKFold는 경주 그룹을 나눈 검증이지 미래 날짜만으로 순방향 평가하는 walk-forward가 아니다.</p>
<h4>LightGBM rank+binary 5-fold 결과</h4><p>평균 ROC-AUC는 <b>{kfold['mean']['roc_auc']:.4f} ± {kfold['std']['roc_auc']:.4f}</b>, 평균 PR-AUC는 <b>{kfold['mean']['pr_auc']:.4f} ± {kfold['std']['pr_auc']:.4f}</b>, 평균 top-1 적중률은 <b>{kfold['mean']['top1_hit_rate']:.4f} ± {kfold['std']['top1_hit_rate']:.4f}</b>다. 다섯 fold의 OOF(각 행이 자신을 학습에 쓰지 않은 모델의 예측) 통합 ROC-AUC는 <b>{kfold['oof']['roc_auc']:.4f}</b>, OOF PR-AUC는 <b>{kfold['oof']['pr_auc']:.4f}</b>다.</p>{kfold_html}
<p>fold 4의 ROC-AUC 0.7289와 fold 5의 0.7594처럼 차이가 존재한다. 평균이 하나의 고정된 성능을 보장하는 것이 아니라, 경주 묶음 구성에 따라 성능이 흔들릴 수 있음을 보여 준다. 이 표는 여섯 모델 전체를 다시 비교하는 표가 아니라, 3번의 최종 Test 결과를 보완하는 LightGBM rank+binary의 내부 안정성 검증이다.</p>
<h2>6. 예상 질문과 해석 안내</h2><p>아래는 보고서 수치를 읽을 때 자주 생길 수 있는 혼동을 미리 해소하기 위한 참고 사항이다.</p>
<h3>모델별 표에 기록된 지표 범위</h3><p>공통 비교표에는 여섯 모델에 공통으로 기록된 지표만 넣었고, 상세 성능·EDGE 진단·학습 설정 표에는 각 모델이 실제로 저장한 모든 지표를 추가했다. 예를 들어 calibration MAE와 top-1 적중률은 LightGBM·Deep에서만 기록됐고, F1은 XGBoost에서만 기록됐다. 기록되지 않은 값은 “—”로 남겼으며 임의로 계산하거나 0으로 취급하지 않았다.</p>
<h3>순위 합의형 피처 중요도 산출 방식</h3><p>Random Forest의 Gini, 부스팅 모델의 gain, CatBoost의 prediction-change, 신경망의 입력 가중치는 원시 단위가 달라 직접 평균하면 왜곡된다. 그래서 6개 일반 모델 각각의 내부 순위를 0~1로 정규화하고 평균·모델별 열지도를 제시했다. 이는 반복적으로 나타나는 예측 신호를 찾는 방법이며 인과적 중요도는 아니다.</p>
<h3>상위 10%·20%·30%의 선택 규칙</h3><p>각 경주에서 최고 EDGE 말 1두만 남긴 뒤 그 후보들을 EDGE 내림차순으로 정렬한다. 상위 10%는 64회, 20%는 127회, 30%는 191회를 선택한다. 이 범위별 일반 모델 원시 ROI는 3번 표·그래프에 제시했다. 4번의 5-fold 검증은 시장 데이터와 ROI가 아닌 확률·순위 성능의 안정성을 점검하는 별도 절차다.</p>
<h2>7. 산출물 목록</h2><p>모델별 JSON 지표, test 예측 CSV, 피처 중요도 CSV, 저장 모델 파일은 <code>models/v5/</code>에 보존되어 있다. 이 보고서와 원본 그래프·비교 CSV·5-fold 원자료는 <code>outputs/reports/</code>에 보존되어 있다.</p>
</body></html>"""
    page = split_html_paragraphs(apply_note_style(page))
    (OUT / "v5_final_model_report.html").write_text(page, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([read_metric(name, path) for name, path in SPECS.items()])
    table = table.sort_values("test_roc_auc", ascending=False).reset_index(drop=True)
    table.insert(0, "roc_rank", range(1, len(table) + 1))
    table.to_csv(OUT / "v5_model_metrics.csv", index=False, encoding="utf-8-sig")
    details = detailed_metrics()
    settings = model_settings_table()
    chart(table)
    consensus = feature_consensus()
    chart_feature_importance(consensus)
    chart_data_split()
    chart_edge_summary(table)
    general_thresholds = general_model_thresholds()
    chart_general_model_thresholds(general_thresholds)
    rows = "\n".join(
        f"| {r.model} | {r.test_roc_auc:.4f} | {('-' if pd.isna(r.test_pr_auc) else f'{r.test_pr_auc:.4f}')} | {r.top10pct_roi:+.1%} | {r.mean_realized_edge:+.4f} | {r.edge_realized_correlation:+.4f} |"
        for r in table.itertuples())
    detail_rows = "\n".join(
        f"| {r.model} | {format_value(r.train_roc_auc)} | {format_value(r.valid_roc_auc)} | {format_value(r.valid_pr_auc)} | {format_value(r.valid_f1)} | {format_value(r.test_roc_auc)} | {format_value(r.test_pr_auc)} | {format_value(r.test_f1)} | {format_value(r.test_cal_mae)} | {format_value(r.test_top1_hit_rate)} | {format_value(r.test_avg_winner_prob)} | {format_value(r.best_iteration, 'int')} | {format_value(r.decision_threshold)} |"
        for r in details.itertuples())
    edge_rows = "\n".join(
        f"| {r.model} | {format_value(r.top10_bets, 'int')} | {format_value(r.mean_predicted_edge)} | {format_value(r.mean_realized_edge)} | {format_value(r.top10_roi, 'pct')} | {format_value(r.positive_edge_races, 'int')} | {format_value(r.positive_edge_roi, 'pct')} | {format_value(r.edge_realized_correlation)} |"
        for r in details.itertuples())
    settings_rows = "\n".join(f"| {r.model} | {r.learning_form} | {r.rerun_configuration} |" for r in settings.itertuples())
    general_threshold_rows = "\n".join(
        f"| {r.model} | 상위 {r.selection_pct}% | {r.bets} | {r.mean_predicted_edge:+.4f} | {r.mean_realized_edge:+.4f} | {r.roi:+.1%} | {r.win_rate:.1%} |"
        for r in general_thresholds.itertuples())
    kfold = json.loads((OUT / "v5_lightgbm_5fold_summary.json").read_text(encoding="utf-8"))
    chart_kfold(kfold)
    kfold_rows = "\n".join(
        f"| {r['fold']} | {r['train_rows']:,} | {r['valid_rows']:,} | {r['train_races']:,} | {r['valid_races']:,} | {r['rank_weight']:.2f} | {r['rank_best_iteration']} | {r['binary_best_iteration']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['calibration_mae']:.4f} | {r['top1_hit_rate']:.4f} |"
        for r in kfold["folds"])
    report = f"""# revised_v5 최종 모델 평가 보고서

![데이터 분할](C:/Users/goqud/source/repos/PythonApplication1/PythonApplication1/outputs/reports/v5_data_split.png)

이 그림은 자료가 시간 순서대로 Train·Validation·Test로 한 번만 분리됐음을 보여 준다. 파란 Train으로 과거 패턴을 학습하고, 주황 Validation으로 반복 횟수·혼합 비중을 고르며, 녹색 Test는 마지막까지 잠가 둔 최종 평가 구간이다. 세 막대가 겹치지 않으므로 미래 경주의 결과나 통계가 과거 학습에 직접 들어가는 누수를 줄인다. 오른쪽 행 수 막대는 Train이 더 크고 Validation·Test가 비슷한 규모임을 보여 준다. 다만 날짜 분할만으로 모든 피처의 시간 누수가 자동으로 사라지는 것은 아니므로, Target Encoding도 과거 경주만 사용하도록 별도로 처리했다.

![v5 모델 비교](C:/Users/goqud/source/repos/PythonApplication1/PythonApplication1/outputs/reports/v5_model_comparison.png)

왼쪽은 모든 test 행을 대상으로 한 ROC-AUC, 오른쪽은 각 경주에서 최고 EDGE 말 한 두를 남긴 뒤 상위 10%인 64회만 선택했을 때의 원시 ROI다. ROC-AUC는 우승마를 전반적으로 더 높은 순위에 두는 능력이고, ROI는 특정 선택 규칙과 배당 결과가 결합된 수익률이라 단위와 목적이 다르다. 점선 시장 기준보다 일반 모델 ROC-AUC가 낮다는 사실과, 일부 모델의 양(+) ROI가 동시에 나타날 수 있다. 이는 작은 선택 표본이나 고배당 적중의 영향 때문일 수 있으므로, 오른쪽 양수 막대를 장기 수익 증명이나 모델의 절대 우위로 읽으면 안 된다.

## 1. 데이터·검증 설계

- 시간 순서 분할: train 19,617행/1,891경주, valid 6,582행/641경주, test 6,639행/635경주.
- 무결성: 세 분할의 스키마가 동일하고 결측치·중복 `entry_id`가 없으며, 모든 경주에 우승마가 정확히 1두다.
- 전처리: 단순 범주형은 One-Hot, 장구류 `tool_set`은 31개 Multi-Hot, 기수·조교사·마주·등급은 학습 구간의 **과거 경주만** 쓰는 스무딩 Target Encoding이다. `track`은 `waterRate`와 중복되어, `meet`·`fold`·`wgBudamBigo`와 함께 제거했다.
- 식별자: `race_id`는 경주 단위 평가를 위한 그룹 키로만 보존하고, `entry_id`·`hrName`은 학습 입력에서 제외했다.
- 시장 기준: test ROC-AUC **0.8122**, PR-AUC **0.3303**.

| 분할 | 기간 | 행 수 | 경주 수 | 용도 |
|---|---|---:|---:|---|
| Train | 2023-08-05~2025-05-11 | 19,617 | 1,891 | 모델 학습 |
| Valid | 2025-05-17~2025-12-27 | 6,582 | 641 | 조기 종료·모델 선택 |
| Test | 2025-12-28~2026-08-09 | 6,639 | 635 | 최종 1회 평가 |

전처리 후 데이터는 146개 열이며, 실제 모델 입력은 121개다. 단순 범주형(`sex`, `weather`, `rcDay`, `budam`, `born`)은 One-Hot, `tool_set`은 31개 장구류 Multi-Hot, `jkName`·`trName`·`owName`·`rank`는 과거 경주만 쓰는 스무딩 Target Encoding으로 처리했다. 원본 `final.csv.gz`의 시장 확률·배당률·결과 열은 일반 모델 입력에서 제외했다.

## 2. 최종 테스트 성능

| 모델 | Test ROC-AUC | Test PR-AUC | 상위 10% EDGE ROI | 평균 실현 EDGE | EDGE-실현 상관 |
|---|---:|---:|---:|---:|---:|
{rows}

모델별 확률의 보정 방식과 목적함수가 다르므로, ROC-AUC와 ROI를 같은 의미의 점수처럼 합산하지 않았다. ROC-AUC는 전반적인 순위 분별력, PR-AUC는 희소한 우승마 탐지, EDGE/ROI는 선택된 베팅 규칙에서의 과거 수익성 지표다.

### 모델별 상세 성능

학습 로그와 지표 JSON에 실제로 기록된 값만 수록했다. 동일한 지표가 기록되지 않은 모델은 `—`로 표시했으며, 이를 0 또는 성능 저하로 해석하면 안 된다.

| 모델 | Train ROC | Valid ROC | Valid PR | Valid F1 | Test ROC | Test PR | Test F1 | Test calibration MAE | Test top-1 hit | 평균 우승마 확률 | 최적 반복 | 분류 임계값 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{detail_rows}

### 모델별 EDGE 백테스트 진단

상위 10%는 test 635경주에서 각 경주의 최고 EDGE 말 1두만 남긴 뒤 64회를 선택한 규칙이다. `Positive-EDGE ROI`는 모든 양(+) EDGE 경주 선택의 ROI이므로 상위 10% ROI와 표본 및 의미가 다르다.

| 모델 | Top-10% 선택 수 | 평균 예측 EDGE | 평균 실현 EDGE | Top-10% ROI | 양(+) EDGE 경주 수 | Positive-EDGE ROI | EDGE-실현 상관 |
|---|---:|---:|---:|---:|---:|---:|---:|
{edge_rows}

### 상위 10%·20%·30% 선택 범위 비교

여섯 일반 모델의 저장된 test 예측을 같은 규칙으로 다시 계산했다. 각 경주에서 최고 EDGE 말 1두만 남기고, 그 경주 후보를 EDGE 내림차순으로 정렬해 상위 범위를 선택했다. 아래 값은 **수수료 미차감 단일 test의 원시 ROI**이므로, 보수적 walk-forward ROI와 혼동하면 안 된다. 선택 범위를 넓혔을 때 결과가 유지되는지 확인하기 위한 표다.

![일반 모델의 선택 범위별 ROI](C:/Users/goqud/source/repos/PythonApplication1/PythonApplication1/outputs/reports/v5_general_model_threshold_comparison.png)

여섯 일반 모델의 저장된 test 예측에서 경주당 최고 EDGE 말 한 두만 고른 뒤, 그 후보를 EDGE 순으로 상위 10%·20%·30%까지 넓혀 계산한 원시 ROI다. 선택 범위를 넓히면 표본은 64회에서 127회, 191회로 커지지만 평균 EDGE는 낮아진다. 따라서 상위 10%의 큰 양수 ROI가 20%·30%에서도 유지되는지 확인하면 특정 소수 경주에만 의존했는지 판단할 수 있다. 이 그림은 수수료를 차감하지 않은 단일 Test 기간의 결과이며, 시간 순서 반복 검증이나 실제 배팅 한도를 반영하지 않는다. 막대의 부호 변화는 모델 신호와 수익률이 선택 범위에 민감하다는 경고로 읽어야 한다.

| 모델 | 선택 범위 | 선택 수 | 평균 예측 EDGE | 평균 실현 EDGE | 원시 Test ROI | 적중률 |
|---|---:|---:|---:|---:|---:|---:|
{general_threshold_rows}

### 재학습 설정

아래는 재실행에 사용한 기본 설정이다. 조기 종료가 적용된 모델은 실제 최적 반복 횟수를 상세 성능 표에 별도로 기록했다.

| 모델 | 학습 형태 | 재학습 설정 |
|---|---|---|
{settings_rows}

| 모델군 | 학습 방식 | 핵심 목적 |
|---|---|---|
| Random Forest / XGBoost / CatBoost | 이진 우승 분류 | 개별 말의 우승 가능성 |
| LightGBM rank+binary | LambdaRank + 이진 분류 블렌드 | 경주 내 순위와 확률 동시 반영 |
| Deep listwise | 5-seed 경주 단위 앙상블 | 경주 내부 확률 합계 1 제약 학습 |
| Plackett-Luce hybrid | listwise + Bradley-Terry | 경주 단위 상대 순위 학습 |

![EDGE 진단](C:/Users/goqud/source/repos/PythonApplication1/PythonApplication1/outputs/reports/v5_edge_diagnostics.png)

세 패널은 같은 64회 선택 결과를 서로 다른 관점에서 분리해 보여 준다. ROI는 실제 배당을 포함한 수익률, 평균 실현 EDGE는 선택 말들이 시장 예상보다 실제로 더 자주 이겼는지, 상관은 큰 예측 EDGE를 준 말일수록 시장 예상보다 더 잘 뛰었는지를 뜻한다. 예를 들어 ROI가 높아도 실현 EDGE나 상관이 약하면 우연한 배당 적중에 민감할 수 있다. 반대로 실현 EDGE가 양수여도 수수료와 배당 구조 때문에 ROI가 음수일 수 있다. 따라서 세 막대를 하나의 점수로 합치지 말고, 방향성과 표본 수 64회를 함께 읽어야 한다.

## 3. 5-fold 경주 그룹 검증

최종 일반 모델과 비교 조건이 다른 별도 실험은 본문에서 제외하고, 원래 Train 기간의 1,891경주만 사용한 **5-fold GroupKFold** 검증을 수행했다. 최종 Validation·Test는 이 검증에 사용하지 않았으며, 각 경주는 통째로 하나의 fold에만 포함된다. 따라서 같은 경주의 말이 학습과 검증으로 섞이지 않는다.

이 검증은 LightGBM rank+binary에 적용했다. 매 fold에서 약 1,512~1,513경주로 새로 학습하고 약 378~379경주로 평가했으며, 범주형 코드도 fold의 학습 경주로만 다시 생성했다. 시장 확률·배당률·결과 누수 열은 모델 입력에 사용하지 않았다. 이는 ROI 검증이 아니라 **확률·순위 성능이 경주 묶음에 따라 얼마나 안정적인지** 확인하는 검증이다.

![LightGBM 5-fold 검증](C:/Users/goqud/source/repos/PythonApplication1/PythonApplication1/outputs/reports/v5_lightgbm_5fold_validation.png)

각 막대는 한 fold의 독립 검증 결과이고 점선은 5개 평균이다. ROC-AUC 평균은 **{kfold['mean']['roc_auc']:.4f} ± {kfold['std']['roc_auc']:.4f}**, PR-AUC 평균은 **{kfold['mean']['pr_auc']:.4f} ± {kfold['std']['pr_auc']:.4f}**, top-1 적중률 평균은 **{kfold['mean']['top1_hit_rate']:.4f} ± {kfold['std']['top1_hit_rate']:.4f}**다. OOF 통합 ROC-AUC는 **{kfold['oof']['roc_auc']:.4f}**, OOF 통합 PR-AUC는 **{kfold['oof']['pr_auc']:.4f}**다. fold 4 ROC-AUC 0.7289와 fold 5 0.7594의 차이는 평균만으로 가려지는 변동성을 보여 준다.

| Fold | Train 행 | Valid 행 | Train 경주 | Valid 경주 | Rank 비중 | Rank 최적 반복 | Binary 최적 반복 | ROC-AUC | PR-AUC | Calibration MAE | Top-1 적중률 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{kfold_rows}

이 표는 6개 모델을 다시 순위화한 결과가 아니다. 2번의 최종 Test 비교를 보완하는 LightGBM rank+binary의 내부 안정성 검증이다. 또한 GroupKFold는 경주 단위 누수를 막지만, 미래 날짜만을 평가하는 순방향 walk-forward와는 다른 검증 방식이다.

## 4. 피처·성과 해석 및 검증 기준

![6개 모델 합의 피처 중요도](C:/Users/goqud/source/repos/PythonApplication1/PythonApplication1/outputs/reports/v5_feature_importance_consensus.png)

왼쪽은 여섯 모델의 평균 순위 점수, 오른쪽은 모델별 순위 점수다. 여러 열에서 진하게 나타나고 `top-20 in n/6` 횟수가 많은 변수는 특정 알고리즘 하나에만 의존하지 않는 반복 신호다.

이 그림은 Random Forest·XGBoost·LightGBM·CatBoost·Deep listwise·Plackett-Luce의 중요도 파일을 모두 사용한다. 각 모델의 **순위**를 0~1 점수로 정규화한 뒤 평균했으므로, 서로 단위가 다른 Gini·gain·CatBoost prediction-change·신경망 입력 가중치를 원시값 그대로 섞지 않는다. 따라서 높은 점수는 “서로 다른 모델에서 반복적으로 상위에 등장한 변수”라는 뜻이다. 이는 예측 모델의 내부 신호 요약이지 인과효과의 증명은 아니다. 특히 Deep/Plackett-Luce의 가중치 기반 중요도는 순위 비교에만 사용했고, 절대적 영향도로 해석하면 안 된다.

### EDGE·ROI의 해석과 검증 기준

**EDGE**는 `모델의 경주 내 정규화 확률 − 시장의 정규화 확률`이다. 양수일수록 모델이 시장보다 해당 말을 더 높게 평가했다는 뜻이다. 실현 EDGE는 `win(0 또는 1) − 시장 확률`이며, 장기적으로 양수이고 예측 EDGE와 양의 상관을 보여야 신호의 타당성이 있다.

**ROI**는 선정된 베팅의 평균 실현수익률, 즉 `win × 단승 배당률 − 1`의 평균이다. 평가에서는 무효 배당률 9999.9을 제외하고, 각 경주의 최대 EDGE 말 1두만 남긴 뒤 상위 10%인 **64회**를 선택했다. 이 규칙은 다중 선택과 비현실적인 배당률로 인한 과장을 막지만, 64회는 여전히 작아 한두 번의 고배당 적중이 ROI를 크게 흔든다. 따라서 ROI가 양수여도 다음 조건을 함께 충족하기 전에는 운영 결론을 내리지 않는다.

#### EDGE 지표 해석 기준

- **평균 실현 EDGE**는 선택한 말들이 실제로 시장 예상보다 더 자주 이겼는지를 본다. 양수일수록 선택 집합 전체가 시장 예상보다 좋았다는 뜻이다.
- **EDGE-실현 상관**은 모델이 더 큰 EDGE를 준 말일수록 실제로 시장 예상보다 더 잘 뛰었는지를 본다. 양수라도 0에 가까우면 신호의 구별력은 약하다.
- **Positive-EDGE ROI**는 예측 EDGE가 0보다 큰 모든 경주 후보를 선택했을 때의 ROI다. 상위 10% ROI와 달리 많은 후보를 포함하므로, 두 값의 차이는 선택 규칙이 결과에 얼마나 민감한지 보여 준다.
- 높은 예측 EDGE는 모델의 주장일 뿐이다. 확률이 과대추정되었거나 배당률·수수료를 이길 만큼의 차이가 아니라면 높은 EDGE도 손실로 이어질 수 있다.

#### 상위 10% 선택 규칙의 표본 한계

일반 모델 test에서 Random Forest·XGBoost의 상위 10% ROI는 각각 +55.0%, LightGBM은 +45.0%였지만, 모두 64회 선택의 단일 구간 결과다. 선택 범위를 20%·30%로 넓힌 일반 모델 비교는 3번 표와 그래프에 수록했다. 예를 들어 Random Forest는 +55.0% → −17.9% → −6.2%, XGBoost는 +55.0% → −3.7% → −23.5%로 변한다. 따라서 상위 10%가 연구할 만한 후보 규칙이라는 의미이지, 자동으로 높은 수익률을 보장한다는 의미는 아니다.

수수료를 반영한 보수적 ROI는 `win × 단승 배당률 × (1 − 수수료율) − 1`로 계산한다. 여기서 “보수적”이라는 말은 수수료, 경주당 1두, 유효 배당률 범위 같은 실제 제약을 넣어 기본 ROI보다 불리한 조건에서 다시 계산한다는 뜻이다. 이 보고서의 5-fold 검증은 시장 배당률을 사용하지 않는 순위·확률 안정성 평가이므로, 보수적 ROI를 산출하지 않는다.

## 5. 최종 결과 정리와 인사이트

### 핵심 결과

- **승률·순위 예측:** Deep listwise ensemble이 일반 모델 중 가장 높은 Test ROC-AUC(**0.7710**)와 경주당 1위 적중률(33.1%)을 기록했다. 다만 시장 기준 ROC-AUC 0.8122보다 낮으므로, 시장 확률을 단독으로 대체할 근거는 부족하다.
- **시장 대비 신호:** Plackett-Luce hybrid는 평균 실현 EDGE **+0.0424**, EDGE-실현 상관 **+0.0471**이 가장 높았다. 모델이 시장보다 높게 평가한 말들이 평균적으로 시장 예상보다 조금 더 잘 뛰는 방향성은 확인했지만, 상관값 자체는 작다.
- **상위 10% 원시 ROI:** Random Forest와 XGBoost가 각각 **+55.0%**, LightGBM이 **+45.0%**를 기록했다. 모두 경주당 1두, 총 64회 선택, 수수료 미차감이라는 같은 규칙에서 산출한 단일 Test 기간의 값이다.

### 결과에서 읽을 수 있는 점

상위 10% ROI가 높다는 사실만으로 모델이 안정적인 수익 신호를 냈다고 보기는 어렵다. Random Forest는 선택 범위를 10%에서 20%·30%로 넓히자 +55.0%에서 −17.9%·−6.2%로, XGBoost는 +55.0%에서 −3.7%·−23.5%로 바뀌었다. 소수의 높은 EDGE 사례 또는 고배당 적중에 결과가 크게 좌우될 수 있다는 뜻이다.

반면 Plackett-Luce는 상위 10% ROI가 +23.3%, 평균 실현 EDGE가 양수이고 EDGE-실현 상관도 양수여서 시장 대비 방향성을 살펴볼 후보로 볼 수 있다. 그러나 이 역시 64회 선택의 단일 Test 결과이므로 운영 성과로 확정하지 않는다. 다음 단계의 판단은 수수료·배팅 한도·유효 배당률을 적용한 여러 시간 구간 검증이 추가된 뒤에만 가능하다.

LightGBM의 5-fold 경주 그룹 검증은 평균 ROC-AUC 0.7457 ± 0.0115로 경주 묶음에 따른 순위 성능 변동 범위를 보여 준다. 이 값은 확률·순위 모델의 내부 안정성을 점검한 결과이며, ROI나 미래 수익성을 검증한 결과는 아니다. 따라서 이 보고서에서는 예측 성능, 시장 대비 EDGE, 선택 규칙별 ROI를 서로 다른 근거로 분리해 해석한다.

## 6. 예상 질문과 해석 안내

아래는 보고서 수치를 읽을 때 자주 생길 수 있는 혼동을 미리 해소하기 위한 안내다. 실제 질의응답이나 대화 기록이 아니다.

### 모델별 표에 기록된 지표 범위

공통 비교표에는 여섯 모델에 공통으로 기록된 지표만 넣었고, 상세 성능·EDGE 진단·학습 설정 표에는 각 모델이 실제로 저장한 모든 지표를 추가했다. 예를 들어 calibration MAE와 top-1 적중률은 LightGBM·Deep에서만 기록됐고, F1은 XGBoost에서만 기록됐다. 기록되지 않은 값은 `—`로 남겼으며 임의로 계산하거나 0으로 취급하지 않았다.

### 순위 합의형 피처 중요도 산출 방식

Random Forest의 Gini, 부스팅 모델의 gain, CatBoost의 prediction-change, 신경망의 입력 가중치는 원시 단위가 달라 직접 평균하면 왜곡된다. 그래서 6개 일반 모델 각각의 내부 순위를 0~1로 정규화하고 평균·모델별 열지도를 제시했다. 이는 반복적으로 나타나는 예측 신호를 찾는 방법이며 인과적 중요도는 아니다.

### 상위 10%·20%·30%의 선택 규칙

각 경주에서 최고 EDGE 말 1두만 남긴 뒤 그 후보들을 EDGE 내림차순으로 정렬한다. 상위 10%는 64회, 20%는 127회, 30%는 191회를 선택한다. 이 범위별 일반 모델 원시 ROI는 2번 표·그래프에 제시했다. 4번의 5-fold 검증은 시장 데이터와 ROI가 아닌 확률·순위 성능의 안정성을 점검하는 별도 절차다.

"""
    report = split_markdown_paragraphs(apply_note_style(report))
    (OUT / "v5_final_model_report.md").write_text(report, encoding="utf-8")
    write_self_contained_html(table, details, settings, general_thresholds, kfold)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
