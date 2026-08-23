from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
REPORTS = ROOT / "reports"
CONFIGS = ROOT / "configs"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def f3(value: float) -> str:
    return f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def export_importance(locked: dict) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for target, feature_sets in locked["selections"].items():
        for feature_set, selection in feature_sets.items():
            bundle = joblib.load(selection["bundle"])
            names = bundle["preprocessor"].get_feature_names_out()
            model = bundle["model"]
            if hasattr(model, "feature_importances_"):
                values = np.asarray(model.feature_importances_, dtype=float)
                importance_type = "impurity_importance"
            elif hasattr(model, "coef_"):
                values = np.abs(np.asarray(model.coef_[0], dtype=float))
                importance_type = "absolute_coefficient"
            else:
                continue
            frame = pd.DataFrame({"feature": names, "importance": values}).sort_values(
                "importance", ascending=False
            )
            frame.insert(0, "importance_type", importance_type)
            frame.insert(0, "candidate", selection["candidate"])
            frame.insert(0, "feature_set", feature_set)
            frame.insert(0, "target", target)
            frame.to_csv(
                TABLES / f"feature_importance_{target}_{feature_set}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            records.append(frame.head(30))
    return pd.concat(records, ignore_index=True)


def build_figures(metrics: pd.DataFrame, roi: pd.DataFrame, importance: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    x = np.array([1, 2, 5, 10, 20, 30, 50, 100])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for ax, target, title in zip(
        axes,
        ["darkhorse", "favorite_bust"],
        ["Darkhorse", "Favorite bust"],
    ):
        for _, row in metrics.loc[metrics["target"].eq(target)].iterrows():
            y = [row[f"lift_{p}pct"] for p in x]
            ax.plot(x, y, marker="o", label=row["feature_set"])
        ax.axhline(1.0, color="black", lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_xticks(x, labels=x)
        ax.set_xlabel("Top score percentile (%)")
        ax.set_ylabel("Lift")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "test_lift_by_percentile.png", dpi=180)
    plt.close(fig)

    roi = roi.copy()
    roi["percentile"] = roi["range"].str.extract(r"(\d+)").astype(int)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(roi["percentile"], roi["expected_roi"] * 100, marker="o", label="Expected ROI")
    ax.plot(roi["percentile"], roi["realized_roi"] * 100, marker="o", label="Realized ROI")
    ax.plot(roi["percentile"], roi["roi_without_top1"] * 100, marker="o", label="Realized ROI, max payout removed")
    ax.fill_between(
        roi["percentile"], roi["roi_ci_low"] * 100, roi["roi_ci_high"] * 100,
        alpha=0.14, label="Realized ROI 95% race bootstrap CI"
    )
    ax.axhline(0, color="black", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(x, labels=x)
    ax.set_xlabel("Cumulative top score percentile (%)")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Darkhorse Core: expected vs. realized place-bet ROI")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "darkhorse_core_roi_by_percentile.png", dpi=180)
    plt.close(fig)

    top = importance.loc[
        importance["target"].eq("darkhorse") & importance["feature_set"].eq("core")
    ].head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"].str.replace(r"^[^_]+__", "", regex=True), top["importance"])
    ax.set_xlabel("Random Forest impurity importance")
    ax.set_title("Darkhorse Core: top 15 model features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "darkhorse_core_feature_importance.png", dpi=180)
    plt.close(fig)


def build_report(locked: dict, metrics: pd.DataFrame, core_roi: pd.DataFrame, op: pd.DataFrame) -> None:
    draft = json.loads((CONFIGS / "selection_draft.json").read_text(encoding="utf-8"))
    audit = json.loads((TABLES / "test_data_audit.json").read_text(encoding="utf-8"))
    valid_rows = []
    for target, feature_sets in draft["selections"].items():
        for feature_set, selection in feature_sets.items():
            m = selection["valid_metrics"]
            valid_rows.append([
                target, feature_set, selection["candidate"], f3(m["roc_auc"]),
                f3(m["pr_auc"]), f3(m["lift_10pct"]),
                f"{selection['stability_lift10_mean']:.3f} ± {selection['stability_lift10_std']:.3f}",
            ])
    test_rows = []
    for _, row in metrics.iterrows():
        test_rows.append([
            row["target"], row["feature_set"], row["candidate"], f3(row["roc_auc"]),
            f3(row["pr_auc"]), pct(row["base_rate"]), pct(row["hit_rate_10pct"]),
            f"{row['lift_10pct']:.2f}",
        ])
    roi_rows = []
    for _, row in core_roi.iterrows():
        roi_rows.append([
            row["range"].replace("top_", "상위 ").replace("pct", "%"), int(row["bets"]),
            pct(row["hit_rate"]), f"{row['lift']:.2f}", pct(row["expected_roi"]),
            pct(row["realized_roi"]), f"{pct(row['roi_ci_low'])} ~ {pct(row['roi_ci_high'])}",
            pct(row["roi_without_top1"]),
        ])
    op_rows = []
    for _, row in op.iterrows():
        op_rows.append([
            row["feature_set"], int(row["bets"]), int(row["hits"]), pct(row["hit_rate"]),
            pct(row["expected_roi"]), pct(row["realized_roi"]),
            f"{pct(row['roi_ci_low'])} ~ {pct(row['roi_ci_high'])}", pct(row["roi_without_top1"]),
        ])

    text = f"""# 이변 예측 모델 및 수익률 검증 결과

> 완료일: 2026-08-22  
> 잠금 설정 해시: `{locked['sha256']}`  
> 평가 원칙: 시간 순서 train/valid/test, test 1회 평가, test 확인 후 재튜닝 금지

## 1. 결론

- **주 모델은 Random Forest를 사용했다.** 다크호스 Core 최종 모델은 `rf_d8_leaf30_mfsqrt`이며 600개 트리로 재학습했다.
- 다크호스 Core의 test ROC-AUC는 **{metrics.query("target == 'darkhorse' and feature_set == 'core'").iloc[0]['roc_auc']:.3f}**, PR-AUC는 **{metrics.query("target == 'darkhorse' and feature_set == 'core'").iloc[0]['pr_auc']:.3f}**이다.
- 모델 점수 상위 10%의 입상률은 **{pct(core_roi.query("range == 'top_10pct'").iloc[0]['hit_rate'])}**, 전체 비인기마 입상률 {pct(audit['targets']['darkhorse']['base_rate'])} 대비 **{core_roi.query("range == 'top_10pct'").iloc[0]['lift']:.2f}배**다.
- 상위 10% 예상 ROI는 **{pct(core_roi.query("range == 'top_10pct'").iloc[0]['expected_roi'])}**, 실현 ROI는 **{pct(core_roi.query("range == 'top_10pct'").iloc[0]['realized_roi'])}**지만 95% CI가 {pct(core_roi.query("range == 'top_10pct'").iloc[0]['roi_ci_low'])}~{pct(core_roi.query("range == 'top_10pct'").iloc[0]['roi_ci_high'])}로 넓고, 최고 수익 한 건을 빼면 **{pct(core_roi.query("range == 'top_10pct'").iloc[0]['roi_without_top1'])}**다. 따라서 **선별력은 재현됐지만 수익성은 확정되지 않았다.**

## 2. 타깃과 데이터

- 다크호스: 인기 하위 50%(`pop_pct >= 0.50`) 중 입상(`place == 1`)
- 인기마 붕괴: 인기 상위 25%(`pop_pct <= 0.25`) 중 착순 하위 50%(`fin_pct >= 0.50`)
- test: 전체 {audit['test_rows']:,}행, 다크호스 후보 {audit['targets']['darkhorse']['rows']:,}행/양성 {audit['targets']['darkhorse']['positives']:,}건, 인기마 후보 {audit['targets']['favorite_bust']['rows']:,}행/양성 {audit['targets']['favorite_bust']['positives']:,}건
- 저장 라벨과 재계산 라벨의 일치율: 두 타깃 모두 100%
- 다크호스 연승배당: 결측·비양수·999 이상 값 0건, 중앙값 {audit['odds']['core']['median_valid']:.1f}, 최댓값 {audit['odds']['core']['max_valid']:.1f}

Core는 당일 시장·결과·식별자와 과거 시장 파생변수를 제외한다. History+는 당일 시장·결과·식별자는 제외하되 과거 시장 파생변수를 포함한다.

## 3. valid 모델 선정

선정 기준은 valid Lift@10%, 동률 시 PR-AUC다. Random Forest는 24개 조합을 200개 트리로 선별한 후, 선택 조합을 600개 트리로 재학습하고 5개 시드로 안정성을 확인했다.

{markdown_table(['타깃', '피처셋', '선정 모델', 'ROC-AUC', 'PR-AUC', 'Lift@10', '5시드 Lift@10'], valid_rows)}

다크호스 주 모델은 Core Random Forest다. History+는 비교용 보조 모델이다. 인기마 붕괴에서는 Core가 Random Forest, History+는 Logistic Regression이 선택됐다.

## 4. 잠금 test 성능

{markdown_table(['타깃', '피처셋', '모델', 'ROC-AUC', 'PR-AUC', '기준률', '상위10% 적중률', 'Lift@10'], test_rows)}

![test lift](../outputs/figures/test_lift_by_percentile.png)

## 5. 예측 상위 퍼센트별 ROI — 다크호스 Core

수익률은 1단위 연승 베팅 기준이다. 예상 ROI는 valid에서 Platt 보정한 확률과 최종 연승배당으로 `p × 배당 - 1`, 실현 ROI는 `입상 × 배당 - 1`로 계산했다. 신뢰구간은 경주를 군집 단위로 5,000회 부트스트랩했다.

{markdown_table(['누적 구간', '베팅 수', '적중률', 'Lift', '예상 ROI', '실현 ROI', '실현 ROI 95% CI', '최고수익 1건 제거'], roi_rows)}

![roi](../outputs/figures/darkhorse_core_roi_by_percentile.png)

상위 2~5% 독립 구간에 배당 295.7의 적중 한 건이 포함돼 누적 5~50%의 실현 ROI를 크게 끌어올렸다. 이 한 건을 제거한 민감도 결과가 대부분 음수이고 모든 주요 누적 구간의 95% CI가 0을 포함하므로, 현재 결과만으로 양의 기대수익을 주장해서는 안 된다.

## 6. 고정 운영 규칙

valid 상위 10% 점수 임계값을 test에 그대로 적용한 뒤, 같은 경주에서는 최고 점수 말 1두만 선택했다.

{markdown_table(['피처셋', '베팅/경주', '적중', '적중률', '예상 ROI', '실현 ROI', '95% CI', '최고수익 1건 제거'], op_rows)}

실전 후보 규칙은 **Core 점수 >= {op.query("feature_set == 'core'").iloc[0]['valid_top10_score_threshold']:.6f}, 경주당 최고점 1두**다. 다만 현재 배당은 최종 확정 배당이므로 예상 ROI를 실제 의사결정에 쓰려면 베팅 시점 배당 스냅샷이 필요하다.

## 7. 해석과 다음 검증

1. 모델은 비인기마 중 입상 가능성이 높은 말을 유의미하게 압축한다. 특히 상위 1~10%에서 Lift가 1.82~2.95다.
2. 수익률은 고배당 한 건의 영향이 매우 크다. 수익성보다 **랭킹 모델의 선별력 검증 완료**로 해석하는 편이 타당하다.
3. 다음 단계는 더 긴 기간의 walk-forward 검증, 베팅 시점 배당 저장, 확률 보정 재검증, 배당 구간별 표본 확대다.
4. 본 test는 잠금 후 한 번 평가했으므로 이 결과를 보고 하이퍼파라미터나 임계값을 바꾸지 않는다. 변경 아이디어는 새 기간의 데이터에서 별도 검증한다.

## 8. 산출물

- 설정: `configs/locked_config.json`, `configs/test_evaluation_marker.json`
- 모델: `outputs/models/`
- test 예측: `outputs/predictions/`
- 모델·ROI·감사 표: `outputs/tables/`
- 그래프: `outputs/figures/`
- 실행 코드: `src/`, `scripts/`, `tests/`
"""
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "final_upset_model_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    locked = json.loads((CONFIGS / "locked_config.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(TABLES / "test_model_metrics.csv")
    core_roi = pd.read_csv(TABLES / "test_roi_cumulative_core.csv")
    op = pd.read_csv(TABLES / "test_operational_summary.csv")
    importance = export_importance(locked)
    build_figures(metrics, core_roi, importance)
    build_report(locked, metrics, core_roi, op)
    print(f"[OK] report: {REPORTS / 'final_upset_model_report.md'}")


if __name__ == "__main__":
    main()
