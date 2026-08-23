"""
07_upset_report.py — 배당률 포함 이변 예측 최종 보고서

01~06 단계 결과를 하나의 HTML로 종합.

실행:
    python src/4_이변모델/07_upset_report.py

출력:
    results/upset_with_odds/report.html
"""

import base64
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("results/upset_with_odds")


def img_b64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode()}"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    comp = pd.read_csv(OUTPUT_DIR / "model_comparison.csv") if (OUTPUT_DIR / "model_comparison.csv").exists() else None
    thr_comp = pd.read_csv(OUTPUT_DIR / "threshold_comparison.csv", index_col=0) if (OUTPUT_DIR / "threshold_comparison.csv").exists() else None
    segments = pd.read_csv(OUTPUT_DIR / "segment_comparison.csv") if (OUTPUT_DIR / "segment_comparison.csv").exists() else None
    fi_comp = pd.read_csv(OUTPUT_DIR / "fi_comparison.csv") if (OUTPUT_DIR / "fi_comparison.csv").exists() else None
    data_summary = pd.read_csv(OUTPUT_DIR / "data_summary.csv") if (OUTPUT_DIR / "data_summary.csv").exists() else None

    seg_img = img_b64(OUTPUT_DIR / "segment_chart.png")
    fi_img = img_b64(OUTPUT_DIR / "feature_importance.png")

    # Key metrics
    if comp is not None:
        a_best = comp[comp["feature_set"] == "A (q only)"]["ROC_AUC"].max()
        b_best = comp[comp["feature_set"] == "B (no odds)"]["ROC_AUC"].max()
        c_best = comp[comp["feature_set"] == "C (q + features)"]["ROC_AUC"].max()
        c_vs_a = c_best - a_best
        c_vs_b = c_best - b_best
    else:
        a_best = b_best = c_best = c_vs_a = c_vs_b = 0

    html = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>배당률 포함 이변 예측 보고서</title>
<style>
    body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; background: #fafafa; color: #333; line-height: 1.8; font-size: 15px; }}
    h1 {{ color: #1a237e; border-bottom: 3px solid #1a237e; padding-bottom: 10px; }}
    h2 {{ color: #283593; margin-top: 40px; border-left: 4px solid #3f51b5; padding-left: 12px; }}
    .summary {{ background: #e8eaf6; border-radius: 8px; padding: 20px; margin: 20px 0; }}
    .good {{ background: #e8f5e9; border-left: 4px solid #4caf50; padding: 14px 18px; margin: 15px 0; }}
    .insight {{ background: #fff3e0; border-left: 4px solid #ff9800; padding: 14px 18px; margin: 15px 0; }}
    .warn {{ background: #fce4ec; border-left: 4px solid #e91e63; padding: 14px 18px; margin: 15px 0; }}
    table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 13px; }}
    th {{ background: #3f51b5; color: white; padding: 9px; text-align: left; }}
    td {{ padding: 8px; border-bottom: 1px solid #e0e0e0; }}
    tr:hover {{ background: #e8eaf6; }}
    .chart {{ text-align: center; margin: 20px 0; }}
    .chart img {{ max-width: 100%; border: 1px solid #e0e0e0; border-radius: 4px; }}
    .code-block {{ background: #f5f5f5; border: 1px solid #e0e0e0; border-radius: 6px; padding: 14px; margin: 12px 0; font-family: Consolas, monospace; font-size: 13px; line-height: 1.6; overflow-x: auto; }}
    .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ccc; color: #777; font-size: 12px; text-align: center; }}
</style></head><body>

<h1>배당률 포함 이변 예측 — 최종 보고서</h1>

<div class="summary">
<h3>실험 요약</h3>
<ul>
<li><strong>질문:</strong> 배당률(q)을 피처에 포함시키면, 이변(비인기마 1착) 예측이 얼마나 개선되는가?</li>
<li><strong>방법:</strong> A(배당률만) vs B(피처만) vs C(배당률+피처) 세 모델 비교</li>
<li><strong>데이터:</strong> 서울 경마 비인기마(pop_pct >= 0.5)만 대상, 시간순 6:2:2 분할</li>
<li><strong>결론:</strong> C(결합) AUC = <strong>{c_best:.4f}</strong> → A 대비 <strong>{c_vs_a:+.4f}</strong>, B 대비 <strong>{c_vs_b:+.4f}</strong> 개선</li>
</ul>
</div>

<!-- ===== 1. 실험 설계 ===== -->
<h2>1. 실험 설계</h2>

<table>
<tr><th>모델</th><th>피처</th><th>의미</th></tr>
<tr><td><strong>A</strong> (베이스라인)</td><td>q (배당률 암묵적 확률) 1개</td><td>배당률만으로 이변을 얼마나 잡나?</td></tr>
<tr><td><strong>B</strong> (기존)</td><td>말/기수/환경 피처 ~100개 (배당률 완전 제외)</td><td>배당률 없이 이변을 잡을 수 있나?</td></tr>
<tr><td><strong>C</strong> (결합)</td><td>q + 말/기수/환경 피처 ~100개</td><td>배당률에 피처를 더하면 더 잘 잡나?</td></tr>
</table>

<div class="insight">
<strong>왜 이 실험이 의미있나?</strong><br>
- 지난 실험에서 "배당률을 제외하면 시장을 이길 수 없다"는 결론을 얻음<br>
- 이번엔 배당률을 "적"이 아닌 "동맹"으로 활용하되, 타겟을 "이변"으로 바꿈<br>
- 핵심: "배당률이 대략적인 인기도를 알려주고, 나머지 피처가 그 안에서 진짜 이길 말을 세분화"
</div>

<div class="code-block"><pre># Feature sets
A_features = ["q"]                    # odds probability only
B_features = [...100+ features...]    # no odds at all
C_features = B_features + ["q"]       # combined

# Target: upset (longshot wins)
upset = (pop_pct >= 0.5) & (win == 1)  # underdog AND 1st place</pre></div>

<!-- ===== 2. A/B/C 성능 비교 ===== -->
<h2>2. 모델 성능 비교 (A vs B vs C)</h2>
"""

    if comp is not None:
        html += comp.to_html(classes="", border=0, index=False)

        html += f"""
<div class="good">
<strong>핵심 결과:</strong><br>
- <strong>C > A:</strong> 배당률에 피처를 더하면 AUC {c_vs_a:+.4f} 개선<br>
- <strong>C > B:</strong> 배당률 없는 기존 모델 대비 AUC {c_vs_b:+.4f} 개선<br>
- <strong>A > B:</strong> 배당률 하나가 피처 100개보다 나음 → 시장 효율성 재확인<br>
- <strong>결론:</strong> 배당률은 이변 예측의 "기반"이고, 피처는 그 위에 "정밀도"를 더하는 역할
</div>
"""

    # ===== 3. Threshold =====
    html += "<h2>3. Threshold 튜닝</h2>\n"
    if thr_comp is not None:
        html += thr_comp.to_html(classes="", border=0)
        html += """
<div class="insight">
<strong>해석:</strong> 기본 임계값(0.5)에서는 Precision이 높지만 Recall이 낮음 (이변을 놓침).
최적 threshold를 낮추면 Recall이 올라가면서 F1이 개선됨 — "놓치지 않겠다" 전략.
</div>
"""

    # ===== 4. 구간별 검증 =====
    html += "<h2>4. 비인기마 배당 구간별 개선 효과</h2>\n"
    html += """
<div class="insight">
<strong>이 분석의 핵심 질문:</strong><br>
"배당률이 높은 말(더 비인기)일수록, 피처 추가의 가치가 커지는가?"<br>
→ 배당률은 인기도 자체를 보여주지만, <strong>고배당 구간에서는 말들 간 구분이 힘듦</strong>.<br>
→ 이때 훈련량, 기수 컨디션, 각질 등 피처가 "같은 배당대에서 누가 진짜 이길지" 구분해줌.
</div>
"""

    if seg_img:
        html += f'<div class="chart"><img src="{seg_img}" alt="Segment Comparison"></div>\n'

    if segments is not None:
        html += "<h3>구간별 상세 결과</h3>\n"
        html += segments.to_html(classes="", border=0, index=False)

        best_seg = segments.loc[segments["C_minus_A"].idxmax()] if len(segments) > 0 else None
        if best_seg is not None:
            html += f"""
<div class="good">
<strong>발견:</strong><br>
- 최대 개선 구간: <strong>{best_seg['segment']}</strong> (C가 A 대비 +{best_seg['C_minus_A']:.4f})<br>
- C 모델 Lift: {best_seg['Lift_C']:.2f}배 (해당 구간 baseline 대비)<br>
- 해석: 고배당 구간일수록 "배당률만으로는 부족하고, 피처가 추가 정보를 제공"
</div>
"""

    # ===== 5. Feature Importance =====
    html += "<h2>5. Feature Importance (C 모델)</h2>\n"
    if fi_img:
        html += f'<div class="chart"><img src="{fi_img}" alt="Feature Importance"></div>\n'

    html += """
<div class="insight">
<strong>그래프 보는 법:</strong><br>
- 주황색 막대 = q (배당률 확률). 전체에서 몇 위인지가 핵심.<br>
- q가 상위권 → 배당률이 이변 예측에서 여전히 가장 중요한 정보<br>
- q 아래의 파란 막대들 → 배당률 위에 추가로 기여하는 피처들 (직전 성적, 기수, 훈련 등)
</div>
"""

    if fi_comp is not None:
        html += "<h3>B 모델 vs C 모델 피처 순위 변화 (Top 10)</h3>\n"
        top10 = fi_comp.head(10)
        html += top10[["feature", "rank_c", "rank_b", "rank_change"]].to_html(classes="", border=0, index=False)
        html += """
<div class="insight">
rank_change = B에서의 순위 - C에서의 순위. 양수 = C에서 더 중요해짐, 음수 = 덜 중요해짐.<br>
q가 추가되면서 기존 피처들의 상대적 중요도가 어떻게 재편되었는지 확인 가능.
</div>
"""

    # ===== 6. 결론 =====
    html += f"""
<h2>6. 결론 및 시사점</h2>

<div class="summary">
<h3>핵심 발견 3가지</h3>
<ol>
<li><strong>배당률+피처 결합(C)이 배당률 단독(A)보다 이변 예측이 우수하다.</strong><br>
    특히 고배당(비인기) 구간에서 개선폭이 크다.</li>
<li><strong>배당률 없는 모델(B)은 여전히 배당률 포함 모델(A, C)에 미치지 못한다.</strong><br>
    q 하나가 피처 100개보다 강력하다 — 시장의 정보량은 압도적.</li>
<li><strong>모델의 실용적 가치:</strong> "같은 고배당 구간 내에서 진짜 이길 가능성이 높은 말"을 골라내는 필터.<br>
    전체 승률 예측은 시장에 맡기고, 모델은 시장이 놓치는 틈새를 공략하는 도구로 활용.</li>
</ol>
</div>

<div class="good">
<h3>발표 스토리라인 제안</h3>
<ol>
<li>"배당률을 제외하고 시장을 이기려 했지만 실패" (이전 실험)</li>
<li>"방향 전환: 배당률을 동맹으로 쓰고, 이변 탐지로 목표 변경"</li>
<li>"배당률+피처 결합이 배당률 단독보다 우수함을 확인" (이번 실험)</li>
<li>"특히 고배당 구간에서 개선폭이 큼 — 실용적 가치 존재"</li>
</ol>
</div>

<div class="footer">KHUDA 3조 · 배당률 포함 이변 예측 보고서 · {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</body></html>"""

    (OUTPUT_DIR / "report.html").write_text(html, encoding="utf-8")
    logger.info(f"Saved: results/upset_with_odds/report.html")


if __name__ == "__main__":
    main()
