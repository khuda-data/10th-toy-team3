# 이변모델 선정 결과

`../이변모델_선정_코드/`를 실행해서 나온 표·그래프입니다.

| 파일 | 무엇인가 | 만든 코드 |
|---|---|---|
| `improvement_results.csv` | 6단계 개선 실험 각 단계의 결과표 | `11_improvement_experiments.py` |
| `final_strategy.csv` | winOdds 구간별 필터 적용 ROI | `12_final_strategy.py` |
| `robustness.csv` | 고배당 제거 로버스트니스 검증 결과 | `11_improvement_experiments.py` |
| `cluster_profiles.csv` | 이변마 K-means 군집별 프로필 | `14_upset_insight_and_allocation.py` |
| `decision_tree_rules.txt` | 이변 여부를 가르는 의사결정나무 규칙 (텍스트) | `14_upset_insight_and_allocation.py` |
| `upset_profile_by_odds.csv` | 배당 구간별 이변마 프로필 | `14_upset_insight_and_allocation.py` |
| `upset_vs_normal_comparison.csv` | 이변마 vs 비이변마 피처 비교표 | `14_upset_insight_and_allocation.py` |
| `upset_vs_normal_distribution.png` | 위 비교의 분포 시각화 | `14_upset_insight_and_allocation.py` |
| `kmeans_elbow.png` | K-means 최적 군집 수 탐색(elbow) 그래프 | `14_upset_insight_and_allocation.py` |
