# 이변모델 선정 결과

[`src/5_이변모델_선정/`](../../../src/5_이변모델_선정)을 실행해서 나온 표·그래프입니다.

파일명이 영문인 것은 코드가 이 이름 그대로 읽고 쓰기 때문입니다. 이름을 바꾸면 재실행 시 코드가 파일을 찾지 못합니다. 아래 표가 한글 대응입니다.

| 파일 | 무엇인가 | 만든 코드 |
|---|---|---|
| `improvement_results.csv` | 6단계 개선 실험 각 단계의 결과표 | `01_improvement_experiments.py` |
| `final_strategy.csv` | winOdds 구간별 필터 적용 ROI | `02_final_strategy.py` |
| `robustness.csv` | 고배당 제거 로버스트니스 검증 결과 | `01_improvement_experiments.py` |
| `cluster_profiles.csv` | 이변마 K-means 군집별 프로필 | `04_upset_insight_and_allocation.py` |
| `decision_tree_rules.txt` | 이변 여부를 가르는 의사결정나무 규칙 (텍스트) | `04_upset_insight_and_allocation.py` |
| `upset_profile_by_odds.csv` | 배당 구간별 이변마 프로필 | `04_upset_insight_and_allocation.py` |
| `upset_vs_normal_comparison.csv` | 이변마 vs 비이변마 피처 비교표 | `04_upset_insight_and_allocation.py` |
| `upset_vs_normal_distribution.png` | 위 비교의 분포 시각화 | `04_upset_insight_and_allocation.py` |
| `kmeans_elbow.png` | K-means 최적 군집 수 탐색(elbow) 그래프 | `04_upset_insight_and_allocation.py` |


---

## 재현성에 대해

**이 폴더의 CSV 는 박준석님이 로컬에서 실행해 얻은 원본 기록입니다.**
저장소의 코드로 다시 돌리면 숫자가 조금 달라집니다. 지우거나 덮어쓰지 않고 그대로 뒀습니다.

2026-08-23 에 확인한 내용입니다.

| 확인 항목 | 결과 |
|---|---|
| 입력 데이터 | 원본 `final.csv` 와 저장소 `race_entries.csv.gz` 의 **sha256 일치** |
| 코드 결정성 | 같은 환경에서 두 번 돌리면 완전히 동일 |
| 이식성 | 리눅스와 윈도우가 같은 값을 냄 |
| **scikit-learn 버전** | **버전마다 결과가 달라짐** |

원인은 scikit-learn 버전입니다. 같은 데이터·같은 코드·같은 `random_state=42` 인데도 이렇습니다.

| 환경 | `1_place_target/win` AUC |
|---|---:|
| 이 폴더의 기록 (준석님 로컬) | 0.741983 |
| sklearn 1.7.2 · 1.8.0 | 0.743686 |
| sklearn 1.9.0 | 0.744294 |

RandomForest 는 부트스트랩으로 트리를 만드는데, 시드를 고정해도 라이브러리 내부 구현이 바뀌면
뽑히는 표본이 달라집니다. 그래서 `requirements.txt` 가 버전을 `==` 로 못박고 있습니다.

### 여기서 얻은 것

**AUC 는 0.3% 안에서 움직이는데 ROI 는 훨씬 크게 흔들립니다.**

모델의 판별 능력 자체는 거의 같은데 수익률만 뒤집힌다는 것은,
**결론이 고배당 적중 한두 건에 좌우된다**는 뜻입니다.
`final_strategy.csv` 의 모든 행에서 신뢰구간 하한이 음수인 것과 같은 이야기를,
이번에는 재현 실험이 직접 보여준 셈입니다.

수익률 수치를 인용할 때는 이 점을 함께 밝혀야 합니다.
