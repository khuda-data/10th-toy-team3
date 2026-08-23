# data/ — 무엇이 저장소에 있고 무엇을 만들어 써야 하나

이 파이프라인의 데이터는 **세 등급**으로 나뉩니다. 저장소에는 앞의 두 등급만 들어 있습니다.

| 등급 | 폴더 | 저장소에 | 이유 |
|---|---|---|---|
| 원본 | `raw/` | ✗ 없음 | 저장소 최상위 `data/raw/` 에 한 벌 있어 중복을 피했습니다 |
| 계약·정책 | `manifests/` · `interim/split_manifest.csv` | ✓ 있음 | **재생성하면 안 되는 동결 값**입니다. split 경계, 피처 승인 목록, 평가 잠금 |
| 중간 산출 | `interim/*.csv.gz` · `predictions/` · `analysis/` | ✗ 없음 | 위 둘로부터 **코드가 다시 만들어냅니다** |

## 왜 중간 산출을 뺐나

`interim/seoul_entries.csv.gz` 하나가 14MB, `predictions/` 16개가 3.9MB, `analysis/` 5개가 0.6MB —
합쳐서 **18.5MB** 였습니다. 셋 다 원본과 manifests만 있으면 아래 명령으로 그대로 복원됩니다.
저장소에는 재생성할 수 없는 것만 남깁니다.

## 복원 순서

먼저 [`raw/README.md`](raw/README.md) 대로 `raw/final.csv.gz` 를 채우세요. 그다음 패키지 루트에서:

```powershell
python -m src.data.validate_schema        # 원본 스키마·체크섬 확인
python -m src.data.build_seoul_interim    # → interim/seoul_entries.csv.gz
python -m src.data.build_ranking_dataset  # → interim/ranking_*_manifest
python -m src.models.market_baseline      # → reports/experiments/m0_market_baseline.json
python -m src.models.train_m1_logistic    # → predictions/m1_*
python -m src.models.train_m2_xgboost     # → predictions/m2_*
python -m src.models.select_normalization
python -m src.models.select_market_blend
python -m src.models.select_temperature
python -m src.models.train_r2_ranker      # → predictions/r2_*
python -m src.models.train_market_gate    # → analysis/stage_26_*
python -m src.evaluation.bootstrap        # → analysis/stage_16_*
```

`build_splits` 는 다시 돌리지 마세요. `interim/split_manifest.csv` 가 **이미 동결된 분할**이고,
재실행하면 Final Test 경계가 바뀝니다.

## 남아 있는 것

```text
manifests/                 20개 — 정책·계약·평가 잠금 (동결)
interim/split_manifest.csv  Train/Calibration/Final Test 경주 배정 (동결)
predictions/stage_18_contract_fixture.csv
                            예측 출력 계약 예시 — 문서가 참조하는 샘플
analysis/stage_22_segment_summary.csv
analysis/stage_26_gate_grid.csv
analysis/stage_27_candidate_comparison.csv
                            최종 후보 선택 근거 요약표 (원본 반복표본은 제외)
raw/README.md               원본을 어디서 가져오는지
```

지운 파일의 원본 바이트가 필요하면 `main` 브랜치 또는 커밋 `7f90843` 에 그대로 있습니다.
