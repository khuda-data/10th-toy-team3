# 테스트 가이드

## 전체 실행

```powershell
python -m unittest discover -s tests -v
```

Windows 편의 스크립트:

```powershell
.\scripts\run_tests.ps1
```

## 파일별 실행

```powershell
python -m unittest tests.test_schema -v
python -m unittest tests.test_seoul_interim -v
python -m unittest tests.test_splits -v
python -m unittest tests.test_feature_registry -v
python -m unittest tests.test_modeling_foundation -v
python -m unittest tests.test_bootstrap -v
python -m unittest tests.test_backtest -v
python -m unittest tests.test_prediction_contract -v
python -m unittest tests.test_experiment_reports -v
python -m unittest tests.test_documentation -v
```

## 테스트 영역

| 테스트 파일 | 핵심 계약 |
|---|---|
| `test_schema.py` | 원본 shape, checksum, ID 문자열 보존, 필수 열 |
| `test_seoul_interim.py` | 서울 범위, 장거리 이변 정의, 시장 확률 합, 비정상 경주 격리 |
| `test_splits.py` | 경주·날짜 단위 Train/Calibration/Test 분리 |
| `test_dataset_policy.py` | legacy 및 행 삭제 데이터셋 사용 차단 |
| `test_feature_registry.py` | 승인 피처만 사용, 결과·시장 누수 차단 |
| `test_modeling_foundation.py` | Train 전처리, walk-forward, 정규화·혼합·온도 수식 |
| `test_bootstrap.py` | paired bootstrap 평균과 신뢰구간 요약 |
| `test_backtest.py` | 동일금액 손익, 빈 선택, 보수적 `no_bet` fallback |
| `test_prediction_contract.py` | 완전 경주, 시점, 경주 전체 거부 |
| `test_experiment_reports.py` | 정책·결과·checksum·확률 합의 통합 회귀 검사 |
| `test_documentation.py` | 필수 문서, 고정 결과 설명, 문서화된 산출물 경로 |

## 중요한 테스트 원칙

- 테스트는 개별 출전마가 아니라 경주 구조를 함께 검증한다.
- Final Test 산출물의 checksum이 바뀌면 테스트가 실패해야 한다.
- 정책 파일과 보고서의 선택값이 다르면 실패해야 한다.
- 결과 열이 예측 출력에 포함되면 실패해야 한다.
- `prediction_rejected` 경주는 확률을 출력하지 않아야 한다.
- 테스트를 통과시키기 위해 동결 manifest나 기대 결과를 임의로 갱신하지 않는다.

## Final Test 관련 주의

`src.models.evaluate_final_test`는 테스트 명령이 아니다. 저장된 Final Test를 최초 1회 평가하는 잠금 명령이며 현재는 이미 실행된 상태다. 일반 회귀 테스트에서는 기존 결과와 checksum만 읽는다.

새 모델을 개발하려면 기존 Test 결과를 재사용해 튜닝하지 말고 새로운 미래 holdout과 새 실험 버전을 만든다.
