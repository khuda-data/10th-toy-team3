# 확률 품질 독립 재현 패키지

이 폴더는 원본 서울 경마 데이터에서 기존 확률 모델과 21~27단계 Top-1 제약 도전자를 독립적으로 재생성하기 위한 최소 패키지다.

파생 데이터, 학습된 모델, 예측 CSV와 실험 보고서는 포함하지 않는다. `data/raw/final.csv.gz` 하나에서 모두 다시 생성한다.

## 재현 범위

다음 과정을 순서대로 재현한다.

1. 원본 스키마와 체크섬 검증
2. 서울 경주 interim 및 시간순 Train/Calibration/Final Test 분할
3. 사전 피처 registry와 Train 전용 전처리
4. 시장 기준선, Logistic, XGBoost 확률 모델
5. 경주 정규화, 시장 혼합, temperature scaling
6. 동결 Final Test, bootstrap 및 경제성 진단
7. 시장/M2 Top-1 불일치 분석
8. pairwise ranker, 확률 변환, 시장 유지형 gate
9. 27단계 Future Holdout 도전자 동결
10. 전체 계약 테스트와 기대 지표 비교

보고서 DOCX·HTML 재생성과 실시간 추론 예시는 공식 재현 범위에서 제외했다.

## 포함 파일

```text
data/raw/final.csv.gz                 유일한 원본 데이터
data/manifests/                       재현 전에 고정된 입력 정책 3개
src/                                  실행 경로의 import 폐쇄에 필요한 소스만
tests/                                데이터·모델·정책·기대 지표 테스트
expected/                             입력 체크섬과 핵심 기대 지표
FUTURE_HOLDOUT_VALIDATION.md          27단계 동결 구성요소
requirements-lock.txt                 Python 의존성 고정
reproduction_config.json              재현 단계와 입력·출력 계약
reproduce.py                          단일 실행 진입점
```

`data/interim`, `data/predictions`, `data/analysis`, `artifacts`, `reports`는 실행 중 생성되며 Git 추적 대상이 아니다.

## 요구 환경

- 64비트 CPython 3.12.x
- 충분한 메모리와 디스크 공간
- 인터넷 연결은 최초 패키지 설치 때만 필요

Python 3.12 이외의 버전은 라이브러리 및 모델 직렬화 차이 때문에 공식 동일 재현으로 인정하지 않는다.

## 실행 방법

### Windows PowerShell

```powershell
cd 확률품질_김찬진
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe reproduce.py --check-inputs
.\.venv\Scripts\python.exe reproduce.py
```

### macOS/Linux

```bash
cd 확률품질_김찬진
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements-lock.txt
./.venv/bin/python reproduce.py --check-inputs
./.venv/bin/python reproduce.py
```

`--check-inputs`는 설치 전에도 사용할 수 있으며 원본·정책 파일 5개의 SHA-256만 확인한다.

## 검증 기준

재현이 끝나면 다음을 자동 검증한다.

- 원본과 고정 정책의 SHA-256
- 원본·interim·split의 행 수, 경주 수, 기간 및 식별자 계약
- 피처 누수 차단과 경주 확률합
- Final Test가 정확히 한 번만 평가됐는지 여부
- 15·26·27단계 핵심 지표가 `expected/metrics.json`과 절대오차 `1e-8` 이내인지 여부
- 도전자 ID, random seed, gate threshold와 lambda

모델 바이너리의 체크섬은 OS와 라이브러리 빌드에 따라 달라질 수 있으므로 공식 교차 환경 판정은 데이터 계약과 평가 지표를 우선한다.

## 재실행 정책

Final Test 일회성 잠금 때문에 `reproduce.py`는 깨끗한 폴더에서 한 번만 실행한다. 일부 결과를 선택적으로 삭제해 다시 실행하지 않는다. 재실행이 필요하면 이 폴더를 새 위치에 다시 복사해 시작한다.

## 제외한 항목

- 이미 생성된 모델 `.joblib`
- interim·예측·분석 CSV
- 단계별 JSON·Markdown 보고서
- 중복 HTML·DOCX·XLSX 보고서
- 보고서 렌더링 전용 Pillow·python-docx 의존성
- 실시간 예측 출력 fixture 및 문서 생성 전용 모듈
- 현재 파이프라인에서 import되지 않는 이전·보조 코드

원본 데이터의 재배포 또는 외부 공개 전에는 데이터 출처의 이용 조건을 별도로 확인해야 한다.

