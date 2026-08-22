# 공유·검증용 자료 모음 (share_for_review)

원본 프로젝트(`10th-toy-team3-main`)에서 **팀원 확인·검증에 필요한 파일만 복제**한 폴더다.
원본 파일은 그대로 남아 있으며, 여기 있는 것은 전부 사본이다.

- 프로젝트: 서울 경마 시장확률 보정 (AI 동아리 토이프로젝트 3팀)
- 제외한 것: `.venv`, `.git`, `__pycache__`, 전처리 데이터셋 CSV(약 200MB), `final.csv` 원본

## 먼저 볼 순서

1. `README.md` — 프로젝트 목표, 현재 결론, 최종 후보 모델과 Final Test 결과
2. `PROJECT_GUIDELINES.md` — 팀 작업 규칙과 실험 원칙 (가장 중요한 가이드라인 문서)
3. `보고서/경주마_시장확률_보정_최종_결과보고서.docx` — 최종 결과 보고서
4. `reports/experiments/stage_*_summary.md` — 단계별 실험 요약 (stage 12 → 27 순)
5. `SETUP.md`, `TESTING.md` — 환경 세팅과 테스트 재현 방법

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `src/` | 파이썬 소스 전체 (data / features / models / evaluation / inference / project) |
| `tests/` | 단위·검증 테스트 18개 |
| `scripts/` | 최종 리포트 빌드 스크립트, 테스트 실행 스크립트 |
| `reports/experiments/` | 단계별 실험 결과 JSON + 요약 md |
| `reports/final/assets/` | 최종 보고서용 그래프 이미지 |
| `data/manifests/` | 데이터·정책·피처 레지스트리 매니페스트 JSON (검증 기준값) |
| `data/predictions/` | 모델별 예측 결과 (.csv.gz) |
| `data/analysis/` | 부트스트랩·백테스트·게이트 분석 결과 |
| `artifacts/models/` | 학습된 모델 파일 (.joblib) |
| `보고서/` | 결과보고서 docx, 데이터명세서 xlsx, HTML 리포트 |
| `보고서/전처리/` | EDA 리포트, 전처리 버전 비교 리포트, split 설명 |

## 문서 파일

- `README.md` — 프로젝트 개요·결론
- `PROJECT_GUIDELINES.md` — 작업 가이드라인
- `SETUP.md` — 환경 구성
- `TESTING.md` — 테스트 가이드
- `FUTURE_HOLDOUT_VALIDATION.md` — 미래 홀드아웃 검증 절차
- `requirements.txt` — 의존성

## 검증 재현 방법

```powershell
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

전처리 데이터셋 CSV와 `final.csv`는 용량 때문에 제외했으므로,
원 데이터가 필요한 테스트는 원본 프로젝트 폴더에서 실행해야 한다.

> 본 자료는 연구 결과이며 실제 베팅 또는 금융 조언이 아니다.
