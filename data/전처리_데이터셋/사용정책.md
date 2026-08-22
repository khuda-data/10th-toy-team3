# 기존 전처리 데이터셋 사용 정책

이 폴더의 CSV는 과거 실험 재현과 비교를 위한 읽기 전용 산출물이다. 신규 시장 확률 보정 파이프라인의 입력으로 사용하지 않는다.

## v1~v4

- 상태: `legacy_read_only`
- 과거 보고서 수치 확인에만 사용
- 신규 학습은 `data/interim/seoul_entries.csv.gz`와 `data/interim/split_manifest.csv`를 사용

## v5~v8

- 상태: `forbidden_model_input`
- 이상치를 행 단위로 삭제하여 같은 경주의 일부 출전마만 남음
- 검증·테스트의 완전한 경주 구조가 파괴됨
- 신규 모델 학습, 확률 평가, 백테스트에 사용 금지

기존 파일은 감사 가능성을 위해 삭제하지 않는다. 사용 허용 여부는 `data/manifests/dataset_policy.json`과 `src/data/dataset_policy.py`가 강제한다.
