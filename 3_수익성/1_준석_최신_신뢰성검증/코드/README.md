# 준석 검증 코드 — 최종 사용 범위

최종 발표에서 필요한 검증 흐름만 남겼습니다.

1. `model_selection_validation/01_build_models.py`: 인기 하위 50% 다크호스와 인기마 부진 Random Forest를 재현하고 Test 예측을 저장합니다.
2. `model_selection_validation/02_bootstrap_and_sensitivity.py`: 상위 10% ROI·Lift를 경주 단위 군집 부트스트랩으로 점검하고, 고배당 적중 제거 민감도를 계산합니다.
3. `model_selection_validation/03_true_holdout.py`: 기존 Test 말단 약 8주를 시간 후행 구간으로 재평가합니다. 완전히 미관측인 잠금 Test로 표현하지 않습니다.
4. `pipeline/config.py`: 공통 열 정의와 시간순 분할 설정입니다.

저장소 루트에서 1→2 순서로 실행할 수 있습니다. 3은 별도의 시간 안정성 참고 점검입니다. 산출물은 저장소 루트의 `results/junseok_final_validation/`에 생성되며 Git 추적 대상이 아닙니다. 과거 결과 수치를 그대로 재사용하지 말고 수정 코드로 다시 실행해 확인해야 합니다.
