# 04/08 수정 내용

- 웨이트 shape 유지
- 50-task 구조 유지
- 실제 학습/평가에서는 12개 task만 사용
- 5070은 OmniGibson/실행 담당
- A100은 정책 추론/학습 담당
- task embedding, stage 관련 입력 형식, flow matching 구조 유지

## 특히 많이 손본 파일
- `src/b1k/models/pi_behavior.py`
- `src/b1k/models/pi_behavior_config.py`
- `src/b1k/models/observation.py`
- `src/b1k/training/config.py`
- `src/b1k/training/data_loader.py`
- `src/b1k/transforms.py`
- `src/b1k/shared/eval_b1k_wrapper.py`
- `scripts/serve_b1k.py`
- `src/b1k/policies/checkpoint_switcher.py`

## 새로 추가한 파일
- `src/b1k/configs/task_subset.py`
- `src/b1k/configs/__init__.py`
- `task_checkpoint_mapping_light12.json`

## 참고
코드의 클래스 이름, 함수 이름, 라이브러리 이름은 실행을 위해 영어 그대로 두었습니다.
대신 주석과 설명 문장은 가능한 한 한글로 바꿨습니다.
