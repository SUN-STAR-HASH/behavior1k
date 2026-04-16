# 수정 요약

이 저장소는 2025 BEHAVIOR Challenge 1등팀 코드를 기반으로,
선택한 12개 태스크만 빠르게 실험할 수 있도록 줄이고 정리한 버전이다.

## 핵심 아이디어

- 원본 BEHAVIOR-1K 데이터셋은 0~49 범위의 global task id를 사용한다.
- 현재 모델은 선택한 12개 태스크만 학습하므로 내부 task embedding은 0~11 범위의 local task id를 사용한다.
- 따라서 데이터 로더와 평가 wrapper에서 global id와 local id를 헷갈리지 않는 것이 가장 중요하다.
- 체크포인트 mapping JSON과 평가 환경은 global task id 기준이다.
- 모델 입력의 `tokenized_prompt`에는 local task id가 들어간다.

## 주석을 자세히 보강한 파일

- `src/b1k/configs/task_subset.py`
  - 12개 태스크 목록과 global/local task id 변환표
- `src/b1k/training/data_loader.py`
  - BEHAVIOR/LeRobot 데이터셋을 batch로 읽는 흐름
- `src/b1k/transforms.py`
  - 데이터셋 key와 task id를 모델 입력 형식으로 바꾸는 transform
- `src/b1k/policies/b1k_policy.py`
  - 카메라 이미지와 로봇 proprioception을 모델 입력으로 바꾸는 부분
- `src/b1k/shared/eval_b1k_wrapper.py`
  - 평가 서버 observation을 모델 입력으로 바꾸고 action chunk를 관리하는 wrapper
- `src/b1k/policies/policy_config.py`
  - 체크포인트에서 추론용 policy를 만드는 부분
- `src/b1k/policies/checkpoint_switcher.py`
  - 태스크별로 사용할 체크포인트를 고르는 부분
- `scripts/serve_b1k.py`
  - 평가용 websocket policy server 실행 흐름

## 실험할 때 특히 조심할 점

- `--task-id`는 global task id 기준이다.
- `task_checkpoint_mapping.json`의 `tasks`도 global task id 기준이다.
- 모델 내부 `task_embeddings`는 12칸뿐이므로 모델 입력에는 반드시 local task id가 들어가야 한다.
- `~/models/checkpoint_1` 같은 경로는 코드에서 `expanduser()`로 처리되도록 수정했다.
- fake smoke config는 실제 OmniGibson 데이터 없이 shape와 train loop 진입을 확인하기 위한 용도다.

## 확인한 기본 검사

- `python -m compileall -q scripts src`
