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

## 2026-04-16 오늘 반영한 내용

오늘은 크게 두 가지를 정리했다.

### 1. 실행 중 오류가 날 수 있는 부분 수정

- `src/b1k/shared/eval_b1k_wrapper.py`
  - 평가 환경에서 들어오는 `task_id`는 원본 BEHAVIOR-1K 기준 global task id로 유지하도록 정리했다.
  - 모델 입력에 넣는 값은 12개 subset 기준 local task id로 따로 관리하도록 `self.local_task_id`를 추가했다.
  - 체크포인트 선택은 global id 기준, 모델 `task_embeddings` 조회는 local id 기준으로 분리했다.
  - 이 수정으로 multi-checkpoint 평가 시 잘못된 체크포인트를 고르거나 embedding index가 어긋나는 문제를 막았다.

- `src/b1k/training/data_loader.py`
  - `_filter_to_selected_tasks()`에 `strict` 인자를 추가했다.
  - 기존에는 함수 내부에서 `strict`를 참조하지만 함수 인자로 받지 않아 fake smoke 또는 필터 실패 상황에서 런타임 에러가 날 수 있었다.
  - 이제 필터 적용 실패를 경고로 넘길지, 즉시 에러로 막을지 선택할 수 있다.

- `src/b1k/policies/policy_config.py`
  - `~/models/checkpoint_1` 같은 로컬 체크포인트 경로를 `expanduser()`로 풀도록 수정했다.
  - Python의 `pathlib.Path("~")`는 shell처럼 자동으로 홈 디렉터리로 바꾸지 않기 때문에 명시 처리가 필요하다.

- `src/b1k/policies/checkpoint_switcher.py`
  - `task_checkpoint_mapping.json` 안의 체크포인트 경로도 `expanduser()`로 처리하도록 수정했다.
  - JSON의 `tasks`는 계속 global task id 기준으로 유지한다.

### 2. 비전공자도 읽을 수 있도록 한글 주석 보강

- `src/b1k/configs/task_subset.py`
  - global task id와 local task id의 차이를 설명했다.
  - 왜 12개 task만 다시 0~11로 번호를 매기는지 적었다.

- `src/b1k/training/data_loader.py`
  - 데이터 로더가 무엇을 하는지, batch가 왜 필요한지 설명했다.
  - subset 필터가 global task id 기준으로 동작한다는 점을 명확히 했다.

- `src/b1k/transforms.py`
  - transform이 데이터 모양을 바꾸는 작은 함수라는 설명을 추가했다.
  - `tokenized_prompt`가 자연어 토큰이 아니라 local task id를 담는다는 점을 정리했다.

- `src/b1k/policies/b1k_policy.py`
  - proprioception이 무엇인지, 긴 로봇 센서값에서 왜 23차원 state만 뽑는지 설명했다.
  - 이미지 형식 `HWC`와 `CHW` 차이도 주석으로 남겼다.

- `src/b1k/shared/eval_b1k_wrapper.py`
  - 평가 환경 observation을 모델 입력으로 바꾸는 전체 흐름을 설명했다.
  - `self.task_id`는 global id, `self.local_task_id`는 모델용 local id라는 점을 주석으로 분리했다.

- `src/b1k/policies/policy_config.py`
  - 체크포인트 폴더에서 추론용 policy를 만드는 과정을 설명했다.
  - 입력 transform, 출력 transform, norm stats, correlation matrix가 각각 왜 필요한지 적었다.

- `src/b1k/policies/checkpoint_switcher.py`
  - 태스크별 체크포인트를 바꾸는 이유와 GPU 메모리 때문에 한 번에 하나만 올리는 이유를 설명했다.

- `scripts/serve_b1k.py`
  - websocket policy server 실행 흐름을 단계별로 적었다.
  - `--task-id`가 global task id 기준이라는 점을 명확히 했다.

### 오늘 만든 커밋

- `ee89c53 Fix task id mapping and checkpoint paths`
- `1df3670 Add detailed Korean code comments`

### 오늘 확인한 검사

- `python -m compileall -q scripts src`
  - Python 문법 컴파일 검사를 통과했다.
  - 실제 OmniGibson 실행이나 전체 학습은 돌리지 않았다.
