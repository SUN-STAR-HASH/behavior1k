# 수정 요약

이 문서는 `SUN-STAR-HASH/behavior1k` 저장소에서 원본 1등팀 코드 대비 수정한 내용을 날짜별로 누적 기록한다.

새 작업은 항상 이 파일의 위쪽에 추가하고, 예전 작업 기록은 지우지 않는다.

## 2026-05-06 10k week config 제거

### 배경

- 실제 baseline 학습은 `70_000 step`으로 완료했고, A100 한 장에서 1주일 이내 학습 가능하다는 것을 확인했다.
- 70k가 12개 task 전체 episode 중 약 10%를 보는 기준이므로, stage tracking 비교도 같은 70k 조건으로 맞추는 것이 맞다.
- 따라서 빠른 확인용으로 만들었던 `10k week` 계열 config는 제거했다.

### 제거한 config

- `pi_behavior_b1k_a100_week`
- `pi_behavior_b1k_a100_week_stage`

### 현재 사용할 config

- 순수 task embedding + flow matching 70k baseline:
  `pi_behavior_b1k_a100_baseline_draft`
- System 2 stage tracking 70k 재학습:
  `pi_behavior_b1k_a100_baseline_stage_draft`

실행 명령:

```bash
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_a100_baseline_stage_draft
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --overwrite
```

정리:

- 현재 비교 축은 `70k baseline` vs `70k stage tracking`이다.
- 10k config는 더 이상 코드에 남기지 않는다.
- 아래 과거 기록에 보이는 10k 관련 내용은 당시 실험 설계 기록이며, 현재 실행 기준은 아니다.

## 2026-05-06 70k stage tracking 재학습 기준 정리

### 배경

- 기존 `task embedding + flow matching` baseline은 `70_000 step`으로 학습했다.
- 이 70k는 12개 task 전체 episode 중 약 10% 정도를 보는 실험 기준으로 잡은 값이다.
- 실제 A100 학습 로그 기준으로 70k 학습이 1주일 이내에 가능하다는 것을 확인했다.
- 평가 점수가 낮게 나와, 단순 task id embedding만으로는 복잡한 장기 task의 진행 단계를 충분히 표현하기 어렵다는 문제가 보였다.
- 그래서 다음 재학습은 같은 70k 조건에서 System 2 stage tracking을 추가하는 방향으로 정리했다.

### 현재 추천 config

```bash
pi_behavior_b1k_a100_baseline_stage_draft
```

이 config는 기존 70k baseline인 `pi_behavior_b1k_a100_baseline_draft`와 비교하기 위해 만든 설정이다.

- 기존 70k baseline: `pi_behavior_b1k_a100_baseline_draft`
- 70k stage tracking: `pi_behavior_b1k_a100_baseline_stage_draft`
- `num_train_steps=70_000`
- `batch_size=28`
- `num_workers=6`
- `subtask_loss_weight=0.1`
- `use_stage_conditioning=True`

실행 명령:

```bash
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_a100_baseline_stage_draft
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --overwrite
```

중간부터 이어서 돌릴 때:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --resume
```

정리:

- `pi_behavior_b1k_a100_week_stage`의 10k는 빠른 구조 확인용이다.
- 실제 baseline과 공정 비교하는 stage tracking 재학습은 `pi_behavior_b1k_a100_baseline_stage_draft`를 사용한다.
- stage tracking 때문에 70k가 10k로 바뀐 것이 아니라, 실험 목적이 다른 config가 둘 다 존재하는 것이다.

## 2026-05-06 System 2 stage tracking 설정 추가

### 목표

- 기존 `task embedding + flow matching only` baseline은 비교용으로 유지한다.
- 새 실험 config에서는 1등팀의 System 2 아이디어처럼 현재 task stage를 모델 입력에 함께 넣는다.
- 학습 중에는 timestamp와 episode 길이로 pseudo stage label을 만들고, stage prediction head를 보조 손실로 학습한다.
- 추론 중에는 wrapper가 현재 stage를 `[local_task_id, current_stage]` 형태로 넣고, 모델이 예측한 stage logit을 voting으로 반영한다.

### 추가 점검 / 오류 수정

- `src/b1k/models/pi_behavior.py`에 `sample_actions()`가 두 번 정의되어 있던 부분을 정리했다.
  - Python은 뒤쪽 함수를 최종 사용하지만, 앞쪽 함수가 남아 있으면 사람이 볼 때 어떤 inference 경로가 실제인지 헷갈린다.
  - 앞쪽 단순 구현은 `_legacy_sample_actions_simple()`로 이름을 바꿔 참고용으로만 남겼다.
- stage loss가 `total_loss`에 잘못 더해질 수 있던 부분을 수정했다.
  - 기존 코드에는 stage logit 자체를 loss처럼 평균해서 더할 위험이 있었다.
  - 이제는 `subtask_loss`로 계산한 cross entropy만 `subtask_loss_weight`와 함께 더한다.
- `eval_b1k_wrapper.py`의 action debug filter 기본 동작을 정리했다.
  - `apply_eval_tricks=False`이면 raw action을 그대로 쓰도록 `B1K_ACTION_DEBUG_MODE=none` 경로를 추가했다.
  - `apply_eval_tricks=True`일 때만 기본 안정화 모드 `eval_stable_v6`가 적용된다.
- 70k baseline과 같은 조건으로 stage tracking만 켠 비교 config를 추가했다.
  - 기존 70k baseline: `pi_behavior_b1k_a100_baseline_draft`
  - 새 70k stage 비교: `pi_behavior_b1k_a100_baseline_stage_draft`

중요한 정리:

- stage tracking을 추가했다고 해서 기존 70k baseline이 10k로 바뀐 것은 아니다.
- `pi_behavior_b1k_a100_week`와 `pi_behavior_b1k_a100_week_stage`는 A100 1장으로 1주 이내에 빠르게 실험하려고 따로 만든 10k config다.
- 공정하게 "70k baseline vs 70k stage tracking"을 비교하려면 `pi_behavior_b1k_a100_baseline_draft`와 `pi_behavior_b1k_a100_baseline_stage_draft`를 비교하면 된다.

### 새로 추가한 config

- 파일: `src/b1k/training/config.py`
- config 이름: `pi_behavior_b1k_a100_week_stage`
- 실험 이름: `a100_week_stage_10k`
- 기본 학습 step: `num_train_steps=10_000`
- 기본 batch size: `batch_size=8`
- stage 보조 손실: `subtask_loss_weight=0.1`
- stage conditioning: `use_stage_conditioning=True`

### 유지한 baseline 성격

- correlated noise: OFF
- FAST auxiliary loss: OFF
- KV transform: OFF
- knowledge insulation: OFF
- delta joint action: OFF
- FAST tokenization: OFF

즉 이번 설정은 `task embedding + flow matching` baseline에 stage tracking만 추가한 비교 실험용이다.

### 수정한 코드 흐름

- `src/b1k/transforms.py`
  - `TaskIndexToTaskId`가 필요할 때 `[local_task_id, stage_id]`를 만들 수 있게 했다.
  - `ComputeSubtaskStateFromMeta`가 global task id를 local task id로 바꾼 뒤 `TASK_NUM_STAGES`를 조회하도록 수정했다.

- `src/b1k/training/data_loader.py`
  - stage 계산 transform이 실제 dataset metadata를 볼 수 있도록 dataset 생성 후 연결한다.
  - 이 연결이 있어야 episode length를 기준으로 timestamp를 stage id로 바꿀 수 있다.

- `src/b1k/models/pi_behavior.py`
  - stage logit의 cross entropy를 `total_loss`에 더하도록 정리했다.
  - 서빙/평가에 필요한 `sample_actions()`를 복원했다.
  - `sample_actions()`는 action chunk와 stage logits를 함께 반환한다.

- `src/b1k/shared/eval_b1k_wrapper.py`
  - `use_stage_tracking=True`이면 모델 입력을 `[local_task_id, current_stage]`로 만든다.
  - 모델이 예측한 stage를 history voting으로 반영해 stage가 너무 흔들리지 않게 했다.

- `scripts/serve_b1k.py`
  - 기본 예시 config를 `pi_behavior_b1k_a100_week_stage`로 바꿨다.
  - `--use-stage-tracking / --no-use-stage-tracking` 옵션으로 stage tracking을 켜고 끌 수 있게 했다.

### 실행 명령

학습:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week_stage --overwrite
```

이어 학습:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week_stage --resume
```

서빙:

```bash
uv run scripts/serve_b1k.py policy:checkpoint \
  --policy.config pi_behavior_b1k_a100_week_stage \
  --policy.dir /path/to/checkpoint
```

## 2026-04-16 A100 1주 이내 baseline 실험 설정 추가

### 목표

- RTX 5070은 OmniGibson / Isaac Sim 실행과 평가에 사용한다.
- A100은 JAX 학습에 사용한다.
- 1등팀의 8xH200 / 200k step 규모를 그대로 재현하지 않고, 12개 task subset에서 1주일 안에 끝낼 수 있는 baseline checkpoint를 먼저 만든다.
- 실험 성격은 "대회 1등 완전 재현"이 아니라 "Pi0.5 backbone + task embedding + flow matching 중심의 축소 baseline"이다.

### 새로 추가한 학습 config

- 파일: `src/b1k/training/config.py`
- config 이름: `pi_behavior_b1k_a100_week`
- 실험 이름: `a100_week_10k`
- 기본 학습 step: `num_train_steps=10_000`
- 기본 batch size: `batch_size=8`
- 단일 A100 기준: `fsdp_devices=1`
- checkpoint 저장 주기: `save_interval=1000`
- 오래 보관할 checkpoint 주기: `keep_period=2000`
- 로그 주기: `log_interval=50`
- flow matching sample 수: `num_flow_samples=1`

### baseline 순도를 위해 꺼둔 항목

- correlated noise: OFF
- FAST auxiliary loss: OFF
- KV transform: OFF
- knowledge insulation: OFF
- subtask / stage loss: OFF
- delta joint action: OFF
- FAST tokenization: OFF

### A100 1주 실험 실행 명령

처음 시작:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week --overwrite
```

중간에 끊겼을 때 이어서 시작:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week --resume
```

첫 500~1000 step 로그를 봤을 때 너무 느리면 step을 더 줄여서 실행:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week \
  --num_train_steps=5000 \
  --save_interval=500 \
  --keep_period=1000 \
  --overwrite
```

메모리가 안정적이고 GPU 사용률이 낮으면 batch size를 올려서 실행:

```bash
uv run scripts/train.py pi_behavior_b1k_a100_week \
  --batch_size=12 \
  --overwrite
```

### README 반영

- 파일: `README.md`
- 기존 README에는 원본 1등팀의 200k step 학습 예시가 먼저 보였다.
- 이제는 이 fork에서 바로 쓸 수 있는 `pi_behavior_b1k_a100_week` 실행 명령을 먼저 적었다.
- 원본 1등팀 규모의 single GPU / multi GPU 명령은 참고용으로 아래에 남겼다.

### 주의할 점

- 실제 1주일 안에 끝나는지는 A100 종류, 데이터 저장장치 속도, batch size, dataloader 병목에 따라 달라진다.
- 첫 500~1000 step에서 step/sec 또는 sec/step을 확인한 뒤, 10k step이 1주를 넘길 것 같으면 5k step으로 줄이는 것이 안전하다.
- 5070은 학습용으로 쓰기보다 OmniGibson 실행 / 평가용으로 쓰는 것이 맞다.

## 2026-04-16 실행 오류 가능성 수정 및 한글 주석 보강

### 실행 중 오류가 날 수 있는 부분 수정

- `src/b1k/shared/eval_b1k_wrapper.py`
  - 평가 환경에서 들어오는 `task_id`는 원래 BEHAVIOR-1K 기준 global task id로 유지하도록 정리했다.
  - 모델 입력에 넣는 값은 12개 subset 기준 local task id로 따로 관리하도록 `self.local_task_id`를 추가했다.
  - checkpoint 선택은 global id 기준, 모델 `task_embeddings` 조회는 local id 기준으로 분리했다.
  - 이 수정으로 multi-checkpoint 평가 때 잘못된 checkpoint를 고르거나 embedding index가 꼬이는 문제를 막는다.

- `src/b1k/training/data_loader.py`
  - `_filter_to_selected_tasks()`에 `strict` 인자를 추가했다.
  - 기존에는 함수 내부에서 `strict`를 참조하지만 함수 인자로 받지 않아 fake smoke 또는 예외 상황에서 바로 에러가 날 수 있었다.
  - 이제 필터 적용 실패를 경고로 넘길지, 즉시 에러로 막을지 선택할 수 있다.

- `src/b1k/policies/policy_config.py`
  - `~/models/checkpoint_1` 같은 로컬 checkpoint 경로를 `expanduser()`로 풀도록 수정했다.
  - Python의 `pathlib.Path("~")`는 shell처럼 자동으로 홈 디렉터리로 바뀌지 않기 때문에 명시 처리가 필요하다.

- `src/b1k/policies/checkpoint_switcher.py`
  - `task_checkpoint_mapping.json` 안의 checkpoint 경로도 `expanduser()`로 처리하도록 수정했다.
  - JSON의 `tasks` 값은 계속 global task id 기준으로 유지한다.

### 비전공자도 읽을 수 있도록 한글 주석을 보강한 파일

- `src/b1k/configs/task_subset.py`
  - global task id와 local task id의 차이를 설명했다.
  - 왜 12개 task만 다시 0~11로 번호를 매기는지 적었다.

- `src/b1k/training/data_loader.py`
  - 데이터 로더가 무엇을 하는지, batch가 왜 필요한지 설명했다.
  - subset 필터가 global task id 기준으로 동작한다는 점을 명확히 했다.

- `src/b1k/transforms.py`
  - transform이 데이터 모양을 바꾸는 작은 함수라는 점을 설명했다.
  - `tokenized_prompt`가 자연어 토큰이 아니라 local task id를 담는다는 점을 정리했다.

- `src/b1k/policies/b1k_policy.py`
  - proprioception이 무엇인지 설명했다.
  - 긴 로봇 센서값에서 왜 23차원 state만 뽑는지 적었다.
  - 이미지 형식 `HWC`와 `CHW` 차이를 주석으로 남겼다.

- `src/b1k/shared/eval_b1k_wrapper.py`
  - 평가 환경 observation을 모델 입력으로 바꾸는 전체 흐름을 설명했다.
  - `self.task_id`는 global id, `self.local_task_id`는 모델용 local id라는 점을 분리했다.

- `src/b1k/policies/policy_config.py`
  - checkpoint 폴더에서 policy를 만드는 과정을 설명했다.
  - 입력 transform, 출력 transform, norm stats, correlation matrix가 각각 왜 필요한지 적었다.

- `src/b1k/policies/checkpoint_switcher.py`
  - task별 checkpoint를 바꾸는 이유와 GPU 메모리 때문에 한 번에 하나만 올리는 이유를 설명했다.

- `scripts/serve_b1k.py`
  - websocket policy server 실행 흐름을 단계별로 적었다.
  - `--task-id`가 global task id 기준이라는 점을 명확히 했다.

### 만든 커밋

- `ee89c53 Fix task id mapping and checkpoint paths`
- `1df3670 Add detailed Korean code comments`
- `d3778cf Update Korean modification summary`

### 확인한 기본 검사

```bash
python -m compileall -q scripts src
```

- Python 문법 컴파일 검사를 통과했다.
- 실제 OmniGibson 실행이나 전체 학습은 돌리지 않았다.

## 2026-04-08 12-task baseline 구조 정리

### 전체 방향

- 원본 2025 BEHAVIOR Challenge 1등팀 코드를 기반으로 삼았다.
- 전체 50개 task를 모두 학습하는 대신, 선택한 12개 task만 빠르게 실험하도록 줄였다.
- 목표는 완전한 1등팀 재현이 아니라, Pi0.5 / openpi 계열 backbone 위에 task embedding과 flow matching을 붙인 baseline을 만드는 것이다.

### 핵심 아이디어

- 원본 BEHAVIOR-1K 데이터셋은 0~49 범위의 global task id를 사용한다.
- 현재 모델은 선택한 12개 task만 학습하므로 모델 내부 task embedding은 0~11 범위의 local task id를 사용한다.
- 데이터 로더와 평가 wrapper에서 global id와 local id를 섞지 않는 것이 중요하다.
- checkpoint mapping JSON과 평가 환경은 global task id 기준이다.
- 모델 입력의 `tokenized_prompt`에는 local task id가 들어간다.

### 12-task subset 관련 파일

- `src/b1k/configs/task_subset.py`
  - 선택한 12개 task 목록과 global/local task id 변환표를 둔다.

- `src/b1k/configs/__init__.py`
  - task subset config를 패키지에서 import할 수 있게 한다.

### 주요 수정 파일

- `src/b1k/models/pi_behavior.py`
  - Pi0.5 계열 모델에 task embedding / flow matching 중심 구조를 유지한다.
  - 1등팀 코드의 여러 실험 기능이 남아 있지만 baseline config에서는 대부분 꺼져 있다.

- `src/b1k/models/pi_behavior_config.py`
  - 모델 크기, action horizon, action dim, task 수, baseline 옵션을 설정한다.

- `src/b1k/models/observation.py`
  - observation에 task id와 필요한 입력 field가 들어가도록 정리한다.

- `src/b1k/training/config.py`
  - smoke, baseline, 5070 debug, A100 smoke 같은 실험 config를 추가했다.
  - 12개 task subset 기반 실험을 쉽게 시작할 수 있도록 했다.

- `src/b1k/training/data_loader.py`
  - 실제 BEHAVIOR / LeRobot 데이터셋에서 12개 task만 골라 학습할 수 있게 한다.

- `src/b1k/transforms.py`
  - dataset key와 task id를 모델 입력 형식으로 바꾼다.

- `src/b1k/shared/eval_b1k_wrapper.py`
  - 평가 서버에서 들어오는 observation을 policy 입력으로 바꾸고 action chunk를 관리한다.

- `scripts/serve_b1k.py`
  - 평가용 websocket policy server 실행 인자를 정리했다.

- `src/b1k/policies/checkpoint_switcher.py`
  - task별 checkpoint mapping을 읽고 필요한 checkpoint를 고를 수 있게 했다.

### 하드웨어 사용 방향

- RTX 5070
  - OmniGibson / Isaac Sim 실행, 환경 검증, 평가 루프 확인에 사용한다.
  - RTX 계열이라 RT core가 필요한 simulator 쪽에 적합하다.

- A100
  - JAX 학습에 사용한다.
  - Isaac Sim / OmniGibson 실행보다는 모델 학습에 쓰는 것이 적합하다.

### 기본 확인 명령

```bash
python -m compileall -q scripts src
```

이 명령은 Python 문법 수준에서 import 가능한 파일들이 깨지지 않았는지 빠르게 확인하기 위한 것이다.
