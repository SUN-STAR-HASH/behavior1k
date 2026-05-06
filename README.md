# BEHAVIOR-1K Solution

이 저장소는 OpenPI 기반의 BEHAVIOR-1K 정책 학습 및 평가 파이프라인입니다.

BEHAVIOR-1K Challenge 환경에서 사용할 수 있는 커스텀 `PiBehavior` 모델, 데이터
전처리, normalization 통계 계산, checkpoint 로딩, OmniGibson 평가용 websocket
policy server를 포함합니다.

RGB 관측, proprioception, task id 기반 conditioning, System 2 stage tracking,
task별 checkpoint switching을 지원합니다.

## 개요

현재 A100 재학습 기본 추천 config는 `pi_behavior_b1k_a100_baseline_stage_draft`입니다.
이 config는 기존 70k baseline과 같은 step / batch 조건에서 System 2 stage tracking만 추가합니다.

주요 특징은 다음과 같습니다.

- OpenPI 기반 VLA policy를 BEHAVIOR-1K에 맞게 수정
- 자연어 prompt 대신 task id 기반 conditioning 사용
- 선택한 12개 task subset을 global id에서 local id로 변환
- 3-view RGB 이미지와 robot proprioception 사용
- Pi0.5 backbone + task embedding + flow matching baseline 유지
- System 2 stage prediction / stage-conditioned token 지원
- A100 단일 GPU에서 검증된 70k baseline 조건을 기준으로 비교
- BEHAVIOR task id별 checkpoint switching 지원

## 저장소 구조

```text
src/b1k/
  models/          PiBehavior 모델 및 모델 설정
  training/        학습 설정, dataloader, checkpoint, weight loader
  policies/        policy 생성, checkpoint switching, inference wrapper
  shared/          normalization, eval wrapper, correction rule

scripts/
  compute_norm_stats.py      normalization 통계 계산
  train_fast_tokenizer.py    FAST tokenizer 학습
  train.py                   PiBehavior policy 학습
  serve_b1k.py               평가용 websocket policy server 실행

BEHAVIOR-1K/       공식 BEHAVIOR-1K / OmniGibson 코드
openpi/            OpenPI dependency
```

## 설치

권장 환경은 다음과 같습니다.

- Linux
- Python 3.11
- CUDA 12.x
- NVIDIA GPU

submodule과 함께 저장소를 clone합니다.

```bash
git clone --recurse-submodules https://github.com/SUN-STAR-HASH/behavior1k.git
cd behavior1k
```

설치 스크립트를 실행합니다.

```bash
bash setup_remote.sh
```

submodule이 비어 있다면 다음 명령을 실행합니다.

```bash
git submodule update --init --recursive
```

## 데이터셋

기본 config는 resized RGB 데이터셋을 사용합니다.

```text
IliaLarchenko/behavior_224_rgb
```

데이터셋 다운로드 예시입니다.

```bash
uv run huggingface-cli login

uv run python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="IliaLarchenko/behavior_224_rgb",
    repo_type="dataset",
    local_dir="./data/behavior_224_rgb",
    local_dir_use_symlinks=False,
)
PY
```

dataloader는 아래 구조의 parquet episode 파일을 기대합니다.

```text
<data_root>/data/task-*/episode_*.parquet
```

필요하면 `src/b1k/training/config.py`에서 경로를 수정합니다.

```python
behavior_dataset_root="./data/behavior_224_rgb"
assets_base_dir="./outputs/assets"
checkpoint_base_dir="./outputs/checkpoints"
```

## 전처리

학습 전에 normalization statistics를 계산해야 합니다.

```bash
uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_a100_baseline_stage_draft
```

FAST tokenizer는 기본 stage config에서 사용하지 않습니다. 나중에 FAST auxiliary를 다시 켤 때만 학습합니다.

```bash
uv run scripts/train_fast_tokenizer.py \
  --config-name pi_behavior_b1k_a100_baseline_stage_draft \
  --encoded-dims="0:6,7:23" \
  --vocab-size=1024
```

## 학습

이미 검증한 70k baseline 조건으로 stage tracking을 다시 학습하는 config입니다.
기존 `baseline_70k` 평가 점수가 낮게 나온 뒤, task embedding만으로는 복잡한 장기 task를 충분히 구분하기 어렵다는 가설을 확인하기 위한 재학습 경로입니다.

```bash
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --overwrite
```

기존 학습을 이어서 실행합니다.

```bash
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --resume
```

기본 설정은 다음과 같습니다.

- `num_train_steps=70000`
- `batch_size=28`
- `fsdp_devices=1`
- `save_interval=1000`
- `keep_period=5000`
- `log_interval=10`
- `num_flow_samples=1`
- `subtask_loss_weight=0.1`

메모리가 안정적이고 GPU 사용률이 낮으면 70k stage config에서 batch size를 올려 볼 수 있습니다.

```bash
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft \
  --batch_size=32 \
  --overwrite
```

순수 task embedding baseline은 `pi_behavior_b1k_a100_baseline_draft`로 남겨 두었습니다.

```bash
uv run scripts/train.py pi_behavior_b1k_a100_baseline_draft --overwrite
```

원래 70k baseline과 같은 길이로 비교하려면 아래 두 config를 사용합니다.

```bash
# 70k 순수 task embedding baseline
uv run scripts/train.py pi_behavior_b1k_a100_baseline_draft --overwrite

# 70k baseline + System 2 stage tracking
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --overwrite
```

즉 현재 비교 축은 `70k 순수 baseline`과 `70k stage tracking` 두 가지입니다.

Weights & Biases logging을 끄려면 다음 옵션을 사용합니다.

```bash
uv run scripts/train.py pi_behavior_b1k_a100_baseline_stage_draft --wandb_enabled=false
```

## Policy Server 실행

학습된 checkpoint를 websocket policy server로 실행합니다.

```bash
uv run scripts/serve_b1k.py \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_a100_baseline_stage_draft \
  --policy.dir /path/to/checkpoint
```

기본 port는 `8000`입니다. 다른 port를 사용하려면 다음처럼 실행합니다.

```bash
uv run scripts/serve_b1k.py \
  --port 8001 \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_a100_baseline_stage_draft \
  --policy.dir /path/to/checkpoint
```

stage tracking을 끄고 순수 task embedding 입력으로 평가하려면 다음 옵션을 추가합니다.

```bash
uv run scripts/serve_b1k.py \
  --no-use-stage-tracking \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_a100_baseline_draft \
  --policy.dir /path/to/checkpoint
```

## Task별 Checkpoint Switching

`task_checkpoint_mapping.json`을 사용하면 BEHAVIOR task id별로 다른 checkpoint를 사용할 수 있습니다.

```bash
uv run scripts/serve_b1k.py \
  --task-checkpoint-mapping task_checkpoint_mapping.json \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_a100_baseline_stage_draft \
  --policy.dir /path/to/initial/checkpoint
```

mapping 파일의 task id는 원본 BEHAVIOR global task id 기준입니다.

## 평가

먼저 policy server를 실행한 뒤, 다른 터미널에서 BEHAVIOR-1K evaluation을 실행합니다.

```bash
python BEHAVIOR-1K/omnigibson/learning/eval.py \
  log_path=./eval_logs \
  policy=websocket \
  model.host=localhost \
  model.port=8000 \
  task.name=make_microwave_popcorn \
  eval_instance_ids="[0,1,2,3]"
```

RTX 5070은 OmniGibson / Isaac Sim 실행과 평가에 사용하고, A100은 JAX 학습에 사용하는 구성을 권장합니다.

## 주의사항

- 이 저장소는 커스텀 JAX `PiBehavior` 모델을 대상으로 합니다.
- PyTorch inference는 이 코드 경로에서 구현되어 있지 않습니다.
- `compute_norm_stats.py`를 먼저 실행하지 않으면 학습 또는 inference에서 필요한 normalization stats를 찾지 못할 수 있습니다.
- 기본 config는 `gs://openpi-assets/checkpoints/pi05_base/params`에서 초기 weight를 읽습니다.
- BEHAVIOR-1K와 OmniGibson은 CUDA, GPU driver, display, streaming 환경에 민감할 수 있습니다.

## References

- BEHAVIOR-1K: https://github.com/StanfordVL/BEHAVIOR-1K
- BEHAVIOR Challenge: https://behavior.stanford.edu/challenge/
- OpenPI: https://github.com/Physical-Intelligence/openpi
