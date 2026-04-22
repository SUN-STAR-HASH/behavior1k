    # BEHAVIOR-1K OpenPI Solution

  이 저장소는 OpenPI 기반의 BEHAVIOR-1K 정책 학습 및 평가 파이프라인입니다.

  BEHAVIOR-1K Challenge 환경에서 사용할 수 있는 커스텀 `PiBehavior` 모델, 데이터
  전처리, normalization 통계 계산, FAST tokenizer 학습, checkpoint 로딩,
  OmniGibson 평가용 websocket policy server를 포함합니다.

  RGB 관측, proprioception, task id 기반 조건부 policy inference, task별
  checkpoint switching을 지원합니다.

  ## 개요

  기본 모델 설정 이름은 `pi_behavior_b1k_fast`입니다.

  주요 특징은 다음과 같습니다.

  - OpenPI 기반 VLA policy를 BEHAVIOR-1K에 맞게 수정
  - 자연어 prompt 대신 task id 기반 conditioning 사용
  - 3-view RGB 이미지와 robot proprioception 사용
  - delta action 학습 및 per-timestamp normalization 적용
  - action correlation statistics 기반 correlated-noise flow matching 지원
  - FAST auxiliary action-token prediction 지원
  - subtask/stage prediction auxiliary loss 지원
  - rolling inpainting, stage voting, action interpolation, correction rule을 포
  함한 평가 wrapper 제공
  - BEHAVIOR task id별 checkpoint switching 지원

  ## 저장소 구조

  ```text
  src/b1k/
    models/          PiBehavior 모델 및 모델 설정
    training/        학습 설정, dataloader, checkpoint, weight loader
    policies/        policy 생성, checkpoint switching, inference wrapper
    shared/          normalization, eval wrapper, correction rule

  scripts/
    compute_norm_stats.py      normalization 및 action correlation 통계 계산
    train_fast_tokenizer.py    action chunk용 FAST tokenizer 학습
    train.py                   PiBehavior policy 학습
    serve_b1k.py               평가용 websocket policy server 실행

  BEHAVIOR-1K/       공식 BEHAVIOR-1K / OmniGibson 코드
  openpi/            OpenPI dependency

  ## 설치

  권장 환경은 다음과 같습니다.

  - Linux
  - Python 3.11
  - CUDA 12.x
  - NVIDIA GPU

  submodule과 함께 저장소를 clone합니다.

  git clone --recurse-submodules https://github.com/SUN-STAR-HASH/behavior1k.git
  cd behavior1k

  설치 스크립트를 실행합니다.

  bash setup_remote.sh

  setup_remote.sh는 system dependency, uv, OpenPI, 현재 패키지, BEHAVIOR-1K /
  OmniGibson 평가 dependency를 설치합니다.

  submodule이 비어 있다면 다음 명령을 실행합니다.

  git submodule update --init --recursive

  ## 데이터셋

  기본 config는 resized RGB 데이터셋을 사용합니다.

  IliaLarchenko/behavior_224_rgb

  데이터셋 다운로드 예시입니다.

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

  dataloader는 아래 구조의 parquet episode 파일을 기대합니다.

  <data_root>/data/task-*/episode_*.parquet

  필요하면 src/b1k/training/config.py에서 경로를 수정합니다.

  behavior_dataset_root="./data/behavior_224_rgb"
  assets_base_dir="./outputs/assets"
  checkpoint_base_dir="./outputs/checkpoints"

  ## 전처리

  학습 전에 normalization statistics를 계산해야 합니다.

  uv run scripts/compute_norm_stats.py \
    --config-name pi_behavior_b1k_fast \
    --correlation

  기본 출력 위치는 다음과 같습니다.

  outputs/assets/pi_behavior_b1k_fast/IliaLarchenko/behavior_224_rgb/

  FAST auxiliary loss를 사용하려면 FAST tokenizer도 학습합니다.

  uv run scripts/train_fast_tokenizer.py \
    --config-name pi_behavior_b1k_fast \
    --encoded-dims="0:6,7:23" \
    --vocab-size=1024

  tokenizer는 같은 asset directory 아래에 저장됩니다.

  ## 학습

  단일 GPU 학습 예시입니다.

  uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=16 \
    --num_train_steps=200000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=100

  멀티 GPU / FSDP 학습 예시입니다.

  uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=2048 \
    --fsdp_devices=8 \
    --num_train_steps=200000 \
    --save_interval=500 \
    --keep_period=2000 \
    --log_interval=25

  기존 학습을 이어서 실행합니다.

  uv run scripts/train.py pi_behavior_b1k_fast --resume

  같은 experiment directory에서 새로 시작하려면 다음 옵션을 사용합니다.

  uv run scripts/train.py pi_behavior_b1k_fast --overwrite

  Weights & Biases logging을 끄려면 다음 옵션을 사용합니다.

  uv run scripts/train.py pi_behavior_b1k_fast --wandb_enabled=false

  기본 checkpoint 저장 위치는 다음과 같습니다.

  outputs/checkpoints/pi_behavior_b1k_fast/openpi/

  ## Policy Server 실행

  학습된 checkpoint를 websocket policy server로 실행합니다.

  uv run scripts/serve_b1k.py \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/checkpoint

  기본 port는 8000입니다. 다른 port를 사용하려면 다음처럼 실행합니다.

  uv run scripts/serve_b1k.py \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/checkpoint

  ## Task별 Checkpoint Switching

  task_checkpoint_mapping.json을 사용하면 BEHAVIOR task id별로 다른 checkpoint를
  사용할 수 있습니다.

  uv run scripts/serve_b1k.py \
    --task-checkpoint-mapping task_checkpoint_mapping.json \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/initial/checkpoint

  mapping 파일은 0부터 49까지 모든 task id를 포함해야 합니다.

  ## 평가

  먼저 policy server를 실행한 뒤, 다른 터미널에서 BEHAVIOR-1K evaluation을 실행
  합니다.

  python BEHAVIOR-1K/omnigibson/learning/eval.py \
    log_path=./eval_logs \
    policy=websocket \
    model.host=localhost \
    model.port=8000 \
    task.name=make_microwave_popcorn \
    eval_instance_ids="[0,1,2,3]"

  평가 wrapper는 OmniGibson observation을 모델 입력으로 변환하고, task stage
  state를 추적하며, correction rule과 action interpolation을 적용해 실행 가능한
  action으로 변환합니다.

  ## Viewer 실행

  BEHAVIOR 환경이 정상적으로 실행되는지 확인하려면 다음 명령을 사용합니다.

  uv run python run_behavior_task_viewer.py

  OmniGibson viewer가 필요하므로 rendering이 가능한 환경에서 실행해야 합니다.

  ## 주의사항

  - 이 저장소는 커스텀 JAX PiBehavior 모델을 대상으로 합니다.
  - PyTorch inference는 이 코드 경로에서 구현되어 있지 않습니다.
  - compute_norm_stats.py를 먼저 실행하지 않으면 학습 또는 inference에서 필요한
    normalization/correlation stats를 찾지 못할 수 있습니다.
  - 기본 config는 gs://openpi-assets/checkpoints/pi05_base/params에서 초기 weig
    ht를 읽습니다. 해당 경로를 사용할 수 없다면 src/b1k/training/config.py의 we
    ight_loader를 수정해야 합니다.
  - BEHAVIOR-1K와 OmniGibson은 CUDA, GPU driver, display, streaming 환경에 민감
    할 수 있습니다.

  ## References

  - BEHAVIOR-1K: https://github.com/StanfordVL/BEHAVIOR-1K
  - BEHAVIOR Challenge: https://behavior.stanford.edu/challenge/
  - OpenPI: https://github.com/Physical-Intelligence/openpi



