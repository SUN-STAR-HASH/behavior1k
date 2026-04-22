  # BEHAVIOR-1K Solution

  BEHAVIOR-1K Challenge용 OpenPI 기반 학습 및 평가 코드입니다. 이 저장소는
  Stanford `BEHAVIOR-1K` 환경과 Physical Intelligence `openpi`를 함께 사용하며,
  BEHAVIOR-1K 50개 challenge task를 위한 `pi_behavior` 정책 모델, 데이터 전처리,
  학습 스크립트, websocket policy server를 포함합니다.

  이 README는 현재 저장소의 코드 기준으로 작성되어 있습니다.

  ## 구성

  - `src/b1k/models/`: BEHAVIOR-1K용 `PiBehavior` 모델과 observation/config 정의
  - `src/b1k/training/`: 학습 config, data loader, checkpoint/weight loader
  - `src/b1k/policies/`: checkpoint에서 policy를 만들고 task별 checkpoint
  switching 처리
  - `src/b1k/shared/`: 평가 wrapper, normalization, correction rule
  - `scripts/train.py`: JAX/FSDP 학습 스크립트
  - `scripts/compute_norm_stats.py`: state/action normalization 및 action
  correlation 통계 생성
  - `scripts/train_fast_tokenizer.py`: FAST auxiliary loss용 action tokenizer 학
  습
  - `scripts/serve_b1k.py`: OmniGibson evaluation에서 호출할 websocket policy
  server
  - `task_checkpoint_mapping.json`: task id별 checkpoint routing 예시
  - `BEHAVIOR-1K/`: 공식 BEHAVIOR-1K / OmniGibson 코드
  - `openpi/`: OpenPI 코드

  ## 모델 요약

  기본 config 이름은 `pi_behavior_b1k_fast`입니다.

  - Pi0.5/OpenPI 계열 VLA 구조를 BEHAVIOR-1K에 맞게 수정
  - 자연어 prompt 대신 50개 task id와 현재 subtask/stage embedding 사용
  - RGB 3-view 이미지와 proprioception state 사용
  - action horizon 30, action dim 32로 학습하며 실제 BEHAVIOR action은 23차원 사
  용
  - delta joint action과 per-timestamp normalization 사용
  - action correlation 기반 correlated noise flow matching 지원
  - FAST auxiliary action token loss 지원
  - subtask/stage prediction auxiliary loss 지원
  - inference wrapper에서 rolling inpainting, action interpolation, stage
  voting, correction rule 적용

  ## 설치

  권장 환경은 Linux, Python 3.11, CUDA 12 계열 GPU 환경입니다.

  ```bash
  git clone --recurse-submodules https://github.com/SUN-STAR-HASH/behavior1k.git
  cd behavior1k

  bash setup_remote.sh

  setup_remote.sh는 system dependency, uv, openpi, 현재 패키지, BEHAVIOR-1K/
  bddl, BEHAVIOR-1K/OmniGibson[eval]를 설치합니다.

  ## 데이터 준비

  기본 config는 resized RGB 데이터셋 IliaLarchenko/behavior_224_rgb를 사용하도록
  되어 있습니다.

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

  사용자 환경에 맞게 src/b1k/training/config.py의 경로를 수정하세요.

  behavior_dataset_root="./data/behavior_224_rgb"
  assets_base_dir="./outputs/assets"
  checkpoint_base_dir="./outputs/checkpoints"

  데이터 loader는 parquet 파일을 다음 구조에서 찾습니다.

  <behavior_dataset_root>/data/task-*/episode_*.parquet

  # 전처리

  학습 전에 normalization stats가 필요합니다.

  uv run scripts/compute_norm_stats.py \
    --config-name pi_behavior_b1k_fast \
    --correlation

  FAST auxiliary loss를 사용하려면 tokenizer도 학습합니다.

  uv run scripts/train_fast_tokenizer.py \
    --config-name pi_behavior_b1k_fast \
    --encoded-dims="0:6,7:23" \
    --vocab-size=1024

  출력은 기본적으로 아래에 저장됩니다.

  outputs/assets/pi_behavior_b1k_fast/IliaLarchenko/behavior_224_rgb/

  # 학습

  단일 GPU 예시:

  uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=16 \
    --num_train_steps=200000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=100

  멀티 GPU/FSDP 예시:

  uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=2048 \
    --fsdp_devices=8 \
    --num_train_steps=200000 \
    --save_interval=500 \
    --keep_period=2000 \
    --log_interval=25

  기존 학습을 이어서 실행:

  uv run scripts/train.py pi_behavior_b1k_fast --resume

  Weights & Biases를 사용하지 않으려면:

  uv run scripts/train.py pi_behavior_b1k_fast --wandb_enabled=false

  기본 checkpoint 저장 위치:

  outputs/checkpoints/pi_behavior_b1k_fast/openpi/

  ## Policy Server

  학습된 checkpoint를 websocket server로 띄웁니다.

  uv run scripts/serve_b1k.py \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/checkpoint

  기본 port는 8000입니다. 변경하려면 --port를 사용합니다.

  task별로 다른 checkpoint를 쓰려면 task_checkpoint_mapping.json의 path를 수정한
  뒤 실행합니다.

  uv run scripts/serve_b1k.py \
    --task-checkpoint-mapping task_checkpoint_mapping.json \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/initial/checkpoint

  mapping 파일은 0부터 49까지 모든 task id를 포함해야 합니다.

  # 평가

  policy server를 먼저 실행한 뒤, 다른 터미널에서 BEHAVIOR-1K evaluation을 실행
  합니다.

  python BEHAVIOR-1K/omnigibson/learning/eval.py \
    log_path=./eval_logs \
    policy=websocket \
    model.host=localhost \
    model.port=8000 \
    task.name=make_microwave_popcorn \
    eval_instance_ids="[0,1,2,3]"

  평가 wrapper는 BEHAVIOR observation을 모델 입력으로 변환하고, task id 기반
  stage state를 유지하며, action sequence를 실행용 action으로 변환합니다.

  ## Viewer

  환경이 제대로 뜨는지 확인하려면:

  uv run python run_behavior_task_viewer.py

  OmniGibson viewer가 필요하므로 GUI/renderer가 가능한 환경에서 실행해야 합니다.

  ## 주의사항

  - 이 저장소의 inference는 PiBehavior JAX checkpoint를 대상으로 합니다. PyTorch
    checkpoint inference는 구현되어 있지 않습니다.
  - compute_norm_stats.py를 먼저 실행하지 않으면 학습과 inference에서
    normalization/correlation stats를 찾지 못합니다.
  - config의 기본 weight loader는
    gs://openpi-assets/checkpoints/pi05_base/params에서 Pi0.5 base weight를 읽
    습니다. 접근이 안 되는 환경에서는 src/b1k/training/config.py의 weight_loade
    r를 로컬 checkpoint 또는 NoOpWeightLoader로 바꿔야 합니다.
  - BEHAVIOR-1K와 OmniGibson 설치는 GPU driver, CUDA, display/streaming 환경 영
    향을 많이 받습니다. 환경 문제는 먼저 공식 BEHAVIOR-1K quickstart가 동작하는
    지 확인하는 것이 좋습니다.

  # References

  - BEHAVIOR-1K: https://github.com/StanfordVL/BEHAVIOR-1K
  - BEHAVIOR Challenge: https://behavior.stanford.edu/challenge/
  - OpenPI: https://github.com/Physical-Intelligence/openpi
