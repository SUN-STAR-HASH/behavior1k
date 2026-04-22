  # BEHAVIOR-1K OpenPI Solution

  This repository contains a BEHAVIOR-1K policy training and evaluation pipeline
  built on top of OpenPI. It includes a custom `PiBehavior` model, BEHAVIOR-1K
  data processing, normalization utilities, FAST tokenizer training, checkpoint
  loading, and a websocket policy server for OmniGibson evaluation.

  The code is tailored for the BEHAVIOR-1K Challenge setup with RGB
  observations, proprioception, task-conditioned policy inference, and optional
  task-specific checkpoint switching.

  ## Overview

  The main model configuration is `pi_behavior_b1k_fast`.

  Key components:

  - OpenPI-based VLA policy adapted for BEHAVIOR-1K
  - Task-id conditioning instead of natural-language prompts
  - 3-view RGB input and robot proprioception
  - Delta action training with per-timestamp normalization
  - Correlated-noise flow matching using dataset action statistics
  - FAST auxiliary action-token prediction
  - Subtask/stage prediction auxiliary loss
  - Evaluation wrapper with rolling inpainting, stage voting, action
  interpolation, and correction rules
  - Optional checkpoint switching by BEHAVIOR task id

  ## Repository Structure

  ```text
  src/b1k/
    models/          PiBehavior model and model configs
    training/        training configs, dataloaders, checkpoints, weight loaders
    policies/        policy creation, checkpoint switching, inference policy
  wrappers
    shared/          normalization, eval wrapper, correction rules

  scripts/
    compute_norm_stats.py      compute normalization and action correlation
  stats
    train_fast_tokenizer.py    train FAST tokenizer for action chunks
    train.py                   train PiBehavior policies
    serve_b1k.py               serve policy over websocket for evaluation

  BEHAVIOR-1K/       official BEHAVIOR-1K / OmniGibson code
  openpi/            OpenPI dependency

  ## Installation

  Recommended environment:

  - Linux
  - Python 3.11
  - CUDA 12.x
  - NVIDIA GPU

  Clone the repository with submodules:

  git clone --recurse-submodules https://github.com/SUN-STAR-HASH/behavior1k.git
  cd behavior1k

  Run the setup script:

  bash setup_remote.sh

  The setup script installs system dependencies, uv, OpenPI, this package, and
  the BEHAVIOR-1K / OmniGibson evaluation dependencies.

  If submodules are missing, initialize them manually:

  git submodule update --init --recursive

  ## Dataset

  The default config uses the resized RGB dataset:

  IliaLarchenko/behavior_224_rgb

  Download it with:

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

  The dataloader expects parquet episodes under:

  <data_root>/data/task-*/episode_*.parquet

  Update paths in src/b1k/training/config.py if needed:

  behavior_dataset_root="./data/behavior_224_rgb"
  assets_base_dir="./outputs/assets"
  checkpoint_base_dir="./outputs/checkpoints"

  ## Preprocessing

  Compute normalization statistics before training:

  uv run scripts/compute_norm_stats.py \
    --config-name pi_behavior_b1k_fast \
    --correlation

  This writes assets to:

  outputs/assets/pi_behavior_b1k_fast/IliaLarchenko/behavior_224_rgb/

  Train the FAST tokenizer:

  uv run scripts/train_fast_tokenizer.py \
    --config-name pi_behavior_b1k_fast \
    --encoded-dims="0:6,7:23" \
    --vocab-size=1024

  The tokenizer is saved under the same asset directory.

  ## Training

  Single-GPU example:

  uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=16 \
    --num_train_steps=200000 \
    --save_interval=2000 \
    --keep_period=10000 \
    --log_interval=100

  Multi-GPU / FSDP example:

  uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=2048 \
    --fsdp_devices=8 \
    --num_train_steps=200000 \
    --save_interval=500 \
    --keep_period=2000 \
    --log_interval=25

  Resume training:

  uv run scripts/train.py pi_behavior_b1k_fast --resume

  Start a fresh run in the same experiment directory:

  uv run scripts/train.py pi_behavior_b1k_fast --overwrite

  Disable Weights & Biases logging:

  uv run scripts/train.py pi_behavior_b1k_fast --wandb_enabled=false

  Default checkpoints are saved to:

  outputs/checkpoints/pi_behavior_b1k_fast/openpi/

  ## Serving a Policy

  Start the websocket policy server:

  uv run scripts/serve_b1k.py \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/checkpoint

  The default port is 8000. To use another port:

  uv run scripts/serve_b1k.py \
    --port 8001 \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/checkpoint

  ## Task-Specific Checkpoint Switching

  task_checkpoint_mapping.json can be used to route different BEHAVIOR task ids
  to different checkpoints.

  uv run scripts/serve_b1k.py \
    --task-checkpoint-mapping task_checkpoint_mapping.json \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir /path/to/initial/checkpoint

  The mapping file must cover all task ids from 0 to 49.

  ## Evaluation

  Run the policy server first, then run BEHAVIOR-1K evaluation in another
  terminal:

  python BEHAVIOR-1K/omnigibson/learning/eval.py \
    log_path=./eval_logs \
    policy=websocket \
    model.host=localhost \
    model.port=8000 \
    task.name=make_microwave_popcorn \
    eval_instance_ids="[0,1,2,3]"

  The evaluation wrapper converts OmniGibson observations into model inputs,
  tracks task stage state, applies optional correction rules, and converts
  predicted action chunks into executable actions.

  ## Viewer

  To sanity-check the BEHAVIOR environment:

  uv run python run_behavior_task_viewer.py

  This requires a working OmniGibson viewer / rendering setup.

  ## Notes

  - This repository targets the custom JAX PiBehavior model.
  - PyTorch inference is not implemented in this codepath.
  - compute_norm_stats.py must be run before training or inference unless the
    required assets are already present in the checkpoint.
  - The default config initializes from
    gs://openpi-assets/checkpoints/pi05_base/params. If that path is unavailabl
    e, update the weight_loader in src/b1k/training/config.py.
  - BEHAVIOR-1K and OmniGibson setup can be sensitive to CUDA, GPU driver,
    display, and streaming configuration.

  ## References

  - BEHAVIOR-1K: https://github.com/StanfordVL/BEHAVIOR-1K
  - BEHAVIOR Challenge: https://behavior.stanford.edu/challenge/
  - OpenPI: https://github.com/Physical-Intelligence/openpi

