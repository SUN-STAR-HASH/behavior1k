import dataclasses
import enum
import logging
import os
import pathlib
import socket

import numpy as np
import tyro

# Set JAX memory allocation before importing JAX (can be overridden by env vars)
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')  # Use 50% of GPU memory
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')  # Platform allocator

from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata

from openpi.policies import policy as _policy

# Import B1K-specific modules
from b1k.policies import policy_config as _policy_config  # Use our custom policy_config
from b1k.policies.checkpoint_switcher import CheckpointSwitcher
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config


# 이 스크립트는 학습이 아니라 "서빙"용이다.
# 서빙은 체크포인트를 메모리에 올려 두고, 평가 환경이 웹소켓으로 관측값을 보내면
# action을 계산해서 다시 돌려주는 과정을 뜻한다.
#
# 실행 흐름:
#   1. tyro가 CLI 인자를 Args dataclass로 읽는다.
#   2. create_policy()가 체크포인트 폴더에서 모델을 복원한다.
#   3. B1KPolicyWrapper가 환경 observation을 모델 입력 형식으로 바꾼다.
#   4. WebsocketPolicyServer가 포트를 열고 평가 환경의 요청을 기다린다.
#
# task_id 주의:
#   CLI의 --task-id는 원본 BEHAVIOR-1K global task id 기준이다.
#   wrapper 내부에서 모델용 local task id로 변환된다.
class EnvMode(enum.Enum):
    # Not used, just kept for compatibility
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """serve_b1k.py 실행 인자 모음.

    tyro를 쓰기 때문에 아래 dataclass 필드들이 자동으로 CLI 옵션이 된다.
    예:
        uv run scripts/serve_b1k.py policy:checkpoint \
            --policy.config pi_behavior_b1k_fast \
            --policy.dir ~/models/checkpoint_1 \
            --port 8000
    """

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default prompt.
    default_prompt: str | None = None
    
    # PI_BEHAVIOR는 텍스트 prompt 대신 task id를 사용한다.
    # 이 값은 원본 BEHAVIOR-1K global task id(0~49)다.
    # 현재 12-task subset에 없는 id를 넣으면 wrapper에서 local id 변환 시 에러가 난다.
    task_id: int | None = None

    # Dataset root, used to retrieve the prompt of the task if taskname is not None.
    dataset_root: str | None = "/scr/behavior/2025-challenge-demos"
    # If provided, will be used to retrieve the prompt of the task, otherwise use turning_on_radio as default.
    task_name: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    
    # B1K Wrapper execution parameters.
    #
    # 모델은 action_horizon 길이만큼 여러 행동을 한 번에 예측한다.
    # 아래 값들은 그 예측 묶음을 실제 환경 step에 어떻게 나눠 실행할지 정한다.
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True  # Enable correction rules and gripper variation checks
    
    # Multi-checkpoint support for PI_BEHAVIOR models (optional).
    # JSON 안의 task id도 global task id 기준이어야 한다.
    task_checkpoint_mapping: str | None = None  # Path to task-checkpoint mapping JSON file


def create_policy(args: Args) -> _policy.Policy:
    """CLI 인자에 적힌 checkpoint를 읽어서 기본 policy를 만든다.

    여기서 만든 policy는 아직 평가 환경 입력을 바로 받을 수 없다.
    아래 main()에서 B1KPolicyWrapper로 한 번 더 감싸야 한다.
    """
    sample_kwargs = {"num_steps": args.num_steps}
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), 
        args.policy.dir, 
        default_prompt=args.default_prompt,
        sample_kwargs=sample_kwargs
    )


def main(args: Args) -> None:
    # B1K only supports PI_BEHAVIOR models (task embeddings, no text prompts).
    # config 이름은 src/b1k/training/config.py의 _CONFIGS 목록에 있어야 한다.
    config = _config.get_config(args.policy.config)
    
    # PI_BEHAVIOR model setup.
    # task_id를 CLI로 고정하지 않으면 평가 환경 observation 안의 task_id를 매번 읽는다.
    # 여러 태스크를 연속 평가할 때는 None으로 두는 편이 안전하다.
    if args.task_id is not None:
        logging.info(f"Using PI_BEHAVIOR model with task_id: {args.task_id}")
        task_id = args.task_id
    else:
        logging.info(f"Using PI_BEHAVIOR model - task_id will be extracted from observations")
        task_id = None
    
    # Placeholder prompt for PI_BEHAVIOR (not actually used by model)
    prompt = "PI_BEHAVIOR model (task-conditioned)"
    logging.info(f"Using prompt: {prompt}")

    # Load initial/default policy.
    # multi-checkpoint mode에서도 서버 시작 시 기본 policy 하나는 먼저 올린다.
    # 이후 task_id가 바뀌면 CheckpointSwitcher가 필요한 checkpoint로 교체한다.
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Create checkpoint switcher if mapping file provided.
    # mapping 파일을 주지 않으면 하나의 checkpoint로 모든 task를 처리한다.
    checkpoint_switcher = None
    if args.task_checkpoint_mapping:
        logging.info(f"Multi-checkpoint mode enabled: {args.task_checkpoint_mapping}")
        
        sample_kwargs = {"num_steps": args.num_steps}
        
        try:
            checkpoint_switcher = CheckpointSwitcher(
                config_path=args.task_checkpoint_mapping,
                training_config=config,
                sample_kwargs=sample_kwargs
            )
            logging.info("Checkpoint switcher initialized - will switch checkpoints based on task_id")
        except Exception as e:
            logging.error(f"Failed to initialize checkpoint switcher: {e}")
            raise
    else:
        logging.info("Single checkpoint mode - using one checkpoint for all tasks")

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Create wrapper configuration.
    # 이 값들은 모델 구조를 바꾸지 않고, 예측된 action chunk를 실행하는 방식만 바꾼다.
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=args.apply_eval_tricks,
    )
    
    logging.info(f"Wrapper config: execute={wrapper_config.actions_to_execute}, keep={wrapper_config.actions_to_keep}, steps={wrapper_config.execute_in_n_steps}, num_steps={wrapper_config.num_steps}")
    
    if wrapper_config.apply_eval_tricks:
        logging.info("Eval tricks ENABLED - correction rules and gripper variation checks active")
    else:
        logging.info("Eval tricks DISABLED (default behavior)")

    # Create B1K wrapper with PI_BEHAVIOR-specific features.
    # wrapper가 없으면 평가 환경의 원본 observation key와 모델 입력 key가 맞지 않는다.
    policy = B1KPolicyWrapper(
        policy, 
        text_prompt=prompt,  # Not used by PI_BEHAVIOR, kept for compatibility
        task_id=task_id,
        config=wrapper_config,
        checkpoint_switcher=checkpoint_switcher
    )
    
    if checkpoint_switcher:
        logging.info("Multi-checkpoint mode: checkpoints will switch based on task_id from observations")
    else:
        logging.info("Rolling inpainting enabled: will use initial_actions from input batch when provided")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
