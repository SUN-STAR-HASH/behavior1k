"""Policy configuration for B1K - loads checkpoints and creates policies.

Exact copy of openpi.policies.policy_config but imports b1k.models.pi_behavior.PiBehavior.

비전공자용 큰 그림:
    학습이 끝나면 체크포인트 폴더 안에 모델 가중치(params)와 정규화 통계/assets가 저장된다.
    이 파일은 그 저장물을 읽어서 "서빙 가능한 policy 객체"로 조립한다.

    policy 객체는 크게 세 부분으로 이루어진다.
        1. 모델: 실제로 action을 예측하는 신경망
        2. 입력 transform: 이미지/상태/action id를 모델이 먹을 수 있는 형태로 바꿈
        3. 출력 transform: 모델이 낸 32차원 action을 환경이 쓰는 23차원 action으로 되돌림

    학습 때 쓰는 transform과 추론 때 쓰는 transform은 완전히 같지 않다.
    예를 들어 학습 때는 action label이 필요하지만, 추론 때는 action label이 없다.
    그래서 여기서는 추론에서 불필요한 transform을 일부 건너뛴다.
"""

import logging
import os
import pathlib
from typing import Any

import numpy as np
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.transforms as transforms

# Import B1K-specific modules
from b1k.models.pi_behavior import PiBehavior
from b1k.policies.pi_behavior_policy import PiBehaviorPolicy
from b1k.training import checkpoints as _checkpoints
from b1k.training import config as _config
from b1k import transforms as b1k_transforms
from b1k.transforms_normalize import NormalizeWithPerTimestamp, UnnormalizeWithPerTimestamp


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """학습된 체크포인트 폴더를 읽어서 추론용 policy를 만든다.

    checkpoint_dir:
        보통 `/home/user/models/checkpoint_1` 같은 로컬 폴더다.
        `gs://...`처럼 원격 URI일 수도 있다.

    norm_stats:
        action/state 정규화 통계다. None이면 checkpoint 안의 assets에서 읽는다.

    반환값:
        평가 서버가 `.infer(...)`로 호출할 수 있는 policy 객체다.
    """
    repack_transforms = repack_transforms or transforms.Group()

    # 사용자가 JSON이나 CLI에 "~/models/..."처럼 적는 경우가 많다.
    # pathlib.Path("~")는 자동으로 홈 디렉터리로 바뀌지 않으므로,
    # 로컬 경로일 때만 expanduser()로 명시적으로 풀어 준다.
    # 반대로 gs://, s3:// 같은 원격 경로는 건드리면 안 되므로 "://"가 있으면 그대로 둔다.
    checkpoint_uri = str(checkpoint_dir)
    if "://" not in checkpoint_uri:
        checkpoint_uri = str(pathlib.Path(checkpoint_uri).expanduser())
    checkpoint_dir = download.maybe_download(checkpoint_uri)

    # 이 프로젝트의 주 경로는 JAX 체크포인트다.
    # 혹시 PyTorch 파일이 들어온 경우를 감지해서 명확히 에러를 낸다.
    is_pytorch = (checkpoint_dir / "pytorch_model.safetensors").exists() or (checkpoint_dir / "pytorch_model.pt").exists()
    
    if is_pytorch:
        raise NotImplementedError("PyTorch inference not supported in b1k")
    
    # JAX model loading.
    # bfloat16으로 읽으면 GPU 메모리를 크게 줄일 수 있다.
    # 학습 때 float32였더라도 추론은 보통 bfloat16으로 충분하다.
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    
    # Get data config
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    # norm_stats는 "정규화/역정규화 기준표"다.
    # 학습 때 action과 state를 평균 0, 표준편차 1 근처로 맞췄다면,
    # 추론 때도 같은 기준을 써야 모델 출력이 실제 로봇 action 단위로 제대로 돌아온다.
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    
    # PiBehavior는 flow matching noise를 만들 때 action 차원 간 상관관계를 사용한다.
    # 이 상관행렬도 norm_stats에서 같이 복원해야 학습 때와 같은 방식으로 샘플링된다.
    if isinstance(model, PiBehavior):
        if norm_stats is None:
            raise ValueError("PiBehavior requires norm_stats but none found.")
        model.load_correlation_matrix(norm_stats)
        logging.info("Loaded correlation matrix for inference")
    
    # Determine the device for PyTorch (not used for b1k but kept for compatibility)
    if is_pytorch and pytorch_device is None:
        try:
            import torch
            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"
    
    # 추론에서는 아래 transform들을 건너뛴다.
    # - ComputeSubtaskStateFromMeta: 학습 데이터의 timestamp로 stage를 계산하는 용도
    # - TaskIndexToTaskId: wrapper가 이미 local task id를 직접 넣어 준다
    # - TokenizeFASTActions: action label을 FAST token으로 바꾸는 학습 보조 손실용
    model_transforms_inputs = []
    for transform in data_config.model_transforms.inputs:
        # Skip training-specific transforms during inference
        if isinstance(transform, (b1k_transforms.ComputeSubtaskStateFromMeta, b1k_transforms.TaskIndexToTaskId, b1k_transforms.TokenizeFASTActions)):
            continue
        model_transforms_inputs.append(transform)
    
    # 입력 transform 순서:
    # 1. wrapper가 만든 dict를 B1K 공통 입력 형식으로 바꿈
    # 2. state/action 정규화
    # 3. 이미지 resize, action padding 같은 모델 입력 최종 정리
    #
    # data_config.repack_transforms는 학습 데이터셋 전용 key mapping이므로
    # 웹소켓 평가 입력에는 그대로 쓰지 않는다.
    input_transforms = [
        *repack_transforms.inputs,
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        NormalizeWithPerTimestamp(norm_stats, use_quantiles=data_config.use_quantile_norm, use_per_timestamp=data_config.use_per_timestamp_norm),
        *model_transforms_inputs,
    ]
    
    # 출력 transform 순서:
    # 모델은 정규화된 action을 내므로 먼저 실제 action 단위로 되돌린다.
    # 마지막에는 B1kOutputs가 환경이 기대하는 23차원 action만 잘라서 반환한다.
    output_transforms = [
        *data_config.model_transforms.outputs,
        UnnormalizeWithPerTimestamp(norm_stats, use_quantiles=data_config.use_quantile_norm, use_per_timestamp=data_config.use_per_timestamp_norm),
        *data_config.data_transforms.outputs,
        *repack_transforms.outputs,
    ]
    
    # PiBehaviorPolicy는 PiBehavior 모델의 반환 형식과 inpainting 옵션을 다루는 전용 policy다.
    # 일반 OpenPI Policy를 그대로 쓰면 B1K Observation/Action 튜플 처리에서 어긋날 수 있다.
    if isinstance(model, PiBehavior):
        return PiBehaviorPolicy(
            model,
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=train_config.policy_metadata,
        )
    else:
        return _policy.Policy(
            model,
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs=sample_kwargs,
            metadata=train_config.policy_metadata,
            is_pytorch=is_pytorch,
            pytorch_device=pytorch_device if is_pytorch else "cpu",
        )

