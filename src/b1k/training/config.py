"""BEHAVIOR-1K 학습 설정 파일.

이 파일은 데이터셋을 어떻게 읽을지, 어떤 변환을 거칠지,
그리고 모델 입력 형식을 어떻게 맞출지를 정한다.
이번 수정의 핵심은 아래와 같다.

1. backbone은 pi0(pi05_base) 웨이트로 초기화한다.
2. 실제 학습/평가에서는 선택한 12개 태스크만 사용한다.
3. 모델 내부 task embedding은 12개만 두고, 전역 task id를 로컬 0~11로 바꿔 넣는다.
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, List, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.shared.download as _download
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms
import openpi.shared.nnx_utils as nnx_utils

# Import from B1K custom modules
from b1k.models import pi_behavior_config
from b1k.policies import b1k_policy
from b1k.shared import normalize as _normalize
from b1k.training import weight_loaders
from b1k import transforms as b1k_transforms
from b1k.configs.task_subset import GLOBAL_TO_LOCAL, SELECTED_TASKS

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.
    """
    # Assets directory. If not provided, the config assets_dirs will be used.
    assets_dir: str | None = None
    # Asset id. If not provided, the repo id will be used.
    asset_id: str | None = None

# [4/8] 현재 B1K-only 흐름에서는 안 쓸 가능성이 크지만, 다른 함수 시그니처에서 참조 중일 수도 있어 주석처리
@dataclasses.dataclass(frozen=True)
class DataConfig:
    # 사용할 LeRobot 데이터셋 이름.
    # None이면 진짜 데이터를 읽지 않고 테스트용 가짜 데이터를 만든다.
    repo_id: str | None = None
    # 정규화 통계나 토크나이저 같은 부가 파일이 들어 있는 하위 폴더 이름.
    asset_id: str | None = None
    # 미리 계산해 둔 정규화 통계. None이면 정규화를 하지 않는다.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # 데이터셋마다 다른 입력 형식을 공통 형식으로 맞추는 변환
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 로봇 특성에 맞는 추가 데이터 변환
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # 모델 전용 변환. 정규화 후에 적용된다.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    
    # True면 분위수 정규화를 쓰고, False면 일반적인 z-score 정규화를 쓴다.
    use_quantile_norm: bool = False
    # True면 행동 시퀀스를 시간축별로 따로 정규화한다.
    use_per_timestamp_norm: bool = False

    # 행동 시퀀스를 만들 때 어떤 키를 읽을지 지정
    action_sequence_keys: Sequence[str] = ("actions",)

    # True면 데이터셋의 task 문자열을 프롬프트로 쓴다. PI_BEHAVIOR에서는 사용하지 않는다.
    prompt_from_task: bool = False

    # RLDS 로더에서만 사용
    rlds_data_dir: str | None = None

    # B1K 로더에서만 사용
    behavior_dataset_root: str | None = None

    # DROID 데이터셋용 action space
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # DROID 데이터 필터 파일 경로
    filter_dict_path: str | None = None

    # 학습에 사용할 episode 번호 목록
    episodes_index: List[int] | None = None

    # True면 전체 50개 태스크 대신 선택한 12개 태스크만 사용한다.
    use_task_subset: bool = False

    # subset 모드일 때 허용할 원래 task id 목록
    allowed_task_ids: List[int] | None = None

class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""

@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """B1K용 모델 입력 변환 묶음을 만든다."""

    # 이 모델은 텍스트 프롬프트 대신 task embedding을 쓰므로 사실상 사용하지 않는다.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        # [4/8] 나중에 prompt 구성이나 meta 처리에서 다시 건드릴 여지가 있어서 주석 처리
        return _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                # 전역 task id(원본 데이터셋 기준)를 subset 로컬 id(0~11)로 바꾼다.
                # stage는 기본 경로에서 사용하지 않으므로 tokenized_prompt는 [task_id]만 만든다.
                b1k_transforms.TaskIndexToTaskId(task_mapping=GLOBAL_TO_LOCAL),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,  # Always use z-score normalization for B1K
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    """Data configuration for BEHAVIOR-1K dataset."""

    action_sequence_keys: Sequence[str] = ("action",)
    use_delta_joint_actions: bool = False
    
    # FAST auxiliary tokenization (only for PI_BEHAVIOR with use_fast_auxiliary)
    use_fast_tokenization: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms for B1K observations
        repack_mapping = {
            "observation/egocentric_camera": "observation.images.rgb.head",
            "observation/wrist_image_left": "observation.images.rgb.left_wrist",
            "observation/wrist_image_right": "observation.images.rgb.right_wrist",
            "observation/state": "observation.state",
            "actions": "action",
            "task_index": "task_index",  # Always preserve task_index
            "timestamp": "timestamp",    # Preserve timestamp for subtask state computation
            "episode_index": "episode_index",  # Preserve episode_index for episode length lookup
            "index": "index",           # Preserve index
        }
            
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        # Prepare data for policy training
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )

        # Delta action transforms
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # 모델 입력용 변환.
        # 여기서 가장 중요한 부분은 원래 전역 task id를
        # 12개 subset 전용 로컬 task id(0~11)로 바꾸는 것이다.
        model_transforms = ModelTransformFactory()(model_config)
        
        # FAST tokenization (if enabled for PI_BEHAVIOR)
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            
            # Get base config to access norm_stats
            base_config = self.create_base_config(assets_dirs, model_config)
            
            # Only add transform if tokenizer directory exists
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(
                    f"FAST tokenizer not found at {tokenizer_path}. "
                    "FAST auxiliary training will be disabled (inference mode)."
                )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            # 아래 두 값은 "모델 구조를 12개로 줄인다"는 뜻이 아니다.
            # 모델은 그대로 50개 구조를 유지하고,
            # 실제로 읽고 학습할 데이터만 12개 태스크로 제한하겠다는 뜻이다.
            use_task_subset=True,
            allowed_task_ids=list(SELECTED_TASKS),
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "B1K"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config (PI_BEHAVIOR only for B1K).
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_behavior_config.PiBehaviorConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Note: PyTorch support removed - JAX only

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotB1KDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int | None = None
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # FSDP configuration for model sharding across devices.
    fsdp_devices: int = 1
    
    # Validation configuration
    val_log_interval: int = 100
    val_batch_size: int | None = None
    val_num_batches: int = 10
    val_repo_id: str | None = None
    val_episodes_index: List[int] | None = None

    # True면 전체 50개 태스크 대신 선택한 12개 태스크만 사용한다.
    use_task_subset: bool = False

    # subset 모드일 때 허용할 원래 task id 목록
    allowed_task_ids: List[int] | None = None
    
    # Number of flow matching samples per training step
    num_flow_samples: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

# B1K Training Configurations
_CONFIGS = [
    # ------------------------------------------------------------------
    # 0) 가장 먼저 돌릴 스모크 테스트용 baseline
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi_behavior_b1k_smoke",
        exp_name="smoke",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,

            # ---- baseline: 추가 기법 OFF ----
            use_correlated_noise=False,
            correlation_beta=0.0,

            use_fast_auxiliary=False,
            fast_loss_weight=0.0,

            use_kv_transform=False,
            use_knowledge_insulation=False,

            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=False,
            ),
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=1e-4,
            decay_steps=200,
            decay_lr=1e-5,
        ),
        num_flow_samples=1,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=200,
        log_interval=10,
        save_interval=100,
        keep_period=200,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=2,
        batch_size=4,
        wandb_enabled=True,
    ),

    # ------------------------------------------------------------------
    # 1) 본 baseline
    # pi0 backbone + 12개 task embedding + flow matching only
    # ------------------------------------------------------------------
    TrainConfig(
        name="pi_behavior_b1k_baseline",
        exp_name="baseline",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,

            # ---- baseline ----
            use_correlated_noise=False,
            correlation_beta=0.0,

            use_fast_auxiliary=False,
            fast_loss_weight=0.0,

            use_kv_transform=False,
            use_knowledge_insulation=False,

            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=False,
            ),
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=1,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=30_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=8,
        batch_size=16,
        save_interval=1000,
        keep_period=5000,
    ),

    TrainConfig(
        name="pi_behavior_b1k_5070_smoke",
        exp_name="5070_smoke",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=False,
            correlation_beta=0.0,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            use_kv_transform=False,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=False,
            ),
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=1e-4,
            decay_steps=200,
            decay_lr=1e-5,
        ),
        num_flow_samples=1,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=200,
        log_interval=10,
        save_interval=100,
        keep_period=200,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=2,
        batch_size=1,
        wandb_enabled=False,
        fsdp_devices=1,
        val_num_batches=1,
    ),

    TrainConfig(
        name="pi_behavior_b1k_5070_debug",
        exp_name="5070_debug",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=False,
            correlation_beta=0.0,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            use_kv_transform=False,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=False,
            ),
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=1e-4,
            decay_steps=1000,
            decay_lr=1e-5,
        ),
        num_flow_samples=1,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=1000,
        log_interval=20,
        save_interval=200,
        keep_period=1000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=4,
        batch_size=2,
        wandb_enabled=False,
        fsdp_devices=1,
        val_num_batches=1,
    ),

    TrainConfig(
        # 이 config 이름으로 CLI에서 실행함
        # 예: python scripts/train.py pi_behavior_b1k_laptop_smoke --overwrite
        name="pi_behavior_b1k_laptop_smoke",
        # 체크포인트/실험 폴더 이름
        exp_name="laptop_smoke",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            # action horizon / action dim은 BEHAVIOR 기본 파이프라인에 맞춤
            action_horizon=30,
            action_dim=32,
            # ---- baseline 성격의 최소 설정 ----
            # 추가 기법은 전부 끄고 구조 smoke만 보려는 목적
            use_correlated_noise=False,
            correlation_beta=0.0,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            use_kv_transform=False,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            # vision backbone은 유지하되, 여기서는 구조 확인이 목적
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            # 핵심: 실데이터 대신 fake dataset 분기로 보내기 위한 repo_id
            repo_id="fake",
            base_config=DataConfig(
                # task 문자열 대신 task embedding 흐름을 사용하므로 False
                prompt_from_task=False,
                # fake smoke에서는 실제 데이터 루트가 필요 없음
                behavior_dataset_root=None,
                # per-timestamp normalization 사용 안 함
                use_per_timestamp_norm=False,
            ),
            # per-timestamp normalization 사용 안 함
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        # 아주 짧은 smoke용 scheduler
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1,
            peak_lr=1e-4,
            decay_steps=2,
            decay_lr=1e-5,
        ),
        # flow sample도 1개로 최소화
        num_flow_samples=1,
        # 핵심: pretrained weight 다운로드/복원을 하지 않음
        weight_loader=weight_loaders.NoOpWeightLoader(),
        # 2 step만 돌아가는 최소 smoke
        num_train_steps=2,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        # fake smoke에서는 멀티워커가 오히려 복잡성만 늘리므로 0
        num_workers=0,
        # batch size도 1로 최소화
        batch_size=1,
        # step마다 바로 저장/로그를 보게 짧게 잡음
        save_interval=1,
        keep_period=1,
        log_interval=1,
        # wandb는 꺼서 오버헤드 제거
        wandb_enabled=False,
        # 단일 GPU 기준
        fsdp_devices=1,
        # validation도 최소
        val_num_batches=1,
    ),

    TrainConfig(
        # full-size smoke가 너무 무거워서,
        # Gemma 300m / 300m 조합으로 줄인 tiny version
        name="pi_behavior_b1k_5070_fake_smoke",
        exp_name="5070_fake_smoke",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            # 핵심: paligemma / action expert 둘 다 300m로 축소
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=False,
            correlation_beta=0.0,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            use_kv_transform=False,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="fake",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root=None,
                use_per_timestamp_norm=False,
            ),
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        # smoke 성격이라 scheduler도 짧게
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=1e-4,
            decay_steps=200,
            decay_lr=1e-5,
        ),
        num_flow_samples=1,
        # pretrained restore 제거
        weight_loader=weight_loaders.NoOpWeightLoader(),
        # tiny는 구조 검증용이므로 2 steps만
        num_train_steps=200,
        log_interval=10,
        save_interval=100,
        keep_period=200,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=0,
        batch_size=1,
        wandb_enabled=False,
        fsdp_devices=1,
        val_num_batches=1,
    ),

    TrainConfig(
        # 최종적으로 5070에서 실제 train loop 진입까지 보려고 만든 최소 dummy config
        name="pi_behavior_b1k_5070_fake_dummy",
        exp_name="5070_fake_dummy",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            # 핵심: 가능한 가장 작은 dummy variant 사용
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=30,
            action_dim=32,
            # 추가 기법 전부 OFF
            use_correlated_noise=False,
            correlation_beta=0.0,
            use_fast_auxiliary=False,
            fast_loss_weight=0.0,
            use_kv_transform=False,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.0,
            # backbone 자체 구조는 남기되 학습은 최소화
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            # 실데이터/OmniGibson 없이 fake raw schema 사용
            repo_id="fake",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root=None,
                use_per_timestamp_norm=False,
            ),
            use_delta_joint_actions=False,
            use_fast_tokenization=False,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1,
            peak_lr=1e-4,
            decay_steps=2,
            decay_lr=1e-5,
        ),
        num_flow_samples=1,
        # pretrained restore 제거
        weight_loader=weight_loaders.NoOpWeightLoader(),
        # EMA도 끔
        # 이유: smoke에서 ema_params까지 들고 있으면 메모리 사용량이 더 커짐
        ema_decay=None,
        # 핵심: PaliGemma 쪽 전체를 freeze해서
        # trainable parameter와 optimizer state를 최대한 줄임
        # 즉, 큰 vision/llm 부분은 forward만 통과하고 gradient/update는 최소화
        freeze_filter=nnx_utils.PathRegex(r".*PaliGemma.*"),
        num_train_steps=2,
        log_interval=1,
        save_interval=1,
        keep_period=1,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=0,
        batch_size=1,
        wandb_enabled=False,
        fsdp_devices=1,
        val_num_batches=1,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

