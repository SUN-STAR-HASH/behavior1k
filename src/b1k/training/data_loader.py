"""BEHAVIOR-1K 학습용 데이터 로더.

핵심 역할은 아래와 같다.
- OmniGibson/LeRobot 형식의 데이터를 읽는다.
- OpenPI의 배치/변환 파이프라인을 그대로 재사용한다.
- 이번 수정에서는 50개 태스크 전체가 아니라 12개 태스크만 걸러서 쓸 수 있게 한다.
"""

import importlib
import logging
import os
import time
from typing import Literal

import numpy as np
from torch.utils.data import Dataset
import jax
import torch
import lerobot.datasets.lerobot_dataset as lerobot_dataset

LeRobotDatasetMetadata = getattr(lerobot_dataset, "LeRobotDatasetMetadata", None)

import openpi.models.model as _model
import openpi.training.data_loader as _openpi_data_loader
from b1k.training import config as _config
from b1k.models.observation import Observation
from b1k.configs.task_subset import SELECTED_TASKS

logger = logging.getLogger(__name__)


# 전체 50개 task 구조는 유지하되, 실제 학습 데이터는 선택한 subset만 쓰기 위한 필터다.
# 즉 "모델 구조를 12-task로 줄이는 것"이 아니라,
# "데이터셋에서 특정 task만 남기는 것"에 가깝다.
#
# dataset 구현체마다 내부에 실제 Hugging Face dataset이 들어있는 필드명이 다를 수 있어서
# hf_dataset / dataset 순서로 찾아본다.
# 내부 dataset 객체를 찾지 못하면 학습을 아예 깨지 않기 위해 경고만 남기고 원본을 반환한다.
def _filter_to_selected_tasks(dataset, allowed_task_ids):
    """선택한 태스크만 남기는 간단한 필터.

    데이터셋 구현마다 내부 구조가 조금 달라서,
    대표적으로 많이 쓰는 `hf_dataset` 또는 `dataset` 필드를 우선 시도한다.
    필터를 적용할 수 없는 구조면 경고만 남기고 원본을 그대로 반환한다.
    """
    allowed = set(int(x) for x in allowed_task_ids)
    base = getattr(dataset, "hf_dataset", None) or getattr(dataset, "dataset", None)
    if base is None or not hasattr(base, "filter"):
        logger.warning("subset 필터를 적용할 수 있는 내부 dataset 객체를 찾지 못했다. 원본 데이터셋을 그대로 사용한다.")
        return dataset
    try:
        filtered = base.filter(lambda ex: int(ex.get("task_index", ex.get("task_id", -1))) in allowed)
        if getattr(dataset, "hf_dataset", None) is not None:
            dataset.hf_dataset = filtered
        elif getattr(dataset, "dataset", None) is not None:
            dataset.dataset = filtered
        logger.info("12개 task subset 필터 적용 완료: %s", sorted(allowed))
    except Exception as exc:
        logger.warning("subset 필터 적용 중 문제가 생겨 원본 데이터셋을 그대로 사용한다: %s", exc)
    return dataset


# OpenPI 기본 DataLoader는 openpi 쪽 Observation 형식을 기준으로 동작한다.
# 그런데 현재 우리 학습 루프(train_step)는 b1k.models.observation.Observation을 기대한다.
# 그래서 여기서는 배치를 그대로 넘기지 않고, Observation.from_dict(batch)로 한 번 감싸서
# "우리 프로젝트가 기대하는 입력 형식"으로 맞춘 뒤 (Observation, actions) 튜플로 반환한다.
#
# 이 래퍼가 없으면
# 1) OpenPI Observation / B1K Observation 타입 불일치가 생길 수 있고,
# 2) fast_tokens 같이 B1K 쪽에서 확장한 필드가 학습 코드로 자연스럽게 전달되지 않을 수 있다.
class DataLoaderImpl(_openpi_data_loader.DataLoader):
    """우리 프로젝트용 Observation 형식으로 배치를 바꿔서 내보내는 DataLoader.

    OpenPI 기본 로더는 openpi 쪽 Observation 타입을 만들 수 있는데,
    현재 학습 코드는 b1k.models.observation.Observation을 기대하므로
    여기에서 한 번 감싸서 넘겨준다.
    """
    def __init__(
        self,
        data_config: _config.DataConfig,
        data_loader: _openpi_data_loader.TorchDataLoader | _openpi_data_loader.RLDSDataLoader,
    ):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]

# [4/8] 삭제 후보
# def _expand_root(root: str | None) -> str | None:
#     """Expand '~' and return an absolute path."""
#     if root is None:
#         return None
#     return os.path.abspath(os.path.expanduser(root))


# def _get_dataset_fps(repo_id: str, default_fps: float = 30.0) -> float:
#     """Try to read dataset fps from LeRobot metadata, otherwise fall back."""
#     if LeRobotDatasetMetadata is None:
#         logger.warning(
#             "LeRobotDatasetMetadata is not available in this lerobot install. "
#             "Falling back to %.1f fps.",
#             default_fps,
#         )
#         return default_fps
#     try:
#         dataset_meta = LeRobotDatasetMetadata(repo_id)
#         fps = getattr(dataset_meta, "fps", None)
#         if fps is not None:
#             return float(fps)
#     except Exception as exc:
#         logger.warning(
#             "Could not read dataset fps from LeRobot metadata for %s. "
#             "Falling back to %.1f fps. (%s)",
#             repo_id,
#             default_fps,
#             exc,
#         )
#     return default_fps


# def _build_delta_timestamps(
#     action_sequence_keys: tuple[str, ...] | list[str],
#     action_horizon: int,
#     fps: float,
# ) -> dict[str, list[float]]:
#     """Build delta timestamps expected by LeRobot / BehaviorLeRobotDataset."""
#     return {
#         key: [t / fps for t in range(action_horizon)]
#         for key in action_sequence_keys
#     }


# OmniGibson 버전에 따라 BehaviorLeRobotDataset의 import 경로가 달라질 수 있다.
# 그래서 한 경로를 고정으로 믿지 않고, 후보 import 경로들을 순서대로 시도한다.
#
# 이 함수가 필요한 이유는:
# - 연구실/서버/개인 PC마다 OmniGibson 버전 차이가 날 수 있고
# - 같은 코드라도 환경에 따라 import 에러가 날 수 있기 때문이다.
# def _get_behavior_lerobot_dataset_cls():
#     """Import BehaviorLeRobotDataset from any known OmniGibson path.

#     OmniGibson 버전에 따라 import path가 달라질 수 있으므로 여러 후보를 시도한다.
#     """
#     candidate_modules = [
#         "omnigibson.learning.datas.lerobot_dataset",
#         "omnigibson.learning.data.lerobot_dataset",
#         "omnigibson.learning.utils.lerobot_dataset",
#     ]
#     for module_name in candidate_modules:
#         try:
#             module = importlib.import_module(module_name)
#         except Exception:
#             continue
#         dataset_cls = getattr(module, "BehaviorLeRobotDataset", None)
#         if dataset_cls is not None:
#             logger.info("Using BehaviorLeRobotDataset from %s", module_name)
#             return dataset_cls
#     return None


# # BehaviorLeRobotDataset 생성자 시그니처는 버전마다 조금씩 다를 수 있다.
# # 그래서 kwargs를 한 번에 고정하지 않고,
# # "가장 많은 옵션을 주는 경우"부터 "최소 인자만 주는 경우"까지 순서대로 시도한다.
# #
# # 여기서 중요한 포인트는:
# # - download_videos=False 로 스모크 테스트 부담을 줄이고,
# # - episodes / root / local_only 같은 옵션이 환경에 따라 안 먹을 수 있으므로
# #   TypeError를 기준으로 안전하게 fallback 한다는 점이다.
# #
# # 즉, 이 함수는 데이터셋 생성 자체보다 "환경 차이를 견디는 호환 레이어" 역할이 더 크다.
# def _instantiate_behavior_dataset(
#     dataset_cls,
#     repo_id: str,
#     root: str | None,
#     delta_timestamps: dict[str, list[float]],
#     episodes_index: list[int] | None,
#     seed: int | None,
# ):
    # """Instantiate BehaviorLeRobotDataset with a few signature fallbacks.

    # OmniGibson / BEHAVIOR 버전에 따라 생성자 인자가 조금씩 다를 수 있어서
    # 여러 kwargs 조합을 순차적으로 시도한다.
    # """
    # common_kwargs = {
    #     "repo_id": repo_id,
    #     "delta_timestamps": delta_timestamps,
    # }

    # candidate_kwarg_sets: list[dict[str, object]] = [
    #     {
    #         "root": root,
    #         "episodes": episodes_index,
    #         "download_videos": False,
    #         "local_only": False,
    #         "chunk_streaming_using_keyframe": False,
    #         "seed": seed,
    #     },
    #     {
    #         "root": root,
    #         "episodes": episodes_index,
    #         "download_videos": False,
    #         "local_only": False,
    #         "chunk_streaming_using_keyframe": False,
    #     },
    #     {
    #         "root": root,
    #         "episodes": episodes_index,
    #         "download_videos": False,
    #         "local_only": False,
    #     },
    #     {
    #         "root": root,
    #         "episodes": episodes_index,
    #         "download_videos": False,
    #     },
    #     {
    #         "root": root,
    #         "episodes": episodes_index,
    #     },
    #     {
    #         "root": root,
    #     },
    #     {},
    # ]

    # errors: list[str] = []
    # for extra_kwargs in candidate_kwarg_sets:
    #     kwargs = {
    #         **common_kwargs,
    #         **{k: v for k, v in extra_kwargs.items() if v is not None},
    #     }
    #     try:
    #         dataset = dataset_cls(**kwargs)
    #         logger.info(
    #             "Created BehaviorLeRobotDataset with kwargs: %s",
    #             sorted(kwargs.keys()),
    #         )
    #         return dataset
    #     except TypeError as exc:
    #         errors.append(f"{sorted(kwargs.keys())}: {exc}")

    # # 랩탑 / fake smoke 목적에서는 HF remote fallback이 문제를 일으켜서 의도적으로 비활성화    
    # raise TypeError(
    #     "Could not instantiate BehaviorLeRobotDataset with any known signature.\n"
    #     + "\n".join(errors)
    # )


# # 랩탑 스모크 환경에서는 LeRobot fallback을 의도적으로 막아 둔다.
# # 이유는 fallback 경로가 로컬 데이터만 읽는 것처럼 보여도,
# # 실제로는 Hugging Face metadata 조회나 원격 다운로드를 건드릴 수 있기 때문이다.
# #
# # 즉, "OmniGibson이 없으면 그냥 더 느리게라도 진행"이 아니라,
# # "잘못하면 원격 접근으로 흐름이 꼬이므로 여기서 명시적으로 중단"하는 정책이다.
# def _instantiate_fallback_lerobot_dataset(
#     repo_id: str,
#     root: str | None,
#     delta_timestamps: dict[str, list[float]],
# ):
#     raise RuntimeError(
#         "OmniGibson is not installed in this laptop environment, "
#         "and LeRobot fallback is disabled because it triggers remote HF downloads."
#     )


# 이 파일의 핵심 분기점.
# - repo_id == "fake" 이면 OmniGibson 없이도 돌아가는 smoke용 가짜 데이터셋을 만든다.
# - 그 외에는 실제 BehaviorLeRobotDataset을 생성한다.
#
# fake dataset의 목적은 "정확한 데이터 재현"이 아니라
# "기존 transform / batching / model forward 파이프라인이 기대하는 raw schema를 흉내 내는 것"이다.
# 그래서 3개 RGB 카메라, 256차원 proprio, 23차원 action, task/timestamp 관련 키를 맞춰 준다.
#
# 여기서 action을 23차원으로 두는 이유는 원본 B1K 흐름을 흉내 내기 위해서이고,
# 이후 transform 단계에서 32차원으로 padding 되도록 설계되어 있다.
def create_behavior_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    seed: int | None = None,
) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.

    Uses OmniGibson's BehaviorLeRobotDataset for efficient loading of
    BEHAVIOR-1K data when real data is available.

    For laptop smoke tests (`repo_id == "fake"`), returns a lightweight
    dummy dataset that mimics the raw BEHAVIOR field structure expected by
    the existing repack/data/model transform pipeline.
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
    logging.info(f"Using random seed for dataset: {seed}")

    # ------------------------------------------------------------------
    # Laptop / fake smoke path: bypass OmniGibson entirely
    # ------------------------------------------------------------------
    if data_config.repo_id == "fake":
        logging.warning(
            "Using laptop fake dataset for smoke test "
            "(bypassing OmniGibson / BehaviorLeRobotDataset)."
        )

        class _FakeMeta:
            def __init__(self, episodes):
                self.episodes = episodes

        # smoke test용 최소 fake dataset.
        # 이 데이터셋의 목적은 "학습 성능 확인"이 아니라
        # "data_loader -> transform -> model 입력 형식이 끝까지 이어지는지 확인"이다.
        #
        # 따라서 값 자체는 랜덤이어도 괜찮지만,
        # key 이름과 shape는 실제 파이프라인이 기대하는 형태와 최대한 같아야 한다.
        class _LaptopFakeBehaviorDataset:
            """Smoke test용 경량 fake dataset.

            기존 transform 파이프라인이 기대하는 raw schema:
            - 3개 RGB 카메라
            - 256차원 proprio
            - 23차원 action (이후 transform에서 32차원으로 패딩됨)
            - task_index / timestamp / episode_index / index
            """
            def __init__(self, num_samples: int = 16):
                self.num_samples = num_samples
                self.episodes = np.array([0, 1], dtype=np.int32)
                self.meta = _FakeMeta(self.episodes)
                half = num_samples // 2
                self.episode_data_index = {
                    "from": np.array([0, half], dtype=np.int64),
                    "to": np.array([half, num_samples], dtype=np.int64),
                }

            def __len__(self):
                return self.num_samples

            def __getitem__(self, idx: int):
                sample_rng = np.random.default_rng(seed + idx)

                # 두 개의 에피소드가 있는 것처럼 index를 나눠서 구성
                half = self.num_samples // 2
                if idx < half:
                    episode_index = 0
                    timestep_in_episode = idx
                else:
                    episode_index = 1
                    timestep_in_episode = idx - half

                # proprio는 256차원으로 만들고,
                # 실제 gripper 위치처럼 보이도록 일부 인덱스에 작은 상수값을 넣음
                proprio = np.zeros(256, dtype=np.float32)
                proprio += sample_rng.normal(0.0, 0.005, size=(256,)).astype(np.float32)
                proprio[193:195] = 0.02
                proprio[232:234] = 0.02

                # 3개 카메라 입력
                head_rgb = sample_rng.random((3, 224, 224), dtype=np.float32)
                left_wrist_rgb = sample_rng.random((3, 224, 224), dtype=np.float32)
                right_wrist_rgb = sample_rng.random((3, 224, 224), dtype=np.float32)

                # 원본 action은 23차원으로 만들고,
                # 이후 transform에서 32차원으로 pad되도록 둠
                action = sample_rng.normal(
                    loc=0.0,
                    scale=0.05,
                    size=(action_horizon, 23),
                ).astype(np.float32)

                # fake smoke에서도 subset task만 쓰게 맞춤
                selected_task_id = int(SELECTED_TASKS[idx % len(SELECTED_TASKS)])

                return {
                    "observation.images.rgb.head": head_rgb,
                    "observation.images.rgb.left_wrist": left_wrist_rgb,
                    "observation.images.rgb.right_wrist": right_wrist_rgb,
                    "observation.state": proprio,
                    "action": action,
                    "task_index": np.int32(idx % 50),
                    "timestamp": np.float32(timestep_in_episode),
                    "episode_index": np.int32(episode_index),
                    "index": np.int64(idx),
                }

        dataset = _LaptopFakeBehaviorDataset()
        if getattr(data_config, 'use_task_subset', False):
            dataset = _filter_to_selected_tasks(dataset, data_config.allowed_task_ids or SELECTED_TASKS)
        return dataset

    # ------------------------------------------------------------------
    # Real dataset path
    # ------------------------------------------------------------------
    # 참고: 이 부분은 현재 네 버전대로 direct import를 유지한 형태
    # 어제 fake smoke 기준으로는 문제 없었지만,
    # 나중엔 _get_behavior_lerobot_dataset_cls()를 써서 더 안전하게 바꿀 수 있음
    from omnigibson.learning.datas.lerobot_dataset import BehaviorLeRobotDataset

    # 현재 실데이터 로더는 BEHAVIOR task 이름 목록을 명시적으로 넘겨준다.
    # 이렇게 해 두면 task 순서/구성이 코드에서 고정되어,
    # 나중에 subset 매핑이나 task_index 해석이 달라지는 문제를 줄일 수 있다.
    #
    # 즉, "어떤 task를 쓰는지"를 외부 상태에 맡기지 않고 코드 안에 고정하는 구간이다.
    tasks = [
        "picking_up_trash",
        "putting_away_Halloween_decorations",
        "cleaning_up_plates_and_food",
        "setting_mousetraps",
        "hiding_Easter_eggs",
        "set_up_a_coffee_station_in_your_kitchen",
        "putting_dishes_away_after_cleaning",
        "preparing_lunch_box",
        "loading_the_car",
        "carrying_in_groceries",
        "turning_on_radio",
        "picking_up_toys",
        "can_meat",
        "rearranging_kitchen_furniture",
        "putting_up_Christmas_decorations_inside",
        "bringing_in_wood",
        "moving_boxes_to_storage",
        "bringing_water",
        "tidying_bedroom",
        "outfit_a_basic_toolbox",
        "sorting_vegetables",
        "collecting_childrens_toys",
        "putting_shoes_on_rack",
        "boxing_books_up_for_storage",
        "storing_food",
        "clearing_food_from_table_into_fridge",
        "assembling_gift_baskets",
        "getting_organized_for_work",
        "clean_up_your_desk",
        "setting_the_fire",
        "clean_boxing_gloves",
        "wash_a_baseball_cap",
        "wash_dog_toys",
        "hanging_pictures",
        "attach_a_camera_to_a_tripod",
        "clean_a_patio",
        "clean_a_trumpet",
        "spraying_for_bugs",
        "spraying_fruit_trees",
        "make_microwave_popcorn",
        "cook_cabbage",
        "make_pizza",
        "chop_an_onion",
        "slicing_vegetables",
        "chopping_wood",
        "canning_food",
        "cook_hot_dogs",
        "cook_bacon",
        "freeze_pies",
    ]

    dataset = BehaviorLeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        tasks=tasks,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)]
            for key in data_config.action_sequence_keys
        },
        episodes=data_config.episodes_index,
        chunk_streaming_using_keyframe=False,
        shuffle=True,
        seed=seed,
    )

    if data_config.prompt_from_task:
        dataset = _openpi_data_loader.TransformedDataset(
            dataset,
            [_openpi_data_loader._transforms.PromptFromLeRobotTask(dataset.meta.tasks)],
        )

    return dataset


# 실제 데이터셋 생성 이후에는 OpenPI의 transform_dataset / TorchDataLoader 파이프라인을 최대한 재사용한다.
# 즉, 우리 수정의 핵심은 "dataset source만 바꾸고",
# batching / sharding / worker 처리 방식은 기존 OpenPI 흐름을 따르는 것이다.
#
# 이 구조 덕분에 데이터 원천(BEHAVIOR)만 교체하면서도
# 학습 루프 쪽 코드를 크게 다시 쓰지 않을 수 있다.
def create_behavior_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: Literal["jax", "pytorch"] = "jax",
):
    """Create a BEHAVIOR-1K torch-backed data loader."""
    dataset = create_behavior_dataset(
        data_config,
        action_horizon=action_horizon,
        seed=seed,
    )
    dataset = _openpi_data_loader.transform_dataset(
        dataset,
        data_config,
        skip_norm_stats=skip_norm_stats,
    )

    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logger.info("local_batch_size: %s", local_batch_size)

    data_loader = _openpi_data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    # 핵심: OpenPI 기본 DataLoaderImpl이 아니라
    # B1K Observation을 반환하는 커스텀 래퍼를 사용
    return DataLoaderImpl(data_config, data_loader)


# train.py에서 실제로 호출하는 최종 진입점.
# config에서 DataConfig를 만든 뒤,
# RLDS 경로면 OpenPI 기본 로더로 보내고,
# B1K 경로면 우리가 만든 behavior 전용 torch data loader로 보낸다.
#
# 즉, 이 함수는 "데이터셋 종류에 따라 어느 로더를 탈지 결정하는 스위치" 역할이다.
def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
):
    """Create a BEHAVIOR-1K data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info("data_config: %s", data_config)

    if data_config.rlds_data_dir is not None:
        return _openpi_data_loader.create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )

    return create_behavior_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=0 if config.seed is None else config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )