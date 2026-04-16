"""BEHAVIOR-1K 전용 데이터 변환 모음.

OpenPI 기본 변환에 더해,
- 전역 task id를 12개 subset용 로컬 task id로 바꾸는 변환
- 필요할 때만 timestamp로 stage를 계산하는 변환
- FAST 보조 토큰을 만드는 변환
을 추가했다.

비전공자용 큰 그림:
    transform은 "데이터 모양을 바꾸는 작은 함수"다.
    학습 데이터는 사람이 보기 편한 이름과 모양으로 저장되어 있지만,
    모델은 정확히 정해진 key 이름, shape, dtype을 기대한다.

    예:
        원본: task_index = 46
        변환 후: tokenized_prompt = [11]

    여기서 tokenized_prompt라는 이름은 OpenPI 코드와 호환하기 위해 쓰지만,
    이 프로젝트에서는 자연어 문장 토큰이 아니라 task id 숫자를 담는다.
"""

import dataclasses
import logging
import numpy as np

# OpenPI에서 기본 변환 도구들을 가져온다.
from openpi.transforms import (
    DataTransformFn,
    DataDict,
    Group,
    CompositeTransform,
    compose,
    RepackTransform,
    Normalize,
    Unnormalize,
    ResizeImages,
    SubsampleActions,
    DeltaActions,
    AbsoluteActions,
    PadStatesAndActions,
    PromptFromLeRobotTask,  # PI_BEHAVIOR에서는 거의 쓰지 않지만 호환성을 위해 남겨둔다.
    flatten_dict,
    unflatten_dict,
    transform_dict,
    apply_tree,
    pad_to_dim,
    make_bool_mask,
)

from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class TaskIndexToTaskId(DataTransformFn):
    """task_index 또는 task_id를 모델용 tokenized_prompt로 바꾼다.

    이 버전의 기본 경로는 stage conditioning을 사용하지 않는다.
    따라서 tokenized_prompt는 보통 [local_task_id] 한 개 토큰만 만든다.

    task_mapping을 주면
    - 학습 시: 전역 task id(예: 18)를 로컬 task id(예: 7)로 바꾼다.
    - 추론 시: 환경이 주는 전역 task id도 같은 방식으로 바꿀 수 있다.
    """

    # task_mapping은 global/original task id -> local subset task id 변환표다.
    # 예: {0: 0, 1: 1, 5: 2, ..., 46: 11}
    # None이면 이미 local id가 들어왔다고 보고 그대로 사용한다.
    task_mapping: dict[int, int] | None = None
    
    def __call__(self, data: DataDict) -> DataDict:
        # 추론에서는 task_index 대신 task_id라는 이름으로 들어올 수 있다.
        # 이 경로는 "이미 모델에 넣을 id가 준비되어 있다"는 의미로 보고 그대로 쓴다.
        # 평가 wrapper는 별도로 local id를 넣어 주므로 보통 이 transform은 추론에서 skip된다.
        if "task_id" in data:
            # Direct task_id provided (inference mode)
            task_id = int(data["task_id"])
        elif "task_index" in data:
            # task_index provided (training mode)
            task_index = int(data["task_index"])
            
            # 학습 데이터는 보통 global task_index를 갖고 있다.
            # 모델 embedding은 local id를 요구하므로 mapping이 있으면 반드시 변환한다.
            if self.task_mapping is not None:
                if task_index not in self.task_mapping:
                    raise ValueError(f"task_index {task_index} not found in mapping")
                task_id = self.task_mapping[task_index]
            else:
                task_id = task_index
        else:
            # During inference, if neither is provided, this transform should be skipped
            # The tokenized_prompt should already be set up correctly
            if "tokenized_prompt" in data:
                return data  # Already has tokenized_prompt, skip this transform
            raise ValueError("Either task_index, task_id, or tokenized_prompt is required for PI_BEHAVIOR model")

        # 기본 경로는 stage 토큰을 따로 붙이지 않는다.
        # 최종적으로 local task id 하나만 모델에 전달한다.
        #
        # shape이 [1]인 이유:
        #   모델 쪽에서 batch를 만들면 [B, 1]이 된다.
        #   두 번째 값(stage id)을 붙이면 [B, 2]가 되고 stage conditioning 경로가 열린다.
        prompt_tokens = np.array([task_id], dtype=np.int32)
        prompt_mask = np.array([True], dtype=bool)

        return {
            **data,
            "tokenized_prompt": prompt_tokens,
            "tokenized_prompt_mask": prompt_mask
        }


@dataclasses.dataclass(frozen=True)
class ComputeSubtaskStateFromMeta(DataTransformFn):
    """timestamp와 episode 길이를 이용해 현재 stage 번호를 계산한다.

    쉬운 설명:
        하나의 episode는 로봇이 태스크를 시작해서 끝낼 때까지의 전체 영상/행동이다.
        stage는 그 episode를 여러 구간으로 나눈 번호다.
        예를 들어 10초짜리 episode를 5개 stage로 나누면,
        앞 2초는 stage 0, 다음 2초는 stage 1처럼 볼 수 있다.

    현재 12-task baseline에서는 stage conditioning을 기본으로 쓰지 않는다.
    그래도 나중에 stage loss나 stage-conditioned 모델을 다시 켤 수 있게 코드를 남겨 둔다.
    
    Divides episode into task-specific number of stages based on episode length.
    Requires dataset reference to access episode metadata.
    
    Args:
        dataset: Dataset instance with meta.episodes containing episode_length
    
    Assumes:
    - data["episode_index"] exists
    - data["timestamp"] exists (frame index within episode)
    - data["task_index"] exists (for task-specific stage count)
    - dataset.meta.episodes contains episode_length for each episode
    
    Creates:
    - data["subtask_state"]: Stage index (0 to num_stages-1)
    """
    
    dataset: object | None = None  # Will be set by data loader
    
    def __call__(self, data: DataDict) -> DataDict:
        if self.dataset is None:
            # During inference or when dataset is not available, use default
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        if "episode_index" not in data or "timestamp" not in data or "task_index" not in data:
            # Missing required fields, default to stage 0
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        episode_index = int(data["episode_index"])
        timestamp = float(data["timestamp"])
        task_index = int(data["task_index"])
        
        # Validate task_index
        if not (0 <= task_index < 50):
            logging.warning(f"Invalid task_index {task_index}, using stage 0")
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        # Get number of stages for this task
        num_stages = TASK_NUM_STAGES[task_index]
        
        # Get episode length from dataset metadata
        if not hasattr(self.dataset, 'meta') or not hasattr(self.dataset.meta, 'episodes'):
            # No metadata available
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
            
        meta_episodes = self.dataset.meta.episodes
        
        if episode_index not in meta_episodes:
            logging.warning(f"Episode {episode_index} not found in metadata, using stage 0")
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
            
        episode_info = meta_episodes[episode_index]
        
        # Try 'length' first (standard key), then 'episode_length' (alternative)
        episode_length = episode_info.get('length', episode_info.get('episode_length', None))
        
        if episode_length is None or episode_length <= 0:
            logging.warning(f"Invalid episode_length for episode {episode_index}, using stage 0")
            data["subtask_state"] = np.array(0, dtype=np.int32)
            return data
        
        episode_length = float(episode_length)
        
        # CRITICAL: Convert timestamp (in seconds) to frames (30 FPS)
        # Dataset provides timestamp in seconds, episode_length is in frames
        current_step = timestamp * 30.0
        
        # Divide episode into num_stages equal parts
        frames_per_stage = episode_length / num_stages
        
        # Compute current stage (0-indexed)
        subtask_state = int(current_step / frames_per_stage)
        
        # Clamp to valid range [0, num_stages-1]
        subtask_state = max(0, min(subtask_state, num_stages - 1))
        
        data["subtask_state"] = np.array(subtask_state, dtype=np.int32)
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeFASTActions(DataTransformFn):
    """Tokenize actions to FAST discrete tokens for PI_BEHAVIOR auxiliary training.
    
    Converts continuous actions to discrete tokens using a trained FAST tokenizer.
    This is used for auxiliary training to improve action quality.
    
    Note: Uses GLOBAL normalization even if per-timestamp normalization is enabled
    elsewhere, to preserve temporal smoothness for better DCT compression.
    
    Args:
        tokenizer_path: Path to trained FAST tokenizer directory
        encoded_dim_ranges: List of (start, end) tuples for dimensions to encode
        max_fast_tokens: Maximum number of tokens (truncate/pad)
        norm_stats: Normalization stats for denorm/renorm if per-timestamp is used
        use_per_timestamp: Whether per-timestamp normalization is being used
    
    Assumes:
    - data["actions"] exists and is normalized
    - data["state"] exists (for delta computation)
    
    Creates:
    - data["fast_tokens"]: Discrete token indices [max_fast_tokens]
    - data["fast_token_mask"]: Boolean mask for valid tokens [max_fast_tokens]
    """
    
    # Path to FAST tokenizer
    tokenizer_path: str
    # Action dimensions to encode: [(start, end), ...]
    encoded_dim_ranges: list[tuple[int, int]]
    # Max tokens to predict
    max_fast_tokens: int = 180
    # Normalization stats (for denorm/renorm if per-timestamp is used)
    norm_stats: dict[str, NormStats] | None = None
    # Whether per-timestamp normalization is being used
    use_per_timestamp: bool = False
    
    def _get_tokenizer(self):
        """Lazy load tokenizer (called in each worker process)."""
        # Check if already loaded in this process
        if not hasattr(self, '_tokenizer'):
            import importlib.util
            from pathlib import Path
            
            tokenizer_dir = Path(self.tokenizer_path)
            processor_file = tokenizer_dir / "processing_action_tokenizer.py"
            
            if not processor_file.exists():
                raise FileNotFoundError(f"Tokenizer processing file not found at {processor_file}")
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location("processing_action_tokenizer", processor_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get the UniversalActionProcessor class
            UniversalActionProcessor = module.UniversalActionProcessor
            
            # Load the tokenizer using from_pretrained
            tokenizer = UniversalActionProcessor.from_pretrained(str(tokenizer_dir))
            
            # Store in this object (won't be pickled due to lazy loading pattern)
            object.__setattr__(self, '_tokenizer', tokenizer)
        
        return self._tokenizer
    
    def __call__(self, data: DataDict) -> DataDict:
        # Only tokenize if actions are present (training time)
        if "actions" not in data:
            return data
        
        actions = data["actions"]  # [H, D] - may be per-timestamp normalized
        
        # Extract encoded dimensions
        encoded_chunks = []
        for start, end in self.encoded_dim_ranges:
            encoded_chunks.append(actions[:, start:end])
        encoded_actions = np.concatenate(encoded_chunks, axis=-1)  # [H, D_encoded]
        
        # Convert to global QUANTILE normalization for FAST (always, regardless of per_timestamp)
        # This clips outliers and provides consistent [-1, 1] range for DCT compression
        if self.norm_stats is not None:
            action_stats = self.norm_stats['actions']
            
            # Extract stats for encoded dimensions only
            # Build dimension indices from ranges
            encoded_dims = []
            for start, end in self.encoded_dim_ranges:
                encoded_dims.extend(range(start, end))
            encoded_dims = np.array(encoded_dims)
            
            # Denormalize to get back to raw delta actions
            if self.use_per_timestamp:
                # Denormalize from per-timestamp MEAN/STD normalization
                per_ts_mean = action_stats.per_timestamp_mean[:, encoded_dims]  # [H, D_encoded]
                per_ts_std = action_stats.per_timestamp_std[:, encoded_dims]    # [H, D_encoded]
                actions_delta = encoded_actions * per_ts_std + per_ts_mean
            else:
                # Denormalize from global MEAN/STD normalization
                global_mean = action_stats.mean[encoded_dims]  # [D_encoded]
                global_std = action_stats.std[encoded_dims]    # [D_encoded]
                actions_delta = encoded_actions * global_std + global_mean
            
            # Renormalize using GLOBAL QUANTILE normalization (for FAST)
            # This clips to [q01, q99] and maps to [-1, 1]
            q01 = action_stats.q01[encoded_dims]  # [D_encoded]
            q99 = action_stats.q99[encoded_dims]  # [D_encoded]
            actions_delta = np.clip(actions_delta, q01, q99)
            encoded_actions = 2.0 * (actions_delta - q01) / np.maximum(q99 - q01, 1e-6) - 1.0
        
        # Tokenize
        tokenizer = self._get_tokenizer()
        tokens = tokenizer(encoded_actions[None])[0]
        
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        
        # Truncate or pad to max_fast_tokens
        if len(tokens) > self.max_fast_tokens:
            tokens = tokens[:self.max_fast_tokens]
            mask = np.ones(self.max_fast_tokens, dtype=bool)
        else:
            mask = np.concatenate([
                np.ones(len(tokens), dtype=bool),
                np.zeros(self.max_fast_tokens - len(tokens), dtype=bool)
            ])
            tokens = np.pad(tokens, (0, self.max_fast_tokens - len(tokens)), constant_values=0)
        
        data["fast_tokens"] = tokens.astype(np.int32)
        data["fast_token_mask"] = mask
        
        return data

