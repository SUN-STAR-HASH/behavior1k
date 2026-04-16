"""태스크별 체크포인트를 바꿔 가며 정책을 쓰는 도구.

원래 코드는 50개 태스크 전체가 모두 매핑되어 있어야 한다고 가정했다.
이번 버전은 실제로 12개 태스크만 사용하는 설정에 맞게 검사 기준을 바꾼다.

비전공자용 큰 그림:
    체크포인트는 "학습된 모델 저장 파일"이라고 보면 된다.
    어떤 체크포인트는 특정 태스크들을 더 잘하도록 fine-tuning되어 있을 수 있다.

    예:
        checkpoint_1 -> microwave, popcorn, ...
        checkpoint_2 -> radio, cabinet, ...

    평가 중 task_id가 바뀌면 이 클래스가 mapping JSON을 보고
    "이번 태스크에는 어떤 체크포인트를 써야 하지?"를 결정한다.

    주의:
        이 클래스는 원본 BEHAVIOR-1K global task id를 기준으로 동작한다.
        모델 내부 local task id와 섞으면 안 된다.
"""

import json
import logging
import gc
import pathlib
from typing import Any, TYPE_CHECKING

from openpi.policies import policy as _policy

# Import B1K-specific policy_config
from b1k.policies import policy_config as _policy_config
from b1k.configs.task_subset import SELECTED_TASKS, SELECTED_TASKS_SET

if TYPE_CHECKING:
    from b1k.training import config as _config


class CheckpointSwitcher:
    """태스크에 따라 다른 체크포인트를 불러오는 클래스.

    메모리에는 한 번에 하나의 체크포인트만 올려 둔다.
    이번 설정에서는 전체 50개가 아니라 선택한 12개 태스크만 모두 매핑되어 있으면 된다.

    왜 하나만 올려 두는가?
        PiBehavior 체크포인트는 크고 GPU 메모리를 많이 쓴다.
        여러 개를 동시에 올리면 24GB GPU에서도 바로 메모리가 부족해질 수 있다.
        그래서 필요할 때 기존 policy를 지우고 새 checkpoint policy를 만든다.
    """
    
    def __init__(
        self,
        config_path: str,
        training_config: "TrainConfig",
        sample_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize the checkpoint switcher.
        
        Args:
            config_path: Path to task_checkpoint_mapping.json (REQUIRED)
            training_config: Training config for loading policies
            sample_kwargs: kwargs for policy sampling (e.g., num_steps)
        """
        if not config_path:
            raise ValueError("config_path is required for checkpoint switching")
        
        self.config_path = config_path
        self.training_config = training_config
        self.sample_kwargs = sample_kwargs or {}
        
        # task_to_checkpoint:
        #   global task id -> checkpoint 이름
        #   예: 40 -> "checkpoint_4"
        self.task_to_checkpoint: dict[int, str] = {}

        # checkpoint_paths:
        #   checkpoint 이름 -> 실제 폴더 경로
        #   예: "checkpoint_4" -> "/home/user/models/checkpoint_4"
        self.checkpoint_paths: dict[str, str] = {}
        
        # Currently loaded checkpoint
        self.current_policy: _policy.Policy | None = None
        self.current_checkpoint_name: str | None = None
        
        # Load and validate mapping
        self._load_mapping()
        self._validate_all_tasks_covered()
        
        logging.info(f"Checkpoint switcher initialized with {len(self.checkpoint_paths)} checkpoints")
        logging.info("현재 사용 중인 12개 태스크가 모두 체크포인트에 연결되어 있다.")
    
    def _load_mapping(self):
        """mapping JSON을 읽어서 빠르게 조회할 수 있는 dict로 바꾼다.

        JSON은 사람이 편하게 편집하기 좋은 형태다.
        하지만 실행 중에는 매번 JSON을 뒤지는 것보다 dict 조회가 빠르고 단순하다.
        그래서 시작할 때 한 번만 읽고 `task_to_checkpoint`를 만들어 둔다.
        """
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            logging.error(f"Checkpoint mapping file not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in checkpoint mapping file: {e}")
            raise
        
        if "checkpoints" not in config:
            raise ValueError("Checkpoint mapping file must contain 'checkpoints' key")
        
        # JSON 예시는 대략 아래 모양이다.
        # {
        #   "checkpoints": {
        #     "checkpoint_1": {
        #       "path": "~/models/checkpoint_1",
        #       "tasks": [2, 3, 5]
        #     }
        #   }
        # }
        seen_tasks = set()
        for checkpoint_name, checkpoint_info in config["checkpoints"].items():
            if "path" not in checkpoint_info:
                raise ValueError(f"Checkpoint '{checkpoint_name}' missing 'path' field")
            if "tasks" not in checkpoint_info:
                raise ValueError(f"Checkpoint '{checkpoint_name}' missing 'tasks' field")
            
            checkpoint_path = checkpoint_info["path"]
            # JSON에 "~/models/..."처럼 적으면 shell이 아니므로 자동 확장되지 않는다.
            # 여기서 홈 디렉터리 절대 경로로 바꿔 두면 뒤쪽 로더가 FileNotFoundError를 덜 낸다.
            if "://" not in checkpoint_path:
                checkpoint_path = str(pathlib.Path(checkpoint_path).expanduser())
            tasks = checkpoint_info["tasks"]
            
            # 같은 task_id가 두 checkpoint에 동시에 들어 있으면 어느 쪽을 써야 할지 모호하다.
            # 그런 설정은 조용히 넘어가지 않고 시작할 때 바로 실패시킨다.
            for task_id in tasks:
                if task_id in seen_tasks:
                    raise ValueError(f"Task {task_id} is assigned to multiple checkpoints")
                seen_tasks.add(task_id)
            
            self.checkpoint_paths[checkpoint_name] = checkpoint_path
            for task_id in tasks:
                self.task_to_checkpoint[task_id] = checkpoint_name
            
            logging.info(f"Checkpoint '{checkpoint_name}' -> {len(tasks)} tasks: {sorted(tasks)}")
    
    def _validate_all_tasks_covered(self):
        """현재 사용할 12개 태스크가 빠짐없이 매핑되었는지 검사한다.

        평가 중 특정 태스크가 왔는데 mapping이 없으면 그 순간 서버가 죽는다.
        오래 걸리는 평가를 중간에 망치지 않기 위해, 시작 시점에 미리 검사한다.
        """
        all_tasks = set(SELECTED_TASKS)
        mapped_tasks = set(self.task_to_checkpoint.keys())
        missing_tasks = sorted(all_tasks - mapped_tasks)
        
        if missing_tasks:
            raise ValueError(
                f"현재 사용하는 12개 태스크는 모두 체크포인트에 연결되어 있어야 한다. "
                f"Missing tasks: {missing_tasks}"
            )
    
    def get_checkpoint_for_task(self, task_id: int) -> str:
        """주어진 global task id를 처리할 checkpoint 이름을 반환한다.
        
        Args:
            task_id: Task ID (0-49)
            
        Returns:
            Checkpoint name
            
        Raises:
            ValueError: If task_id not mapped (should never happen after validation)
        """
        if task_id not in SELECTED_TASKS_SET:
            raise ValueError(f"Task {task_id} 는 현재 사용 설정의 12개 subset에 없다.")
        if task_id not in self.task_to_checkpoint:
            raise ValueError(f"Task {task_id} not mapped to any checkpoint")
        
        return self.task_to_checkpoint[task_id]
    
    def get_policy_for_task(self, task_id: int) -> _policy.Policy:
        """Get policy for task_id, loading new checkpoint if needed.
        
        Args:
            task_id: Task ID (0-49)
            
        Returns:
            Policy for the requested task
        """
        target_checkpoint = self.get_checkpoint_for_task(task_id)
        
        # 이미 같은 checkpoint가 올라와 있으면 다시 로드하지 않는다.
        # 로딩은 느리고 GPU 메모리도 흔들리므로 가능한 재사용한다.
        if self.current_checkpoint_name == target_checkpoint and self.current_policy is not None:
            logging.debug(f"Task {task_id} using already-loaded checkpoint '{target_checkpoint}'")
            return self.current_policy
        
        # 다른 checkpoint가 필요하다면 현재 policy를 내리고 새 policy를 만든다.
        logging.info(f"Task {task_id} requires checkpoint '{target_checkpoint}'")
        
        # Unload current policy and free JAX/GPU memory
        if self.current_policy is not None:
            logging.info(f"Unloading checkpoint '{self.current_checkpoint_name}'")
            del self.current_policy
            self.current_policy = None
            
            # Python 객체 참조를 지운 뒤 gc.collect()로 메모리 회수를 요청한다.
            # JAX/GPU 메모리는 즉시 비워지지 않을 수 있어 아래 clear_caches도 같이 호출한다.
            gc.collect()
            
            # Clear JAX compilation cache and device memory
            try:
                import jax
                jax.clear_caches()
                logging.info("Cleared JAX caches and device memory")
            except Exception as e:
                logging.warning(f"Could not clear JAX caches: {e}")
        
        # Load new checkpoint
        checkpoint_path = self.checkpoint_paths[target_checkpoint]
        logging.info(f"Loading checkpoint from: {checkpoint_path}")
        
        try:
            self.current_policy = _policy_config.create_trained_policy(
                self.training_config,
                checkpoint_path,
                sample_kwargs=self.sample_kwargs
            )
            self.current_checkpoint_name = target_checkpoint
            logging.info(f"Successfully loaded checkpoint '{target_checkpoint}'")
        except Exception as e:
            logging.error(f"Failed to load checkpoint '{target_checkpoint}': {e}")
            raise
        
        return self.current_policy

