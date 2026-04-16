"""B1K Policy Transforms

Transforms BEHAVIOR-1K observations to model format.

Reference: https://github.com/wensi-ai/openpi/blob/behavior/src/openpi/policies/b1k_policy.py

비전공자용 큰 그림:
    이 파일은 "환경/데이터셋에서 온 observation dict"를
    "PiBehavior 모델이 먹는 observation dict"로 바꾸는 곳이다.

    로봇 상태(proprioception)는 256개 정도의 긴 센서값으로 들어온다.
    하지만 모델이 action 예측에 쓰는 핵심 값은 그중 일부다.
    extract_state_from_proprio()가 필요한 인덱스만 뽑아 23차원 state로 줄인다.

    이미지도 저장 방식이 조금씩 다르다.
    어떤 데이터는 [H, W, C]이고 어떤 데이터는 [C, H, W]다.
    _parse_image()가 모델 transform이 기대하는 [H, W, C] uint8 이미지로 맞춘다.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
from b1k.shared.proprioception_indices import PROPRIOCEPTION_INDICES


def make_b1k_example() -> dict:
    """테스트용 가짜 입력 예시를 만든다.

    실제 학습에는 쓰지 않고, policy 인터페이스가 어떤 key를 기대하는지
    빠르게 확인할 때 사용할 수 있다.
    """
    return {
        "observation/egocentric_camera": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(23),
        "prompt": "do something",
    }

def extract_state_from_proprio(proprio_data):
    """긴 proprioception 벡터에서 모델이 사용할 23차원 state만 뽑는다.

    proprioception은 로봇 내부 센서값 전체를 뜻한다.
    여기에는 base 속도, 몸통 위치, 양팔 관절, 그리퍼 폭 등 많은 값이 들어 있다.

    모델은 모든 센서를 그대로 보지 않고 아래 순서의 23차원만 사용한다.
        base velocity 3
        trunk position 4
        left arm joint 7
        left gripper width 1
        right arm joint 7
        right gripper width 1

    두 손가락 그리퍼는 양쪽 finger 값을 합쳐 하나의 "그리퍼가 얼마나 열렸는지"로 본다.
    """
    # PROPRIOCEPTION_INDICES는 256차원 벡터에서 어느 구간이 어떤 센서인지 알려 주는 표다.
    # 예를 들어 base_qvel이 [0, 1, 2]라면 그 세 값을 base 속도로 사용한다.
    base_qvel = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_left_qpos"]]  #  7
    arm_right_qpos = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["arm_right_qpos"]]  #  7
    
    # 그리퍼는 손가락 두 개의 위치값으로 저장된다.
    # 모델 action space는 그리퍼를 하나의 값으로 다루므로 두 finger 값을 합친다.
    left_gripper_raw = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_left_qpos"]].sum(axis=-1, keepdims=True)
    right_gripper_raw = proprio_data[..., PROPRIOCEPTION_INDICES["R1Pro"]["gripper_right_qpos"]].sum(axis=-1, keepdims=True)
    
    # 실제 gripper 폭은 대략 [0, 0.1] meter 범위다.
    # 모델은 action/state 값을 [-1, 1] 근처로 보는 쪽이 안정적이라 이 범위로 선형 변환한다.
    # 공식: normalized = 2 * (raw / max_width) - 1
    MAX_GRIPPER_WIDTH = 0.1  # From statistics q99 values
    left_gripper_width = 2.0 * (left_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0
    right_gripper_width = 2.0 * (right_gripper_raw / MAX_GRIPPER_WIDTH) - 1.0

    # 이 순서는 학습 때 action/state 통계와 맞아야 한다.
    # 순서가 바뀌면 숫자 shape는 같아도 의미가 바뀌어서 모델 성능이 크게 무너질 수 있다.
    return np.concatenate([
        base_qvel,
        trunk_qpos,
        arm_left_qpos,
        left_gripper_width,    # Now normalized [-1, 1]
        arm_right_qpos,
        right_gripper_width,   # Now normalized [-1, 1]
    ], axis=-1)


def _parse_image(image) -> np.ndarray:
    """이미지를 모델 전처리가 기대하는 uint8 HWC 형식으로 맞춘다.

    HWC: height, width, channel 순서다. 일반 이미지 라이브러리가 자주 쓰는 형식이다.
    CHW: channel, height, width 순서다. PyTorch 계열 데이터셋이 자주 쓰는 형식이다.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class B1kInputs(transforms.DataTransformFn):
    # Determines which model will be used (not actually used in B1K, kept for compatibility)
    model_type: _model.ModelType | str = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:

        proprio_data = data["observation/state"]
        # 긴 로봇 센서 벡터를 모델이 보는 23차원 state로 압축한다.
        state = extract_state_from_proprio(proprio_data)
        if "actions" in data:
            action =  data["actions"]

        # LeRobot 데이터셋은 이미지를 float32 CHW로 줄 때가 있고,
        # 평가 wrapper는 uint8 HWC로 줄 때가 있다.
        # 두 경우 모두 모델 전처리가 받을 수 있게 여기서 통일한다.
        base_image = _parse_image(data["observation/egocentric_camera"])
        wrist_image_left = _parse_image(data["observation/wrist_image_left"])
        wrist_image_right = _parse_image(data["observation/wrist_image_right"])

        # B1K 모델은 세 카메라를 모두 쓴다.
        # 이름은 Observation 클래스와 pi_behavior 모델이 기대하는 key와 반드시 같아야 한다.
        names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        images = (base_image, wrist_image_left, wrist_image_right)
        image_masks = (np.True_, np.True_, np.True_)

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = action

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
            
        # task_index는 뒤 transform에서 global -> local task id로 바꾸는 데 필요하다.
        if "task_index" in data:
            inputs["task_index"] = data["task_index"]
            
        # tokenized_prompt가 이미 있으면 그대로 보존한다.
        # 추론 wrapper는 여기에 local task id를 직접 넣어 준다.
        if "tokenized_prompt" in data:
            inputs["tokenized_prompt"] = data["tokenized_prompt"]
        if "tokenized_prompt_mask" in data:
            inputs["tokenized_prompt_mask"] = data["tokenized_prompt_mask"]
            
        # Preserve subtask_state for PI_BEHAVIOR model
        if "subtask_state" in data:
            inputs["subtask_state"] = data["subtask_state"]
            
        # Preserve timestamp and episode_index for subtask state computation
        if "timestamp" in data:
            inputs["timestamp"] = data["timestamp"]
        if "episode_index" in data:
            inputs["episode_index"] = data["episode_index"]
            
        # initial_actions는 rolling inpainting용 힌트다.
        # 모델 action_dim은 32지만 환경 action은 23차원이라 부족한 뒤쪽은 0으로 채운다.
        if "initial_actions" in data:
            initial_actions = data["initial_actions"]
            # Pad initial_actions from 23 dimensions to 32 dimensions (model's action_dim)
            if initial_actions.shape[-1] < 32:
                padding_dim = 32 - initial_actions.shape[-1]
                padding = np.zeros(initial_actions.shape[:-1] + (padding_dim,))
                initial_actions = np.concatenate([initial_actions, padding], axis=-1)
            inputs["initial_actions"] = initial_actions

        return inputs


@dataclasses.dataclass(frozen=True)
class B1kOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Return actions (truncated to 23 dims) and preserve subtask predictions
        result = {"actions": np.asarray(data["actions"][:, :23])}
        
        # Preserve subtask prediction fields for PI_BEHAVIOR models
        if "subtask_logits" in data:
            result["subtask_logits"] = data["subtask_logits"]
        if "predicted_stage" in data:
            result["predicted_stage"] = data["predicted_stage"]
            
        return result
