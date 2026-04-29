"""평가/서빙 시 정책 출력에 후처리를 적용하는 wrapper.

이번 설정은 pi0 + task embedding + flow matching only 경로에 맞춘다.
기본적으로 stage 추적, 평가용 보정 규칙, 다중 체크포인트는 사용하지 않는다.

비전공자용 큰 그림:
    이 파일은 "로봇 환경에서 들어온 관측값"과 "모델이 원하는 입력값" 사이의
    통역기 역할을 한다.

    환경이 주는 값:
        - 카메라 이미지 3장
        - 로봇 관절/그리퍼 상태
        - 원본 BEHAVIOR-1K 기준 task_id

    모델이 원하는 값:
        - 224x224로 맞춘 이미지
        - 23차원으로 정리된 로봇 상태
        - 12개 subset 기준 local task_id

    특히 task_id가 중요하다.
    체크포인트 선택은 원본 global task_id 기준으로 해야 하지만,
    모델의 task_embeddings 표는 12칸뿐이라 모델 입력은 local task_id여야 한다.
    그래서 이 wrapper는 self.task_id(global)와 self.local_task_id(local)를
    일부러 따로 들고 간다.
"""

import os
import logging
import numpy as np
import torch
import dataclasses
from collections import deque

from openpi_client.base_policy import BasePolicy
from openpi_client.image_tools import resize_with_pad
from b1k.policies.b1k_policy import extract_state_from_proprio
from b1k.configs.task_subset import map_global_to_local
from b1k.shared.proprioception_indices import PROPRIOCEPTION_INDICES

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224


@dataclasses.dataclass
class B1KWrapperConfig:
    # [수정일: 2026-04-29]
    # [수정 이유]
    # 현재 70k checkpoint에서 26→20 action compression과 inpainting을 켜면
    # base/arm action이 과하게 들어가 로봇이 넘어지는 현상이 발생한다.
    #
    # 따라서 먼저 안정성 확인을 위해 가장 보수적인 eval 설정으로 되돌린다.
    # - action compression 끔: actions_to_execute == execute_in_n_steps
    # - inpainting 끔: actions_to_keep = 0
    # - eval tricks 끔: action 보정 rule 영향 제거
    #
    # 이 설정에서 로봇이 안 넘어지면,
    # 원인은 policy 자체보다는 compression/inpainting 설정일 가능성이 크다.
    actions_to_execute: int = 12
    actions_to_keep: int = 0
    execute_in_n_steps: int = 12

    history_len: int = 1
    votes_to_promote: int = 1
    time_threshold_inpaint: float = 0.3
    num_steps: int = 8
    apply_eval_tricks: bool = False

class B1KPolicyWrapper():
    """PI_BEHAVIOR 모델을 BEHAVIOR 평가 서버 형식에 맞춰 감싸는 클래스.

    policy 자체는 "이미 전처리된 입력을 받아 action을 예측하는 객체"다.
    그런데 실제 평가 서버는 원본 카메라 이름, 원본 proprioception 이름,
    원본 task_id를 보낸다. 이 클래스가 그 차이를 맞춰 준다.
    """
    
    def __init__(
        self, 
        policy: BasePolicy,
        text_prompt: str = "PI_BEHAVIOR model (task-conditioned)",  # Not used, kept for compatibility
        action_horizon: int = 30,
        task_id: int | None = None,
        config: B1KWrapperConfig = None,
        checkpoint_switcher = None,
    ) -> None:
        self.base_policy = policy
        self.policy = policy
        self.checkpoint_switcher = checkpoint_switcher
        self.text_prompt = text_prompt
        self.action_horizon = action_horizon
        self.config = config if config is not None else B1KWrapperConfig()
        
        # Validate configuration
        if self.config.actions_to_execute + self.config.actions_to_keep > self.action_horizon:
            raise ValueError(
                f"actions_to_execute + actions_to_keep exceeds action_horizon"
            )
        
        # PI_BEHAVIOR specific (always True for B1K).
        #
        # self.task_id:
        #   BEHAVIOR-1K 원본 번호다. 예: 5, 40, 46
        #   체크포인트 mapping JSON과 correction rule은 이 번호를 기준으로 작성되어 있다.
        #
        # self.local_task_id:
        #   SELECTED_TASKS 안에서 다시 매긴 번호다. 예: global 5 -> local 2
        #   모델의 task_embeddings는 12행뿐이므로 반드시 이 번호를 넣어야 한다.
        self.task_id = task_id
        self.local_task_id = map_global_to_local(task_id) if task_id is not None else None
        self.current_stage = 0
        self.prediction_history = deque([], maxlen=self.config.history_len)
        
        # Control loop variables
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
    
    def reset(self):
        """Reset policy state."""
        self.policy.reset()
        self.last_actions = None
        self.action_index = 0
        self.step_count = 0
        self.prediction_count = 0
        self.next_initial_actions = None
        self.current_stage = 0
        self.prediction_history.clear()
        logger.info(f"Policy reset - Task ID: {self.task_id}, Action horizon: {self.action_horizon}")
    
    def _handle_task_change(self, new_task_id):
        """환경이 다른 task를 시작했을 때 wrapper 내부 상태를 초기화한다.

        new_task_id는 반드시 원본 global task id다.
        여기에서 local id를 따로 계산한 뒤, 체크포인트 스위처에는 global id를 넘긴다.
        이렇게 해야 JSON mapping과 모델 embedding lookup이 서로 섞이지 않는다.
        """
        if self.task_id != new_task_id:
            old_task_id = self.task_id
            self.task_id = new_task_id
            self.local_task_id = map_global_to_local(new_task_id)
            
            logger.info(f"🔄 Task change detected: {old_task_id} → {new_task_id}")
            
            if self.checkpoint_switcher:
                new_policy = self.checkpoint_switcher.get_policy_for_task(new_task_id)
                if new_policy is not self.policy:
                    logger.info(f"📦 Switching checkpoint: task {old_task_id} → {new_task_id}")
                    self.base_policy = new_policy
                    self.policy = new_policy
                    self.policy.reset()
            
            self.current_stage = 0
            self.prediction_history.clear()
            self.last_actions = None
            self.action_index = 0
            self.next_initial_actions = None

    def process_obs(self, obs: dict) -> dict:
        """평가 환경의 원본 observation을 모델 입력 이름으로 바꾼다.

        BEHAVIOR 환경은 카메라 이름이 길고 시뮬레이터 내부 경로처럼 생겼다.
        모델 쪽 transform은 더 짧은 공통 이름을 기대한다.
        여기서는 이미지 크기와 key 이름만 맞추고, 정규화는 뒤 transform에서 처리한다.
        """
        prop_state = obs["robot_r1::proprio"]
        
        head_original = obs["robot_r1::robot_r1:zed_link:Camera:0::rgb"][..., :3]
        left_original = obs["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"][..., :3]
        right_original = obs["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"][..., :3]
        
        # 모델 backbone은 224x224 이미지를 기준으로 학습되었다.
        # resize_with_pad는 이미지를 찌그러뜨리지 않도록 비율을 유지하고 빈 공간을 padding한다.
        head_resized = resize_with_pad(head_original, RESIZE_SIZE, RESIZE_SIZE)
        left_resized = resize_with_pad(left_original, RESIZE_SIZE, RESIZE_SIZE)
        right_resized = resize_with_pad(right_original, RESIZE_SIZE, RESIZE_SIZE)
        
        return {
            "observation/egocentric_camera": head_resized,
            "observation/wrist_image_left": left_resized,
            "observation/wrist_image_right": right_resized,
            "observation/state": prop_state,
            "prompt": self.text_prompt,
        }
    
    def update_current_stage(self, predicted_subtask_logits):
        """이 baseline에서는 stage 추적을 사용하지 않는다."""
        return
    
    def prepare_batch_for_pi_behavior(self, batch):
        """모델 입력에 로컬 task id만 추가한다.

        이 모델은 텍스트 프롬프트를 읽지 않는다.
        대신 "몇 번째 태스크인지"를 숫자로 넣고, 모델 내부 embedding 표에서
        해당 태스크 벡터를 꺼내 쓴다.
        """
        task_id = self.local_task_id if self.local_task_id is not None else -1
        batch_copy = batch.copy()
        if "prompt" in batch_copy:
            del batch_copy["prompt"]

        # PI_BEHAVIOR 기본 경로에서는 텍스트 프롬프트를 쓰지 않는다.
        # tokenized_prompt라는 이름은 OpenPI 코드 흐름과 맞추기 위해 유지하지만,
        # 실제 내용은 자연어 토큰이 아니라 local task id 하나다.
        batch_copy["tokenized_prompt"] = np.array([task_id], dtype=np.int32)
        batch_copy["tokenized_prompt_mask"] = np.array([True], dtype=bool)
        return batch_copy
    
    def _interpolate_actions(self, actions, target_steps):
        """Interpolate actions using cubic spline."""
        from scipy.interpolate import interp1d
        
        original_indices = np.linspace(0, len(actions)-1, len(actions))
        target_indices = np.linspace(0, len(actions)-1, target_steps)
        
        interpolated = np.zeros((target_steps, actions.shape[1]))
        for dim in range(actions.shape[1]):
            f = interp1d(original_indices, actions[:, dim], kind='cubic')
            interpolated[:, dim] = f(target_indices)
        
        return interpolated

    def act(self, obs: dict) -> torch.Tensor:
        """Main action function."""
        
        # Extract task_id from observations
        if "task_id" in obs:
            # 환경은 원래 전역 task id(예: 5, 40, 46)를 준다.
            # 여기서는 global id 그대로 상태 변경 함수에 넘긴다.
            # local id 변환은 _handle_task_change()와 prepare_batch_for_pi_behavior()가 담당한다.
            raw_task_id = int(obs["task_id"][0])
            self._handle_task_change(raw_task_id)
        
        raw_state = obs["robot_r1::proprio"]
        current_state = extract_state_from_proprio(raw_state)
        
        # 모델 예측은 비싸기 때문에 매 simulator step마다 새로 예측하지 않는다.
        # last_actions가 없거나, 이미 실행할 만큼 실행했을 때만 새 action chunk를 만든다.
        if self.last_actions is None or self.action_index >= self.config.execute_in_n_steps:
            
            # Process observation
            model_input = self.process_obs(obs)
            model_input = self.prepare_batch_for_pi_behavior(model_input)
            
            # Add rolling inpainting if available
            if self.next_initial_actions is not None and ("initial_actions" not in model_input or model_input["initial_actions"] is None):
                model_input["initial_actions"] = self.next_initial_actions
            
            # Get prediction
            if "initial_actions" in model_input and model_input["initial_actions"] is not None:
                output = self.policy.infer(model_input, initial_actions=model_input["initial_actions"])
            else:
                output = self.policy.infer(model_input)
            
            actions = output["actions"]
            
            # Ensure correct shape
            if len(actions.shape) == 3:
                actions = actions[0]
            if actions.shape[1] > 23:
                actions = actions[:, :23]
            
            # Apply eval tricks if enabled
            should_compress = self.config.execute_in_n_steps < self.config.actions_to_execute
            
            if False and self.config.apply_eval_tricks:
                if self.task_id is not None:
                    actions_before = actions.copy()
                    actions, corrected_stage = apply_correction_rules(
                        self.task_id, self.current_stage, current_state, actions
                    )
                    
                    # Log if stage was corrected
                    if corrected_stage != self.current_stage:
                        logger.info(f"🔧 Correction rule: Stage corrected {self.current_stage} → {corrected_stage} (task {self.task_id}, step {self.step_count})")
                        self.current_stage = corrected_stage
                        self.prediction_history.clear()
                    
                    # Log if actions were modified
                    if not np.allclose(actions_before, actions, rtol=1e-3):
                        max_diff = np.max(np.abs(actions_before - actions))
                        logger.info(f"🔧 Correction rule: Actions modified (max diff: {max_diff:.4f}, task {self.task_id}, stage {self.current_stage})")
                
                if should_compress:
                    has_high_variation, mean_var, max_var = check_gripper_variation(
                        actions, self.config.actions_to_execute
                    )
                    if has_high_variation:
                        should_compress = False
                        logger.info(f"🔧 Gripper variation: Compression disabled (mean: {mean_var:.4f}, max: {max_var:.4f})")
            
            # Determine execution parameters
            actions_to_execute = self.config.actions_to_execute if should_compress else self.config.execute_in_n_steps
            execute_steps = self.config.execute_in_n_steps
            
            # Save actions for next inpainting (before compression)
            inpainting_start = actions_to_execute
            inpainting_end = inpainting_start + self.config.actions_to_keep
            
            if len(actions) >= inpainting_end:
                self.next_initial_actions = actions[inpainting_start:inpainting_end].copy()
            else:
                self.next_initial_actions = None
            
            # Extract and compress actions
            self.last_actions = actions[:actions_to_execute].copy()
            
            if should_compress:
                compressed_actions = self._interpolate_actions(self.last_actions, execute_steps)
                compression_factor = actions_to_execute / execute_steps
                compressed_actions[:, :3] *= compression_factor  # Scale velocities
                self.last_actions = compressed_actions
            
            self.action_index = 0
            self.prediction_count += 1
            
            # Log prediction details (at lower frequency, every 10 predictions)
            if self.prediction_count % 10 == 0:
                compression_status = f"compressed {actions_to_execute}→{execute_steps}" if should_compress else f"uncompressed ({execute_steps})"
                logger.info(f"🎯 Prediction #{self.prediction_count} | Actions: {compression_status} | Inpainting: {self.next_initial_actions is not None}")
            
            # Update stage based on model predictions
            if "subtask_logits" in output:
                self.update_current_stage(output["subtask_logits"])
        
        # Get current action from sequence
        if self.action_index >= len(self.last_actions):
            self.action_index = 0
            
        current_action = self.last_actions[self.action_index]

        # [수정일: 2026-04-29]
        # [디버그 목적]
        # 로봇이 넘어지는 원인을 action 영역별로 분리하기 위한 테스트 코드.
        #
        # 사용 방법:
        # A100 policy server 실행 전에 아래 환경변수 설정:
        #
        # export B1K_ACTION_DEBUG_MODE=zero_all
        # export B1K_ACTION_DEBUG_MODE=base_only
        # export B1K_ACTION_DEBUG_MODE=arm_only
        # export B1K_ACTION_DEBUG_MODE=safe_clip
        #
        # mode 설명:
        # - zero_all  : 모든 action을 0으로 고정. 이 상태에서도 넘어지면 sim/init 문제.
        # - base_only : base action만 아주 작게 허용, arm/gripper는 0. base 때문에 넘어지는지 확인.
        # - arm_only  : base는 0, arm만 작게 허용. arm/torso 때문에 넘어지는지 확인.
        # - safe_clip : base와 arm을 모두 작게 제한. 실제 안정화 후보.
        #
        # 주의:
        # 이 코드는 성능 향상용이 아니라 원인 분리용 임시 safety filter다.
        current_action = current_action.copy()

        debug_mode = os.environ.get("B1K_ACTION_DEBUG_MODE", "safe_clip")

        if debug_mode == "zero_all":
            current_action[:] = 0.0

        elif debug_mode == "base_only":
            # base만 아주 작게 움직이고 arm/gripper는 고정
            current_action[:] = 0.0
            current_action[:3] = np.clip(
                self.last_actions[self.action_index][:3] * 0.08,
                -0.03,
                0.03,
            )

        elif debug_mode == "arm_only":
            # base는 완전히 고정하고, arm/joint만 조금 더 크게 허용한다.
            # 방금 arm_only에서 팔이 거의 안 움직였으므로 clip 범위를 키워서 확인한다.
            current_action[:3] = 0.0
            current_action[3:-1] = np.clip(current_action[3:-1], -0.3, 0.3)
            current_action[-1] = 0.0

        elif debug_mode == "safe_clip":
            # 현재 가장 보수적인 안정화 후보
            current_action[:3] = np.clip(current_action[:3] * 0.08, -0.03, 0.03)
            current_action[3:-1] = np.clip(current_action[3:-1], -0.08, 0.08)
            current_action[-1] = 0.0

        else:
            raise ValueError(f"Unknown B1K_ACTION_DEBUG_MODE: {debug_mode}")

        # [수정일: 2026-04-29]
        # [디버그 목적]
        # arm_only / safe_clip 상태에서 실제 action 값이 어느 channel에 나오는지 확인한다.
        # 팔 카메라가 움직이지 않는 원인이
        # 1) arm action 값이 거의 0인 것인지
        # 2) 우리가 arm이라고 생각한 current_action[3:-1]이 실제 팔 channel이 아닌 것인지
        # 확인하기 위한 로그다.
        if self.step_count % 20 == 0:
            raw_action = self.last_actions[self.action_index]
            logger.info(
                "[ACTION DEBUG] "
                f"mode={debug_mode}, "
                f"shape={raw_action.shape}, "
                f"raw_base={raw_action[:3]}, "
                f"raw_mid_min={raw_action[3:-1].min():.4f}, "
                f"raw_mid_max={raw_action[3:-1].max():.4f}, "
                f"raw_mid_mean={raw_action[3:-1].mean():.4f}, "
                f"raw_gripper={raw_action[-1]:.4f}, "
                f"final_base={current_action[:3]}, "
                f"final_mid_min={current_action[3:-1].min():.4f}, "
                f"final_mid_max={current_action[3:-1].max():.4f}, "
                f"final_gripper={current_action[-1]:.4f}"
            )

        self.action_index += 1
        self.step_count += 1
        
        # Log progress every 100 steps
        if self.step_count % 100 == 0:
            logger.info(f"📊 Step {self.step_count} | Local task: {self.task_id} | Predictions: {self.prediction_count}")
        
        # Convert to torch tensor
        action_tensor = torch.from_numpy(current_action).float()
        if len(action_tensor) > 23:
            action_tensor = action_tensor[:23]
        
        return action_tensor

