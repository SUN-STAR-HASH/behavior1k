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
from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.shared.proprioception_indices import PROPRIOCEPTION_INDICES

logger = logging.getLogger(__name__)

RESIZE_SIZE = 224

# ============================================================
# R1Pro 23D action mapping
# ============================================================
# b1k_policy.py의 state 구성 순서와 동일하게 맞춘다.
# 0:3   base velocity
# 3:7   torso/trunk 4D
# 7:14  left arm 7D
# 14    left gripper
# 15:22 right arm 7D
# 22    right gripper

ACTION_DIM = 23

BASE = slice(0, 3)
TORSO = slice(3, 7)
LEFT_ARM = slice(7, 14)
LEFT_GRIPPER = 14
RIGHT_ARM = slice(15, 22)
RIGHT_GRIPPER = 22

GRIPPER_OPEN_VALUE = 1.0
GRIPPER_CLOSE_VALUE = -1.0
GRIPPER_THRESHOLD = 0.25


def _threshold_gripper_value(g: np.ndarray | float) -> np.ndarray:
    """
    Gripper output을 연속값 그대로 쓰지 않고 open/close로 이산화한다.

    현재 repo의 correction_rules.py 기준:
    -1.0 = closed
     1.0 = open
    """
    return np.where(
        g < -GRIPPER_THRESHOLD,
        GRIPPER_CLOSE_VALUE,
        np.where(g > GRIPPER_THRESHOLD, GRIPPER_OPEN_VALUE, GRIPPER_OPEN_VALUE),
    )


def postprocess_action_eval_stable_v6(action: np.ndarray) -> np.ndarray:
    """
    eval 전용 action 안정화 후처리.

    목적:
    1. base가 너무 커서 넘어지는 문제 방지
    2. torso/trunk가 자세를 흔드는 문제 방지
    3. arm은 너무 죽이지 않아서 radio/object interaction 가능하게 유지
    4. left/right gripper를 각각 threshold 처리
    """
    a = np.asarray(action, dtype=np.float32).copy()

    if a.shape[-1] < ACTION_DIM:
        raise ValueError(f"Expected action dim >= {ACTION_DIM}, got shape={a.shape}")

    # NaN / Inf 방어
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)

    raw = a.copy()

    # ----------------------------
    # 1) base velocity: 이동은 살리되 yaw는 매우 작게
    # ----------------------------
    a[..., 0] = np.clip(raw[..., 0] * 0.22, -0.18, 0.18)
    a[..., 1] = np.clip(raw[..., 1] * 0.40, -0.30, 0.30)
    a[..., 2] = np.clip(raw[..., 2] * 0.020, -0.018, 0.018)

    # ----------------------------
    # 2) torso/trunk: 자세 불안정 방지를 위해 약하게
    # ----------------------------
    a[..., TORSO] = np.clip(raw[..., TORSO] * 0.10, -0.20, 0.20)

    # ----------------------------
    # 3) arms: base보다 강하게 허용
    # ----------------------------
    a[..., LEFT_ARM] = np.clip(raw[..., LEFT_ARM] * 1.25, -1.20, 1.20)
    a[..., RIGHT_ARM] = np.clip(raw[..., RIGHT_ARM] * 1.25, -1.20, 1.20)

    # ----------------------------
    # 4) grippers: left/right 둘 다 threshold 처리
    # ----------------------------
    a[..., LEFT_GRIPPER] = _threshold_gripper_value(raw[..., LEFT_GRIPPER])
    a[..., RIGHT_GRIPPER] = _threshold_gripper_value(raw[..., RIGHT_GRIPPER])

    return a

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
    actions_to_execute: int = 20
    actions_to_keep: int = 0
    execute_in_n_steps: int = 20

    history_len: int = 1
    votes_to_promote: int = 1
    time_threshold_inpaint: float = 0.3
    num_steps: int = 8
    apply_eval_tricks: bool = False
    use_stage_tracking: bool = True

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
        """모델이 예측한 stage logit을 voting으로 부드럽게 반영한다.

        한 번의 예측만 믿고 stage를 바로 바꾸면 화면이 애매한 순간에 앞뒤로 흔들릴 수 있다.
        그래서 최근 예측을 history에 모아두고, 같은 다음 stage가 일정 횟수 이상 나왔을 때만
        current_stage를 올린다.
        """
        if not self.config.use_stage_tracking or self.local_task_id is None:
            return

        logits = np.asarray(predicted_subtask_logits)
        if logits.ndim > 1:
            logits = logits[0]

        max_stage = TASK_NUM_STAGES[self.local_task_id] - 1
        predicted_stage = int(np.argmax(logits))
        predicted_stage = max(0, min(predicted_stage, max_stage))
        self.prediction_history.append(predicted_stage)

        if len(self.prediction_history) < self.config.history_len:
            return

        next_stage = self.current_stage + 1
        if next_stage <= max_stage:
            votes_for_next = sum(1 for pred in self.prediction_history if pred == next_stage)
            votes_to_skip = sum(1 for pred in self.prediction_history if pred == next_stage + 1)

            if votes_for_next >= self.config.votes_to_promote:
                old_stage = self.current_stage
                self.current_stage = next_stage
                self.prediction_history.clear()
                logger.info(
                    "Stage advanced: %s -> %s (global task %s, local task %s, step %s)",
                    old_stage,
                    self.current_stage,
                    self.task_id,
                    self.local_task_id,
                    self.step_count,
                )
            elif votes_to_skip == self.config.history_len and next_stage < max_stage:
                old_stage = self.current_stage
                self.current_stage = next_stage
                self.prediction_history.clear()
                logger.info(
                    "Stage skipped: %s -> %s (global task %s, local task %s, step %s)",
                    old_stage,
                    self.current_stage,
                    self.task_id,
                    self.local_task_id,
                    self.step_count,
                )

        prev_stage = self.current_stage - 1
        if prev_stage >= 0:
            votes_to_go_back = sum(1 for pred in self.prediction_history if pred == prev_stage)
            if votes_to_go_back == self.config.history_len:
                old_stage = self.current_stage
                self.current_stage = prev_stage
                self.prediction_history.clear()
                logger.info(
                    "Stage went back: %s -> %s (global task %s, local task %s, step %s)",
                    old_stage,
                    self.current_stage,
                    self.task_id,
                    self.local_task_id,
                    self.step_count,
                )
    
    def prepare_batch_for_pi_behavior(self, batch):
        """모델 입력에 로컬 task id와 선택적으로 현재 stage id를 추가한다.

        이 모델은 텍스트 프롬프트를 읽지 않는다.
        대신 "몇 번째 태스크인지"와 "현재 몇 번째 stage인지"를 숫자로 넣고,
        모델 내부 embedding 표에서 해당 벡터를 꺼내 쓴다.
        """
        task_id = self.local_task_id if self.local_task_id is not None else -1
        batch_copy = batch.copy()
        if "prompt" in batch_copy:
            del batch_copy["prompt"]

        if self.config.use_stage_tracking:
            batch_copy["tokenized_prompt"] = np.array(
                [task_id, self.current_stage], dtype=np.int32
            )
            batch_copy["tokenized_prompt_mask"] = np.array([True, True], dtype=bool)
            batch_copy["subtask_state"] = np.array(self.current_stage, dtype=np.int32)
        else:
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

        debug_mode = os.environ.get("B1K_ACTION_DEBUG_MODE")
        if debug_mode is None:
            debug_mode = "eval_stable_v6" if self.config.apply_eval_tricks else "none"

        if debug_mode in ("none", "off", "raw"):
            pass

        elif debug_mode == "zero_all":
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

        elif debug_mode == "eval_stable_v1":
            # [수정일: 2026-04-30]
            # [실험 목적]
            # probe_sweep 결과:
            # - index 0~2는 base 계열로 보이며, 큰 base action은 넘어짐을 유발할 가능성이 큼.
            # - index 22는 gripper로 거의 확정.
            # - index 3~21은 arm/body 계열로 예상되지만, 너무 작게 clip하면 움직임이 거의 안 보임.
            #
            # 따라서 실제 eval 실험에서는:
            # 1) base는 아주 작게만 허용
            # 2) arm/body 계열은 기존 safe_clip보다 크게 허용
            # 3) gripper는 일단 0으로 고정해서 불필요한 여닫힘을 막음
            current_action[:3] = np.clip(current_action[:3] * 0.03, -0.02, 0.02)
            current_action[3:22] = np.clip(current_action[3:22], -0.8, 0.8)
            current_action[22] = 0.0
        
        elif debug_mode == "eval_stable_v2":
            # [수정일: 2026-04-30]
            # [실험 목적]
            # eval_stable_v1에서 넘어짐은 해결됐지만 base 이동이 너무 작아 task 진전이 거의 없음.
            #
            # probe / eval 로그 기준:
            # - index 0~2는 base 계열
            # - 특히 base[1] 값이 계속 크게 나오므로 전후 이동 후보로 보고 더 열어준다.
            # - base[2]는 회전/자세 불안정 가능성이 있으므로 작게 유지한다.
            # - gripper는 아직 불필요한 여닫힘을 막기 위해 고정한다.
            base_action = current_action[:3].copy()

            # base[0]: 좌우 또는 전후 후보. v1보다 조금만 증가.
            current_action[0] = np.clip(base_action[0] * 0.04, -0.03, 0.03)

            # base[1]: 전후 이동 후보. task 진전을 위해 v1보다 크게 허용.
            current_action[1] = np.clip(base_action[1] * 0.10, -0.07, 0.07)

            # base[2]: 회전/yaw 후보. 넘어짐 방지를 위해 작게 유지.
            current_action[2] = np.clip(base_action[2] * 0.025, -0.02, 0.02)

            # arm/body 계열은 너무 작게 자르면 팔이 거의 안 움직이므로 v1과 동일하게 유지.
            current_action[3:22] = np.clip(current_action[3:22], -0.8, 0.8)

            # gripper는 일단 고정.
            current_action[22] = 0.0

        elif debug_mode == "eval_stable_v3":
            # [수정일: 2026-04-30]
            # [실험 목적]
            # eval_stable_v2에서 넘어짐은 해결됐지만 base 이동이 너무 작아 task 진전이 거의 없음.
            #
            # 이번 v3에서는:
            # 1) base x/y 이동을 v2보다 확실히 키운다.
            # 2) 회전/yaw 후보인 base[2]는 여전히 작게 제한한다.
            # 3) arm/body는 v2보다 약간 더 열어준다.
            # 4) gripper는 여전히 고정한다.
            base_action = current_action[:3].copy()

            # base[0], base[1] 둘 중 어느 쪽이 전후 이동인지 아직 완전히 확정되지 않았으므로
            # 둘 다 v2보다 열어준다.
            current_action[0] = np.clip(base_action[0] * 0.10, -0.08, 0.08)
            current_action[1] = np.clip(base_action[1] * 0.20, -0.15, 0.15)

            # 회전/yaw는 넘어짐 또는 불안정 원인이 될 수 있으므로 조금만 허용한다.
            current_action[2] = np.clip(base_action[2] * 0.035, -0.03, 0.03)

            # 팔/몸통 계열은 v2보다 조금 더 허용한다.
            current_action[3:22] = np.clip(current_action[3:22], -1.0, 1.0)

            # gripper는 불필요한 여닫힘 방지.
            current_action[22] = 0.0
        elif debug_mode == "eval_stable_v4":
            # [수정일: 2026-04-30]
            # [실험 목적]
            # eval_stable_v3에서도 넘어지지는 않았지만 base 이동이 너무 작아 task 진전이 거의 없음.
            #
            # v4에서는:
            # 1) base x/y 이동을 v3보다 더 크게 허용한다.
            # 2) 회전/yaw 후보인 base[2]는 계속 작게 유지한다.
            # 3) arm/body는 v3와 동일하게 둔다.
            # 4) gripper는 계속 고정한다.
            base_action = current_action[:3].copy()

            # base 이동 강화
            current_action[0] = np.clip(base_action[0] * 0.18, -0.14, 0.14)
            current_action[1] = np.clip(base_action[1] * 0.35, -0.25, 0.25)

            # yaw/회전은 아직 위험하므로 작게 유지
            current_action[2] = np.clip(base_action[2] * 0.03, -0.025, 0.025)

            # 팔/몸통 계열
            current_action[3:22] = np.clip(current_action[3:22], -1.0, 1.0)

            # gripper 고정
            current_action[22] = 0.0

        elif debug_mode == "eval_stable_v5":
            # [수정일: 2026-04-30]
            # [실험 목적]
            # eval_stable_v4에서 넘어지지는 않았지만 base 이동이 아직 부족하여
            # turning_on_radio task에서 라디오 앞까지 접근하지 못함.
            #
            # v5에서는:
            # 1) base x/y 이동을 v4보다 더 크게 허용한다.
            # 2) yaw/회전은 넘어짐 방지를 위해 아주 작게 유지한다.
            # 3) arm/body는 v4보다 약간 더 허용한다.
            # 4) gripper는 완전 고정하지 않고 약하게만 허용한다.
            base_action = current_action[:3].copy()

            # base 이동 강화
            current_action[0] = np.clip(base_action[0] * 0.25, -0.20, 0.20)
            current_action[1] = np.clip(base_action[1] * 0.60, -0.40, 0.40)

            # yaw/회전은 계속 작게 제한
            current_action[2] = np.clip(base_action[2] * 0.015, -0.012, 0.012)

            # 팔/몸통 계열은 v4보다 조금 더 허용
            current_action[3:22] = np.clip(current_action[3:22], -1.2, 1.2)

            # gripper는 완전 고정하지 않고 약하게만 허용
            current_action[22] = np.clip(current_action[22] * 0.15, -0.25, 0.25)

        elif debug_mode == "eval_stable_v6":
            # [수정일: 2026-05-06]
            # [실험 목적]
            # v5는 3:22를 한 덩어리로 처리하고 right gripper(22)만 약하게 허용했다.
            # v6는 action mapping을 명시적으로 반영한다.
            #
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper
            current_action = postprocess_action_eval_stable_v6(current_action)

        elif debug_mode == "eval_radio_v7":
            # [수정일: 2026-05-06]
            # [실험 목적]
            # v6에서는 gripper 14, 22를 threshold 처리했는데,
            # 모델 출력 노이즈 때문에 gripper가 계속 open/close 반복하는 문제가 발생했다.
            #
            # turning_on_radio task는 물체를 집는 작업이 아니므로,
            # 이번 v7에서는 양쪽 gripper를 완전히 고정한다.

            raw_action = current_action.copy()

            # 1) base 이동은 v5 수준으로 다시 살림
            current_action[0] = np.clip(raw_action[0] * 0.25, -0.20, 0.20)
            current_action[1] = np.clip(raw_action[1] * 0.60, -0.40, 0.40)

            # yaw는 계속 작게. yaw가 크면 넘어지거나 빙글빙글 돌 가능성이 큼
            current_action[2] = np.clip(raw_action[2] * 0.015, -0.012, 0.012)

            # 2) torso/trunk는 자세 흔들림 방지를 위해 약하게
            current_action[3:7] = np.clip(raw_action[3:7] * 0.10, -0.20, 0.20)

            # 3) arm은 radio 조작을 위해 어느 정도 허용
            current_action[7:14] = np.clip(raw_action[7:14] * 1.20, -1.20, 1.20)
            current_action[15:22] = np.clip(raw_action[15:22] * 1.20, -1.20, 1.20)

            # 4) 핵심 수정: 양쪽 gripper 완전 고정
            # v6의 threshold 방식은 gripper open/close 진동을 유발했으므로 제거한다.
            current_action[14] = 0.0
            current_action[22] = 0.0

        elif debug_mode == "eval_radio_nav_v8":
            # [수정일: 2026-05-06]
            # [목적]
            # v6/v7에서 yaw를 너무 작게 제한해서 radio를 찾지 못하는 문제가 있었다.
            # v8은 gripper를 완전히 고정하고, navigation 단계에서 base/yaw를 다시 살린다.
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base navigation 강화
            # ----------------------------
            base = np.zeros(3, dtype=np.float32)

            base[0] = np.clip(raw_action[0] * 0.32, -0.26, 0.26)
            base[1] = np.clip(raw_action[1] * 0.75, -0.45, 0.45)
            base[2] = np.clip(raw_action[2] * 0.08, -0.055, 0.055)

            # base smoothing: 갑자기 흔들리거나 넘어지는 것 방지
            if not hasattr(self, "_radio_v8_last_base"):
                self._radio_v8_last_base = np.zeros(3, dtype=np.float32)

            base = 0.65 * self._radio_v8_last_base + 0.35 * base
            self._radio_v8_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk는 약하게 유지
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.08, -0.15, 0.15)

            # ----------------------------
            # 3) 초기 navigation 동안 arm은 줄임
            # ----------------------------
            nav_phase = self.step_count < 180

            if nav_phase:
                current_action[7:14] = np.clip(raw_action[7:14] * 0.25, -0.35, 0.35)
                current_action[15:22] = np.clip(raw_action[15:22] * 0.25, -0.35, 0.35)
            else:
                current_action[7:14] = np.clip(raw_action[7:14] * 1.20, -1.20, 1.20)
                current_action[15:22] = np.clip(raw_action[15:22] * 1.20, -1.20, 1.20)

            # ----------------------------
            # 4) gripper 완전 고정
            # ----------------------------
            current_action[14] = 0.0
            current_action[22] = 0.0

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_radio_nav_v8] "
                    f"step={self.step_count}, "
                    f"nav_phase={nav_phase}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"g14={current_action[14]:.3f}, "
                    f"g22={current_action[22]:.3f}"
                )

        elif debug_mode == "eval_radio_nav_v9":
            # [수정일: 2026-05-06]
            # [목적]
            # v8은 radio를 찾는 데 성공했지만, 찾는 속도가 느리고 이후 넘어지는 문제가 있었다.
            # v9는 base 이동은 조금 더 빠르게 만들되, lateral/yaw/arm을 안정화한다.
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base navigation: 빠르지만 안정적으로
            # ----------------------------
            base = np.zeros(3, dtype=np.float32)

            # forward는 v8보다 조금 살림
            base[0] = np.clip(raw_action[0] * 0.42, -0.34, 0.34)

            # lateral은 v8보다 줄임
            # v8의 y clip ±0.45는 넘어짐을 유발할 가능성이 컸음
            base[1] = np.clip(raw_action[1] * 0.50, -0.28, 0.28)

            # yaw도 v8보다 줄임
            # v8의 ±0.055는 회전 탐색은 됐지만 넘어질 위험이 컸음
            base[2] = np.clip(raw_action[2] * 0.06, -0.038, 0.038)

            # x/y 동시 이동이 너무 커지는 것을 방지
            xy_norm = np.linalg.norm(base[0:2])
            max_xy_norm = 0.34
            if xy_norm > max_xy_norm:
                base[0:2] = base[0:2] / (xy_norm + 1e-6) * max_xy_norm

            # 초반에는 조금 더 빠르게 찾고,
            # 후반에는 팔 조작을 위해 base를 줄인다.
            if self.step_count < 240:
                base[0:3] *= 1.10
            elif self.step_count < 360:
                base[0:3] *= 0.75
            else:
                base[0:3] *= 0.45

            # base smoothing 강화
            if self.step_count == 0 or not hasattr(self, "_radio_v9_last_base"):
                self._radio_v9_last_base = np.zeros(3, dtype=np.float32)

            base = 0.78 * self._radio_v9_last_base + 0.22 * base
            self._radio_v9_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk는 더 약하게
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.05, -0.10, 0.10)

            # ----------------------------
            # 3) arm은 늦게, 천천히 풀기
            # ----------------------------
            if self.step_count < 300:
                arm_scale = 0.20
                arm_clip = 0.30
            elif self.step_count < 420:
                arm_scale = 0.55
                arm_clip = 0.65
            else:
                arm_scale = 0.85
                arm_clip = 0.90

            current_action[7:14] = np.clip(raw_action[7:14] * arm_scale, -arm_clip, arm_clip)
            current_action[15:22] = np.clip(raw_action[15:22] * arm_scale, -arm_clip, arm_clip)

            # ----------------------------
            # 4) gripper 완전 고정
            # ----------------------------
            current_action[14] = 0.0
            current_action[22] = 0.0

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_radio_nav_v9] "
                    f"step={self.step_count}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"arm_scale={arm_scale:.2f}, "
                    f"g14={current_action[14]:.3f}, "
                    f"g22={current_action[22]:.3f}"
                )

        elif debug_mode == "eval_radio_nav_v10":
            # [수정일: 2026-05-06]
            # [목적]
            # v8은 radio를 찾았지만 넘어졌고, v9은 너무 느려져서 radio를 못 찾았다.
            # v10은 v8의 탐색 능력을 다시 살리되, lateral/yaw/arm을 단계적으로 제한한다.
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base navigation
            # ----------------------------
            base = np.zeros(3, dtype=np.float32)

            # v8보다 forward는 조금 더 빠르게
            base[0] = np.clip(raw_action[0] * 0.45, -0.36, 0.36)

            # v9은 y가 너무 작아서 탐색이 죽었음
            # v8보다는 작고, v9보다는 크게
            base[1] = np.clip(raw_action[1] * 0.68, -0.38, 0.38)

            # yaw도 v9보다 다시 살림
            # v8 수준에 가깝게 하되, clip은 약간만 줄임
            base[2] = np.clip(raw_action[2] * 0.085, -0.052, 0.052)

            # x/y 동시 이동이 너무 커지는 것 방지
            xy_norm = np.linalg.norm(base[0:2])
            max_xy_norm = 0.42
            if xy_norm > max_xy_norm:
                base[0:2] = base[0:2] / (xy_norm + 1e-6) * max_xy_norm

            # 단계별 속도 조절
            # 초반: 라디오 찾기 위해 적극 이동
            # 중반: 접근 유지
            # 후반: 넘어짐 방지를 위해 감속
            if self.step_count < 220:
                base[0:3] *= 1.15
            elif self.step_count < 360:
                base[0:3] *= 0.95
            else:
                base[0:3] *= 0.60

            # smoothing은 v9보다 약하게 해서 반응성을 살림
            if self.step_count == 0 or not hasattr(self, "_radio_v10_last_base"):
                self._radio_v10_last_base = np.zeros(3, dtype=np.float32)

            if self.step_count < 260:
                smooth_prev = 0.55
                smooth_new = 0.45
            else:
                smooth_prev = 0.70
                smooth_new = 0.30

            base = smooth_prev * self._radio_v10_last_base + smooth_new * base
            self._radio_v10_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk 안정화
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.06, -0.12, 0.12)

            # ----------------------------
            # 3) arm은 너무 늦게 풀지 않되, 급격히 풀지 않음
            # ----------------------------
            if self.step_count < 240:
                arm_scale = 0.18
                arm_clip = 0.28
            elif self.step_count < 360:
                arm_scale = 0.45
                arm_clip = 0.55
            else:
                arm_scale = 0.75
                arm_clip = 0.80

            current_action[7:14] = np.clip(raw_action[7:14] * arm_scale, -arm_clip, arm_clip)
            current_action[15:22] = np.clip(raw_action[15:22] * arm_scale, -arm_clip, arm_clip)

            # ----------------------------
            # 4) gripper 고정
            # ----------------------------
            current_action[14] = 0.0
            current_action[22] = 0.0

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_radio_nav_v10] "
                    f"step={self.step_count}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"arm_scale={arm_scale:.2f}, "
                    f"g14={current_action[14]:.3f}, "
                    f"g22={current_action[22]:.3f}"
                )

        elif debug_mode == "eval_radio_approach_v11":
            # [수정일: 2026-05-06]
            # [목적]
            # radio task 전용 if / phase 분기를 제거한 일반 action 후처리 모드.
            # 모든 task에 같은 방식으로 적용된다.
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # tunable parameters
            # ----------------------------
            # 가정:
            # action[0] = forward/back
            # action[1] = yaw
            # action[2] = lateral
            forward_axis = int(os.environ.get("B1K_FORWARD_AXIS", "0"))
            yaw_axis = int(os.environ.get("B1K_YAW_AXIS", "1"))
            lateral_axis = int(os.environ.get("B1K_LATERAL_AXIS", "2"))

            forward_sign = float(os.environ.get("B1K_FORWARD_SIGN", "1.0"))
            forward_bias = float(os.environ.get("B1K_FORWARD_BIAS", "0.08"))

            forward_scale = float(os.environ.get("B1K_FORWARD_SCALE", "0.34"))
            yaw_scale = float(os.environ.get("B1K_YAW_SCALE", "0.035"))
            lateral_scale = float(os.environ.get("B1K_LATERAL_SCALE", "0.28"))

            forward_max = float(os.environ.get("B1K_FORWARD_MAX", "0.28"))
            yaw_max = float(os.environ.get("B1K_YAW_MAX", "0.025"))
            lateral_max = float(os.environ.get("B1K_LATERAL_MAX", "0.18"))
            planar_max = float(os.environ.get("B1K_PLANAR_MAX", "0.32"))

            ramp_start = float(os.environ.get("B1K_FORWARD_RAMP_START", "180"))
            ramp_len = float(os.environ.get("B1K_FORWARD_RAMP_LEN", "140"))

            # step 기반 ramp. task-specific if 없이 0~1로 부드럽게 증가.
            ramp = np.clip((float(self.step_count) - ramp_start) / (ramp_len + 1e-6), 0.0, 1.0)

            # ----------------------------
            # 1) base navigation
            # ----------------------------
            base = np.zeros(3, dtype=np.float32)

            # forward/back
            base[forward_axis] = np.clip(
                raw_action[forward_axis] * forward_scale + forward_sign * forward_bias * ramp,
                -forward_max,
                forward_max,
            )

            # yaw는 작게 제한. 기존 코드에서 action[1]을 크게 살린 게 넘어짐 원인일 수 있음.
            base[yaw_axis] = np.clip(
                raw_action[yaw_axis] * yaw_scale,
                -yaw_max,
                yaw_max,
            )

            # lateral은 허용하되 작게 제한
            base[lateral_axis] = np.clip(
                raw_action[lateral_axis] * lateral_scale,
                -lateral_max,
                lateral_max,
            )

            # forward + lateral 평면 속도 제한
            planar_norm = np.linalg.norm([base[forward_axis], base[lateral_axis]])
            planar_scale = np.minimum(1.0, planar_max / (planar_norm + 1e-6))
            base[forward_axis] *= planar_scale
            base[lateral_axis] *= planar_scale

            # smoothing. if문 없이 getattr 기본값 사용.
            last_base = getattr(self, "_general_v12_last_base", np.zeros(3, dtype=np.float32))
            base = 0.62 * last_base + 0.38 * base
            self._general_v12_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk 안정화
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.05, -0.10, 0.10)

            # ----------------------------
            # 3) arm도 if 없이 ramp로 천천히 풀기
            # ----------------------------
            arm_ramp = np.clip((float(self.step_count) - 260.0) / 220.0, 0.0, 1.0)

            arm_scale = 0.18 + 0.57 * arm_ramp
            arm_clip = 0.28 + 0.52 * arm_ramp

            current_action[7:14] = np.clip(raw_action[7:14] * arm_scale, -arm_clip, arm_clip)
            current_action[15:22] = np.clip(raw_action[15:22] * arm_scale, -arm_clip, arm_clip)

            # ----------------------------
            # 4) gripper 고정
            # ----------------------------
            current_action[14] = 0.0
            current_action[22] = 0.0

            logger.info(
                f"[eval_general_forward_v12] "
                f"step={self.step_count}, "
                f"ramp={ramp:.3f}, "
                f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                f"forward_axis={forward_axis}, "
                f"forward_sign={forward_sign:.1f}, "
                f"forward_bias={forward_bias:.3f}, "
                f"arm_scale={arm_scale:.2f}, "
                f"g14={current_action[14]:.3f}, "
                f"g22={current_action[22]:.3f}"
            )

        elif debug_mode == "eval_reach_grip_v13":
            # [수정일: 2026-05-06]
            # [목적]
            # 전진축은 action[0]으로 확정된 상황에서,
            # 전진 bias는 줄이고, radio 근처에서 팔을 단계적으로 풀고,
            # gripper를 open -> close -> open 형태로 한 번 동작시킨다.
            #
            # 현재 가정:
            # action[0] = forward/back
            # action[1] = yaw
            # action[2] = lateral
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # tunable parameters
            # ----------------------------
            forward_axis = int(os.environ.get("B1K_FORWARD_AXIS", "0"))
            yaw_axis = int(os.environ.get("B1K_YAW_AXIS", "1"))
            lateral_axis = int(os.environ.get("B1K_LATERAL_AXIS", "2"))

            forward_sign = float(os.environ.get("B1K_FORWARD_SIGN", "1.0"))

            # 전진은 맞았으니 기존보다 약하게
            forward_bias = float(os.environ.get("B1K_FORWARD_BIAS", "0.06"))
            forward_scale = float(os.environ.get("B1K_FORWARD_SCALE", "0.30"))
            forward_max = float(os.environ.get("B1K_FORWARD_MAX", "0.24"))

            yaw_scale = float(os.environ.get("B1K_YAW_SCALE", "0.030"))
            yaw_max = float(os.environ.get("B1K_YAW_MAX", "0.020"))

            lateral_scale = float(os.environ.get("B1K_LATERAL_SCALE", "0.22"))
            lateral_max = float(os.environ.get("B1K_LATERAL_MAX", "0.14"))

            planar_max = float(os.environ.get("B1K_PLANAR_MAX", "0.28"))

            forward_ramp_start = float(os.environ.get("B1K_FORWARD_RAMP_START", "180"))
            forward_ramp_len = float(os.environ.get("B1K_FORWARD_RAMP_LEN", "140"))

            # 팔을 풀기 시작하는 시점
            reach_start = float(os.environ.get("B1K_REACH_START", "260"))
            reach_len = float(os.environ.get("B1K_REACH_LEN", "180"))

            # gripper 동작 시점
            grip_close_start = float(os.environ.get("B1K_GRIP_CLOSE_START", "360"))
            grip_open_start = float(os.environ.get("B1K_GRIP_OPEN_START", "520"))
            grip_ramp_len = float(os.environ.get("B1K_GRIP_RAMP_LEN", "60"))

            # gripper command 값
            # 방향이 반대면 A100 실행 시 OPEN/CLOSE 값을 서로 바꾸면 된다.
            gripper_open_cmd = float(os.environ.get("B1K_GRIPPER_OPEN", "0.25"))
            gripper_close_cmd = float(os.environ.get("B1K_GRIPPER_CLOSE", "-0.35"))

            # arm gain
            left_arm_gain = float(os.environ.get("B1K_LEFT_ARM_GAIN", "0.85"))
            right_arm_gain = float(os.environ.get("B1K_RIGHT_ARM_GAIN", "1.15"))

            # ----------------------------
            # 1) base navigation
            # ----------------------------
            base = np.zeros(3, dtype=np.float32)

            forward_ramp = np.clip(
                (float(self.step_count) - forward_ramp_start) / (forward_ramp_len + 1e-6),
                0.0,
                1.0,
            )

            base[forward_axis] = np.clip(
                raw_action[forward_axis] * forward_scale + forward_sign * forward_bias * forward_ramp,
                -forward_max,
                forward_max,
            )

            base[yaw_axis] = np.clip(
                raw_action[yaw_axis] * yaw_scale,
                -yaw_max,
                yaw_max,
            )

            base[lateral_axis] = np.clip(
                raw_action[lateral_axis] * lateral_scale,
                -lateral_max,
                lateral_max,
            )

            planar_norm = np.linalg.norm([base[forward_axis], base[lateral_axis]])
            planar_scale = np.minimum(1.0, planar_max / (planar_norm + 1e-6))
            base[forward_axis] *= planar_scale
            base[lateral_axis] *= planar_scale

            last_base = getattr(self, "_reach_grip_v13_last_base", np.zeros(3, dtype=np.float32))
            base = 0.70 * last_base + 0.30 * base
            self._reach_grip_v13_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk 안정화
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.04, -0.08, 0.08)

            # ----------------------------
            # 3) arm을 단계적으로 풀기
            # ----------------------------
            reach_ramp = np.clip(
                (float(self.step_count) - reach_start) / (reach_len + 1e-6),
                0.0,
                1.0,
            )

            # 초반에는 0.20 수준, reach 이후에는 1.05까지 증가
            arm_scale = 0.20 + 0.85 * reach_ramp
            arm_clip = 0.30 + 0.70 * reach_ramp

            current_action[7:14] = np.clip(
                raw_action[7:14] * arm_scale * left_arm_gain,
                -arm_clip,
                arm_clip,
            )

            current_action[15:22] = np.clip(
                raw_action[15:22] * arm_scale * right_arm_gain,
                -arm_clip,
                arm_clip,
            )

            # ----------------------------
            # 4) gripper open -> close -> open
            # ----------------------------
            close_gate = np.clip(
                (float(self.step_count) - grip_close_start) / (grip_ramp_len + 1e-6),
                0.0,
                1.0,
            )

            reopen_gate = np.clip(
                (float(self.step_count) - grip_open_start) / (grip_ramp_len + 1e-6),
                0.0,
                1.0,
            )

            # close_gate가 1이면 close, reopen_gate가 1이면 다시 open
            grip_gate = close_gate * (1.0 - reopen_gate)

            gripper_cmd = gripper_open_cmd * (1.0 - grip_gate) + gripper_close_cmd * grip_gate

            current_action[14] = gripper_cmd
            current_action[22] = gripper_cmd

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_reach_grip_v13] "
                    f"step={self.step_count}, "
                    f"forward_ramp={forward_ramp:.3f}, "
                    f"reach_ramp={reach_ramp:.3f}, "
                    f"grip_gate={grip_gate:.3f}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"arm_scale={arm_scale:.2f}, "
                    f"arm_clip={arm_clip:.2f}, "
                    f"gripper_cmd={gripper_cmd:.3f}"
                )

        elif debug_mode == "eval_arm_gripper_v14":
            # [수정일: 2026-05-06]
            # [목적]
            # v13에서는 팔/그리퍼를 너무 늦게 풀어서 실제 움직임이 거의 없었다.
            # v14는 팔을 초반부터 바로 활성화하고, gripper 고정을 제거한다.
            #
            # 현재 확인된 base mapping:
            # action[0] = forward/back
            # action[1] = yaw
            # action[2] = lateral
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base: 전진은 맞으므로 약하게 유지
            # ----------------------------
            forward_axis = int(os.environ.get("B1K_FORWARD_AXIS", "0"))
            yaw_axis = int(os.environ.get("B1K_YAW_AXIS", "1"))
            lateral_axis = int(os.environ.get("B1K_LATERAL_AXIS", "2"))

            forward_sign = float(os.environ.get("B1K_FORWARD_SIGN", "1.0"))

            forward_bias = float(os.environ.get("B1K_FORWARD_BIAS", "0.04"))
            forward_scale = float(os.environ.get("B1K_FORWARD_SCALE", "0.28"))
            forward_max = float(os.environ.get("B1K_FORWARD_MAX", "0.22"))

            yaw_scale = float(os.environ.get("B1K_YAW_SCALE", "0.025"))
            yaw_max = float(os.environ.get("B1K_YAW_MAX", "0.018"))

            lateral_scale = float(os.environ.get("B1K_LATERAL_SCALE", "0.20"))
            lateral_max = float(os.environ.get("B1K_LATERAL_MAX", "0.12"))

            planar_max = float(os.environ.get("B1K_PLANAR_MAX", "0.26"))

            forward_ramp_start = float(os.environ.get("B1K_FORWARD_RAMP_START", "160"))
            forward_ramp_len = float(os.environ.get("B1K_FORWARD_RAMP_LEN", "120"))

            forward_ramp = np.clip(
                (float(self.step_count) - forward_ramp_start) / (forward_ramp_len + 1e-6),
                0.0,
                1.0,
            )

            base = np.zeros(3, dtype=np.float32)

            base[forward_axis] = np.clip(
                raw_action[forward_axis] * forward_scale + forward_sign * forward_bias * forward_ramp,
                -forward_max,
                forward_max,
            )

            base[yaw_axis] = np.clip(
                raw_action[yaw_axis] * yaw_scale,
                -yaw_max,
                yaw_max,
            )

            base[lateral_axis] = np.clip(
                raw_action[lateral_axis] * lateral_scale,
                -lateral_max,
                lateral_max,
            )

            planar_norm = np.linalg.norm([base[forward_axis], base[lateral_axis]])
            planar_scale = np.minimum(1.0, planar_max / (planar_norm + 1e-6))
            base[forward_axis] *= planar_scale
            base[lateral_axis] *= planar_scale

            last_base = getattr(self, "_arm_gripper_v14_last_base", np.zeros(3, dtype=np.float32))
            base = 0.72 * last_base + 0.28 * base
            self._arm_gripper_v14_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk: 넘어짐 방지용으로 계속 작게
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.04, -0.08, 0.08)

            # ----------------------------
            # 3) arm: 초반부터 바로 활성화
            # ----------------------------
            arm_scale = float(os.environ.get("B1K_ARM_SCALE", "1.15"))
            arm_clip = float(os.environ.get("B1K_ARM_CLIP", "1.10"))

            left_arm_gain = float(os.environ.get("B1K_LEFT_ARM_GAIN", "1.00"))
            right_arm_gain = float(os.environ.get("B1K_RIGHT_ARM_GAIN", "1.25"))

            current_action[7:14] = np.clip(
                raw_action[7:14] * arm_scale * left_arm_gain,
                -arm_clip,
                arm_clip,
            )

            current_action[15:22] = np.clip(
                raw_action[15:22] * arm_scale * right_arm_gain,
                -arm_clip,
                arm_clip,
            )

            # ----------------------------
            # 4) gripper: 고정 제거, 모델 출력 증폭
            # ----------------------------
            gripper_scale = float(os.environ.get("B1K_GRIPPER_SCALE", "2.50"))
            gripper_max = float(os.environ.get("B1K_GRIPPER_MAX", "0.60"))
            gripper_deadband = float(os.environ.get("B1K_GRIPPER_DEADBAND", "0.02"))

            g14 = raw_action[14] * gripper_scale
            g22 = raw_action[22] * gripper_scale

            if abs(g14) < gripper_deadband:
                g14 = 0.0
            if abs(g22) < gripper_deadband:
                g22 = 0.0

            current_action[14] = np.clip(g14, -gripper_max, gripper_max)
            current_action[22] = np.clip(g22, -gripper_max, gripper_max)

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_arm_gripper_v14] "
                    f"step={self.step_count}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"arm_scale={arm_scale:.2f}, "
                    f"arm_clip={arm_clip:.2f}, "
                    f"raw_g14={raw_action[14]:.3f}, "
                    f"raw_g22={raw_action[22]:.3f}, "
                    f"final_g14={current_action[14]:.3f}, "
                    f"final_g22={current_action[22]:.3f}"
                )

        elif debug_mode == "eval_selected12_v15":
            # [수정일: 2026-05-06]
            # [목적]
            # 12개 task 평가용 공통 action 후처리 모드.
            # radio 전용 if / task별 if 없이 모든 task에 동일하게 적용한다.
            #
            # 현재까지 확인한 base mapping:
            # action[0] = forward/back
            # action[1] = yaw
            # action[2] = lateral
            #
            # action mapping:
            # 0~2   base velocity
            # 3~6   torso/trunk
            # 7~13  left arm
            # 14    left gripper
            # 15~21 right arm
            # 22    right gripper

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base stabilization
            # ----------------------------
            forward_axis = int(os.environ.get("B1K_FORWARD_AXIS", "0"))
            yaw_axis = int(os.environ.get("B1K_YAW_AXIS", "1"))
            lateral_axis = int(os.environ.get("B1K_LATERAL_AXIS", "2"))

            forward_scale = float(os.environ.get("B1K_FORWARD_SCALE", "0.26"))
            forward_max = float(os.environ.get("B1K_FORWARD_MAX", "0.20"))

            yaw_scale = float(os.environ.get("B1K_YAW_SCALE", "0.025"))
            yaw_max = float(os.environ.get("B1K_YAW_MAX", "0.018"))

            lateral_scale = float(os.environ.get("B1K_LATERAL_SCALE", "0.18"))
            lateral_max = float(os.environ.get("B1K_LATERAL_MAX", "0.10"))

            planar_max = float(os.environ.get("B1K_PLANAR_MAX", "0.24"))

            base = np.zeros(3, dtype=np.float32)

            base[forward_axis] = np.clip(
                raw_action[forward_axis] * forward_scale,
                -forward_max,
                forward_max,
            )

            base[yaw_axis] = np.clip(
                raw_action[yaw_axis] * yaw_scale,
                -yaw_max,
                yaw_max,
            )

            base[lateral_axis] = np.clip(
                raw_action[lateral_axis] * lateral_scale,
                -lateral_max,
                lateral_max,
            )

            planar_norm = np.linalg.norm([base[forward_axis], base[lateral_axis]])
            planar_scale = np.minimum(1.0, planar_max / (planar_norm + 1e-6))
            base[forward_axis] *= planar_scale
            base[lateral_axis] *= planar_scale

            last_base = getattr(self, "_selected12_v15_last_base", np.zeros(3, dtype=np.float32))
            base = 0.72 * last_base + 0.28 * base
            self._selected12_v15_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk stabilization
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.04, -0.08, 0.08)

            # ----------------------------
            # 3) arm activation
            # ----------------------------
            arm_scale = float(os.environ.get("B1K_ARM_SCALE", "1.60"))
            arm_clip = float(os.environ.get("B1K_ARM_CLIP", "1.40"))

            left_arm_gain = float(os.environ.get("B1K_LEFT_ARM_GAIN", "1.10"))
            right_arm_gain = float(os.environ.get("B1K_RIGHT_ARM_GAIN", "1.40"))

            current_action[7:14] = np.clip(
                raw_action[7:14] * arm_scale * left_arm_gain,
                -arm_clip,
                arm_clip,
            )

            current_action[15:22] = np.clip(
                raw_action[15:22] * arm_scale * right_arm_gain,
                -arm_clip,
                arm_clip,
            )

            # ----------------------------
            # 4) gripper activation
            # ----------------------------
            gripper_scale = float(os.environ.get("B1K_GRIPPER_SCALE", "3.00"))
            gripper_max = float(os.environ.get("B1K_GRIPPER_MAX", "0.70"))
            gripper_deadband = float(os.environ.get("B1K_GRIPPER_DEADBAND", "0.005"))

            g14 = raw_action[14] * gripper_scale
            g22 = raw_action[22] * gripper_scale

            if abs(g14) < gripper_deadband:
                g14 = 0.0
            if abs(g22) < gripper_deadband:
                g22 = 0.0

            current_action[14] = np.clip(g14, -gripper_max, gripper_max)
            current_action[22] = np.clip(g22, -gripper_max, gripper_max)

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_selected12_v15] "
                    f"step={self.step_count}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"arm_scale={arm_scale:.2f}, "
                    f"arm_clip={arm_clip:.2f}, "
                    f"raw_g14={raw_action[14]:.3f}, "
                    f"raw_g22={raw_action[22]:.3f}, "
                    f"final_g14={current_action[14]:.3f}, "
                    f"final_g22={current_action[22]:.3f}"
                )

        elif debug_mode == "probe_sweep":
            # [수정일: 2026-04-29]
            # [디버그 목적]
            # action index mapping을 찾기 위해 index를 자동으로 바꿔가며
            # 하나의 channel만 강제로 움직인다.
            #
            # 사용 예:
            # export B1K_ACTION_DEBUG_MODE=probe_sweep
            # export B1K_PROBE_START=3
            # export B1K_PROBE_END=23
            # export B1K_PROBE_INTERVAL=50
            # export B1K_PROBE_VALUE=0.3
            #
            # 의미:
            # - 50 step 동안 index 3만 움직임
            # - 다음 50 step 동안 index 4만 움직임
            # - ...
            # - index 22까지 확인
            #
            # 영상에서 어느 index일 때 wrist camera / arm / base / gripper가
            # 움직이는지 확인하기 위한 디버그 모드다.

            current_action[:] = 0.0

            probe_start = int(os.environ.get("B1K_PROBE_START", "3"))
            probe_end = int(os.environ.get("B1K_PROBE_END", "23"))
            probe_interval = int(os.environ.get("B1K_PROBE_INTERVAL", "50"))
            probe_value = float(os.environ.get("B1K_PROBE_VALUE", "0.3"))

            num_probe_channels = max(1, probe_end - probe_start)
            probe_slot = (self.step_count // probe_interval) % num_probe_channels
            probe_index = probe_start + probe_slot

            phase = self.step_count % probe_interval
            sign = 1.0 if phase < (probe_interval // 2) else -1.0

            if 0 <= probe_index < len(current_action):
                current_action[probe_index] = sign * probe_value

            if self.step_count % 20 == 0:
                logger.info(
                    f"[PROBE SWEEP] step={self.step_count}, "
                    f"probe_index={probe_index}, "
                    f"value={current_action[probe_index]:.4f}, "
                    f"range=[{probe_start}, {probe_end}), "
                    f"interval={probe_interval}"
                )

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
                f"raw_left_gripper={raw_action[LEFT_GRIPPER]:.4f}, "
                f"raw_right_gripper={raw_action[RIGHT_GRIPPER]:.4f}, "
                f"final_base={current_action[:3]}, "
                f"final_mid_min={current_action[3:-1].min():.4f}, "
                f"final_mid_max={current_action[3:-1].max():.4f}, "
                f"final_left_gripper={current_action[LEFT_GRIPPER]:.4f}, "
                f"final_right_gripper={current_action[RIGHT_GRIPPER]:.4f}"
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

