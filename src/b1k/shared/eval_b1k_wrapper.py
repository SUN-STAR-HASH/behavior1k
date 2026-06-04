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

# [2026-05-20 추가] 1등팀 wrapper 스타일 로그 필터
class _B1KFirstTeamStyleLogFilter(logging.Filter):
    def filter(self, record):
        try:
            import os as _os
            import re as _re

            if _os.environ.get("B1K_FIRSTTEAM_LOG_STYLE", "1") != "1":
                return True

            msg = record.getMessage()

            # action 수치 로그는 숨김
            noisy_prefixes = (
                "[ACTION DEBUG]",
                "[ACTION_DEBUG]",
                "[DEEP_ACTION_DEBUG]",
                "[DEEP_ACTION_DEBUG_RAW]",
                "[DEEP_ACTION_DEBUG_ACTION]",
                "[DEEP_ACTION_DEBUG_SPLIT]",
                "[DEEP_ACTION_DEBUG_STATS]",
                "[DEEP_ACTION_DEBUG_LAST_ACTIONS]",
                "[DEEP_ACTION_DEBUG_INITIAL]",
                "[eval_1stlike_v17]",
            )
            if any(msg.startswith(p) for p in noisy_prefixes):
                return False

            # 기존 Local task 형식 step 로그는 숨기고, 아래 first-team style progress 로그만 남김
            if msg.startswith("📊 Step ") and "Local task:" in msg:
                return False

            # stage 로그를 1등팀 스타일로 통일
            m = _re.search(
                r"Stage advanced:\s*(\d+)\s*(?:->|→)\s*(\d+).*?(?:global task|task)\s*(\d+).*?step\s*(\d+)",
                msg,
            )
            if m:
                a, b, task, step = m.groups()
                record.msg = f"⬆️  Stage advanced: {a} → {b} (task {task}, step {step})"
                record.args = ()
                return True

            m = _re.search(
                r"Stage went back:\s*(\d+)\s*(?:->|→)\s*(\d+).*?(?:global task|task)\s*(\d+).*?step\s*(\d+)",
                msg,
            )
            if m:
                a, b, task, step = m.groups()
                record.msg = f"⬅️  Stage went back: {a} → {b} (task {task}, step {step})"
                record.args = ()
                return True

            m = _re.search(
                r"Stage skipped:\s*(\d+)\s*(?:->|→)\s*(\d+).*?(?:global task|task)\s*(\d+).*?step\s*(\d+)",
                msg,
            )
            if m:
                a, b, task, step = m.groups()
                record.msg = f"⏭️  Stage skipped: {a} → {b} (task {task}, step {step})"
                record.args = ()
                return True

            return True
        except Exception:
            return True


def _b1k_firstteam_style_progress_log(self):
    """1등팀 wrapper와 같은 위치/형식의 progress log."""
    import os as _os

    if _os.environ.get("B1K_FIRSTTEAM_LOG_STYLE", "1") != "1":
        return

    step = int(getattr(self, "step_count", 0))
    if step <= 0 or step % 100 != 0:
        return

    if getattr(self, "_b1k_last_firstteam_progress_step", None) == step:
        return
    self._b1k_last_firstteam_progress_step = step

    task_id = getattr(self, "task_id", None)
    if task_id is None:
        task_id = getattr(self, "current_task_id", None)
    if task_id is None:
        task_id = getattr(self, "_current_task_id", None)

    stage = getattr(self, "current_stage", None)
    if stage is None:
        stage = getattr(self, "stage", None)
    if stage is None:
        stage = 0

    pred = getattr(self, "prediction_count", None)
    if pred is None:
        pred = getattr(self, "predictions", None)
    if pred is None:
        pred = "?"

    max_stage = None
    try:
        from b1k.models.pi_behavior_config import TASK_NUM_STAGES
        if task_id is not None:
            max_stage = TASK_NUM_STAGES[int(task_id)] - 1
    except Exception:
        max_stage = None

    stage_text = f"{stage}/{max_stage}" if max_stage is not None else f"{stage}/?"

    logger.info(
        f"📊 Step {step} | Task: {task_id} | Stage: {stage_text} | Predictions: {pred}"
    )


if not any(isinstance(f, _B1KFirstTeamStyleLogFilter) for f in logger.filters):
    logger.addFilter(_B1KFirstTeamStyleLogFilter())


# [2026-05-20 추가] 1등팀 스타일 로그 필터 + 진행 로그
class _B1KTeamStyleLogFilter(logging.Filter):
    def filter(self, record):
        try:
            import os as _os
            import re as _re

            if _os.environ.get("B1K_TEAMSTYLE_LOGS", "1") != "1":
                return True

            msg = record.getMessage()

            # action 수치 로그 제거
            noisy_prefixes = (
                "[ACTION DEBUG]",
                "[ACTION_DEBUG]",
                "[DEEP_ACTION_DEBUG]",
                "[DEEP_ACTION_DEBUG_RAW]",
                "[DEEP_ACTION_DEBUG_ACTION]",
                "[DEEP_ACTION_DEBUG_SPLIT]",
                "[DEEP_ACTION_DEBUG_STATS]",
                "[DEEP_ACTION_DEBUG_LAST_ACTIONS]",
                "[DEEP_ACTION_DEBUG_INITIAL]",
                "[eval_1stlike_v17]",
            )
            if any(msg.startswith(p) for p in noisy_prefixes):
                return False

            # 기존 우리 progress 로그는 teamstyle progress helper가 다시 찍게 숨김
            if msg.startswith("📊 Step ") and "Local task:" in msg:
                return False

            # stage 로그를 1등팀 스타일로 변환
            m = _re.search(
                r"Stage advanced:\s*(\d+)\s*(?:->|→)\s*(\d+).*?(?:global task|task)\s*(\d+).*?step\s*(\d+)",
                msg,
            )
            if m:
                a, b, task, step = m.groups()
                record.msg = f"⬆️  Stage advanced: {a} → {b} (task {task}, step {step})"
                record.args = ()
                return True

            m = _re.search(
                r"Stage went back:\s*(\d+)\s*(?:->|→)\s*(\d+).*?(?:global task|task)\s*(\d+).*?step\s*(\d+)",
                msg,
            )
            if m:
                a, b, task, step = m.groups()
                record.msg = f"⬅️  Stage went back: {a} → {b} (task {task}, step {step})"
                record.args = ()
                return True

            m = _re.search(
                r"Stage skipped:\s*(\d+)\s*(?:->|→)\s*(\d+).*?(?:global task|task)\s*(\d+).*?step\s*(\d+)",
                msg,
            )
            if m:
                a, b, task, step = m.groups()
                record.msg = f"⏭️  Stage skipped: {a} → {b} (task {task}, step {step})"
                record.args = ()
                return True

            return True
        except Exception:
            return True


def _b1k_get_attr_any(obj, names, default=None):
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value is not None:
                return value
    return default


def _b1k_teamstyle_progress_log(self):
    """1등팀 스타일 진행 로그.
    예: 📊 Step 100 | Task: 19 | Stage: 0/11 | Predictions: 5
    """
    if os.environ.get("B1K_TEAMSTYLE_LOGS", "1") != "1":
        return

    try:
        every = int(os.environ.get("B1K_TEAMSTYLE_PROGRESS_EVERY", "100"))
    except Exception:
        every = 100

    step = int(getattr(self, "step_count", 0))
    if every > 0 and step % every != 0:
        return

    # 같은 step 중복 출력 방지
    if getattr(self, "_b1k_last_teamstyle_progress_step", None) == step:
        return
    self._b1k_last_teamstyle_progress_step = step

    task_id = _b1k_get_attr_any(
        self,
        ["task_id", "current_task_id", "_current_task_id", "global_task_id", "local_task_id"],
        None,
    )

    stage = _b1k_get_attr_any(
        self,
        ["current_stage", "stage", "_current_stage"],
        0,
    )

    # max stage 후보들. 없으면 ? 로 둔다.
    max_stage = _b1k_get_attr_any(
        self,
        ["max_stage", "max_stages", "num_stages", "n_stages", "_max_stage", "_max_stages"],
        None,
    )

    # task별 stage table이 있는 경우 최대한 읽기
    if max_stage is None:
        for name in ["task_max_stages", "max_stages_by_task", "stage_counts", "_task_max_stages"]:
            if hasattr(self, name):
                table = getattr(self, name)
                try:
                    if task_id in table:
                        max_stage = table[task_id]
                        break
                except Exception:
                    pass

    prediction = _b1k_get_attr_any(
        self,
        ["prediction_count", "predictions", "_prediction_count"],
        None,
    )

    stage_text = f"{stage}/{max_stage}" if max_stage is not None else f"{stage}/?"
    pred_text = prediction if prediction is not None else "?"

    logger.info(
        f"📊 Step {step} | Task: {task_id} | Stage: {stage_text} | Predictions: {pred_text}"
    )


if not any(isinstance(f, _B1KTeamStyleLogFilter) for f in logger.filters):
    logger.addFilter(_B1KTeamStyleLogFilter())


def _b1k_compact_action_debug_log(self, current_action):
    """[2026-05-20 추가] 1등팀 기본 ACTION_DEBUG 스타일의 compact logger.

    출력 예:
    [ACTION_DEBUG] step=50 task=5 norm=... base=[...] arm_mean_abs=... gripper22=...
    """
    if os.environ.get("B1K_COMPACT_ACTION_DEBUG", "1") != "1":
        return

    try:
        every = int(os.environ.get("B1K_COMPACT_ACTION_DEBUG_EVERY", "50"))
    except Exception:
        every = 50

    step = int(getattr(self, "step_count", 0))
    if every > 0 and step % every != 0:
        return

    try:
        action = np.asarray(current_action)

        task_id = getattr(self, "task_id", None)
        if task_id is None:
            task_id = getattr(self, "current_task_id", None)
        if task_id is None:
            task_id = getattr(self, "_current_task_id", None)

        if action.ndim >= 1 and action.shape[0] >= 23:
            arms = np.concatenate([action[7:14], action[15:22]])
            arm_mean_abs = float(np.mean(np.abs(arms)))
            base = np.array2string(action[0:3], precision=4, suppress_small=True)
            gripper22 = float(action[22])
            norm = float(np.linalg.norm(action))

            logger.info(
                f"[ACTION_DEBUG] step={step} task={task_id} "
                f"norm={norm:.6f} base={base} "
                f"arm_mean_abs={arm_mean_abs:.6f} gripper22={gripper22}"
            )
    except Exception as e:
        logger.warning(f"[ACTION_DEBUG] compact log failed: {e}")


def _b1k_deep_action_debug_log(self, current_action, raw_action=None):
    """[2026-05-18 추가] 1등팀 로그 형식에 맞춘 deep action debug logger.

    출력:
    - DEEP_ACTION_DEBUG
    - DEEP_ACTION_DEBUG_RAW
    - DEEP_ACTION_DEBUG_ACTION
    - DEEP_ACTION_DEBUG_SPLIT
    - DEEP_ACTION_DEBUG_STATS
    - DEEP_ACTION_DEBUG_LAST_ACTIONS
    - DEEP_ACTION_DEBUG_INITIAL

    주의:
    - action 자체는 수정하지 않고 로그만 찍는다.
    - self 내부에 last_actions / initial_actions 계열 attribute가 있으면 같이 출력한다.
    """
    if os.environ.get("B1K_DEEP_ACTION_DEBUG", "0") != "1":
        return

    try:
        every = int(os.environ.get("B1K_DEEP_ACTION_DEBUG_EVERY", "20"))
    except Exception:
        every = 20

    step = int(getattr(self, "step_count", 0))
    if every > 0 and step % every != 0:
        return

    try:
        np.set_printoptions(precision=4, suppress=True)

        action = np.asarray(current_action)
        raw = None if raw_action is None else np.asarray(raw_action)

        task_id = getattr(self, "task_id", None)
        if task_id is None:
            task_id = getattr(self, "current_task_id", None)
        if task_id is None:
            task_id = getattr(self, "_current_task_id", None)

        stage = getattr(self, "current_stage", None)
        if stage is None:
            stage = getattr(self, "stage", None)

        prediction = getattr(self, "prediction_count", None)
        if prediction is None:
            prediction = getattr(self, "predictions", None)

        action_index = getattr(self, "action_index", None)
        if action_index is None:
            action_index = getattr(self, "_action_index", 0)

        logger.info(
            "[DEEP_ACTION_DEBUG] "
            f"step={step} task={task_id} stage={stage} "
            f"prediction={prediction} action_index={action_index} "
            f"shape={action.shape}"
        )

        if raw is not None and raw.ndim >= 1 and os.environ.get("B1K_DEEP_LOG_RAW", "1") == "1":
            logger.info(
                "[DEEP_ACTION_DEBUG_RAW] "
                f"raw_0_22={np.array2string(raw[:23], precision=4, suppress_small=True)}"
            )

        logger.info(
            "[DEEP_ACTION_DEBUG_ACTION] "
            f"action_0_22={np.array2string(action[:23], precision=4, suppress_small=True)}"
        )

        if action.ndim >= 1 and action.shape[0] >= 23:
            logger.info(
                "[DEEP_ACTION_DEBUG_SPLIT] "
                f"base_0_1_2={np.array2string(action[0:3], precision=4, suppress_small=True)} "
                f"torso_3_6={np.array2string(action[3:7], precision=4, suppress_small=True)} "
                f"left_arm_7_13={np.array2string(action[7:14], precision=4, suppress_small=True)} "
                f"left_gripper_14={float(action[14]):.4f} "
                f"right_arm_15_21={np.array2string(action[15:22], precision=4, suppress_small=True)} "
                f"right_gripper_22={float(action[22]):.4f}"
            )

            logger.info(
                "[DEEP_ACTION_DEBUG_STATS] "
                f"base_min={float(action[0:3].min()):.4f} "
                f"base_max={float(action[0:3].max()):.4f} "
                f"left_arm_min={float(action[7:14].min()):.4f} "
                f"left_arm_max={float(action[7:14].max()):.4f} "
                f"right_arm_min={float(action[15:22].min()):.4f} "
                f"right_arm_max={float(action[15:22].max()):.4f} "
                f"all_min={float(action.min()):.4f} "
                f"all_max={float(action.max()):.4f} "
                f"norm={float(np.linalg.norm(action)):.4f}"
            )

        # 1등팀 로그의 last_actions_shape / first / current / last 최대한 따라가기
        last_actions = None
        for name in (
            "last_actions",
            "_last_actions",
            "actions",
            "_actions",
            "action_queue",
            "_action_queue",
            "action_buffer",
            "_action_buffer",
            "last_action_chunk",
            "_last_action_chunk",
        ):
            if hasattr(self, name):
                candidate = getattr(self, name)
                if candidate is not None:
                    arr = np.asarray(candidate)
                    if arr.ndim == 2 and arr.shape[-1] >= 23 and arr.shape[0] > 0:
                        last_actions = arr
                        break

        if last_actions is not None:
            try:
                idx = int(action_index)
            except Exception:
                idx = 0
            idx = max(0, min(idx, last_actions.shape[0] - 1))

            logger.info(
                "[DEEP_ACTION_DEBUG_LAST_ACTIONS] "
                f"last_actions_shape={last_actions.shape} "
                f"first={np.array2string(last_actions[0, :23], precision=4, suppress_small=True)} "
                f"current={np.array2string(last_actions[idx, :23], precision=4, suppress_small=True)} "
                f"last={np.array2string(last_actions[-1, :23], precision=4, suppress_small=True)}"
            )

        # 1등팀 로그의 next_initial_actions_shape / tail_keep 형식
        initial_actions = None
        for name in (
            "initial_actions",
            "_initial_actions",
            "next_initial_actions",
            "_next_initial_actions",
        ):
            if hasattr(self, name):
                candidate = getattr(self, name)
                if candidate is not None:
                    arr = np.asarray(candidate)
                    if arr.ndim == 2 and arr.shape[-1] >= 23 and arr.shape[0] > 0:
                        initial_actions = arr
                        break

        if initial_actions is not None:
            keep = min(4, initial_actions.shape[0])
            logger.info(
                "[DEEP_ACTION_DEBUG_INITIAL] "
                f"next_initial_actions_shape={initial_actions.shape} "
                f"tail_keep={np.array2string(initial_actions[-keep:, :23], precision=4, suppress_small=True)}"
            )

    except Exception as e:
        logger.warning(f"[DEEP_ACTION_DEBUG] failed: {e}")


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
    # [2026-05-18 수정]
    # behavior-v2 / 1st-team-like checkpoint 평가용 기본 rolling 설정.
    # 30 horizon 중 앞 26개를 실행 후보로 쓰고, 뒤 4개는 다음 inference의 initial_actions로 넘긴다.
    # 26개 action을 20 simulator step에 압축 실행해서 1등팀 wrapper 쪽 rolling/inpainting 구조에 가깝게 맞춘다.
    actions_to_execute: int = 26
    actions_to_keep: int = 4
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

        elif debug_mode == "eval_radio_arm_sign_v16":
            # [수정일: 2026-05-08]
            # [목적]
            # Turning On Radio task에서 base는 절대 건드리지 않고,
            # probe로 확인된 팔 앞/뒤 축만 보정한다.
            #
            # 확인된 probe 결과:
            # - action[7]  = left arm forward/backward axis
            # - action[15] = right arm forward/backward axis
            # - +1을 주면 팔이 뒤로 뻗음
            #
            # 따라서 이 모드는 action[7], action[15]만 sign flip한다.
            # action[0:3] base는 원본 current_action 그대로 유지한다.
            raw_action = current_action.copy()

            left_fb_sign = float(os.environ.get("B1K_LEFT_FB_SIGN", "-1.0"))
            right_fb_sign = float(os.environ.get("B1K_RIGHT_FB_SIGN", "-1.0"))
            fb_clip = float(os.environ.get("B1K_ARM_FB_CLIP", "0.45"))
            fb_bias = float(os.environ.get("B1K_ARM_FB_BIAS", "0.0"))

            # base는 건드리지 않음:
            # current_action[0:3] = raw_action[0:3] 그대로 유지

            # 팔 앞/뒤 축만 보정
            # +가 backward였으므로 기본 sign=-1.0으로 뒤집는다.
            current_action[7] = np.clip(
                raw_action[7] * left_fb_sign + fb_bias,
                -fb_clip,
                fb_clip,
            )
            current_action[15] = np.clip(
                raw_action[15] * right_fb_sign + fb_bias,
                -fb_clip,
                fb_clip,
            )

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_radio_arm_sign_v16] "
                    f"step={self.step_count}, "
                    f"base_raw_kept=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"raw_7={raw_action[7]:.3f}, final_7={current_action[7]:.3f}, "
                    f"raw_15={raw_action[15]:.3f}, final_15={current_action[15]:.3f}, "
                    f"fb_clip={fb_clip:.3f}, fb_bias={fb_bias:.3f}"
                )

        elif debug_mode == "eval_radio_v16":
            # [수정일: 2026-05-08]
            # [목적]
            # 수요일에 안 넘어지게 만든 eval_selected12_v15의 base 안정화는 그대로 사용하고,
            # probe로 확인한 팔 앞/뒤 축 7, 15만 sign flip한다.
            #
            # 확인된 사항:
            # - action[0] = forward/back
            # - action[1] = yaw
            # - action[2] = lateral
            # - action[7] = left arm forward/backward axis
            # - action[15] = right arm forward/backward axis
            # - 7, 15에 +1을 주면 팔이 뒤로 뻗음
            #
            # 따라서:
            # - base 안정화는 eval_selected12_v15와 동일하게 유지
            # - torso/trunk 안정화도 동일하게 유지
            # - arm 전체 gain도 동일하게 유지
            # - 단, 7번/15번만 sign flip

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base stabilization
            #    eval_selected12_v15와 동일
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

            last_base = getattr(self, "_radio_v16_last_base", np.zeros(3, dtype=np.float32))
            base = 0.72 * last_base + 0.28 * base
            self._radio_v16_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk stabilization
            #    eval_selected12_v15와 동일
            # ----------------------------
            current_action[3:7] = np.clip(raw_action[3:7] * 0.04, -0.08, 0.08)

            # ----------------------------
            # 3) arm activation
            #    eval_selected12_v15와 동일 gain을 쓰되,
            #    7번/15번 forward-backward 축만 sign flip
            # ----------------------------
            arm_scale = float(os.environ.get("B1K_ARM_SCALE", "1.60"))
            arm_clip = float(os.environ.get("B1K_ARM_CLIP", "1.40"))
            left_arm_gain = float(os.environ.get("B1K_LEFT_ARM_GAIN", "1.10"))
            right_arm_gain = float(os.environ.get("B1K_RIGHT_ARM_GAIN", "1.40"))

            left_arm = raw_action[7:14].copy()
            right_arm = raw_action[15:22].copy()

            # 핵심 수정:
            # probe 결과 + 방향이 backward였으므로 앞쪽으로 보내기 위해 sign flip
            left_arm[0] *= -1.0     # global action index 7
            right_arm[0] *= -1.0    # global action index 15

            current_action[7:14] = np.clip(
                left_arm * arm_scale * left_arm_gain,
                -arm_clip,
                arm_clip,
            )
            current_action[15:22] = np.clip(
                right_arm * arm_scale * right_arm_gain,
                -arm_clip,
                arm_clip,
            )

            # ----------------------------
            # 4) gripper activation
            #    eval_selected12_v15와 동일
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
            # [2026-05-13] v16_yawfix: repeated spinning suppression
            base_caps = np.asarray([
                float(os.environ.get("B1K_V16_BASE0_MAX", "0.025")),
                float(os.environ.get("B1K_V16_BASE1_MAX", "0.050")),
                float(os.environ.get("B1K_V16_BASE2_MAX", "0.0015")),
            ], dtype=np.float32)
            base_dead = np.asarray([
                float(os.environ.get("B1K_V16_BASE0_DEAD", "0.006")),
                float(os.environ.get("B1K_V16_BASE1_DEAD", "0.008")),
                float(os.environ.get("B1K_V16_BASE2_DEAD", "0.001")),
            ], dtype=np.float32)
            base_alpha = float(os.environ.get("B1K_V16_BASE_EMA_ALPHA", "0.10"))
            base = np.asarray(current_action[:3], dtype=np.float32).copy()
            base = np.clip(base, -base_caps, base_caps)
            base[np.abs(base) < base_dead] = 0.0
            if not hasattr(self, "_eval_radio_base_ema"):
                self._eval_radio_base_ema = np.zeros(3, dtype=np.float32)
            self._eval_radio_base_ema = (1.0 - base_alpha) * self._eval_radio_base_ema + base_alpha * base
            current_action[:3] = self._eval_radio_base_ema

            if self.step_count % 20 == 0:
                logger.info(
                    f"[eval_radio_v16] "
                    f"step={self.step_count}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"raw_7={raw_action[7]:.3f}, final_7={current_action[7]:.3f}, "
                    f"raw_15={raw_action[15]:.3f}, final_15={current_action[15]:.3f}, "
                    f"g14={current_action[14]:.3f}, "
                    f"g22={current_action[22]:.3f}"
                )

        elif debug_mode == "eval_1stlike_v17":
            # [2026-05-18 추가]
            # [목적]
            # behavior-v2 / 1st-team-like checkpoint용 action 후처리.
            #
            # 현재 관찰:
            # - arm은 너무 크게 튀고 있음
            # - yaw는 거의 죽어 있음
            # - behavior-v2 checkpoint는 correlated noise / FAST aux / KV transform 등
            #   1등팀 계열 요소가 들어갔으므로 wrapper도 rolling/inpainting을 더 살리는 쪽이 맞음
            #
            # 방향:
            # - base/yaw를 eval_selected12_v15/v16보다 더 살림
            # - arm gain/clip은 v15/v16보다 낮춤
            # - gripper는 완전 고정하지 않고 약하게만 허용
            # - task-specific if 없이 12 task 공통으로 사용 가능하게 둠

            raw_action = current_action.copy()

            # ----------------------------
            # 1) base / yaw / lateral
            # ----------------------------
            # 현재 코드 주석 기준:
            # action[0] = forward/back
            # action[1] = yaw
            # action[2] = lateral
            #
            # 실제 축 매핑이 완전히 확정된 것은 아니므로,
            # 나중에 필요하면 환경변수로 axis만 바꿔서 다시 테스트할 수 있게 둔다.
            forward_axis = int(os.environ.get("B1K_FORWARD_AXIS", "0"))
            yaw_axis = int(os.environ.get("B1K_YAW_AXIS", "1"))
            lateral_axis = int(os.environ.get("B1K_LATERAL_AXIS", "2"))

            forward_scale = float(os.environ.get("B1K_FORWARD_SCALE", "0.38"))
            forward_max = float(os.environ.get("B1K_FORWARD_MAX", "0.30"))

            # v15/v16의 yaw_max=0.018 수준은 너무 작아서 라디오 탐색이 죽는 것으로 보임.
            # 그래서 yaw를 크게 살리되, smoothing과 clip으로 넘어짐을 억제한다.
            yaw_scale = float(os.environ.get("B1K_YAW_SCALE", "0.22"))
            yaw_max = float(os.environ.get("B1K_YAW_MAX", "0.14"))

            lateral_scale = float(os.environ.get("B1K_LATERAL_SCALE", "0.22"))
            lateral_max = float(os.environ.get("B1K_LATERAL_MAX", "0.16"))

            planar_max = float(os.environ.get("B1K_PLANAR_MAX", "0.38"))

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

            # forward/lateral 평면 속도 제한
            planar_norm = np.linalg.norm([base[forward_axis], base[lateral_axis]])
            if planar_norm > planar_max:
                base[forward_axis] = base[forward_axis] / (planar_norm + 1e-6) * planar_max
                base[lateral_axis] = base[lateral_axis] / (planar_norm + 1e-6) * planar_max

            # base smoothing
            # step_count==0이면 새 task/eval 시작으로 보고 smoothing state 초기화
            if self.step_count == 0 or not hasattr(self, "_firstlike_v17_last_base"):
                self._firstlike_v17_last_base = np.zeros(3, dtype=np.float32)

            base_smooth_prev = float(os.environ.get("B1K_BASE_SMOOTH_PREV", "0.60"))
            base_smooth_new = 1.0 - base_smooth_prev
            base = base_smooth_prev * self._firstlike_v17_last_base + base_smooth_new * base
            self._firstlike_v17_last_base = base.copy()

            current_action[0:3] = base

            # ----------------------------
            # 2) torso/trunk 안정화
            # ----------------------------
            torso_scale = float(os.environ.get("B1K_TORSO_SCALE", "0.04"))
            torso_clip = float(os.environ.get("B1K_TORSO_CLIP", "0.08"))
            current_action[3:7] = np.clip(
                raw_action[3:7] * torso_scale,
                -torso_clip,
                torso_clip,
            )

            # ----------------------------
            # 3) arms: v15/v16보다 줄임
            # ----------------------------
            # 기존 v15/v16은 arm_scale=1.60, clip=1.40이라 현재 영상처럼 팔 폭주가 생길 수 있음.
            # v17은 arm을 적당히 살리되, 먼저 폭주를 줄이는 방향.
            arm_scale = float(os.environ.get("B1K_ARM_SCALE", "0.65"))
            arm_clip = float(os.environ.get("B1K_ARM_CLIP", "0.75"))

            left_arm_gain = float(os.environ.get("B1K_LEFT_ARM_GAIN", "1.00"))
            right_arm_gain = float(os.environ.get("B1K_RIGHT_ARM_GAIN", "1.00"))

            left_arm = raw_action[7:14].copy()
            right_arm = raw_action[15:22].copy()

            # radio task에서 확인했던 팔 앞/뒤 축 sign flip은 옵션으로만 둔다.
            # 필요하면 A100 서버 실행 전에 B1K_FLIP_ARM_FB=1 로 켠다.
            if os.environ.get("B1K_FLIP_ARM_FB", "0") == "1":
                left_arm[0] *= -1.0   # global action index 7
                right_arm[0] *= -1.0  # global action index 15

            current_action[7:14] = np.clip(
                left_arm * arm_scale * left_arm_gain,
                -arm_clip,
                arm_clip,
            )
            current_action[15:22] = np.clip(
                right_arm * arm_scale * right_arm_gain,
                -arm_clip,
                arm_clip,
            )

            # ----------------------------
            # 4) grippers: 완전 고정하지 않고 약하게 허용
            # ----------------------------
            gripper_scale = float(os.environ.get("B1K_GRIPPER_SCALE", "1.50"))
            gripper_max = float(os.environ.get("B1K_GRIPPER_MAX", "0.50"))
            gripper_deadband = float(os.environ.get("B1K_GRIPPER_DEADBAND", "0.02"))

            g14 = raw_action[14] * gripper_scale
            g22 = raw_action[22] * gripper_scale

            if abs(g14) < gripper_deadband:
                g14 = 0.0
            if abs(g22) < gripper_deadband:
                g22 = 0.0

            current_action[14] = np.clip(g14, -gripper_max, gripper_max)
            current_action[22] = np.clip(g22, -gripper_max, gripper_max)


            if os.environ.get("B1K_VERBOSE_ACTION_DEBUG", "0") == "1" and self.step_count % int(os.environ.get("B1K_VERBOSE_ACTION_DEBUG_EVERY", "20")) == 0:
                logger.info(
                    f"[eval_1stlike_v17] "
                    f"step={self.step_count}, "
                    f"base=({current_action[0]:.3f}, {current_action[1]:.3f}, {current_action[2]:.3f}), "
                    f"raw_base=({raw_action[0]:.3f}, {raw_action[1]:.3f}, {raw_action[2]:.3f}), "
                    f"forward_axis={forward_axis}, yaw_axis={yaw_axis}, lateral_axis={lateral_axis}, "
                    f"arm_scale={arm_scale:.2f}, arm_clip={arm_clip:.2f}, "
                    f"raw_g14={raw_action[14]:.3f}, raw_g22={raw_action[22]:.3f}, "
                    f"final_g14={current_action[14]:.3f}, final_g22={current_action[22]:.3f}"
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
        if os.environ.get("B1K_VERBOSE_ACTION_DEBUG", "0") == "1" and self.step_count % int(os.environ.get("B1K_VERBOSE_ACTION_DEBUG_EVERY", "20")) == 0:
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
        
        # Log progress every 100 steps - 1st team style
        if self.step_count % 100 == 0:
            try:
                if self.local_task_id is not None:
                    max_stage = TASK_NUM_STAGES[int(self.local_task_id)] - 1
                else:
                    max_stage = "?"
            except Exception:
                max_stage = "?"

            logger.info(
                f"📊 Step {self.step_count} | Task: {self.task_id} | "
                f"Stage: {self.current_stage}/{max_stage} | "
                f"Predictions: {self.prediction_count}"
            )
        
        # Convert to torch tensor
        action_tensor = torch.from_numpy(current_action).float()
        if len(action_tensor) > 23:
            action_tensor = action_tensor[:23]
        
        return action_tensor

