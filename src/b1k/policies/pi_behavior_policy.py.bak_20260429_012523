"""PiBehavior 모델 전용 Policy 클래스 (pi_behavior_policy.py)

이 파일의 역할:
    PiBehavior 모델은 sample_actions() 함수가 (actions, subtask_logits) 튜플을 반환한다.
    그러나 openpi의 기본 Policy 클래스는 단순 배열 반환을 기대한다.

    이 클래스는 그 차이를 해결하는 최소한의 래퍼(wrapper)이다.

    주요 추가 기능:
        1. 튜플 언패킹: (actions, subtask_logits) → 각각 분리
        2. initial_actions 지원: 롤링 inpainting을 위해 이전 step의 action을
           초기 추측(initial guess)으로 사용할 수 있다.
        3. subtask_logits 출력: 모델이 예측한 stage 분포를 출력에 포함한다.
        4. predicted_stage: argmax(subtask_logits)를 편의 필드로 추가한다.

사용법:
    policy = PiBehaviorPolicy(
        model=pi_behavior_model,
        ...
    )
    output = policy.infer(obs)
    # output['actions']: 예측된 action [action_horizon, action_dim]
    # output['subtask_logits']: stage 분포 [MAX_NUM_STAGES]
    # output['predicted_stage']: 예측된 stage id (int)
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.policies.policy import Policy
# openpi의 기본 Observation이 아니라 FAST 필드가 추가된 B1K 버전 사용
from b1k.models.observation import Observation


class PiBehaviorPolicy(Policy):
    """PiBehavior 모델을 위한 Policy 클래스.

    openpi의 기본 Policy 클래스에서 딱 하나만 다르다:
        sample_actions()가 (actions, subtask_logits) 튜플을 반환하므로
        이를 언패킹해서 'actions'와 'subtask_logits'로 분리한다.

    그 외 input transform, output transform, 배치 처리는 모두 부모 클래스와 동일하다.
    """

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        initial_actions: np.ndarray | None = None,
    ) -> dict:
        """모델 추론을 실행하고 action과 보조 정보를 반환한다.

        부모 Policy.infer()와의 차이점:
            1. initial_actions 매개변수 지원:
               이전 step에서 예측한 action을 이번 step의 초기 추측으로 사용한다.
               롤링 추론(rolling inference)에서 연속성을 유지하기 위해 사용한다.
            2. (actions, subtask_logits) 튜플 언패킹:
               PiBehavior의 sample_actions()가 튜플을 반환하기 때문에 필요하다.

        처리 흐름:
            1. obs 딕셔너리를 input transform으로 전처리한다.
            2. 배치 차원을 추가한다 (단일 샘플 추론).
            3. initial_actions가 있으면 normalize해서 sample_kwargs에 추가한다.
            4. sample_actions() 호출 → (actions, subtask_logits) 언패킹.
            5. output transform으로 후처리한다.
            6. 편의 필드(predicted_stage, policy_timing)를 추가한다.

        Args:
            obs: 현재 step의 관측값 딕셔너리.
                 키: "observation/egocentric_camera" 등 카메라 이미지,
                     "observation/state" 로봇 상태, "task_index" 등.
            noise: 선택적으로 제공하는 초기 noise.
                   None이면 내부에서 랜덤 생성한다.
            initial_actions: 이전 step에서 예측된 action.
                             None이면 순수 noise에서 시작한다.

        Returns:
            출력 딕셔너리:
                'actions':        예측 action [action_horizon, action_dim] numpy array.
                'subtask_logits': stage 분류 logit [MAX_NUM_STAGES] numpy array.
                'predicted_stage': argmax(subtask_logits) 정수값.
                'state':          입력 state (echo-back).
                'policy_timing':  {'infer_ms': 추론 시간(밀리초)}.
        """
        # 입력을 JAX/NumPy가 처리할 수 있도록 복사한다.
        inputs = jax.tree.map(lambda x: x, obs)

        # input_transform: repack, normalize 등 전처리를 적용한다.
        inputs = self._input_transform(inputs)

        # 단일 샘플을 배치 차원을 가진 배열로 변환한다.
        # [H, W, C] → [1, H, W, C] 등
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        # 난수 키를 분리한다: 이번 스텝용과 다음 스텝용
        self._rng, sample_rng = jax.random.split(self._rng)

        # ── Sample kwargs 준비 ───────────────────────────────────────────
        sample_kwargs = dict(self._sample_kwargs)  # 기본 sample 파라미터 복사

        # noise를 외부에서 제공한 경우 처리한다.
        # flow matching에서 초기 noise를 고정하면 동일한 obs에서 결정론적 결과를 얻을 수 있다.
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                # [H, D] → [1, H, D]: 배치 차원 추가
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        # initial_actions: 롤링 inpainting을 위한 이전 step action
        if initial_actions is not None:
            # initial_actions를 모델 입력과 같은 방식으로 정규화해야 한다.
            # 그러려면 원본 obs 딕셔너리 + actions를 묶어서 transform을 통과시킨다.
            training_obs = {}

            # 키 이름 매핑: 평가 환경 키 → 학습 데이터 키
            # 평가 환경에서는 "observation/state", 학습 데이터는 다른 키를 쓸 수 있다.
            if "observation/state" in obs:
                training_obs["observation/state"] = obs["observation/state"]
            elif "state" in obs:
                training_obs["observation/state"] = obs["state"]

            if "observation/egocentric_camera" in obs:
                training_obs["observation/egocentric_camera"] = obs["observation/egocentric_camera"]
            elif "image" in obs and "base_0_rgb" in obs["image"]:
                training_obs["observation/egocentric_camera"] = obs["image"]["base_0_rgb"]

            if "observation/wrist_image_left" in obs:
                training_obs["observation/wrist_image_left"] = obs["observation/wrist_image_left"]
            elif "image" in obs and "left_wrist_0_rgb" in obs["image"]:
                training_obs["observation/wrist_image_left"] = obs["image"]["left_wrist_0_rgb"]

            if "observation/wrist_image_right" in obs:
                training_obs["observation/wrist_image_right"] = obs["observation/wrist_image_right"]
            elif "image" in obs and "right_wrist_0_rgb" in obs["image"]:
                training_obs["observation/wrist_image_right"] = obs["image"]["right_wrist_0_rgb"]

            # 나머지 키들도 그대로 복사한다 (tokenized_prompt, subtask_state 등).
            for key in obs:
                if key not in training_obs and key not in ["image", "state"]:
                    training_obs[key] = obs[key]

            # actions 필드를 추가해서 transform이 정규화할 수 있게 한다.
            initial_batch = {
                **training_obs,
                "actions": initial_actions,
            }

            # input_transform을 통해 actions도 정규화한다.
            # (normalize: z-score 또는 quantile 정규화)
            transformed_batch = self._input_transform(initial_batch)
            normalized_initial_actions = transformed_batch["actions"]

            # 배치 차원 추가
            initial_actions = jnp.asarray(normalized_initial_actions)
            if initial_actions.ndim == 2:
                initial_actions = initial_actions[None, ...]  # [1, H, D]

            sample_kwargs["initial_actions"] = initial_actions

        # ── 모델 추론 ─────────────────────────────────────────────────────
        # Observation 구조체로 변환한다.
        # FAST 필드가 추가된 B1K 버전의 Observation.from_dict()를 사용한다.
        observation = Observation.from_dict(inputs)

        start_time = time.monotonic()

        # PiBehavior.sample_actions()는 (actions, subtask_logits) 튜플을 반환한다.
        # 이것이 이 클래스가 존재하는 유일한 이유이다.
        # 부모 클래스는 단순 배열 반환을 기대하므로 여기서 언패킹한다.
        actions, subtask_logits = self._sample_actions(sample_rng, observation, **sample_kwargs)

        model_time = time.monotonic() - start_time

        # ── 출력 딕셔너리 구성 ────────────────────────────────────────────
        outputs = {
            "state": inputs["state"],       # echo-back: 입력 state 반환
            "actions": actions,              # 예측 action [1, H, D] JAX array
            "subtask_logits": subtask_logits,  # stage logit [1, MAX_NUM_STAGES] JAX array
        }

        # JAX array를 NumPy array로 변환하고 배치 차원을 제거한다.
        # outputs[k][0]: 배치 차원 제거 ([1, H, D] → [H, D])
        outputs = {
            k: np.asarray(v[0, ...]) if isinstance(v, (jnp.ndarray, np.ndarray)) else v
            for k, v in outputs.items()
        }

        # output_transform: denormalize (z-score 역변환 등) 적용
        outputs = self._output_transform(outputs)

        # 편의 필드 추가: stage 예측 결과를 정수 하나로 요약한다.
        outputs["predicted_stage"] = int(np.argmax(outputs["subtask_logits"]))

        # 추론 시간 기록 (밀리초 단위)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        """Policy 메타데이터를 반환한다."""
        return self._metadata
