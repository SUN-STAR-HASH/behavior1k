"""BEHAVIOR-1K 핵심 모델 정의 (pi_behavior.py)

이 파일의 역할:
    PiBehavior 모델 클래스를 정의한다.
    Pi0(pi05_base)를 기반으로 다음 기능을 추가했다.

        1. Task embedding:
           텍스트 프롬프트 대신 학습 가능한 task embedding 벡터를 사용한다.
           num_tasks=12개 태스크를 각각 고정된 크기의 벡터로 표현한다.

        2. Stage conditioning:
           각 태스크 안에서 현재 subtask(stage)를 조건으로 넣을 수 있다.
           stage 정보는 sin/cos positional encoding + task-stage embedding 테이블을 통해
           여러 종류의 조건 벡터를 만든다.

        3. Flow matching:
           Pi0과 동일한 linear ODE 기반 flow matching으로 action을 예측한다.
           노이즈 action x_t → clean action x_0 방향으로 복원한다.

        4. Pi0 가중치 재사용:
           PaliGemma, SigLIP, ActionExpert 가중치는 pi0 체크포인트에서 가져온다.
           task embedding, stage head 등 신규 파라미터만 랜덤 초기화한다.

전체 forward pass 흐름:
    [이미지] ──→ SigLIP ──→ 이미지 토큰
    [task id] ─→ task_embeddings ──→ task 토큰        ┐
    [stage id] → task_stage_embeddings ──→            │
                 fuse_task_and_subtask ──→ stage 토큰  │ prefix
    [state] ───→ discretize + embed ──→ 상태 토큰     ┘
         ↓ (모두 concatenate → PaliGemma forward → KV cache 생성)
    [노이즈 action + timestep t] ──→ embed_suffix ──→ suffix 토큰
         ↓ (KV cache 재사용하며 suffix forward)
    [예측된 velocity v_t]
         ↓ (ODE: x_{t-dt} = x_t - dt × v_t, 반복)
    [clean action x_0] ← 최종 출력
"""

import logging
import pathlib
import os #[4/9 추가]

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.models import siglip as _siglip
from openpi.models.pi0 import make_attn_mask, posemb_sincos
from openpi.shared import array_typing as at

# 이 프로젝트에서 정의한 모듈들
from b1k.models import pi_behavior_config
from b1k.models.observation import Observation, preprocess_observation
from b1k.models.pi_behavior_config import (
    TASK_NUM_STAGES,               # 태스크별 stage 수 튜플 (길이 12)
    MAX_NUM_STAGES,                # 가장 많은 stage 수 (= 15)
    TOTAL_TASK_STAGE_EMBEDDINGS,   # 모든 (task, stage) 쌍의 총 개수 (= 143)
    TASK_STAGE_OFFSETS,            # 태스크별 stage embedding 시작 오프셋
)

logger = logging.getLogger("b1k")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# KV Cache 변환 모듈
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class KVCacheTransform(nnx.Module):
    """여러 VLM 레이어의 KV cache를 선형 결합해서 새로운 KV cache를 만드는 모듈.

    기본 Pi0의 attention 구조:
        action expert layer i → VLM layer i의 KV cache 참조 (1:1 대응)

    이 모듈을 쓰면:
        action expert layer i → 여러 VLM layer들의 KV cache 가중합 참조

    왜 이렇게 하는가?
        - VLM의 각 레이어는 서로 다른 추상화 수준의 정보를 가진다.
          (얕은 레이어: 저수준 시각 특징, 깊은 레이어: 고수준 의미 정보)
        - action expert가 한 레이어만 보는 것보다, 여러 레이어를 유연하게 조합하면
          더 풍부한 정보를 참조할 수 있다.

    초기화 전략 (identity 초기화):
        - k_coeffs, v_coeffs를 항등행렬(I)로 초기화한다.
        - 이렇게 하면 학습 초반에는 "없는 것과 동일한" 동작에서 시작한다.
        - 즉, 처음에는 기존 Pi0와 동일하게 작동하다가,
          점점 유용한 mixing을 학습한다.

    Attributes:
        k_coeffs: K 변환 가중치 [num_layers, num_layers]. 초기값: 단위행렬.
        k_bias:   K 변환 편향 [num_layers, num_kv_heads, head_dim]. 초기값: 0.
        v_coeffs: V 변환 가중치 [num_layers, num_layers]. K와 독립적으로 학습.
        v_bias:   V 변환 편향 [num_layers, num_kv_heads, head_dim]. 초기값: 0.
    """

    def __init__(
        self,
        num_layers: int,        # VLM 레이어 수 (= PaliGemma depth)
        head_dim: int,          # attention head 1개의 차원
        num_kv_heads: int,      # KV head 수 (grouped query attention의 경우 < num_heads)
        rngs: nnx.Rngs,
    ):
        # K 변환 가중치: [도착 레이어 수, 원본 레이어 수]
        # k_new[dst] = Σ_src k_coeffs[dst, src] × cache_k[src]
        # identity 초기화: 처음에는 dst 레이어가 동일한 src 레이어만 참조한다.
        self.k_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))

        # K 편향: [레이어, KV head 수, head 차원]
        # 0으로 초기화해서 처음에는 편향이 없다.
        self.k_bias = nnx.Param(
            jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32)
        )

        # V 변환은 K와 별도로 독립적으로 학습한다.
        # K와 V는 역할이 달라서(K: 위치 결정, V: 내용 추출) 각각 다른 mixing이 유리하다.
        self.v_coeffs = nnx.Param(jnp.eye(num_layers, dtype=jnp.float32))
        self.v_bias = nnx.Param(
            jnp.zeros((num_layers, num_kv_heads, head_dim), dtype=jnp.float32)
        )

    def __call__(
        self,
        kv_cache: tuple[jnp.ndarray, jnp.ndarray],
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """여러 레이어의 KV cache를 선형 결합해서 새로운 KV cache를 만든다.

        수식:
            k_new[dst, b, t, h, d] = Σ_src k_coeffs[dst, src] × cache_k[src, b, t, h, d]
                                       + k_bias[dst, h, d]
            v_new: 동일한 방식으로 v_coeffs, v_bias 사용.

        Args:
            kv_cache: (cache_k, cache_v) 튜플.
                각각의 shape: [num_layers, batch, seq_len, num_kv_heads, head_dim]

        Returns:
            변환된 (k_new, v_new). 입력과 동일한 shape와 dtype.
        """
        cache_k, cache_v = kv_cache
        # shape: [레이어 수, 배치, 시퀀스 길이, KV head 수, head 차원]

        # bfloat16 등 dtype이 유지되도록 원래 dtype을 기억해 둔다.
        # einsum은 float32로 계산하는 경우가 많으므로 마지막에 원래 dtype으로 복원한다.
        original_dtype = cache_k.dtype

        # K 변환:
        # 'ds,sbtkh->dbtkh'의 의미:
        #   d = 도착 레이어, s = 원본 레이어, b = 배치, t = 시퀀스, k = KV head, h = head 차원
        #   각 도착 레이어 d는 모든 원본 레이어 s의 가중합으로 계산된다.
        k_new = jnp.einsum('ds,sbtkh->dbtkh', self.k_coeffs.value, cache_k)
        # 편향 추가: [레이어, head, dim] → broadcast to [레이어, 배치, 시퀀스, head, dim]
        k_new = k_new + self.k_bias.value[:, None, None, :, :]

        # V 변환: K와 동일한 방식이지만 v_coeffs, v_bias를 사용한다.
        v_new = jnp.einsum('ds,sbtkh->dbtkh', self.v_coeffs.value, cache_v)
        v_new = v_new + self.v_bias.value[:, None, None, :, :]

        # 원래 dtype으로 복원 (bfloat16 학습 시 필요)
        k_new = k_new.astype(original_dtype)
        v_new = v_new.astype(original_dtype)

        return (k_new, v_new)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 메인 모델 클래스
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PiBehavior(_model.BaseModel):
    """BEHAVIOR-1K 전용 로봇 정책 모델.

    Pi0(pi05_base)를 기반으로 task embedding과 flow matching을 추가한 버전이다.

    모델 구성:
        PaliGemma:
            llm: Gemma 2B (prefix) + Gemma 300M action expert (suffix)
                 두 모델이 같은 Forward 함수 안에서 처리되지만,
                 adaRMS conditioning은 action expert 쪽에만 적용된다.
            img: SigLIP vision encoder. 이미지를 시각 토큰으로 변환한다.

        task_embeddings:
            num_tasks × task_embedding_dim 크기의 embedding 테이블.
            각 행이 하나의 태스크를 나타낸다.
            pi0 체크포인트에는 없으므로 랜덤 초기화된다.

        task_stage_embeddings:
            TOTAL_TASK_STAGE_EMBEDDINGS × (task_embedding_dim/2) 크기.
            각 (task, stage) 쌍마다 고유한 embedding을 가진다.

        stage_pred_from_vlm:
            VLM 출력 → stage 분류 head.
            VLM이 base task 토큰에서 현재 stage를 예측하도록 보조 학습한다.

        gate_sincos, gate_task_stage, gate_task:
            task + subtask 정보를 섞을 때 각 성분의 기여도를 학습하는 게이트.

        fusion_layer1, fusion_layer2:
            task + stage 정보를 합쳐 하나의 조건 벡터로 만드는 2층 MLP.

        action_in_proj, action_out_proj:
            action 차원 ↔ action expert hidden 차원 변환.

        time_mlp_in, time_mlp_out:
            timestep → adaRMS conditioning 벡터 변환 MLP.
    """

    def __init__(self, config: pi_behavior_config.PiBehaviorConfig, rngs: nnx.Rngs):
        """모델 구성요소를 초기화한다.

        Args:
            config: 모든 하이퍼파라미터를 담은 PiBehaviorConfig 인스턴스.
            rngs: 파라미터 초기화에 사용할 난수 키들.
        """
        # BaseModel 초기화: action_dim, action_horizon, max_token_len을 저장한다.
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)

        # 나중에 여러 메서드에서 참조할 설정을 인스턴스 변수로 저장한다.
        self.config = config

        # PaliGemma와 ActionExpert의 Transformer 설정을 가져온다.
        # 레이어 수, hidden 차원, head 수 등이 여기서 결정된다.
        paligemma_config = _gemma.get_config(config.paligemma_variant)   # gemma_2b 설정
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # gemma_300m 설정

        # ── PaliGemma (LLM + Vision) 초기화 ────────────────────────────────
        # flax.linen 모듈을 nnx로 감싸는 브릿지 패턴이다.
        # Pi0 코드베이스가 linen API로 작성되어 있어서 이 방식이 필요하다.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                # embed_dtype: embedding 연산에 사용할 dtype (bfloat16)
                embed_dtype=config.dtype,
                # adarms=True: action expert 쪽에 adaRMS normalization을 활성화한다.
                #   adaRMS: timestep 조건(t)으로 LayerNorm의 scale/shift를 조절한다.
                #   이를 통해 "현재 diffusion step t에서 어느 정도 노이즈가 있는가"를
                #   각 레이어에 직접 알려줄 수 있다.
                adarms=True,
            )
        )
        # lazy_init: 실제 데이터가 들어오는 첫 forward 시점에 파라미터를 초기화한다.
        # use_adarms=[False, True]: PaliGemma(False)는 adaRMS 없이, action expert(True)는 있이 초기화.
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])

        # SigLIP vision encoder 초기화
        # So400m/14: 400M 파라미터, 14×14 patch 크기의 SigLIP 변형
        # pool_type="none": 전역 풀링 없이 모든 패치 토큰을 유지한다.
        #   224×224 이미지 → (224/14)² = 256개 패치 토큰
        # scan=True: XLA scan을 사용해 레이어를 효율적으로 연산한다.
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,  # 출력 차원을 PaliGemma hidden dim에 맞춤
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        # fake_obs(): 초기화용 더미 이미지 배열. 실제 데이터를 필요로 하지 않는다.
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # PaliGemma를 nnx.Dict로 묶는다.
        # "PaliGemma.llm"과 "PaliGemma.img"로 접근한다.
        # 이렇게 묶으면 체크포인트 저장/로딩 시 같은 경로 prefix 아래 관리된다.
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # ── KV cache 변환 모듈 (선택적) ────────────────────────────────────
        # use_kv_transform=True일 때만 생성한다.
        # 이 모듈이 있으면 action expert가 여러 VLM 레이어를 조합해서 참조할 수 있다.
        if config.use_kv_transform:
            self.kv_transform = KVCacheTransform(
                num_layers=paligemma_config.depth,     # VLM 레이어 수
                head_dim=paligemma_config.head_dim,    # attention head 차원
                num_kv_heads=paligemma_config.num_kv_heads,  # KV head 수
                rngs=rngs,
            )
        else:
            self.kv_transform = None

        # ── Task embedding 테이블 ────────────────────────────────────────────
        # 12개 태스크를 각각 하나의 학습 가능한 벡터로 표현한다.
        # 텍스트 프롬프트 대신 이 벡터가 "어떤 태스크를 수행하는가"를 나타낸다.
        # shape: [num_tasks=12, task_embedding_dim=2048]
        #
        # 왜 dim을 PaliGemma hidden width와 맞추는가?
        # → 추가 projection 없이 바로 prefix 토큰으로 사용할 수 있다.
        self.task_embeddings = nnx.Embed(
            num_embeddings=config.num_tasks,         # 12개 태스크
            features=config.task_embedding_dim,       # 2048 차원 (PaliGemma width)
            rngs=rngs,
        )

        # ── Stage prediction head ────────────────────────────────────────────
        # VLM의 base task 토큰 출력으로부터 현재 stage를 분류하는 head.
        # 출력은 MAX_NUM_STAGES(=15)개의 logit이다.
        # 해당 태스크에 없는 stage는 -inf masking으로 처리한다.
        #
        # 보조 손실(subtask_loss)로 학습한다.
        # subtask_loss_weight=0.0이면 이 head는 학습되지 않는다.
        self.stage_pred_from_vlm = nnx.Linear(
            paligemma_config.width,  # VLM hidden 차원 입력
            MAX_NUM_STAGES,          # stage 수 출력
            rngs=rngs,
        )

        # ── Stage encoding 차원 ──────────────────────────────────────────────
        # sin/cos encoding과 task-stage embedding의 차원.
        # task_embedding_dim(2048)의 절반으로 설정해,
        # 나중에 두 개를 concatenate하면 task_embedding_dim과 같아진다.
        self.subtask_encoding_dim = config.task_embedding_dim // 2  # 1024

        # ── Task-stage embedding 테이블 ──────────────────────────────────────
        # 각 (task, stage) 쌍마다 고유한 embedding 벡터를 가진다.
        # 태스크를 가로질러 같은 번호 stage가 공유되지 않는다.
        # (예: 태스크 0의 stage 2 ≠ 태스크 1의 stage 2)
        # shape: [TOTAL_TASK_STAGE_EMBEDDINGS=143, subtask_encoding_dim=1024]
        self.task_stage_embeddings = nnx.Embed(
            num_embeddings=TOTAL_TASK_STAGE_EMBEDDINGS,  # 12개 태스크의 stage 합 = 143
            features=self.subtask_encoding_dim,           # 1024 차원
            rngs=rngs,
        )

        # ── Task + Subtask 정보 fusion 게이트와 MLP ──────────────────────────
        # fusion 입력 차원 계산:
        #   task_embedding (2048) + sin/cos encoding (1024) + task_stage_embedding (1024)
        #   = 4096 차원
        fusion_input_dim = config.task_embedding_dim + 2 * self.subtask_encoding_dim  # 4096

        # 게이트들: 각 성분의 기여도를 0~1 범위로 학습한다.
        # sigmoid 활성화 함수로 출력이 0~1이 되게 한다.

        # sin/cos encoding에 적용할 게이트: [4096 → 1024]
        self.gate_sincos = nnx.Linear(fusion_input_dim, self.subtask_encoding_dim, rngs=rngs)

        # task-stage embedding에 적용할 게이트: [4096 → 1024]
        self.gate_task_stage = nnx.Linear(fusion_input_dim, self.subtask_encoding_dim, rngs=rngs)

        # task embedding 자체에 적용할 게이트: [4096 → 2048]
        self.gate_task = nnx.Linear(fusion_input_dim, config.task_embedding_dim, rngs=rngs)

        # 균형 잡힌 fusion을 위한 2층 MLP:
        # 4096 → 4096(relu) → 2048
        self.fusion_layer1 = nnx.Linear(fusion_input_dim, config.task_embedding_dim * 2, rngs=rngs)
        self.fusion_layer2 = nnx.Linear(config.task_embedding_dim * 2, config.task_embedding_dim, rngs=rngs)

        # stage 위주 표현을 만드는 projection: 2048 → 2048
        # sin/cos + task-stage 두 벡터(각 1024)를 합쳐서 task 차원(2048)으로 투영한다.
        self.stage_projection = nnx.Linear(2 * self.subtask_encoding_dim, config.task_embedding_dim, rngs=rngs)

        # ── Action / Time embedding 레이어 ──────────────────────────────────
        # action을 action expert hidden 차원으로 투영한다.
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        # timestep t → adaRMS conditioning 벡터:
        # 단순 선형 층 두 개 + swish 활성화로 비선형 변환을 추가한다.
        self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)

        # action expert 출력 → action 차원으로 역투영한다.
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # ── Correlated noise 관련 변수 ────────────────────────────────────────
        # action들 사이의 시간적 상관관계를 반영한 noise 생성에 사용한다.
        # 체크포인트에 직접 저장하지 않고 norm_stats에서 불러온다.
        flat_dim = config.action_horizon * config.action_dim  # 30 × 32 = 960

        # nnx.Intermediate: 체크포인트에 저장되지 않는 중간 버퍼.
        # 아직 불러오기 전에는 단위행렬을 임시값으로 사용한다.
        # (단위행렬 = 독립 noise와 동일한 Cholesky 인수분해)
        self.action_correlation_cholesky = nnx.Intermediate(jnp.eye(flat_dim))
        self.correlation_loaded = False  # load_correlation_matrix() 호출 여부 플래그
        self.use_correlated_noise = config.use_correlated_noise
        self.correlation_beta = config.correlation_beta

        # inpainting 보정 행렬을 캐시해 두는 딕셔너리
        # key: inpainted steps 수, value: 보정 계산에 필요한 행렬 묶음
        self.inpainting_cache = {}

        # ── FAST 보조 학습 구성요소 (선택적) ──────────────────────────────────
        if config.use_fast_auxiliary:
            # FAST 토큰 embedding 테이블: [fast_vocab_size=1024, PaliGemma_width=2048]
            # prefix의 다른 토큰과 같은 공간에 투영되도록 paligemma width 사용.
            self.fast_token_embedding = nnx.Embed(
                num_embeddings=config.fast_vocab_size,
                features=paligemma_config.width,
                rngs=rngs,
            )

            # FAST 토큰 예측 head: PaliGemma 출력 → vocab logit
            # PaliGemma가 prefix를 처리하면서 다음 FAST 토큰을 예측하도록 학습한다.
            self.fast_token_proj = nnx.Linear(
                paligemma_config.width,
                config.fast_vocab_size,
                rngs=rngs,
            )

            logger.info(f"FAST 보조 학습 활성화, vocab_size={config.fast_vocab_size}")

        # 학습/추론 모드 플래그.
        # train() / eval() 호출로 자동 변경된다.
        self.deterministic = True  # 기본값은 추론 모드(dropout 등 비활성화)

    # [4/9 추가] 디버그 trace 출력용 helper
    # 환경변수 B1K_DEBUG_TRACE=1 일 때만 shape 로그를 찍어서
    # 평소 실행에서는 로그가 너무 많아지지 않게 한다.
    def _debug_trace(self, msg: str):
        if os.environ.get("B1K_DEBUG_TRACE", "0") == "1":
            logger.info(f"[TRACE] {msg}")
    #####################

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Stage 인코딩
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def encode_subtask_state(
        self,
        subtask_state: at.Int[at.Array, " b"],   # 현재 stage index [B]
        task_ids: at.Int[at.Array, " b"],         # 로컬 task id [B]
    ) -> at.Float[at.Array, "b {self.subtask_encoding_dim}"]:
        """Stage를 태스크별 진행률로 정규화한 뒤 sin/cos positional encoding으로 변환한다.

        왜 단순 정수 id를 쓰지 않는가?
            - stage 3이라는 숫자의 절대값보다 "현재 태스크 진행률이 어느 정도인가"를
              표현하는 것이 더 의미 있다.
            - 태스크마다 stage 수가 다르므로 같은 stage 번호도 진행률이 다르다.
              예: 5-stage 태스크의 stage 3 = 75% 진행
                  15-stage 태스크의 stage 3 = 21% 진행
            - sin/cos encoding은 연속적인 위치 표현이므로 "stage 2와 3은 가깝다"는
              귀납적 편향(inductive bias)을 자연스럽게 모델에 넣을 수 있다.

        처리 흐름:
            1. task_ids로 각 배치 샘플의 태스크 stage 수를 조회한다.
            2. stage를 [0, 1] 범위로 정규화한다 (0 = 시작, 1 = 마지막 stage).
            3. sin/cos positional encoding으로 변환한다.

        Args:
            subtask_state: 현재 stage index [B]. 0-indexed.
            task_ids: 각 샘플의 로컬 task id [B]. 0~11 범위.

        Returns:
            sin/cos positional encoding [B, subtask_encoding_dim=1024].
        """
        # JAX 배열로 변환한다. 모듈 로드 시점이 아니라 forward 시점에 변환해야
        # 기기(device) 할당 문제를 피할 수 있다.
        task_num_stages_array = jnp.array(TASK_NUM_STAGES, dtype=jnp.int32)

        # 각 배치 샘플의 태스크에 해당하는 stage 수를 조회한다.
        # task_ids: [B] → task_num_stages: [B]
        task_num_stages = task_num_stages_array[task_ids]

        # Stage를 [0, 1] 범위로 정규화한다.
        # - stage 0 → 0.0 (시작)
        # - stage (n-1) → 1.0 (마지막)
        # jnp.maximum으로 단일 stage 태스크(n=1)의 0나누기를 방지한다.
        normalized_state = subtask_state.astype(jnp.float32) / jnp.maximum(
            task_num_stages.astype(jnp.float32) - 1.0,
            1.0,  # 최소 1로 클리핑해서 0 나누기 방지
        )

        # sin/cos positional encoding 적용.
        # 시계열 위치 인코딩과 동일한 방식으로, 연속적인 위치 표현을 만든다.
        # min_period ~ max_period 범위의 다양한 주파수 성분을 사용한다.
        return posemb_sincos(
            normalized_state,
            self.subtask_encoding_dim,
            min_period=1e-3,
            max_period=1.0,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Correlated noise 관련 메서드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def load_correlation_matrix(self, norm_stats: dict):
        """norm_stats에서 action 공분산 Cholesky 인수분해를 로드하고 shrinkage를 적용한다.

        왜 shrinkage를 적용하는가?
            - 실측 공분산 행렬은 노이즈나 outlier 때문에 ill-conditioned할 수 있다.
            - ill-conditioned 행렬의 Cholesky 분해는 수치적으로 불안정하다.
            - beta × Σ + (1-beta) × I 형태로 항등행렬과 섞으면
              고유값이 양수가 되도록 정칙화(regularize)할 수 있다.

        Args:
            norm_stats: normalize.load()로 읽은 정규화 통계 딕셔너리.
                'actions' 키 아래 action_correlation_cholesky 필드가 있어야 한다.

        Raises:
            TypeError: norm_stats가 딕셔너리가 아닐 때.
            ValueError: use_correlated_noise=True인데 필요한 키/필드가 없을 때.
            RuntimeError: shrinkage 적용 후에도 Cholesky 분해가 실패할 때.
        """
        if not self.use_correlated_noise:
            logger.info("correlated noise 비활성화 상태: 공분산 행렬 로딩을 건너뜁니다.")
            return

        # norm_stats 타입 검증
        if not isinstance(norm_stats, dict):
            raise TypeError(
                f"norm_stats는 dict여야 합니다. 현재 타입: {type(norm_stats).__name__}. "
                "openpi.shared.normalize.load()로 로드했는지 확인하세요."
            )

        # 'actions' 키 존재 확인
        if 'actions' not in norm_stats:
            raise ValueError(
                "use_correlated_noise=True인데 norm_stats에 'actions' 키가 없습니다. "
                f"현재 키 목록: {list(norm_stats.keys())}. "
                "compute_norm_stats.py에 --correlation 옵션을 추가해서 재실행하세요."
            )

        actions_stats = norm_stats['actions']

        # 딕셔너리 또는 속성 접근으로 Cholesky 행렬을 가져온다.
        if isinstance(actions_stats, dict):
            chol_matrix = actions_stats.get('action_correlation_cholesky')
        elif hasattr(actions_stats, 'action_correlation_cholesky'):
            chol_matrix = actions_stats.action_correlation_cholesky
        else:
            raise TypeError(
                f"norm_stats['actions']의 타입 {type(actions_stats).__name__}에서 "
                "'action_correlation_cholesky'에 접근할 수 없습니다."
            )

        if chol_matrix is None:
            raise ValueError(
                "action_correlation_cholesky가 None입니다. "
                "compute_norm_stats.py --correlation 옵션으로 먼저 계산하세요."
            )

        # shape 검증: (action_horizon × action_dim) × (action_horizon × action_dim) 행렬이어야 한다.
        expected_dim = self.action_horizon * self.action_dim  # 30 × 32 = 960
        L = jnp.array(chol_matrix)

        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError(f"Cholesky 행렬이 정방행렬이 아닙니다: shape={L.shape}")

        if L.shape[0] != expected_dim:
            raise ValueError(
                f"Cholesky 행렬 크기가 맞지 않습니다: {L.shape[0]}×{L.shape[0]}, "
                f"기대값: {expected_dim}×{expected_dim} "
                f"(action_horizon={self.action_horizon} × action_dim={self.action_dim})"
            )

        # Cholesky 인수분해 L에서 공분산 행렬 Σ = L @ L.T 를 재구성한다.
        Sigma = L @ L.T  # [960, 960]

        # Shrinkage 정칙화: Σ_reg = beta × Σ + (1-beta) × I
        # beta=0.5이면 실측 공분산 50% + 독립 50%를 섞는다.
        beta = self.correlation_beta
        logger.info(f"Shrinkage 정칙화 적용: beta={beta:.2f}")
        Sigma_reg = beta * Sigma + (1 - beta) * jnp.eye(Sigma.shape[0])

        # 정칙화된 공분산 행렬의 Cholesky 분해를 다시 계산한다.
        # 이 L_reg를 저장해 두고 noise 생성 시 사용한다.
        try:
            L_reg = jnp.linalg.cholesky(Sigma_reg)
        except Exception as e:
            raise RuntimeError(
                f"정칙화된 공분산의 Cholesky 분해 실패: {e}. "
                f"현재 beta={beta:.2f}. beta를 줄여서 정칙화를 더 강하게 하세요."
            )

        # nnx.Intermediate의 값을 업데이트한다.
        self.action_correlation_cholesky.value = L_reg
        self.correlation_loaded = True

        logger.info(
            f"✓ 공분산 Cholesky 행렬 로드 완료: shape={L_reg.shape}, "
            f"메모리={L_reg.nbytes / 1024 / 1024:.2f} MB (beta={beta:.2f})"
        )

    def generate_correlated_noise(
        self,
        rng: at.KeyArrayLike,
        batch_size: int,
    ) -> at.Float[at.Array, "b {self.action_horizon} {self.action_dim}"]:
        """action 공분산 구조를 반영한 correlated noise를 생성한다.

        생성 방법:
            1. 표준 정규분포 ε ~ N(0, I) 샘플링
            2. L을 곱해서 correlated noise 생성: η = ε @ L.T
               (η의 공분산 = L @ E[ε ε.T] @ L.T = L @ I @ L.T = L @ L.T = Σ)
            3. [batch, flat_dim] → [batch, action_horizon, action_dim] 으로 reshape

        Args:
            rng: JAX 난수 키.
            batch_size: 생성할 noise 샘플 수.

        Returns:
            correlated noise [batch_size, action_horizon, action_dim].

        Raises:
            RuntimeError: use_correlated_noise=True인데 공분산 행렬이 로드되지 않았을 때.
        """
        if not self.use_correlated_noise:
            # correlated noise 비활성화: 독립 가우시안 noise 반환 (Pi0 기본 동작)
            return jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if not self.correlation_loaded:
            raise RuntimeError(
                "use_correlated_noise=True인데 공분산 행렬이 로드되지 않았습니다. "
                "load_correlation_matrix()를 먼저 호출하세요."
            )

        flat_dim = self.action_horizon * self.action_dim  # 960

        # 표준 정규분포 샘플링: [batch_size, flat_dim]
        standard_normal = jax.random.normal(rng, (batch_size, flat_dim))

        # Cholesky를 이용한 correlated noise 생성:
        # correlated_flat = ε @ L.T → 공분산이 Σ인 noise
        correlated_flat = standard_normal @ self.action_correlation_cholesky.value.T

        # [batch_size, flat_dim] → [batch_size, action_horizon, action_dim]
        correlated_noise = correlated_flat.reshape(batch_size, self.action_horizon, self.action_dim)
        return correlated_noise

    def _precompute_correction_matrix(
        self,
        O_indices: at.Int[at.Array, " nO"],  # inpaint된 차원의 평탄화 인덱스 [|O|]
        U_indices: at.Int[at.Array, " nU"],  # 자유 차원의 평탄화 인덱스 [|U|]
    ) -> dict:
        """상관관계를 고려한 inpainting 보정 행렬을 미리 계산한다.

        수식:
            보정 행렬 M = Σ_{UO} @ Σ_{OO}^{-1}
            δ_U = M @ δ_O

        의미:
            - O 영역(inpainted 차원)에서 발생한 변화 δ_O를
            - U 영역(자유 차원)으로 공분산 구조에 따라 전파한다.
            - 이렇게 하면 inpainting이 action의 시간적 일관성을 더 잘 유지한다.

        Args:
            O_indices: inpaint된 차원의 flat index들 [|O|].
            U_indices: 자유 차원의 flat index들 [|U|].

        Returns:
            {O_indices, U_indices, correction_matrix} 딕셔너리.
                correction_matrix: [|U|, |O|] shape.

        Raises:
            RuntimeError: 공분산 행렬이 로드되지 않았을 때.
        """
        if not self.correlation_loaded:
            raise RuntimeError(
                "보정 행렬을 계산하려면 먼저 load_correlation_matrix()를 호출해야 합니다."
            )

        L = self.action_correlation_cholesky.value
        Sigma = L @ L.T  # 전체 공분산 행렬 [flat_dim, flat_dim]

        # O-O 서브행렬과 U-O 서브행렬을 추출한다.
        Sigma_OO = Sigma[jnp.ix_(O_indices, O_indices)]  # [|O|, |O|]
        Sigma_UO = Sigma[jnp.ix_(U_indices, O_indices)]  # [|U|, |O|]

        # Σ_{OO}가 수치적으로 안정하도록 작은 값을 더해준다.
        eps_OO = 1e-6 * jnp.maximum(jnp.mean(jnp.diag(Sigma_OO)), 1.0)
        Sigma_OO_reg = Sigma_OO + eps_OO * jnp.eye(Sigma_OO.shape[0])

        # 보정 행렬 M = Σ_{UO} @ Σ_{OO}^{-1} 계산:
        # 직접 역행렬을 구하는 대신 선형 시스템을 푼다.
        # Σ_{OO}_reg @ X = Σ_{UO}.T 를 풀어서 X = Σ_{OO}^{-1} @ Σ_{UO}.T
        # M = X.T = Σ_{UO} @ Σ_{OO}^{-1}
        correction_matrix = jax.scipy.linalg.solve(
            Sigma_OO_reg, Sigma_UO.T, assume_a='pos'  # 'pos': 양정치 행렬이므로 Cholesky 분해 사용
        ).T  # [|U|, |O|]

        return {
            'O_indices': O_indices,
            'U_indices': U_indices,
            'correction_matrix': correction_matrix,
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Task + Subtask 정보 Fusion
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def fuse_task_and_subtask(
        self,
        task_embedding: at.Float[at.Array, "b d"],   # 기본 task embedding [B, 2048]
        task_ids: at.Int[at.Array, " b"],             # 로컬 task id [B]
        subtask_state: at.Int[at.Array, " b"],        # 현재 stage index [B]
    ) -> at.Float[at.Array, "b n d"]:
        """Task embedding과 stage 정보를 여러 방식으로 결합해 4개의 조건 벡터를 만든다.

        왜 하나가 아니라 4개의 표현을 만드는가?
            - 모델이 "task는 같지만 stage가 다를 때"를 한 가지 표현으로만 처리하면
              표현 능력이 제한된다.
            - 4개의 서로 다른 fusion 방식으로 조건 벡터를 만들면,
              downstream Transformer가 각 표현에서 필요한 정보를 선택적으로 가져갈 수 있다.

        4가지 조건 벡터:
            1. task_gated (task_embedding × gate_task):
               task embedding을 stage 정보로 조절한 표현.
               "이 task를 수행하는데 현재 stage를 고려해 어떤 부분이 중요한가"

            2. balanced_fusion (fusion MLP 통과):
               task + stage 정보를 균형 있게 합친 표현.
               모든 입력을 2층 MLP로 비선형 결합.

            3. stage_dominant (게이팅된 stage 특징의 projection):
               stage 위주 표현. sin/cos와 task-stage embedding 두 종류의
               stage 표현을 게이팅 후 투영.

            4. pure_stage (sin/cos + task-stage embedding의 단순 concatenation):
               가장 직접적인 stage 표현. 게이트나 MLP 없이 단순 연결.

        모든 표현의 차원은 task_embedding_dim(=2048)로 통일된다.

        Args:
            task_embedding: 기본 task embedding [B, 2048].
            task_ids: 각 샘플의 로컬 task id [B].
            subtask_state: 각 샘플의 현재 stage index [B].

        Returns:
            4개의 조건 벡터를 쌓은 텐서 [B, 4, 2048].
        """
        # sin/cos positional encoding으로 stage를 연속적 벡터로 변환한다.
        sincos_encoding = self.encode_subtask_state(subtask_state, task_ids)  # [B, 1024]

        # (task, stage) 쌍별 embedding 테이블에서 현재 stage의 embedding을 가져온다.
        # TASK_STAGE_OFFSETS[task] + stage_id = 전체 embedding 테이블의 global index
        task_stage_offsets_array = jnp.array(TASK_STAGE_OFFSETS, dtype=jnp.int32)
        task_stage_offsets = task_stage_offsets_array[task_ids]  # [B] 각 태스크의 오프셋
        task_stage_idx = task_stage_offsets + subtask_state      # [B] global stage index
        task_stage_embedding = self.task_stage_embeddings(task_stage_idx)  # [B, 1024]

        # 모든 입력을 concatenate해서 fusion의 입력으로 사용한다.
        # [task_embedding(2048) | sincos(1024) | task_stage(1024)] = [4096]
        all_inputs = jnp.concatenate([
            task_embedding,       # [B, 2048]
            sincos_encoding,      # [B, 1024]
            task_stage_embedding, # [B, 1024]
        ], axis=-1)  # [B, 4096]

        # 게이트 계산: sigmoid로 각 성분의 기여도를 0~1 범위로 만든다.
        gate_sincos = nnx.sigmoid(self.gate_sincos(all_inputs))       # [B, 1024]
        gate_task_stage = nnx.sigmoid(self.gate_task_stage(all_inputs))  # [B, 1024]
        gate_task = nnx.sigmoid(self.gate_task(all_inputs))            # [B, 2048]

        # ── 표현 1: task_gated ─────────────────────────────────────────────
        # task embedding을 stage 정보로 조절(modulate)한다.
        # gate가 1이면 원래 task embedding 그대로, 0이면 0 벡터.
        task_gated = task_embedding * gate_task  # [B, 2048]

        # ── 표현 2: balanced_fusion ────────────────────────────────────────
        # 모든 입력(task + stage)을 2층 MLP로 비선형 결합한다.
        # ReLU 활성화로 비선형성을 추가한다.
        x = self.fusion_layer1(all_inputs)  # [B, 4096]
        x = nnx.relu(x)
        balanced_fusion = self.fusion_layer2(x)  # [B, 2048]

        # ── 표현 3: stage_dominant ─────────────────────────────────────────
        # stage 신호 위주의 표현을 만든다.
        # 게이팅된 sin/cos와 task-stage embedding을 합쳐 투영한다.
        gated_stage_features = jnp.concatenate([
            sincos_encoding * gate_sincos,           # [B, 1024] 게이팅된 sin/cos
            task_stage_embedding * gate_task_stage,  # [B, 1024] 게이팅된 task-stage
        ], axis=-1)  # [B, 2048]
        stage_dominant = self.stage_projection(gated_stage_features)  # [B, 2048]

        # ── 표현 4: pure_stage ─────────────────────────────────────────────
        # 가장 직접적인 stage 표현: 두 벡터를 단순히 이어 붙인다.
        # 게이트나 MLP 없이 원시적인 stage 정보를 보존한다.
        pure_stage = jnp.concatenate([sincos_encoding, task_stage_embedding], axis=-1)  # [B, 2048]

        # 4개의 표현을 새로운 차원(axis=1)에 쌓는다.
        # [B, 2048] × 4 → [B, 4, 2048]
        fused_embeddings = jnp.stack(
            [task_gated, balanced_fusion, stage_dominant, pure_stage], axis=1
        )

        return fused_embeddings  # [B, 4, 2048]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Prefix / Suffix embedding
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @at.typecheck
    def embed_prefix(
        self,
        obs: Observation,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],  # 토큰 텐서 [B, seq_len, hidden_dim]
        at.Bool[at.Array, "b s"],        # 입력 마스크 [B, seq_len]
        at.Bool[at.Array, " s"],         # AR(autoregressive) 마스크 [seq_len]
    ]:
        """Observation을 prefix 토큰 시퀀스로 변환한다.

        Prefix 구성 순서:
            1. 이미지 토큰:     SigLIP(이미지) → [B, 256×num_cams, 2048]
            2. Task 토큰:       task_embeddings[task_id] → [B, 1, 2048]
            3. Stage 토큰:      fuse_task_and_subtask(...) → [B, 4, 2048]
               (tokenized_prompt.shape[1] > 1 일 때만. 기본 경로는 task 토큰만.)
            4. 상태 토큰:       discretize(state) + embed → [B, state_dim, 2048]
            5. FAST 토큰:       fast_token_embedding(fast_tokens) → [B, T, 2048]
               (use_fast_auxiliary=True이고 obs.fast_tokens가 있을 때만.)

        AR 마스크 의미:
            False: 이 토큰은 이전 토큰과 이후 토큰 모두를 볼 수 있다 (양방향 attention).
            True:  이 토큰은 자신보다 앞에 있는 토큰만 볼 수 있다 (단방향 attention).
            이미지/task/state 토큰은 False(양방향), FAST 토큰은 True(자기회귀).

        Args:
            obs: 전처리된 Observation 인스턴스.

        Returns:
            (tokens, input_mask, ar_mask) 튜플.
        """
        input_mask = []  # 유효한 토큰 위치 마스크를 모을 리스트
        ar_mask = []     # AR 마스크를 모을 리스트
        tokens = []      # 토큰 텐서를 모을 리스트

        # ── 1. 이미지 토큰 embedding ────────────────────────────────────────
        image_token_list = []

        # freeze_vision_backbone: True이면 항상 추론 모드(train=False)로 SigLIP을 실행한다.
        # 이렇게 하면 SigLIP의 BatchNorm, Dropout이 비활성화되고
        # gradient가 SigLIP으로 흘러가지 않는다 (사실상 학습 고정과 동일한 효과).
        vision_train_mode = (not self.deterministic) and (not self.config.freeze_vision_backbone)

        for name in obs.images:
            # SigLIP forward: [B, H, W, C] → [B, num_patches, hidden_dim]
            # So400m/14 기준: 224×224 이미지 → 256개 패치 토큰
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=vision_train_mode)
            image_token_list.append(image_tokens)

            tokens.append(image_tokens)

            # 이미지 마스크를 시퀀스 차원으로 broadcast한다.
            # obs.image_masks[name]: [B] (이미지 존재 여부)
            # → [B, num_patches] (각 패치의 유효성)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 이미지 토큰은 서로를 볼 수 있다 (양방향 attention).
            ar_mask += [False] * image_tokens.shape[1]

        # ── 2. Task / Stage 토큰 embedding ────────────────────────────────
        if obs.tokenized_prompt is not None:
            # tokenized_prompt[:, 0]: 로컬 task id (0~11)
            task_ids = obs.tokenized_prompt[:, 0]
            # task_embeddings 테이블에서 해당 task의 embedding 벡터를 가져온다.
            base_task_embedding = self.task_embeddings(task_ids)  # [B, 2048]

            if obs.tokenized_prompt.shape[1] > 1:
                # ── Stage conditioning 경로 ───────────────────────────────
                # tokenized_prompt[:, 1]: 현재 stage id (0~max_stages-1)
                subtask_state = obs.tokenized_prompt[:, 1]

                # 4가지 fusion 표현을 만든다.
                fused_task_embeddings = self.fuse_task_and_subtask(
                    base_task_embedding, task_ids, subtask_state
                )  # [B, 4, 2048]

                # base task 토큰(1개) + stage-conditioned 토큰(4개) = 5개 토큰
                task_sequence = jnp.concatenate([
                    base_task_embedding[:, None, :],  # [B, 1, 2048]
                    fused_task_embeddings,            # [B, 4, 2048]
                ], axis=1)  # [B, 5, 2048]
                tokens.append(task_sequence)

                task_mask = jnp.ones((obs.tokenized_prompt.shape[0], 5), dtype=jnp.bool_)
                input_mask.append(task_mask)

                # AR 마스크:
                # base task 토큰(False=양방향) + stage 토큰들([True, False, False, False])
                # stage 토큰 중 첫 번째만 AR로 두어 stage 예측 보조 학습 구조와 맞춘다.
                ar_mask += [False] + [True, False, False, False]
            else:
                # ── 기본 경로 (12-task baseline) ──────────────────────────
                # stage 정보 없이 task embedding만 사용한다.
                # Pi0 + task embedding + flow matching 구조.
                tokens.append(base_task_embedding[:, None, :])  # [B, 1, 2048]
                task_mask = jnp.ones((obs.tokenized_prompt.shape[0], 1), dtype=jnp.bool_)
                input_mask.append(task_mask)
                ar_mask += [False]  # task 토큰 1개, 양방향

        # ── 3. 상태 토큰 embedding ─────────────────────────────────────────
        # Pi0.5 스타일: state를 이산화(discretize)한 뒤 토큰 ID로 취급해 embed한다.
        #
        # 처리 방법:
        #   1. jnp.digitize: state의 각 차원을 [-1, 1] 범위의 256개 bin으로 이산화
        #   2. 각 차원을 별도의 토큰 ID로 취급해 PaliGemma embed 층에 통과
        #   3. 각 차원의 토큰을 시퀀스로 이어 붙임

        # jnp.linspace(-1, 1, 257)[:-1]: 256개 bin의 왼쪽 경계값
        # 각 값이 어느 bin에 속하는지 결정한다.
        discretized_state = jnp.digitize(
            obs.state,
            bins=jnp.linspace(-1, 1, 256 + 1)[:-1]
        ) - 1  # 0-indexed로 만들기 위해 1을 뺀다.

        # 범위를 [0, 255]로 클리핑 (edge case 처리)
        discretized_state = jnp.clip(discretized_state, 0, 255)

        # 각 차원을 독립적으로 embed한다.
        state_tokens = []
        for i in range(obs.state.shape[-1]):
            # PaliGemma의 embed 층에 scalar token id를 통과시킨다.
            # [B, 1] → [B, 1, 2048]
            state_dim_tokens = self.PaliGemma.llm(
                discretized_state[:, i:i + 1], method="embed"
            )
            state_tokens.append(state_dim_tokens)

        if state_tokens:
            # [B, 1, 2048] × state_dim → [B, state_dim, 2048]
            state_tokens = jnp.concatenate(state_tokens, axis=1)
            tokens.append(state_tokens)
            input_mask.append(
                jnp.ones((obs.state.shape[0], obs.state.shape[-1]), dtype=jnp.bool_)
            )
            # 상태 토큰은 모든 prefix 토큰과 양방향 attention한다.
            ar_mask += [False] * state_tokens.shape[1]

        # ── 4. FAST 토큰 embedding (선택적) ────────────────────────────────
        if self.config.use_fast_auxiliary and obs.fast_tokens is not None:
            fast_tokens = obs.fast_tokens       # [B, T] int32
            fast_token_mask = obs.fast_token_mask  # [B, T] bool

            # Teacher forcing: 입력을 오른쪽으로 한 칸 shift한다.
            # 원본:  [tok_0, tok_1, ..., tok_{T-1}]
            # shift: [BOS,   tok_0, ..., tok_{T-2}]
            # 이렇게 하면 모델이 tok_i를 예측할 때 tok_{i-1}까지만 볼 수 있다.
            bos_token = jnp.zeros((fast_tokens.shape[0], 1), dtype=jnp.int32)  # BOS = 0
            shifted_tokens = jnp.concatenate([bos_token, fast_tokens[:, :-1]], axis=1)

            # 마스크도 동일하게 shift한다.
            bos_mask = jnp.ones((fast_tokens.shape[0], 1), dtype=jnp.bool_)
            shifted_mask = jnp.concatenate([bos_mask, fast_token_mask[:, :-1]], axis=1)

            # PaliGemma embed 층 대신 별도의 FAST embedding 층을 사용한다.
            # FAST 토큰의 어휘 공간(1024)이 PaliGemma 어휘 공간과 다르기 때문이다.
            fast_token_emb = self.fast_token_embedding(shifted_tokens)  # [B, T, 2048]

            tokens.append(fast_token_emb)
            input_mask.append(shifted_mask)
            # FAST 토큰은 전부 인과적(causal/AR): 앞 토큰만 볼 수 있다.
            ar_mask += [True] * shifted_tokens.shape[1]

        # [4/9 추가] prefix 토큰 구성이 끝난 뒤 shape를 남긴다.
        # 여기서 확인하고 싶은 건:
        # - 최종 prefix token 길이
        # - input mask 길이
        # - autoregressive mask 길이
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        self._debug_trace(
            f"embed_prefix done: "
            f"tokens={tokens.shape}, "
            f"input_mask={input_mask.shape}, "
            f"ar_mask={ar_mask.shape}"
        ) # [total_seq] bool
        ############################

        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: Observation,
        noisy_actions: _model.Actions,                    # 노이즈 섞인 action [B, H, D]
        timestep: at.Float[at.Array, " b"],               # flow matching 시간 t [B]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],   # suffix 토큰 [B, action_horizon, hidden_dim]
        at.Bool[at.Array, "b s"],         # 입력 마스크 [B, action_horizon]
        at.Bool[at.Array, " s"],          # AR 마스크 [action_horizon]
        at.Float[at.Array, "b emb"],      # adaRMS conditioning 벡터 [B, hidden_dim]
    ]:
        """Noisy action과 timestep을 suffix 토큰으로 변환한다.

        Suffix 구성:
            - noisy_actions를 action expert hidden 차원으로 선형 투영한다.
            - timestep t를 sin/cos encoding → MLP → adaRMS conditioning 벡터로 변환한다.
            - adaRMS conditioning은 action expert의 각 레이어 LayerNorm을 조절한다.
              즉, "현재 diffusion step t에서 noise 수준이 얼마나 되는가"를
              action expert 전체에 직접 알려준다.

        AR 마스크:
            - 첫 번째 action 토큰: True (앞만 참조, prefix KV cache를 봄)
            - 나머지 토큰: False (서로를 볼 수 있음, but 실제로는 시퀀스 내 attention)
            - 이미지/task/state 토큰은 action 토큰을 볼 수 없다 (정보 오염 방지).

        Args:
            obs: Observation (현재 suffix embedding에서 직접 사용하지는 않지만
                 나중에 확장을 위해 인자로 받는다).
            noisy_actions: t 시점에서 noise가 섞인 action [B, action_horizon, action_dim].
            timestep: 각 배치 샘플의 flow matching 시간 [B]. 범위: [0, 1].

        Returns:
            (tokens, input_mask, ar_mask, adarms_cond) 튜플.
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # ── Noisy action 토큰화 ────────────────────────────────────────────
        # noisy_actions: [B, action_horizon, action_dim]
        # action_in_proj: [action_dim → action_expert_width]
        # action_tokens: [B, action_horizon, action_expert_width]
        action_tokens = self.action_in_proj(noisy_actions)

        # ── Timestep conditioning ──────────────────────────────────────────
        # Pi0.5 스타일 adaRMS conditioning:
        # 1) sin/cos positional encoding으로 t를 벡터로 변환
        #    (min_period=4e-3, max_period=4.0은 Pi0 논문과 동일)
        # 2) MLP(swish 활성화)로 비선형 변환
        # 3) 결과 벡터를 action expert의 adaRMS scale/shift 파라미터로 사용
        time_emb = posemb_sincos(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )  # [B, action_expert_width]

        # 2층 MLP with swish: 시간 정보를 더 풍부하게 변환한다.
        # swish(x) = x × sigmoid(x): ReLU보다 부드러운 활성화 함수.
        time_emb = self.time_mlp_in(time_emb)   # [B, width]
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)  # [B, width]
        time_emb = nnx.swish(time_emb)

        # adaRMS conditioning 벡터: 각 레이어의 LayerNorm scale/shift를 이 벡터로 조절한다.
        adarms_cond = time_emb  # [B, action_expert_width]

        tokens.append(action_tokens)
        input_mask.append(jnp.ones(action_tokens.shape[:2], dtype=jnp.bool_))

        # AR 마스크:
        # [True, False, False, ...]: 첫 토큰만 AR(인과적), 나머지는 양방향.
        # 첫 번째가 True인 이유: prefix(이미지/task/state)가 action을 볼 수 없도록.
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # [4/9 추가] suffix(action 쪽) 토큰 구성이 끝난 뒤 shape를 남긴다.
        # 여기서는 noisy action이 projection된 뒤 token 길이와
        # time embedding에서 나온 adarms_cond shape까지 같이 본다.
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)

        self._debug_trace(
            f"embed_suffix done: "
            f"tokens={tokens.shape}, "
            f"input_mask={input_mask.shape}, "
            f"ar_mask={ar_mask.shape}, "
            f"adarms_cond={adarms_cond.shape}"
        )                       # [action_horizon]
        ####################

        return tokens, input_mask, ar_mask, adarms_cond


    # [2026-04-17] PiBehavior 정식 sample_actions 추가
    # 목적:
    #   OpenPI BaseModel 인터페이스를 만족시키고,
    #   Pi0와 같은 flow-matching ODE 샘플링 경로를 PiBehavior에서도 사용한다.
    #
    # 설계:
    #   1) observation을 inference용 전처리
    #   2) prefix를 한 번 forward 해서 KV cache 생성
    #   3) x_t ~ N(0, I) 에서 시작
    #   4) t=1 -> 0 방향으로 suffix forward를 반복하며 velocity v_t 예측
    #   5) x_{t-dt} = x_t + dt * v_t   (dt < 0)
    #
    # 참고:
    #   dt를 음수로 두는 이유는 Pi0 구현과 동일하게
    #   "noise(1.0) -> clean(0.0)" 방향으로 적분하기 위해서다.
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        # [2026-04-17]
        # Pi0의 sample_actions와 동일하게 inference용 observation 전처리를 거친다.
        # 현재 PiBehavior는 b1k.models.observation 쪽 preprocess_observation을 import하고 있으므로
        # 여기서는 그 경로를 우선 사용한다.
        observation = preprocess_observation(observation, train=False)

        # diffusion 관례:
        #   t=1.0 -> pure noise
        #   t=0.0 -> target(clean action)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        if noise is None:
            noise = jax.random.normal(
                rng,
                (batch_size, self.action_horizon, self.action_dim),
                dtype=jnp.float32,
            )

        # ---------------------------------------------------------------
        # 1) prefix forward 1회로 KV cache 생성
        # ---------------------------------------------------------------
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1

        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )

        # ---------------------------------------------------------------
        # 2) reverse ODE loop
        # ---------------------------------------------------------------
        def step(carry):
            x_t, time = carry

            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
            )

            # suffix 내부 attention mask
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

            # suffix query가 prefix key/value 전체를 볼 수 있게 하는 mask
            prefix_to_suffix_mask = einops.repeat(
                prefix_mask,
                "b p -> b s p",
                s=suffix_tokens.shape[1],
            )

            full_attn_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attn_mask],
                axis=-1,
            )

            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )

            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None

            # suffix hidden -> predicted velocity
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            # dt < 0 이므로 x_t + dt * v_t 는 t=1 -> 0 방향 적분
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            # floating-point 오차에 robust 하게 종료
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 손실 함수
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        """BaseModel의 compute_loss를 오버라이드.

        이 메서드는 사용하지 않는다. 학습에는 compute_detailed_loss()를 써야 한다.
        보조 손실(FAST, stage classification)을 분리해 반환하기 위해 더 상세한 메서드가 필요하다.
        """
        raise NotImplementedError(
            "PiBehavior는 compute_loss() 대신 compute_detailed_loss()를 사용하세요. "
            "FAST 보조 손실과 stage 분류 손실을 따로 받으려면 이 메서드가 필요합니다."
        )

    @override
    def compute_detailed_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        num_flow_samples: int = 1,
    ) -> dict[str, at.Float[at.Array, "*b"]]:
        """Flow matching + 보조 손실을 계산한다.

        핵심 최적화: "비싼 prefix 계산은 한 번, flow sample은 여러 번"
            - Prefix(이미지/task/state) forward를 한 번만 실행해서 KV cache를 만든다.
            - num_flow_samples 개의 서로 다른 (noise, timestep) 쌍으로 suffix를 반복 계산한다.
            - 각 샘플의 손실을 평균해서 최종 flow 손실을 만든다.
            - FAST 손실과 stage 손실은 prefix 계산 결과에서 바로 계산한다.

        Flow matching 원리:
            - 목표: noise x_1 ~ N(0, I)에서 clean action x_0 ~ p(action)로 가는 ODE를 학습한다.
            - 학습 신호: t 시점의 noisy action x_t = (1-t)×x_0 + t×x_1 에서
                         velocity v_t = x_0 - x_1 를 예측하도록 학습한다.
            - 손실: ||predicted_v_t - target_v_t||² (MSE)

        Args:
            rng: 전체 학습 스텝의 난수 키.
            observation: 배치 Observation.
            actions: 정답 action [B, action_horizon, action_dim].
            train: True이면 데이터 증강과 dropout을 활성화한다.
            num_flow_samples: 한 배치에서 몇 개의 (noise, t) 쌍을 샘플링할지.
                              많을수록 정확하지만 메모리와 시간이 늘어난다.

        Returns:
            손실 딕셔너리:
                'flow': flow matching MSE 손실 [B, action_horizon]
                'subtask': stage 분류 cross-entropy 손실 [B] (or 0)
                'fast': FAST 보조 cross-entropy 손실 [B] (or 0)
        """
        losses = {}

        # 이미지 전처리(resize, 증강)를 적용한다.
        preprocess_rng, rng = jax.random.split(rng)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_size = actions.shape[0]

        # ── 1. Prefix 토큰 embedding ───────────────────────────────────────
        # 이미지 + task + state (+FAST if enabled) 토큰을 만든다.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # prefix_tokens: [B, prefix_len, 2048]
        # prefix_mask:   [B, prefix_len] - 유효한 토큰 위치
        # prefix_ar_mask: [prefix_len] - AR 마스크

        # [4/9 추가] embed_prefix 결과 shape 확인
        # prefix 쪽에서 토큰 길이가 비정상적으로 늘어나거나 mask 길이가 안 맞는지
        # early stage에서 바로 확인하려는 로그다.
        self._debug_trace(
            f"after embed_prefix: "
            f"prefix_tokens={prefix_tokens.shape}, "
            f"prefix_mask={prefix_mask.shape}, "
            f"prefix_ar_mask={prefix_ar_mask.shape}"
        )
        ######################

        # ── 2. Prefix KV cache 계산 (한 번만!) ────────────────────────────
        # Attention mask를 만들고 PaliGemma LLM을 실행해서 KV cache를 생성한다.
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        # positions: 각 토큰의 position embedding 위치.
        # cumsum(mask) - 1: 마스킹된 토큰은 무시하고 실제 위치를 계산한다.
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1

        (prefix_out, _), kv_cache_full = self.PaliGemma.llm(
            [prefix_tokens, None],   # [prefix 토큰, suffix 없음(None)]
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )
        # prefix_out: [B, prefix_len, 2048] - 각 prefix 토큰의 출력
        # kv_cache_full: 모든 prefix 레이어의 KV cache

        # [4/9 추가] prefix를 LLM에 통과시킨 뒤 출력과 KV cache shape를 기록
        # 여기서 KV cache 길이/헤드 수가 맞는지 보면
        # 이후 suffix attention에서 어디서 꼬이는지 빨리 찾을 수 있다.
        cache_k, cache_v = kv_cache_full
        self._debug_trace(
            f"after prefix llm: "
            f"prefix_out={prefix_out.shape}, "
            f"kv_cache_k={cache_k.shape}, "
            f"kv_cache_v={cache_v.shape}"
        )
        #####################

        # ── 3. Stage prediction (보조 학습) ────────────────────────────────
        # stage conditioning 경로(tokenized_prompt.shape[1] > 1)에서만 실행된다.
        # 기본 12-task baseline에서는 이 분기가 실행되지 않는다.
        if observation.tokenized_prompt.shape[1] > 1:
            # prefix 시퀀스에서 base task 토큰의 출력을 찾는다.
            # first_stage_token_idx: stage 토큰 중 첫 번째(AR=True)의 위치
            # base_task_token_idx: 그 바로 앞이 base task 토큰
            first_stage_token_idx = jnp.argmax(prefix_ar_mask)
            base_task_token_idx = first_stage_token_idx - 1
            base_task_output = prefix_out[:, base_task_token_idx, :]  # [B, 2048]

            # VLM 출력에서 stage logit을 예측한다.
            subtask_logits = self.stage_pred_from_vlm(base_task_output)  # [B, MAX_NUM_STAGES=15]

            # 해당 태스크에 없는 stage는 -inf로 마스킹한다.
            # (softmax 시 확률 0이 되도록)
            task_ids = observation.tokenized_prompt[:, 0]
            task_num_stages_array = jnp.array(TASK_NUM_STAGES, dtype=jnp.int32)
            task_num_stages = task_num_stages_array[task_ids]  # [B]

            stage_range = jnp.arange(MAX_NUM_STAGES)  # [0, 1, 2, ..., 14]
            # valid_mask[b, s] = (s < task_num_stages[b])
            valid_mask = stage_range[None, :] < task_num_stages[:, None]  # [B, 15]
            subtask_logits = jnp.where(valid_mask, subtask_logits, -jnp.inf)
        else:
            # 기본 경로: stage 손실이 0인 더미 logit 반환
            subtask_logits = jnp.zeros(
                (prefix_out.shape[0], MAX_NUM_STAGES), dtype=prefix_out.dtype
            )

        # ── 4. KV cache 변환 (선택적) ──────────────────────────────────────
        # use_kv_transform=True이면 여러 레이어의 KV를 섞어 새로운 cache를 만든다.
        kv_cache = kv_cache_full
        if self.kv_transform is not None:
            kv_cache = self.kv_transform(kv_cache)

        # ── 5. Flow matching 손실 (num_flow_samples 번 반복) ───────────────
        # 각 반복에서 다른 (noise, timestep)을 사용한다.
        flow_losses = []
        for i in range(num_flow_samples):
            sample_rng, rng = jax.random.split(rng)
            noise_rng, time_rng = jax.random.split(sample_rng)

            # Timestep t ~ Uniform(0, 1) 샘플링
            t = jax.random.uniform(time_rng, shape=(batch_size,))  # [B]

            # Noise 생성 (독립 또는 correlated)
            noise = self.generate_correlated_noise(noise_rng, batch_size)  # [B, H, D]

            # Noisy action 생성: x_t = (1-t)×x_0 + t×x_1
            # x_0: 정답 action (clean), x_1: noise
            t_expanded = t[:, None, None]  # [B, 1, 1] for broadcasting
            noisy_actions = (1.0 - t_expanded) * actions + t_expanded * noise

            # Target velocity: v = x_0 - x_1 = action - noise
            target_velocity = actions - noise  # [B, H, D]

            # ── Suffix 계산 ─────────────────────────────────────────────
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, noisy_actions, t
            )

            # Suffix-to-prefix attention mask:
            # suffix의 각 토큰이 전체 시퀀스(prefix + suffix)의 어디를 볼 수 있는지.
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # prefix 부분 마스크: suffix 모든 토큰이 prefix를 볼 수 있다.
            prefix_attn_mask_for_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            # 전체 attention mask: [B, suffix_len, prefix_len + suffix_len]
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask_for_suffix, suffix_attn_mask], axis=-1
            )

            # Suffix position: prefix 길이 이후부터 시작
            positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]  # prefix 유효 토큰 수
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )

            # Action expert forward (KV cache 재사용):
            # prefix_out=None이면 prefix를 재계산하지 않고 kv_cache를 사용한다.
            (prefix_out_suffix, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],  # action expert에만 conditioning 적용
            )

            # Velocity 예측: suffix 출력의 마지막 action_horizon 토큰에서 velocity를 예측한다.
            predicted_velocity = self.action_out_proj(
                suffix_out[:, -self.action_horizon:]
            )  # [B, action_horizon, action_dim]

            # Flow matching MSE 손실: ||predicted_v - target_v||²
            # 차원별로 MSE를 계산하고 action_dim 축으로 평균 낸다.
            flow_loss = jnp.mean(
                (predicted_velocity - target_velocity) ** 2,
                axis=-1,
            )  # [B, action_horizon]
            flow_losses.append(flow_loss)

        # num_flow_samples 개 샘플의 손실을 평균한다.
        losses['flow'] = jnp.mean(jnp.stack(flow_losses, axis=0), axis=0)  # [B, H]
        losses['subtask'] = subtask_logits  # stage logit (나중에 cross-entropy 계산)

        return losses
