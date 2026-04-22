"""
Training script for BEHAVIOR-1K solution.

Based on https://github.com/PhysicalIntelligence/openpi/blob/behavior/openpi/scripts/train.py with custom modifications.
"""

import subprocess # [4/9 추가]
import dataclasses
import functools
import logging
import os
import platform
import time
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

# Configure JAX memory allocation to prevent OOM errors
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')

# Configure OpenBLAS to prevent thread creation errors
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

# Import B1K-specific modules
from b1k.training import checkpoints as _checkpoints  # Use our custom checkpoints (not openpi's!)
from b1k.training import config as _config
from b1k.training import data_loader as _data_loader
from b1k.training import weight_loaders as _weight_loaders
from b1k.models.pi_behavior import PiBehavior
from b1k.models.pi_behavior_config import PiBehaviorConfig
from b1k.models.observation import Observation



def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)

# [2026-04-22 수정]
# 설명:
# - baseline / 비교실험용으로 W&B에 필요한 scalar metric만 최소 로깅
# - action_loss, subtask_loss, learning_rate, step, GPU peak까지 포함
# - 없는 키는 자동으로 건너뜀

def build_minimal_wandb_payload(
    reduced_info: dict[str, Any],
    *,
    step: int | None = None,
    step_time_sec: float | None = None,
    gpu_mem_peak_mib: float | None = None,
    learning_rate: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    keep_keys = [
        "loss",
        "total_loss",
        "action_loss",
        "subtask_loss",
        "grad_norm",
        "param_norm",
        "subtask_accuracy",
        "grad_norm_vlm",
        "grad_norm_action_expert",
    ]

    for key in keep_keys:
        if key in reduced_info:
            payload[key] = reduced_info[key]

    if step is not None:
        payload["step"] = int(step)

    if step_time_sec is not None:
        payload["step_time"] = float(step_time_sec)

    if gpu_mem_peak_mib is not None:
        payload["gpu_mem_peak_mib"] = float(gpu_mem_peak_mib)

    if learning_rate is not None:
        payload["learning_rate"] = float(learning_rate)

    return payload

# [2026-04-19 수정]
# 목적:
# - 기존에는 "현재 시점의 GPU 메모리"만 찍었음
# - 이제는 실행 중 관측된 최대값(peak_seen)도 같이 기록해서
#   bs16 테스트에서 어느 구간이 가장 위험한지 바로 확인하려는 용도
_GPU_MEM_PEAK_MIB = 0

def reset_gpu_mem_peak():
    """[2026-04-19 수정] peak 추적값을 새 구간 시작 전에 초기화한다."""
    global _GPU_MEM_PEAK_MIB
    _GPU_MEM_PEAK_MIB = 0

def log_gpu_mem(tag: str):
    """[2026-04-19 수정]
    현재 GPU 메모리 사용량과 지금까지 관측한 최대 사용량(peak_seen)을 함께 로그로 찍는다.
    """
    global _GPU_MEM_PEAK_MIB
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().splitlines()[0]

        used, total = [int(x.strip()) for x in out.split(",")]
        _GPU_MEM_PEAK_MIB = max(_GPU_MEM_PEAK_MIB, used)

        logging.info(
            f"[GPU MEM] {tag}: {used} MiB / {total} MiB "
            f"(peak_seen={_GPU_MEM_PEAK_MIB} MiB)"
        )
    except Exception as e:
        logging.info(f"[GPU MEM] {tag}: unavailable ({e})")

def block_and_log(tag: str, x=None):
    """기존과 동일하게 JAX 계산을 끝까지 block한 뒤 GPU 메모리 로그를 찍는다."""
    if x is not None:
        jax.block_until_ready(x)
    log_gpu_mem(tag)

def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    
    # Filter out nnx.Intermediate fields from both sides (they're not params, excluded from checkpoints)
    # This allows loading old checkpoints that didn't have these fields
    def filter_intermediate_fields(params_dict):
        flat = traverse_util.flatten_dict(params_dict)
        # List of field names that are nnx.Intermediate (excluded from checkpoints)
        intermediate_field_names = [
            'action_correlation_cholesky',  # Legacy full correlation matrix
            'L_spatial',                     # Separable spatial correlation
            'L_temporal',                    # Separable temporal correlation
            'cached_num_inpaint_actions',    # Conditional sampling cache
            'cached_input_action_dim',       # Conditional sampling cache
            'cached_Sigma_uo_Sigma_oo_inv',  # Conditional sampling cache
            'cached_L_cond_free',            # Conditional sampling cache
            'cached_Sigma_ou_Sigma_uu_inv',  # Conditional sampling cache
            'cached_L_cond_inp',             # Conditional sampling cache
        ]
        filtered = {k: v for k, v in flat.items() 
                   if not any(field in str(k) for field in intermediate_field_names)}
        return traverse_util.unflatten_dict(filtered)
    
    # Validate loaded params structure  
    params_shape_filtered = filter_intermediate_fields(params_shape)
    loaded_params_filtered = filter_intermediate_fields(loaded_params)
    at.check_pytree_equality(expected=params_shape_filtered, got=loaded_params_filtered, check_shapes=True, check_dtypes=True)
    
    # Remove jax.ShapeDtypeStruct and Intermediate fields from the loaded params
    def should_exclude(k, v):
        if isinstance(v, jax.ShapeDtypeStruct):
            return True
        # Exclude all intermediate fields
        intermediate_field_names = [
            'action_correlation_cholesky', 'L_spatial', 'L_temporal',
            'cached_num_inpaint_actions', 'cached_input_action_dim',
            'cached_Sigma_uo_Sigma_oo_inv', 'cached_L_cond_free',
            'cached_Sigma_ou_Sigma_uu_inv', 'cached_L_cond_inp',
        ]
        return any(field in str(k) for field in intermediate_field_names)
    
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() 
         if not should_exclude(k, v)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, 
    init_rng: at.KeyArrayLike, 
    mesh: jax.sharding.Mesh, 
    *, 
    resume: bool,
    norm_stats: dict | None = None
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)
        
        # Load correlation matrix into PiBehavior models BEFORE creating graphdef
        if isinstance(model, PiBehavior) and norm_stats is not None:
            model.load_correlation_matrix(norm_stats)
            logging.info("Loaded correlation matrix during model initialization")

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    
    # Log KV transform coefficients for PiBehavior models
    model = nnx.merge(train_state.model_def, train_state.params)
    if isinstance(model, PiBehavior) and hasattr(model, 'kv_transform') and model.kv_transform is not None:
        logging.info("KV Transform Coefficients (after loading):")
        logging.info("=" * 80)
        
        k_coeffs = model.kv_transform.k_coeffs.value
        v_coeffs = model.kv_transform.v_coeffs.value
        
        logging.info("K Coefficients (each layer attends to all VLM layers):")
        for i in range(k_coeffs.shape[0]):
            coeffs_str = ", ".join([f"{float(c):.2f}" for c in k_coeffs[i]])
            logging.info(f"  Layer {i:2d}: [{coeffs_str}]")
        
        logging.info("")
        logging.info("V Coefficients (each layer attends to all VLM layers):")
        for i in range(v_coeffs.shape[0]):
            coeffs_str = ", ".join([f"{float(c):.2f}" for c in v_coeffs[i]])
            logging.info(f"  Layer {i:2d}: [{coeffs_str}]")
        
        logging.info("=" * 80)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck  
    def loss_fn(
        model: PiBehavior, rng: at.KeyArrayLike, observation: Observation, actions: _model.Actions
    ):
        losses_dict = model.compute_detailed_loss(rng, observation, actions, train=True, num_flow_samples=config.num_flow_samples)
        total_loss = jnp.mean(losses_dict["total_loss"])
        return total_loss, losses_dict

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, losses_dict), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, observation, actions)
    
    # Knowledge insulation gradient monitoring
    if config.model.use_knowledge_insulation:
        # Helper functions to identify parameter groups
        def is_action_expert_param(path_str):
            # Action expert parameters: 
            # - Second LLM expert (300M params, marked with _1 suffix)
            # - Action projections, time MLPs, kv_transform
            return any(x in path_str for x in [
                "_1",  # All second expert parameters
                "action_in_proj",
                "action_out_proj",
                "time_mlp_in",
                "time_mlp_out",
                "kv_transform"
            ])
        
        def is_vlm_param(path_str):
            # VLM parameters: everything else (first expert, img, FAST, task modules)
            return not is_action_expert_param(path_str)
        
        # Compute gradient norms for monitoring only (no scaling applied)
        def compute_group_norm(grads_state, predicate):
            """Compute norm for gradients matching predicate."""
            flat_grads = []
            for path, value in jax.tree_util.tree_flatten_with_path(grads_state.to_pure_dict())[0]:
                path_str = "/".join(str(k) for k in path)
                if predicate(path_str):
                    if hasattr(value, 'value'):
                        flat_grads.append(value.value if hasattr(value, 'value') else value)
                    else:
                        flat_grads.append(value)
            
            if flat_grads:
                return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in flat_grads))
            return 0.0
        
        grad_norm_vlm = compute_group_norm(grads, is_vlm_param)
        grad_norm_action = compute_group_norm(grads, is_action_expert_param)
    else:
        grad_norm_vlm = None
        grad_norm_action = None

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    
    # Add gradient norm breakdown for knowledge insulation monitoring
    if grad_norm_vlm is not None:
        info["grad_norm_vlm"] = grad_norm_vlm
        info["grad_norm_action_expert"] = grad_norm_action

    # Add detailed loss components to info
    for key, value in losses_dict.items():
        if isinstance(value, (float, int)) or (hasattr(value, 'ndim') and value.ndim == 0):
            info[key] = value
        else:
            info[key] = jnp.mean(value)
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # Generate random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed for JAX RNG: {seed}")
    
    rng = jax.random.key(seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # [4/9 수정]
    log_gpu_mem("before data_loader create")

    data_loader = _data_loader.create_behavior_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    log_gpu_mem("after data_loader create")

    data_iter = iter(data_loader)
    batch = next(data_iter)

    log_gpu_mem("after first batch")
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    
    ###############################
    # [2026-04-22 수정]
    # 설명:
    # - baseline 장기 학습에서는 첫 batch 이미지 업로드를 끔
    # - W&B는 숫자(metric)만 남기고, 이미지 로깅 오버헤드는 피하기 위함
    if config.wandb_enabled:
        logging.info("W&B image logging is disabled for baseline/minimal tracking.")
    ###################################

    # Get norm_stats for correlation matrix loading
    data_config = data_loader.data_config()
    is_fake_smoke = getattr(data_config, "repo_id", None) == "fake"

    if data_config.norm_stats is None:
        if is_fake_smoke:
            logging.info("fake smoke 경로이므로 norm_stats 없이 계속 진행합니다.")
            norm_stats = None
        else:
            raise ValueError(
                "norm_stats not found. Run compute_norm_stats.py to generate normalization statistics."
            )
    else:
        norm_stats = data_config.norm_stats

    # [4/9 수정]
    train_state, train_state_sharding = init_train_state(
        config, init_rng, mesh, resume=resuming, norm_stats=norm_stats
    )

    block_and_log("after init_train_state", train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    #################

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
        
        # [4/9 추가]
        block_and_log("after restore_state", train_state)
        ##############

        # correlation matrix는 norm_stats가 있을 때만 다시 로드
        if norm_stats is not None:
            model = nnx.merge(train_state.model_def, train_state.params)
            model.load_correlation_matrix(norm_stats)
            logging.info("Reloaded correlation matrix after checkpoint restore")
            train_state = dataclasses.replace(train_state, model_def=nnx.graphdef(model))

    lr_fn = config.lr_schedule.create()

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
        # [2026-04-22 추가]
        # lr schedule config 객체 -> 실제 callable schedule 함수
    )

    # [2026-04-19 수정]
    # 목적:
    # - 첫 ptrain_step은 보통 compile + 실제 첫 forward/backward가 겹쳐서 메모리 피크가 크게 나타날 수 있음
    # - 그래서 이 구간만 따로 peak를 초기화하고, 끝난 직후 peak를 요약 출력
    reset_gpu_mem_peak()
    log_gpu_mem("before first ptrain_step")

    try:
        train_state, info = ptrain_step(train_rng, train_state, batch)
        block_and_log("after first ptrain_step", info["loss"])
        logging.info(f"[GPU MEM] first ptrain_step peak={_GPU_MEM_PEAK_MIB} MiB")
        logging.info(f"[TRACE] first step loss={float(jax.device_get(info['loss'])):.6f}")
    except Exception:
        logging.exception("[TRACE] failed during first ptrain_step")
        raise

    start_step = int(train_state.step)

    # [2026-04-19 수정]
    # 목적:
    # - 첫 ptrain_step peak와, 이후 train loop 전체 peak를 분리해서 보기 위함
    reset_gpu_mem_peak()

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        step_start_time = time.time()

        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        step_time_sec = time.time() - step_start_time
        infos.append(info)
        
        # [2026-04-19 수정]
        # 목적:
        # - 각 step 직후 메모리 사용량을 남겨서
        #   bs16에서 특정 step부터 급격히 증가하는지 확인
        block_and_log(f"step {step}", info["loss"])

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            current_lr = float(lr_fn(step))
            
            # Create a concise console log with main metrics
            main_metrics = {
                k: v for k, v in reduced_info.items()
                if k in [
                    "loss",
                    "total_loss",
                    "action_loss",
                    "subtask_loss",
                    "subtask_accuracy",
                    "grad_norm",
                    "param_norm",
                    "grad_norm_vlm",
                    "grad_norm_action_expert",
                ]
            }
            main_metrics["learning_rate"] = current_lr
            main_metrics["step_time"] = step_time_sec
            main_metrics["gpu_mem_peak_mib"] = _GPU_MEM_PEAK_MIB
            def _fmt_metric(v):
                if isinstance(v, (float, int)):
                    return f"{v:.8g}"
                return str(v)

            info_str = ", ".join(f"{k}={_fmt_metric(v)}" for k, v in main_metrics.items())
            pbar.write(f"Step {step}: {info_str}")
            
            if config.wandb_enabled:
                # [2026-04-22 수정]
                # 설명:
                # - W&B에는 비교실험에 필요한 최소 scalar만 기록
                # - reduced_info 전체 업로드 대신 필요한 항목만 선별
                wandb_payload = build_minimal_wandb_payload(
                    reduced_info,
                    step=step,
                    step_time_sec=step_time_sec,
                    gpu_mem_peak_mib=_GPU_MEM_PEAK_MIB,  # 현재 step에서 관측된 GPU 메모리 피크값
                    learning_rate=current_lr,
                )
                logging.info(f"[W&B PAYLOAD] step={step} payload_keys={list(wandb_payload.keys())} payload={wandb_payload}")
                wandb.log(wandb_payload, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

    # [2026-04-19 수정]
    # 목적:
    # - 전체 train loop에서 관측된 최대 GPU 메모리를 마지막에 한 번 더 요약
    logging.info(f"[GPU MEM] max observed during train loop: {_GPU_MEM_PEAK_MIB} MiB")

if __name__ == "__main__":
    main(_config.cli())