#!/usr/bin/env bash
# Planner (outline) RL with GRPO and a frozen executor. Hyperparameters are the paper recipe.
#
# Usage (from anywhere): bash rl/train_1.7b.sh
#
# Needs: data from `python rl/prepare_data.py` in $DATA_DIR, and the modified verl dependency installed
# (`pip install -e verl`). Checkpoints go to $CKPT_DIR/global_step_*; the paper used step 240.
set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)

PLANNER_MODEL=Qwen/Qwen3-1.7B-Base
EXECUTOR_MODEL=Qwen/Qwen3-1.7B
EXP_NAME=${EXP_NAME:-ptts-planner-1.7b}
PROJECT_NAME=${PROJECT_NAME:-ptts}
N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}
DATA_DIR=${DATA_DIR:-${REPO_ROOT}/data}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-${DATA_DIR}/test.parquet}
CKPT_DIR=${CKPT_DIR:-${REPO_ROOT}/checkpoints/${EXP_NAME}}
LOGGER=${LOGGER:-'["console"]'}   # e.g. '["console","wandb"]' or '["console","swanlab"]'
TOTAL_STEPS=${TOTAL_STEPS:-2000}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}

# ---- paper recipe ----------------------------------------------------------------------------
num_outlines=4                         # outlines per planner generation
max_prompt_length=$((1024 * 6))
max_response_length=$((1024 * 4))      # cap for both planner and executor generations
max_total_length=$((1024 * 12))
n_resp_per_prompt=16                   # GRPO group size
train_prompt_bsz=16
train_prompt_mini_bsz=16
lr=1e-6
clip_ratio_low=0.2
clip_ratio_high=0.28
temperature=1.0
top_p=1.0
top_k=-1
val_temperature=0.7
outline_failed_reward=-1.0             # planner output could not be parsed into $num_outlines outlines
outline_format_reward=0.0              # parsed fine, but every executor solution was wrong
# performance
use_dynamic_bsz=True
actor_ppo_max_token_len=$((1024 * 32))
infer_ppo_max_token_len=$((1024 * 160))
offload=True
# ----------------------------------------------------------------------------------------------

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${PLANNER_MODEL}" \
    +actor_rollout_ref.model.override_config.attention_dropout=0. \
    +actor_rollout_ref.model.override_config.embd_pdrop=0. \
    +actor_rollout_ref.model.override_config.resid_pdrop=0. \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.ref.entropy_checkpointing=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${max_total_length} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    custom_reward_function.path="${REPO_ROOT}/rl/math_reward.py" \
    custom_reward_function.name=reward_fn \
    outline.training_component=outline \
    outline.num_outlines=${num_outlines} \
    outline.outline_max_response_length=${max_response_length} \
    outline.frozen_solution_rollout.enable=True \
    outline.frozen_solution_rollout.model_path="${EXECUTOR_MODEL}" \
    outline.outline_failed_reward=${outline_failed_reward} \
    outline.outline_format_reward=${outline_format_reward} \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_before_train=False \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${TOTAL_STEPS} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.validation_data_dir="${CKPT_DIR}/validation_data" \
    trainer.rollout_data_dir="${CKPT_DIR}/rollout_data" \
    trainer.resume_mode=auto \
    +ray_kwargs.ray_init.address=local
