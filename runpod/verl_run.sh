#!/bin/bash
# In-pod entrypoint for a verl GRPO run using this project's noisy reward function.
#
# Not meant to be run by hand: runpod/launch.py passes this file as the pod's
# entrypoint and forwards every knob below as an environment variable.
#
# Layout on the pod's /workspace (a RunPod network volume, so it survives pod
# termination and the expensive environment build happens once):
#   /workspace/verl/                   verl checkout + its uv-built .venv  (cached)
#   /workspace/data/gsm8k/             preprocessed GSM8K parquet          (cached)
#   /workspace/team_repo/              this repo, checked out at REPO_REF  (per run)
#   /workspace/runs/<RUN_NAME>/        everything one run produces:
#       progress.txt  train_log.txt  manifest.txt  mixup_logs/  rollouts/
# An http.server on :8080 serves /workspace so a run can be watched and its files
# fetched (launch.py fetch) without SSH.
#
# Knobs (defaults in the block below). Reward-noise knobs are MIXUP_*, documented
# in reward_functions/persistence_reward.py.

export DEBIAN_FRONTEND=noninteractive
export WANDB_MODE=${WANDB_MODE:-disabled}
export PATH="$HOME/.local/bin:$PATH"

REPO_URL=${REPO_URL:-https://github.com/RoroTheDev/rlvr-verifier-degradation}
REPO_REF=${REPO_REF:-main}            # branch, tag or commit of this repo; pin a tag for real runs
VERL_REF=${VERL_REF:-}                # only used when verl is first installed onto the volume
RUN_NAME=${RUN_NAME:-run}
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
SEED=${SEED:-42}                      # training seed: data order, rollout sampling, loader
STEPS=${STEPS:-2}
TRAIN_BATCH=${TRAIN_BATCH:-16}
MINI_BATCH=${MINI_BATCH:-16}
MICRO_BATCH=${MICRO_BATCH:-2}
ROLLOUT_N=${ROLLOUT_N:-4}
MAX_PROMPT=${MAX_PROMPT:-512}
MAX_RESPONSE=${MAX_RESPONSE:-512}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.4}
DUMP_ROLLOUTS=${DUMP_ROLLOUTS:-0}     # 1 => dump per-sample generations + reward extras
export MIXUP_MODE=${MIXUP_MODE:-clean}
export MIXUP_LOG_DIR=/workspace/runs/${RUN_NAME}/mixup_logs

RUN_DIR=/workspace/runs/${RUN_NAME}
PROGRESS=${RUN_DIR}/progress.txt
mkdir -p "$RUN_DIR"
cd /workspace
python3 -m http.server 8080 >/dev/null 2>&1 &

stage () { echo "STAGE:$1 $(date -u +%FT%TZ)" >> "$PROGRESS"; }

{
  stage BOOT

  # --- this repo, at the pinned ref (the reward function comes from here) ---
  stage TEAM_REPO
  rm -rf /workspace/team_repo
  git clone "$REPO_URL" /workspace/team_repo >"$RUN_DIR/team_repo_clone.log" 2>&1
  git -C /workspace/team_repo checkout "$REPO_REF" >>"$RUN_DIR/team_repo_clone.log" 2>&1
  echo "TEAM_REPO_SHA:$(git -C /workspace/team_repo rev-parse HEAD)" >> "$PROGRESS"

  # --- verl + environment, built once onto the volume ---
  if [ -x /workspace/verl/.venv/bin/python ]; then
    stage REUSING_CACHED_ENV
  else
    stage VERL_INSTALL
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >"$RUN_DIR/uv_install.log" 2>&1
    git clone --depth 1 https://github.com/volcengine/verl /workspace/verl >"$RUN_DIR/verl_clone.log" 2>&1
    if [ -n "$VERL_REF" ]; then
      git -C /workspace/verl fetch --depth 1 origin "$VERL_REF" >>"$RUN_DIR/verl_clone.log" 2>&1
      git -C /workspace/verl checkout FETCH_HEAD >>"$RUN_DIR/verl_clone.log" 2>&1
    fi
  fi
  cd /workspace/verl
  UV_RUN="uv run --frozen --all-packages --extra vllm --extra fsdp"

  mkdir -p reward_functions
  cp /workspace/team_repo/reward_functions/persistence_reward.py reward_functions/persistence_reward.py

  if [ ! -f /workspace/data/gsm8k/train.parquet ]; then
    stage DATA_PREP
    $UV_RUN python3 examples/data_preprocess/gsm8k.py --local_save_dir /workspace/data/gsm8k \
      >"$RUN_DIR/data_prep.log" 2>&1
    echo "DATA_PREP_EXIT:$?" >> "$PROGRESS"
  fi

  # --- provenance: enough to say exactly what produced a number ---
  stage MANIFEST
  {
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "run_name=$RUN_NAME"
    echo "team_repo_url=$REPO_URL"
    echo "team_repo_ref=$REPO_REF"
    echo "team_repo_sha=$(git -C /workspace/team_repo rev-parse HEAD)"
    echo "reward_file_sha256=$(sha256sum reward_functions/persistence_reward.py | cut -d' ' -f1)"
    echo "verl_sha=$(git -C /workspace/verl rev-parse HEAD)"
    echo "verl_uv_lock_sha256=$(sha256sum uv.lock | cut -d' ' -f1)"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    /workspace/verl/.venv/bin/python - <<'PYEOF'
import importlib.metadata as m, sys
print("python=" + sys.version.split()[0])
for p in ("torch", "vllm", "ray", "transformers", "tensordict", "flash-attn", "numpy", "datasets"):
    try:
        print(f"{p}={m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p}=NOT_INSTALLED")
PYEOF
    env | grep -E '^(MIXUP_|MODEL=|SEED=|STEPS=|TRAIN_BATCH=|MINI_BATCH=|MICRO_BATCH=|ROLLOUT_N=|MAX_PROMPT=|MAX_RESPONSE=|GPU_MEM_UTIL=|DUMP_ROLLOUTS=)' | sort
  } > "$RUN_DIR/manifest.txt" 2>&1

  # --- train ---
  EXTRA=()
  if [ "$DUMP_ROLLOUTS" = "1" ]; then
    EXTRA+=("trainer.rollout_data_dir=${RUN_DIR}/rollouts")
  fi

  stage TRAIN_START
  PYTHONUNBUFFERED=1 $UV_RUN python3 -m verl.trainer.main_ppo \
    data.train_files=/workspace/data/gsm8k/train.parquet \
    data.val_files=/workspace/data/gsm8k/test.parquet \
    data.train_batch_size="$TRAIN_BATCH" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$MINI_BATCH" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.data_loader_seed="$SEED" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.seed="$SEED" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEM_UTIL" \
    actor_rollout_ref.rollout.n="$ROLLOUT_N" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward.custom_reward_function.path=reward_functions/persistence_reward.py \
    reward.custom_reward_function.name=compute_score \
    "ray_kwargs.ray_init.runtime_env.py_executable=${UV_RUN}" \
    trainer.critic_warmup=0 \
    "trainer.logger=[console]" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.total_training_steps="$STEPS" \
    "${EXTRA[@]}" \
    > "$RUN_DIR/train_log.txt" 2>&1
  TRAIN_EXIT=$?
  echo "TRAIN_EXIT:${TRAIN_EXIT}" >> "$RUN_DIR/train_log.txt"
  echo "STAGE:DONE exit=${TRAIN_EXIT} $(date -u +%FT%TZ)" >> "$PROGRESS"
} &
wait
