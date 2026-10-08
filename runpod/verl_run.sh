#!/bin/bash
# In-pod entrypoint for a verl GRPO run using this project's noisy reward function.
#
# Not meant to be run by hand: runpod/launch.py passes this file as the pod's
# entrypoint and forwards every knob below as an environment variable.
#
# Design: pods are STATELESS. Everything is rebuilt from pinned commits on each
# launch (~4 min), and results are pulled off with `launch.py fetch` before the
# pod is stopped. No persistent volume is needed.
#
# Where things live, and why it matters:
#   $WORK (default /root/work)  -- the container's LOCAL disk. verl, its uv-built
#       .venv, the uv/HF caches and Ray's temp files all go here.
#   /workspace                  -- on RunPod Secure Cloud this is a network FUSE
#       filesystem (MooseFS), even for a plain "pod volume". Building the env
#       there was ~3x slower (3.5 min vs 1.3 min), so it only holds small outputs.
#       (It is NOT what caused the startup hangs seen in earlier runs: those were
#       TransferQueue starving Ray of CPUs -- see TQ_STORAGE_UNITS below.)
#         /workspace/runs/<RUN_NAME>/
#             progress.txt train_log.txt manifest.txt mixup_logs/ rollouts/ ray_logs/
#       An http.server on :8080 serves /workspace so a run can be watched and
#       its files fetched without SSH.
#
# Knobs (defaults in the block below). Reward-noise knobs are MIXUP_*, documented
# in reward_functions/persistence_reward.py.

export DEBIAN_FRONTEND=noninteractive
export PATH="$HOME/.local/bin:$PATH"

WORK=${WORK:-/root/work}
REPO_URL=${REPO_URL:-https://github.com/RoroTheDev/rlvr-verifier-degradation}
REPO_REF=${REPO_REF:-main}            # branch, tag or commit of this repo; pin a tag for real runs
VERL_REF=${VERL_REF:-}                # verl commit to install; EMPTY MEANS LATEST MAIN -- pin it for real runs
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
# verl's v1 trainer always starts a TransferQueue whose default is 8 storage units
# (+1 controller), each pinning 1 Ray CPU. Ray only sees the pod's CPU quota
# (~10 on a 4090 pod, not the host's 64), so the default leaves too few CPUs to
# place the FSDP worker group and startup hangs silently at "worker group kwargs".
TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-1}
EPOCHS=${EPOCHS:-1}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}   # -1 = whole train set; set to TRAIN_BATCH with EPOCHS>1 to revisit the same tasks
# Task / data. gsm8k = verl's stock converter (smoke tests). mbpp = the Plesner et al.
# setting: data_prep/mbpp_prep.py writes MBPP_PASSES back-to-back passes over the 374
# train problems (each row carries its `encounter` number, so group_resampled noise can
# be redrawn per encounter) and verl runs with data.shuffle=False, one epoch. The pass
# order depends on SEED, so training seeds see different data orders.
TASK=${TASK:-gsm8k}
MBPP_PASSES=${MBPP_PASSES:-34}          # 34 passes x 374 problems / 48 prompts per step = 264 steps
LR=${LR:-1e-6}
CLIP_HIGH=${CLIP_HIGH:-0.2}             # the paper uses 0.28 (clip-higher) with clip_low 0.2
ENABLE_THINKING=${ENABLE_THINKING:-}    # empty = model default; 0/1 = Qwen3 chat-template enable_thinking
VAL_FREQ=${VAL_FREQ:--1}                # validate every N steps; -1 = never
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
VAL_N=${VAL_N:-1}                       # samples per validation prompt (paper: 16)
VAL_TEMP=${VAL_TEMP:-0}                 # paper's eval script samples at 0.7
VAL_SAMPLE=${VAL_SAMPLE:-False}
export MIXUP_MODE=${MIXUP_MODE:-clean}
# Which tasks a persistent mask corrupts. Defaults to the training seed so seed-to-seed
# spread includes which tasks were corrupted; set MIXUP_MASK_SEED to pin one mask.
export MIXUP_MASK_SEED=${MIXUP_MASK_SEED:-$SEED}
export MIXUP_LOG_DIR=/workspace/runs/${RUN_NAME}/mixup_logs

# Weights & Biases: off by default. WANDB=1 needs WANDB_API_KEY in the pod env
# (launch.py --wandb forwards it from your local shell; it is never written to the
# manifest or logs). Pods are non-interactive, so this env var replaces
# `wandb login`, and wandb is already a verl dependency (no `uv add` needed).
# Optional: WANDB_ENTITY (team/user to log under), WANDB_PROJECT.
WANDB=${WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-rlvr-verifier-degradation}
if [ "$WANDB" = "1" ]; then
  export WANDB_MODE=online
  LOGGER="[console,wandb]"
else
  export WANDB_MODE=disabled
  LOGGER="[console]"
fi

# Keep every cache next to the venv, on the same local filesystem (uv hardlinks
# from its cache into the venv; across filesystems it falls back to slow copies).
mkdir -p "$WORK"
export XDG_CACHE_HOME="$WORK/.cache"
export UV_CACHE_DIR="$WORK/.cache/uv"
export HF_HOME="$WORK/.cache/huggingface"

RUN_DIR=/workspace/runs/${RUN_NAME}
PROGRESS=${RUN_DIR}/progress.txt
mkdir -p "$RUN_DIR"
cd /workspace
python3 -m http.server 8080 >/dev/null 2>&1 &

stage () { echo "STAGE:$1 $(date -u +%FT%TZ)" >> "$PROGRESS"; }

# Mirror Ray's logs next to the other outputs so a stalled run can be diagnosed
# after the fact (they otherwise live in /tmp and die with the pod).
(
  while true; do
    sleep 60
    if [ -d /tmp/ray/session_latest/logs ]; then
      mkdir -p "$RUN_DIR/ray_logs"
      cp -ru /tmp/ray/session_latest/logs/. "$RUN_DIR/ray_logs/" 2>/dev/null
    fi
  done
) &

{
  stage BOOT

  # --- this repo, at the pinned ref (the reward function comes from here) ---
  stage TEAM_REPO
  rm -rf "$WORK/team_repo"
  git clone "$REPO_URL" "$WORK/team_repo" >"$RUN_DIR/team_repo_clone.log" 2>&1
  git -C "$WORK/team_repo" checkout "$REPO_REF" >>"$RUN_DIR/team_repo_clone.log" 2>&1
  echo "TEAM_REPO_SHA:$(git -C "$WORK/team_repo" rev-parse HEAD)" >> "$PROGRESS"

  # --- verl + environment ---
  if [ -x "$WORK/verl/.venv/bin/python" ]; then
    stage REUSING_LOCAL_ENV
  else
    stage VERL_INSTALL
    command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >"$RUN_DIR/uv_install.log" 2>&1
    git clone --depth 1 https://github.com/volcengine/verl "$WORK/verl" >"$RUN_DIR/verl_clone.log" 2>&1
    if [ -n "$VERL_REF" ]; then
      git -C "$WORK/verl" fetch --depth 1 origin "$VERL_REF" >>"$RUN_DIR/verl_clone.log" 2>&1
      git -C "$WORK/verl" checkout FETCH_HEAD >>"$RUN_DIR/verl_clone.log" 2>&1
    fi
  fi
  cd "$WORK/verl"
  UV_RUN="uv run --frozen --all-packages --extra vllm --extra fsdp"

  mkdir -p reward_functions
  cp "$WORK/team_repo/reward_functions/persistence_reward.py" "$WORK/team_repo/reward_functions/mbpp_scoring.py" reward_functions/

  EXTRA=()
  if [ "$TASK" = "mbpp" ]; then
    DATA_DIR="$WORK/data/mbpp-seed${SEED}-passes${MBPP_PASSES}"
    if [ ! -f "$DATA_DIR/train.parquet" ]; then
      stage DATA_PREP
      mkdir -p data_prep
      cp "$WORK/team_repo/data_prep/mbpp_prep.py" data_prep/mbpp_prep.py
      $UV_RUN python3 data_prep/mbpp_prep.py --out "$DATA_DIR" --epochs "$MBPP_PASSES" --order-seed "$SEED" \
        >"$RUN_DIR/data_prep.log" 2>&1
      echo "DATA_PREP_EXIT:$?" >> "$PROGRESS"
      grep -E "^(train|val) rows" "$RUN_DIR/data_prep.log" >> "$PROGRESS"
    fi
    TRAIN_FILE="$DATA_DIR/train.parquet"
    VAL_FILE="$DATA_DIR/val.parquet"
    EXTRA+=("data.shuffle=False")      # the file is already in the order we want
  else
    DATA_DIR="$WORK/data/gsm8k"
    if [ ! -f "$DATA_DIR/train.parquet" ]; then
      stage DATA_PREP
      $UV_RUN python3 examples/data_preprocess/gsm8k.py --local_save_dir "$DATA_DIR" \
        >"$RUN_DIR/data_prep.log" 2>&1
      echo "DATA_PREP_EXIT:$?" >> "$PROGRESS"
    fi
    TRAIN_FILE="$DATA_DIR/train.parquet"
    VAL_FILE="$DATA_DIR/test.parquet"
  fi
  if [ "$ENABLE_THINKING" = "0" ]; then
    EXTRA+=("+data.apply_chat_template_kwargs.enable_thinking=False")
  elif [ "$ENABLE_THINKING" = "1" ]; then
    EXTRA+=("+data.apply_chat_template_kwargs.enable_thinking=True")
  fi

  # --- provenance: enough to say exactly what produced a number ---
  stage MANIFEST
  {
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "run_name=$RUN_NAME"
    echo "team_repo_url=$REPO_URL"
    echo "team_repo_ref=$REPO_REF"
    echo "team_repo_sha=$(git -C "$WORK/team_repo" rev-parse HEAD)"
    echo "reward_file_sha256=$(sha256sum reward_functions/persistence_reward.py | cut -d' ' -f1)"
    echo "mbpp_scoring_sha256=$(sha256sum reward_functions/mbpp_scoring.py | cut -d' ' -f1)"
    echo "task=$TASK"
    echo "train_parquet_sha256=$(sha256sum "$TRAIN_FILE" | cut -d' ' -f1)"
    echo "val_parquet_sha256=$(sha256sum "$VAL_FILE" | cut -d' ' -f1)"
    echo "verl_ref_requested=${VERL_REF:-<latest main, unpinned>}"
    echo "verl_sha=$(git -C "$WORK/verl" rev-parse HEAD)"
    echo "verl_uv_lock_sha256=$(sha256sum uv.lock | cut -d' ' -f1)"
    echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    # host facts that explain run-to-run startup variance
    echo "host_nproc=$(nproc)"
    echo "host_mem_gb=$(free -g | awk '/^Mem:/{print $2}')"
    echo "work_fs=$(df -T "$WORK" | awk 'NR==2{print $2" "$3}')"
    echo "workspace_fs=$(df -T /workspace | awk 'NR==2{print $2" "$3}')"
    echo "workspace_mount=$(grep ' /workspace ' /proc/mounts | head -1 | cut -d' ' -f1,3)"
    "$WORK/verl/.venv/bin/python" - <<'PYEOF'
import importlib.metadata as m, sys
print("python=" + sys.version.split()[0])
for p in ("torch", "vllm", "ray", "transformers", "tensordict", "flash-attn", "numpy", "datasets"):
    try:
        print(f"{p}={m.version(p)}")
    except m.PackageNotFoundError:
        print(f"{p}=NOT_INSTALLED")
PYEOF
    env | grep -E '^(MIXUP_|MODEL=|SEED=|STEPS=|TRAIN_BATCH=|MINI_BATCH=|MICRO_BATCH=|ROLLOUT_N=|MAX_PROMPT=|MAX_RESPONSE=|GPU_MEM_UTIL=|DUMP_ROLLOUTS=|TQ_STORAGE_UNITS=|EPOCHS=|TRAIN_MAX_SAMPLES=|WANDB=|WANDB_PROJECT=|WANDB_ENTITY=|TASK=|MBPP_PASSES=|LR=|CLIP_HIGH=|ENABLE_THINKING=|VAL_FREQ=|VAL_N=|VAL_TEMP=|VAL_SAMPLE=|VAL_BEFORE_TRAIN=)' | sort
  } > "$RUN_DIR/manifest.txt" 2>&1

  # --- train ---
  if [ "$DUMP_ROLLOUTS" = "1" ]; then
    EXTRA+=("trainer.rollout_data_dir=${RUN_DIR}/rollouts")
  fi

  stage TRAIN_START
  PYTHONUNBUFFERED=1 $UV_RUN python3 -m verl.trainer.main_ppo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size="$TRAIN_BATCH" \
    data.train_max_samples="$TRAIN_MAX_SAMPLES" \
    data.max_prompt_length="$MAX_PROMPT" \
    data.max_response_length="$MAX_RESPONSE" \
    data.seed="$SEED" \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.actor.optim.lr="$LR" \
    actor_rollout_ref.actor.clip_ratio_high="$CLIP_HIGH" \
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
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMP" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_SAMPLE" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$MICRO_BATCH" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    transfer_queue.backend.SimpleStorage.num_data_storage_units="$TQ_STORAGE_UNITS" \
    reward.custom_reward_function.path=reward_functions/persistence_reward.py \
    reward.custom_reward_function.name=compute_score \
    "ray_kwargs.ray_init.runtime_env.py_executable=${UV_RUN}" \
    trainer.critic_warmup=0 \
    "trainer.logger=${LOGGER}" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$RUN_NAME" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq="$VAL_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    trainer.total_epochs="$EPOCHS" \
    trainer.total_training_steps="$STEPS" \
    "${EXTRA[@]}" \
    > "$RUN_DIR/train_log.txt" 2>&1
  TRAIN_EXIT=$?
  echo "TRAIN_EXIT:${TRAIN_EXIT}" >> "$RUN_DIR/train_log.txt"
  echo "STAGE:DONE exit=${TRAIN_EXIT} $(date -u +%FT%TZ)" >> "$PROGRESS"
} &
wait
