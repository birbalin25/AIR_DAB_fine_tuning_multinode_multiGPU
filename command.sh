#!/usr/bin/env bash
set -euo pipefail

cd "$CODE_SOURCE_PATH"

# AI Runtime applies the job's environment_variables only for a user run_as, NOT
# for a service-principal run_as (the launcher logs "Applying 0 user env var(s)").
# So fall back to defaults for every value this script needs beyond the ones AI
# Runtime always injects (NODE_RANK, MASTER_ADDR, LOCAL_WORLD_SIZE, CODE_SOURCE_PATH).
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_JOIN_RETRIES="${RAY_JOIN_RETRIES:-30}"
RAY_JOIN_SLEEP_SECONDS="${RAY_JOIN_SLEEP_SECONDS:-5}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-train_llama_ray_fsdp.py}"

GPUS_PER_NODE="$LOCAL_WORLD_SIZE"

if [ "$NODE_RANK" = "0" ]; then
  # Node 0 hosts the Ray head and runs the Ray Train driver.
  ray start --head --port="$RAY_HEAD_PORT" --num-gpus="$GPUS_PER_NODE" --dashboard-host=0.0.0.0
  trap 'ray stop >/dev/null 2>&1 || true' EXIT
  python "$TRAINING_SCRIPT"
  # On exit the trap stops the head, which the worker nodes detect below.
else
  # Join the head (retry until it is up).
  joined=0
  for attempt in $(seq 1 "$RAY_JOIN_RETRIES"); do
    if ray start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" --num-gpus="$GPUS_PER_NODE"; then
      joined=1
      break
    fi
    echo "Ray head not ready; retry in ${RAY_JOIN_SLEEP_SECONDS}s (attempt ${attempt}/${RAY_JOIN_RETRIES})" >&2
    sleep "$RAY_JOIN_SLEEP_SECONDS"
  done
  [ "$joined" = 1 ] || { echo "Unable to join the Ray head" >&2; exit 1; }

  # Stay alive while the head is reachable, then exit once node 0 finishes and
  # stops it. This replaces `ray start ... --block`, which did NOT return after
  # the head stopped and left the whole task hanging in teardown until timeout.
  # Require 3 consecutive health failures so a transient blip during training
  # does not drop this worker out of the cluster.
  misses=0
  while true; do
    if ray status >/dev/null 2>&1; then
      misses=0
    else
      misses=$((misses + 1))
      [ "$misses" -ge 3 ] && break
    fi
    sleep "$RAY_JOIN_SLEEP_SECONDS"
  done
  ray stop >/dev/null 2>&1 || true
fi
