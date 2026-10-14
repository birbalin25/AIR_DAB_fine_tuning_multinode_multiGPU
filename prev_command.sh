#!/usr/bin/env bash
# ============================================================================
# ARCHIVED — PREVIOUS VERSION OF command.sh (do NOT deploy this).
# Kept only for before/after comparison: `diff prev_command.sh command.sh`.
#
# This is the version that was live when the 24-GPU / 3-node run HUNG in
# teardown. Two problems, both fixed in the current command.sh:
#   1. Worker nodes used `ray start ... --block`, which did not return after
#      node 0 stopped the Ray head -> the worker command.sh never exited -> the
#      AI Runtime task stayed RUNNING until the 2h timeout (even though training
#      finished and node 0 exited 0).
#   2. It hard-referenced $RAY_HEAD_PORT / $RAY_JOIN_* / $TRAINING_SCRIPT under
#      `set -u`. AI Runtime does NOT apply the job's environment_variables under
#      a service-principal run_as ("Applying 0 user env var(s)"), so those were
#      unbound -> `line 10: RAY_HEAD_PORT: unbound variable` -> exit 1.
# ============================================================================
set -euo pipefail

cd "$CODE_SOURCE_PATH"

GPUS_PER_NODE="$LOCAL_WORLD_SIZE"

if [ "$NODE_RANK" = "0" ]; then
  # Node 0 hosts the Ray head and runs the Ray Train driver.
  ray start --head --port="$RAY_HEAD_PORT" --num-gpus="$GPUS_PER_NODE" --dashboard-host=0.0.0.0
  trap 'ray stop >/dev/null 2>&1 || true' EXIT
  python "$TRAINING_SCRIPT"
else
  # Other nodes join the head and block until training completes.
  for attempt in $(seq 1 "$RAY_JOIN_RETRIES"); do
    if ray start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" --num-gpus="$GPUS_PER_NODE" --block; then
      exit 0
    fi
    sleep "$RAY_JOIN_SLEEP_SECONDS"
  done
  echo "Unable to join the Ray head" >&2
  exit 1
fi
