#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"
INIT_FROM="${INIT_FROM:?INIT_FROM must point to a completed training workdir}"
OUTDIR="${OUTDIR:?OUTDIR is required}"
SELECTION_JSON="${SELECTION_JSON:-$PROJECT_DIR/runs/fid_v51_cfg_selection.json}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-2592000}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"
CLASS_IDS="${CLASS_IDS:-22,88,95,108,207,283,296,386,483,562,604,698,717,805,908,985}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-4}"
SEED="${SEED:-0}"
CHECKPOINT_META="$INIT_FROM/params_ema/metadata.json"
CHECKPOINT_PARAMS="$INIT_FROM/params_ema/ema_params.msgpack"

wait_for_file() {
  local path="$1"
  local waited=0
  while [[ ! -s "$path" ]]; do
    if (( waited >= WAIT_TIMEOUT_SECONDS )); then
      echo "[ERR] Timed out waiting for $path" >&2
      exit 3
    fi
    echo "[WAIT] prerequisite not ready: $path (${waited}s elapsed)"
    sleep "$WAIT_INTERVAL_SECONDS"
    waited=$((waited + WAIT_INTERVAL_SECONDS))
  done
}

wait_for_file "$SELECTION_JSON"
wait_for_file "$CHECKPOINT_META"
wait_for_file "$CHECKPOINT_PARAMS"

CFG_SCALE="$({
  "$VENV_DIR/bin/python" - "$SELECTION_JSON" "$CHECKPOINT_META" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    selected_cfg = float(json.load(handle)["selected_cfg"])
with open(sys.argv[2], encoding="utf-8") as handle:
    step = int(json.load(handle).get("step", -1))
if step != 30000:
    raise SystemExit(f"checkpoint is not final: expected step 30000, got {step}")
print(selected_cfg)
PY
})"

echo "[INFO] selected_cfg=$CFG_SCALE"
echo "[INFO] selection_record=$SELECTION_JSON"
echo "[INFO] init_from=$INIT_FROM"
echo "[INFO] outdir=$OUTDIR"

exec /usr/bin/env \
  INIT_FROM="$INIT_FROM" \
  CLASS_IDS="$CLASS_IDS" \
  SAMPLES_PER_CLASS="$SAMPLES_PER_CLASS" \
  CFG_SCALE="$CFG_SCALE" \
  SEED="$SEED" \
  BATCH_SIZE=8 \
  OUTDIR="$OUTDIR" \
  /bin/bash "$PROJECT_DIR/scripts/run_inference_a800_hpc.sh"
