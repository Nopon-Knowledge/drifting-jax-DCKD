#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"
INIT_FROM="${INIT_FROM:?INIT_FROM must point to a completed training workdir}"
RUN_LABEL="${RUN_LABEL:?RUN_LABEL is required, for example v51 or v51_seed123}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
VAE_DECODE_BATCH_SIZE="${VAE_DECODE_BATCH_SIZE:-32}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-172800}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-60}"

case "$RUN_LABEL" in
  *[!A-Za-z0-9_-]*|'')
    echo "[ERR] RUN_LABEL contains unsupported characters: $RUN_LABEL" >&2
    exit 2
    ;;
esac

SCREEN_20="$PROJECT_DIR/runs/fid_v51_cfg2p0_10k/results_v51_cfg2p0_10k.json"
SCREEN_25="$PROJECT_DIR/runs/fid_v51_cfg2p5_10k/results_v51_cfg2p5_10k.json"
SCREEN_30="$PROJECT_DIR/runs/fid_v51_cfg3p0_10k/results_v51_cfg3p0_10k.json"
SELECTION_JSON="$PROJECT_DIR/runs/fid_v51_cfg_selection.json"
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

wait_for_file "$SCREEN_20"
wait_for_file "$SCREEN_25"
wait_for_file "$SCREEN_30"
wait_for_file "$CHECKPOINT_META"
wait_for_file "$CHECKPOINT_PARAMS"

"$VENV_DIR/bin/python" - "$CHECKPOINT_META" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    metadata = json.load(handle)
step = int(metadata.get("step", -1))
if step != 30000:
    raise SystemExit(f"checkpoint is not final: expected step 30000, got {step}")
PY

SELECTED_CFG="$({
  "$VENV_DIR/bin/python" - "$SELECTION_JSON" "$SCREEN_20" "$SCREEN_25" "$SCREEN_30" <<'PY'
import json
import os
import sys
import tempfile

selection_path = sys.argv[1]
paths = dict(zip(("2.0", "2.5", "3.0"), sys.argv[2:]))
scores = {}
for cfg, path in paths.items():
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    scores[cfg] = {
        "fid": float(result["fid"]),
        "isc_mean": float(result["isc_mean"]),
        "source": path,
    }

best_cfg = min(scores, key=lambda cfg: (scores[cfg]["fid"], -scores[cfg]["isc_mean"]))
payload = {
    "selection_rule": "minimum 10k FID; higher IS breaks an exact FID tie",
    "selected_cfg": float(best_cfg),
    "scores": scores,
}
os.makedirs(os.path.dirname(selection_path), exist_ok=True)
fd, temporary_path = tempfile.mkstemp(prefix=".v51_cfg_selection.", dir=os.path.dirname(selection_path))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, selection_path)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
print(best_cfg)
PY
})"

case "$SELECTED_CFG" in
  2.0) CFG_TAG="2p0" ;;
  2.5) CFG_TAG="2p5" ;;
  3.0) CFG_TAG="3p0" ;;
  *)
    echo "[ERR] Unexpected selected CFG: $SELECTED_CFG" >&2
    exit 4
    ;;
esac

SAMPLE_TAG="$((NUM_SAMPLES / 1000))k"
WORKDIR="${WORKDIR:-$PROJECT_DIR/runs/fid_${RUN_LABEL}_cfg${CFG_TAG}_${SAMPLE_TAG}}"
JSON_OUT="${JSON_OUT:-$WORKDIR/results_${RUN_LABEL}_cfg${CFG_TAG}_${SAMPLE_TAG}.json}"

echo "[INFO] selected_cfg=$SELECTED_CFG"
echo "[INFO] selection_record=$SELECTION_JSON"
echo "[INFO] init_from=$INIT_FROM"
echo "[INFO] workdir=$WORKDIR"
echo "[INFO] json_out=$JSON_OUT"

if [[ -s "$JSON_OUT" ]]; then
  echo "[SKIP] Existing result: $JSON_OUT"
  cat "$JSON_OUT"
  exit 0
fi

exec /usr/bin/env \
  INIT_FROM="$INIT_FROM" \
  CFG_SCALE="$SELECTED_CFG" \
  NUM_SAMPLES="$NUM_SAMPLES" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  VAE_DECODE_BATCH_SIZE="$VAE_DECODE_BATCH_SIZE" \
  WORKDIR="$WORKDIR" \
  JSON_OUT="$JSON_OUT" \
  /bin/bash "$PROJECT_DIR/scripts/run_fid_a800_hpc.sh"
