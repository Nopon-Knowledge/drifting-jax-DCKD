#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"
SOURCE_CONFIG="${SOURCE_CONFIG:-$PROJECT_DIR/configs/gen/latent_ablation_a800_v5_1_dckd_smooth_rank8.yaml}"
ABLATION="${ABLATION:?ABLATION is required}"
WORKDIR="${WORKDIR:-$PROJECT_DIR/runs/gen_latent_a800_v5_1_ablate_${ABLATION}}"
TRAIN_SEED="${TRAIN_SEED:-42}"
CONFIG_SNAPSHOT="$WORKDIR/config_${ABLATION}.yaml"

case "$ABLATION" in
  mode_global|rank4|reference01|multipliers_wide|clamp_v41|gradclip2) ;;
  *)
    echo "[ERR] Unsupported v5.1 ablation: $ABLATION" >&2
    exit 2
    ;;
esac

mkdir -p "$WORKDIR"

"$VENV_DIR/bin/python" - "$SOURCE_CONFIG" "$CONFIG_SNAPSHOT" "$ABLATION" <<'PY'
import copy
import os
import sys
import tempfile

import yaml

source_path, output_path, ablation = sys.argv[1:]
with open(source_path, encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config = copy.deepcopy(config)
loss = config["train"]["loss_kwargs"]

if ablation == "mode_global":
    loss["adaptive_radius_mode"] = "global"
elif ablation == "rank4":
    loss["adaptive_rank"] = 4
elif ablation == "reference01":
    loss["adaptive_reference_weight"] = 0.1
elif ablation == "multipliers_wide":
    loss["adaptive_multipliers"] = [0.5, 1.0, 2.0]
elif ablation == "clamp_v41":
    loss["adaptive_min_radius"] = 0.01
    loss["adaptive_max_radius"] = 0.5
elif ablation == "gradclip2":
    config["train"]["max_grad_norm"] = 2.0
else:
    raise SystemExit(f"unsupported ablation: {ablation}")

config["experiment"] = {
    "parent": "v5.1 DCKD-Smooth-Rank8",
    "ablation": ablation,
    "design": "single-factor control; all unspecified fields inherit v5.1",
}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
fd, temporary_path = tempfile.mkstemp(prefix=".config.", dir=os.path.dirname(output_path))
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    os.replace(temporary_path, output_path)
finally:
    if os.path.exists(temporary_path):
        os.unlink(temporary_path)
PY

echo "[INFO] ablation=$ABLATION"
echo "[INFO] source_config=$SOURCE_CONFIG"
echo "[INFO] config_snapshot=$CONFIG_SNAPSHOT"
echo "[INFO] workdir=$WORKDIR"
echo "[INFO] train_seed=$TRAIN_SEED"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY-RUN] Configuration generated; training was not started."
  exit 0
fi

export CONFIG_PATH="$CONFIG_SNAPSHOT"
export WORKDIR
export TRAIN_SEED
exec /bin/bash "$PROJECT_DIR/scripts/run_latent_replay_a800_hpc.sh"
