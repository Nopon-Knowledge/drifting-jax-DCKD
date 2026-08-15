#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"

export HF_ROOT="${HF_ROOT:-$HOME/datasets/hf_cache}"
export HF_HOME="${HF_HOME:-$HF_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_ROOT/hub}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-$HF_ROOT/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_ROOT/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

cd "$PROJECT_DIR"

echo "[INFO] host=$(hostname)"
echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] python=$VENV_DIR/bin/python"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

"$VENV_DIR/bin/python" - <<'PY'
import ssl

import jax
import torch

print("ssl", ssl.OPENSSL_VERSION)
print("jax", jax.__version__, jax.devices())
print(
    "torch",
    torch.__version__,
    torch.cuda.is_available(),
    torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda",
)
PY
