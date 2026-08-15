#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"

export IMAGENET_PATH="${IMAGENET_PATH:-$HOME/datasets/imagenet-1k}"
export IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-$HOME/datasets/imagenet_latent_cache}"

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
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

LATENT_BATCH_SIZE="${LATENT_BATCH_SIZE:-64}"
LATENT_NUM_WORKERS="${LATENT_NUM_WORKERS:-8}"
LATENT_PREFETCH_FACTOR="${LATENT_PREFETCH_FACTOR:-2}"
LATENT_SAVE_WORKERS="${LATENT_SAVE_WORKERS:-4}"
CONFIG_PATH="${CONFIG_PATH:-configs/gen/latent_ablation_a800_hpc.yaml}"
WORKDIR="${WORKDIR:-runs/gen_latent_a800_hpc}"

cd "$PROJECT_DIR"

echo "[INFO] host=$(hostname)"
echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] python=$VENV_DIR/bin/python"
echo "[INFO] imagenet=$IMAGENET_PATH"
echo "[INFO] latent_cache=$IMAGENET_CACHE_PATH"
echo "[INFO] hf_root=$HF_ROOT"
echo "[INFO] config=$CONFIG_PATH"
echo "[INFO] workdir=$WORKDIR"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

cache_has_split() {
  local split="$1"
  [[ -d "$IMAGENET_CACHE_PATH/$split" ]] && find "$IMAGENET_CACHE_PATH/$split" -name '*.pt' -print -quit | grep -q .
}

if cache_has_split train && cache_has_split val; then
  echo "[STEP] Reuse existing VAE latent cache"
else
  echo "[STEP] Build VAE latent cache"
  "$VENV_DIR/bin/python" -m dataset.latent \
    --backend cuda \
    --data-path "$IMAGENET_PATH" \
    --target-path "$IMAGENET_CACHE_PATH" \
    --local-batch-size "$LATENT_BATCH_SIZE" \
    --num-workers "$LATENT_NUM_WORKERS" \
    --prefetch-factor "$LATENT_PREFETCH_FACTOR" \
    --pin-memory \
    --save-workers "$LATENT_SAVE_WORKERS"
fi

echo "[STEP] Train latent generator"
"$VENV_DIR/bin/python" main.py \
  --gen \
  --config "$CONFIG_PATH" \
  --workdir "$WORKDIR"
