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
XLA_GPU_AUTOTUNE_LEVEL="${XLA_GPU_AUTOTUNE_LEVEL:-0}"
INFERENCE_XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=$XLA_GPU_AUTOTUNE_LEVEL"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax-drifting-inference}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

INIT_FROM="${INIT_FROM:-$PROJECT_DIR/runs/gen_latent_a800_hpc}"
CLASS_IDS="${CLASS_IDS:-95,22,88,108,386,296,483,698}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-1}"
CFG_SCALE="${CFG_SCALE:-2.0}"
SEED="${SEED:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OUTDIR="${OUTDIR:-$PROJECT_DIR/runs/inference_samples}"

cd "$PROJECT_DIR"

echo "[INFO] host=$(hostname)"
echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] python=$VENV_DIR/bin/python"
echo "[INFO] init_from=$INIT_FROM"
echo "[INFO] class_ids=$CLASS_IDS"
echo "[INFO] outdir=$OUTDIR"
echo "[INFO] xla_gpu_autotune_level=$XLA_GPU_AUTOTUNE_LEVEL"
echo "[INFO] jax_compilation_cache=$JAX_COMPILATION_CACHE_DIR"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

env XLA_FLAGS="$INFERENCE_XLA_FLAGS" "$VENV_DIR/bin/python" scripts/sample_classes.py \
  --init-from "$INIT_FROM" \
  --class-ids "$CLASS_IDS" \
  --samples-per-class "$SAMPLES_PER_CLASS" \
  --cfg-scale "$CFG_SCALE" \
  --seed "$SEED" \
  --batch-size "$BATCH_SIZE" \
  --vae-backend cuda \
  --outdir "$OUTDIR"
