#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"

export IMAGENET_PATH="${IMAGENET_PATH:-$HOME/datasets/imagenet-1k}"
export IMAGENET_FID_NPZ="${IMAGENET_FID_NPZ:-$HOME/datasets/imagenet_256_fid_stats.npz}"
export IMAGENET_PR_NPZ="${IMAGENET_PR_NPZ:-$HOME/datasets/imagenet_val_prc_arr0.npz}"
export IMAGENET_PRDC_NPZ="${IMAGENET_PRDC_NPZ:-$HOME/datasets/imagenet_val_inception_features_50k.npz}"

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
GENERATION_XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=$XLA_GPU_AUTOTUNE_LEVEL"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax-drifting-fid}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT="${WANDB_SILENT:-true}"
export INCEPTION_NUM_CLASSES="${INCEPTION_NUM_CLASSES:-1000}"
export INCEPTION_PARAMS_PATH="${INCEPTION_PARAMS_PATH:-$HOME/datasets/inception_params_torchvision.pkl}"

INIT_FROM="${INIT_FROM:-hf://latent_L_sota}"
CFG_SCALE="${CFG_SCALE:-1.0}"
TRAIN_SEED="${TRAIN_SEED:-}"
GENERATION_SEED="${GENERATION_SEED:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
VAE_DECODE_BATCH_SIZE="${VAE_DECODE_BATCH_SIZE:-32}"
FID_REF_BATCH_SIZE="${FID_REF_BATCH_SIZE:-512}"
FID_REF_WORKERS="${FID_REF_WORKERS:-8}"
WORKDIR="${WORKDIR:-$PROJECT_DIR/runs/fid_latent_L_sota}"
JSON_OUT="${JSON_OUT:-$WORKDIR/results_latent_L_sota_fid.json}"
SAMPLES_DIR="${SAMPLES_DIR:-$WORKDIR/generated_samples}"
FEATURE_ARTIFACT_PATH="${FEATURE_ARTIFACT_PATH:-$WORKDIR/generated_inception_artifacts.npz}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-}"
RUN_PROTOCOL_ID="${RUN_PROTOCOL_ID:-}"
RUN_PROTOCOL_PATH="${RUN_PROTOCOL_PATH:-}"
SOURCE_SNAPSHOT_ID="${SOURCE_SNAPSHOT_ID:-}"
EVAL_PRDC="${EVAL_PRDC:-0}"
PRDC_NEAREST_K="${PRDC_NEAREST_K:-5}"
PRDC_ROW_BATCH_SIZE="${PRDC_ROW_BATCH_SIZE:-1024}"
PRDC_COL_BATCH_SIZE="${PRDC_COL_BATCH_SIZE:-1024}"
KEEP_GENERATED_SAMPLES="${KEEP_GENERATED_SAMPLES:-1}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

cd "$PROJECT_DIR"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

echo "[INFO] host=$(hostname)"
echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] python=$VENV_DIR/bin/python"
echo "[INFO] imagenet=$IMAGENET_PATH"
echo "[INFO] fid_npz=$IMAGENET_FID_NPZ"
echo "[INFO] hf_root=$HF_ROOT"
echo "[INFO] inception_params=$INCEPTION_PARAMS_PATH"
echo "[INFO] init_from=$INIT_FROM"
echo "[INFO] cfg_scale=$CFG_SCALE"
echo "[INFO] train_seed=${TRAIN_SEED:-unspecified}"
echo "[INFO] generation_seed=$GENERATION_SEED"
echo "[INFO] num_samples=$NUM_SAMPLES"
echo "[INFO] eval_batch_size=$EVAL_BATCH_SIZE"
echo "[INFO] vae_decode_batch_size=$VAE_DECODE_BATCH_SIZE"
echo "[INFO] xla_gpu_autotune_level=$XLA_GPU_AUTOTUNE_LEVEL"
echo "[INFO] jax_compilation_cache=$JAX_COMPILATION_CACHE_DIR"
echo "[INFO] workdir=$WORKDIR"
echo "[INFO] samples_dir=$SAMPLES_DIR"
echo "[INFO] feature_artifact=$FEATURE_ARTIFACT_PATH"
echo "[INFO] eval_prdc=$EVAL_PRDC"
echo "[INFO] prdc_reference=$IMAGENET_PRDC_NPZ"
echo "[INFO] source_snapshot_id=${SOURCE_SNAPSHOT_ID:-unspecified}"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

mkdir -p "$WORKDIR"

if [ ! -s "$INCEPTION_PARAMS_PATH" ]; then
  echo "[ERR] Missing local Inception params: $INCEPTION_PARAMS_PATH" >&2
  exit 1
fi

if [ ! -s "$IMAGENET_FID_NPZ" ] \
  || { [ "$EVAL_PRDC" = "1" ] && [ ! -s "$IMAGENET_PRDC_NPZ" ]; }; then
  echo "[STEP] Build ImageNet-256 FID/feature reference"
  "$VENV_DIR/bin/python" scripts/prepare_imagenet_fid_stats.py \
    --data-path "$IMAGENET_PATH" \
    --out "$IMAGENET_FID_NPZ" \
    --feature-out "$IMAGENET_PRDC_NPZ" \
    --batch-size "$FID_REF_BATCH_SIZE" \
    --num-workers "$FID_REF_WORKERS" \
    --pin-memory
else
  echo "[SKIP] Existing frozen evaluation reference artifacts"
fi

echo "[STEP 1/2] Generate samples with bounded XLA autotuning"
GENERATE_ARGS=(
  "$VENV_DIR/bin/python" inference.py
  --init-from "$INIT_FROM"
  --cfg-scale "$CFG_SCALE"
  --train-seed "$TRAIN_SEED"
  --generation-seed "$GENERATION_SEED"
  --num-samples "$NUM_SAMPLES"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --vae-decode-batch-size "$VAE_DECODE_BATCH_SIZE"
  --vae-backend cuda
  --workdir "$WORKDIR"
  --samples-dir "$SAMPLES_DIR"
  --config-path "$EVAL_CONFIG_PATH"
  --protocol-id "$RUN_PROTOCOL_ID"
  --protocol-path "$RUN_PROTOCOL_PATH"
  --source-snapshot-id "$SOURCE_SNAPSHOT_ID"
  --generate-only
)
if [ "$ALLOW_OVERWRITE" = "1" ]; then
  GENERATE_ARGS+=(--allow-overwrite)
fi
env XLA_FLAGS="$GENERATION_XLA_FLAGS" "${GENERATE_ARGS[@]}"

echo "[STEP 2/2] Compute FID/IS with default cuDNN autotuning"
METRIC_ARGS=(
  "$VENV_DIR/bin/python" inference.py
  --init-from "$INIT_FROM"
  --cfg-scale "$CFG_SCALE"
  --train-seed "$TRAIN_SEED"
  --generation-seed "$GENERATION_SEED"
  --num-samples "$NUM_SAMPLES"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --vae-decode-batch-size "$VAE_DECODE_BATCH_SIZE"
  --workdir "$WORKDIR"
  --samples-dir "$SAMPLES_DIR"
  --source-snapshot-id "$SOURCE_SNAPSHOT_ID"
  --feature-artifact-path "$FEATURE_ARTIFACT_PATH"
  --metrics-only
  --json-out "$JSON_OUT"
)
if [ "$EVAL_PRDC" = "1" ]; then
  METRIC_ARGS+=(
    --eval-prdc
    --prdc-reference-path "$IMAGENET_PRDC_NPZ"
    --prdc-nearest-k "$PRDC_NEAREST_K"
    --prdc-row-batch-size "$PRDC_ROW_BATCH_SIZE"
    --prdc-col-batch-size "$PRDC_COL_BATCH_SIZE"
  )
fi
if [ "$ALLOW_OVERWRITE" = "1" ]; then
  METRIC_ARGS+=(--allow-overwrite)
fi
env -u XLA_FLAGS "${METRIC_ARGS[@]}"

if [ "$KEEP_GENERATED_SAMPLES" != "1" ]; then
  echo "[CLEANUP] Remove only raw samples.npy after verified metric artifacts"
  rm -f "$SAMPLES_DIR/samples.npy"
fi
