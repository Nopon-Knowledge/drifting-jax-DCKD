#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PROJECT_DIR="${PROJECT_DIR:-${HOME:?}/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-${HOME:?}/venvs/drifting_jax_env}"
INIT_FROM="${INIT_FROM:?INIT_FROM must point to a completed local training workdir}"
METHOD="${METHOD:?METHOD is required, for example baseline or v4.3}"
CFG_SCALE="${CFG_SCALE:?CFG_SCALE is required}"

case "$METHOD" in
  *[!A-Za-z0-9._-]*|'')
    echo "[ERR] METHOD contains unsupported characters: $METHOD" >&2
    exit 2
    ;;
esac

METHOD_TAG="${METHOD//./p}"
SEED="${SEED:-314159}"
EXPECTED_STEP="${EXPECTED_STEP:-30000}"
LATENCY_BATCH_SIZE="${LATENCY_BATCH_SIZE:-1}"
THROUGHPUT_BATCH_SIZE="${THROUGHPUT_BATCH_SIZE:-128}"
WARMUP_ITERATIONS="${WARMUP_ITERATIONS:-5}"
LATENCY_ITERATIONS="${LATENCY_ITERATIONS:-30}"
THROUGHPUT_ITERATIONS="${THROUGHPUT_ITERATIONS:-20}"
HSDP_DIM="${HSDP_DIM:-8}"
NVIDIA_SMI_INTERVAL="${NVIDIA_SMI_INTERVAL:-0.1}"
WORKDIR="${WORKDIR:-$PROJECT_DIR/runs/pr_dckd_v1/efficiency/$METHOD_TAG}"
JSON_OUT="${JSON_OUT:-$WORKDIR/${METHOD_TAG}_generator_efficiency.json}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"

export HF_ROOT="${HF_ROOT:-${HOME:?}/datasets/hf_cache}"
export HF_HOME="${HF_HOME:-$HF_ROOT}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_ROOT/hub}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${HOME:?}/.cache/jax-drifting-efficiency}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="${JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES:--1}"
export EFFICIENCY_BENCHMARK_WRAPPER="$PROJECT_DIR/scripts/run_efficiency_benchmark_a800_hpc.sh"
XLA_GPU_AUTOTUNE_LEVEL="${XLA_GPU_AUTOTUNE_LEVEL:-0}"
BENCHMARK_XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_autotune_level=$XLA_GPU_AUTOTUNE_LEVEL"

if [[ -s "$JSON_OUT" && "$ALLOW_OVERWRITE" != "1" ]]; then
  echo "[ERR] Refusing to overwrite existing result: $JSON_OUT" >&2
  exit 3
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[ERR] Python environment not found: $VENV_DIR/bin/python" >&2
  exit 4
fi

mkdir -p "$WORKDIR" "$JAX_COMPILATION_CACHE_DIR"
cd "$PROJECT_DIR"

echo "[INFO] host=$(hostname)"
echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] init_from=$INIT_FROM"
echo "[INFO] method=$METHOD"
echo "[INFO] cfg_scale=$CFG_SCALE"
echo "[INFO] expected_step=$EXPECTED_STEP"
echo "[INFO] latency=batch${LATENCY_BATCH_SIZE},iterations=${LATENCY_ITERATIONS}"
echo "[INFO] throughput=batch${THROUGHPUT_BATCH_SIZE},iterations=${THROUGHPUT_ITERATIONS}"
echo "[INFO] warmup_iterations=$WARMUP_ITERATIONS"
echo "[INFO] xla_gpu_autotune_level=$XLA_GPU_AUTOTUNE_LEVEL"
echo "[INFO] jax_enable_compilation_cache=$JAX_ENABLE_COMPILATION_CACHE"
echo "[INFO] json_out=$JSON_OUT"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

exec env XLA_FLAGS="$BENCHMARK_XLA_FLAGS" "$VENV_DIR/bin/python" \
  scripts/benchmark_generator_efficiency.py \
  --init-from "$INIT_FROM" \
  --method "$METHOD" \
  --cfg-scale "$CFG_SCALE" \
  --json-out "$JSON_OUT" \
  --seed "$SEED" \
  --expected-step "$EXPECTED_STEP" \
  --latency-batch-size "$LATENCY_BATCH_SIZE" \
  --throughput-batch-size "$THROUGHPUT_BATCH_SIZE" \
  --warmup-iterations "$WARMUP_ITERATIONS" \
  --latency-iterations "$LATENCY_ITERATIONS" \
  --throughput-iterations "$THROUGHPUT_ITERATIONS" \
  --hsdp-dim "$HSDP_DIM" \
  --nvidia-smi-interval "$NVIDIA_SMI_INTERVAL"
