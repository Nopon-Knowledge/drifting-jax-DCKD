#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"
VENV_DIR="${VENV_DIR:-$HOME/venvs/drifting_jax_env}"
REFERENCE_DIR="${REFERENCE_DIR:-$HOME/datasets/pr_dckd_v1}"
export IMAGENET_PATH="${IMAGENET_PATH:-$HOME/datasets/imagenet-1k}"
export IMAGENET_FID_NPZ="${IMAGENET_FID_NPZ:-$REFERENCE_DIR/imagenet_256_fid_stats_50k.npz}"
export IMAGENET_PRDC_NPZ="${IMAGENET_PRDC_NPZ:-$REFERENCE_DIR/imagenet_val_inception_features_50k.npz}"
export INCEPTION_PARAMS_PATH="${INCEPTION_PARAMS_PATH:-$HOME/datasets/inception_params_torchvision.pkl}"
export INCEPTION_NUM_CLASSES="${INCEPTION_NUM_CLASSES:-1000}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

LEGACY_FID_NPZ="${LEGACY_FID_NPZ:-$HOME/datasets/imagenet_256_fid_stats.npz}"
VALIDATION_JSON="${VALIDATION_JSON:-$REFERENCE_DIR/reference_validation.json}"
FID_REF_BATCH_SIZE="${FID_REF_BATCH_SIZE:-512}"
FID_REF_WORKERS="${FID_REF_WORKERS:-8}"
ALLOW_REUSE_VALIDATED="${ALLOW_REUSE_VALIDATED:-0}"

cd "$PROJECT_DIR"
mkdir -p "$REFERENCE_DIR"

echo "[INFO] host=$(hostname)"
echo "[INFO] project=$PROJECT_DIR"
echo "[INFO] imagenet=$IMAGENET_PATH"
echo "[INFO] fid_out=$IMAGENET_FID_NPZ"
echo "[INFO] features_out=$IMAGENET_PRDC_NPZ"
echo "[INFO] validation=$VALIDATION_JSON"
nvidia-smi

if [ "$ALLOW_REUSE_VALIDATED" != "1" ] \
  && { [ -e "$IMAGENET_FID_NPZ" ] || [ -e "$IMAGENET_PRDC_NPZ" ] || [ -e "$VALIDATION_JSON" ]; }; then
  echo "[ERR] Refusing to overwrite a reference artifact." >&2
  exit 1
fi

if [ "$ALLOW_REUSE_VALIDATED" != "1" ]; then
  "$VENV_DIR/bin/python" scripts/prepare_imagenet_fid_stats.py \
    --data-path "$IMAGENET_PATH" \
    --out "$IMAGENET_FID_NPZ" \
    --feature-out "$IMAGENET_PRDC_NPZ" \
    --batch-size "$FID_REF_BATCH_SIZE" \
    --num-workers "$FID_REF_WORKERS" \
    --pin-memory \
    --max-samples 50000 \
    --expected-samples 50000
fi

"$VENV_DIR/bin/python" scripts/validate_pr_reference.py \
  --fid "$IMAGENET_FID_NPZ" \
  --features "$IMAGENET_PRDC_NPZ" \
  --legacy-fid "$LEGACY_FID_NPZ" \
  --expected-samples 50000 \
  --expected-dim 2048 \
  --out "$VALIDATION_JSON"

echo "[DONE] Frozen ImageNet FID/PRDC reference passed validation."
