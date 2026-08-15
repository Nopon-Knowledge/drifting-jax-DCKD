#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C
export JAX_PLATFORMS=gpu,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

export IMAGENET_PATH="${IMAGENET_PATH:-$HOME/datasets/imagenet-1k}"
export IMAGENET_CACHE_PATH="${IMAGENET_CACHE_PATH:-$HOME/datasets/imagenet_latent_cache}"
export HF_ROOT="${HF_ROOT:-$HOME/datasets/hf_cache}"

cd "${PROJECT_DIR:-$HOME/python_project/drifting-jax}"

exec "$HOME/venvs/drifting_jax_env/bin/python" main.py \
  --gen \
  --config configs/gen/pixel_a800_offline.yaml \
  --workdir runs/gen_pixel_a800_offline
