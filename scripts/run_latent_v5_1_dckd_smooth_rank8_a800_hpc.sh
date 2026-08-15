#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"

export CONFIG_PATH="${CONFIG_PATH:-configs/gen/latent_ablation_a800_v5_1_dckd_smooth_rank8.yaml}"
export WORKDIR="${WORKDIR:-runs/gen_latent_a800_v5_1_dckd_smooth_rank8}"
export XLA_FLAGS="${XLA_FLAGS:---xla_gpu_autotune_level=0}"

exec /usr/bin/env bash "$PROJECT_DIR/scripts/run_latent_replay_a800_hpc.sh"
