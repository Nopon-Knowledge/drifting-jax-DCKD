#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/python_project/drifting-jax}"

export INIT_FROM="${INIT_FROM:-$PROJECT_DIR/runs/gen_latent_a800_replay}"
export CFG_SCALE="${CFG_SCALE:-1.0}"
export NUM_SAMPLES="${NUM_SAMPLES:-50000}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
export WORKDIR="${WORKDIR:-$PROJECT_DIR/runs/fid_gen_latent_a800_replay}"
export JSON_OUT="${JSON_OUT:-$WORKDIR/results_replay_fid.json}"

exec /usr/bin/env bash "$PROJECT_DIR/scripts/run_fid_a800_hpc.sh"
