"""Global paths for the public Drift release."""

from __future__ import annotations

import os

_DATASETS_ROOT = os.environ.get(
    "DRIFTING_DATASETS_ROOT",
    os.path.join(os.path.expanduser("~"), "datasets"),
)

IMAGENET_PATH = os.environ.get("IMAGENET_PATH", os.path.join(_DATASETS_ROOT, "imagenet-1k"))
IMAGENET_CACHE_PATH = os.environ.get(
    "IMAGENET_CACHE_PATH",
    os.path.join(_DATASETS_ROOT, "imagenet_latent_cache"),
)
IMAGENET_FID_NPZ = os.environ.get(
    "IMAGENET_FID_NPZ",
    os.path.join(_DATASETS_ROOT, "imagenet_256_fid_stats.npz"),
)
IMAGENET_PR_NPZ = os.environ.get(
    "IMAGENET_PR_NPZ",
    os.path.join(_DATASETS_ROOT, "imagenet_val_prc_arr0.npz"),
)
IMAGENET_PRDC_NPZ = os.environ.get(
    "IMAGENET_PRDC_NPZ",
    os.path.join(_DATASETS_ROOT, "imagenet_val_inception_features_50k.npz"),
)
INCEPTION_PARAMS_PATH = os.environ.get(
    "INCEPTION_PARAMS_PATH",
    "/tmp/inception_params.pkl",
)

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("HF_ROOT", os.path.join(_DATASETS_ROOT, "hf_cache"))
