#!/usr/bin/env python3
"""Convert torchvision Inception v3 weights to the pickle layout used by jax_fid."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


def _np(tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _conv_kernel(state, prefix: str) -> np.ndarray:
    # Torch Conv2d is OIHW; Flax Conv is HWIO.
    return np.transpose(_np(state[f"{prefix}.conv.weight"]), (2, 3, 1, 0))


def _basic_conv(state, prefix: str) -> dict:
    return {
        "conv": {
            "kernel": _conv_kernel(state, prefix),
        },
        "bn": {
            "bias": _np(state[f"{prefix}.bn.bias"]),
            "scale": _np(state[f"{prefix}.bn.weight"]),
            "mean": _np(state[f"{prefix}.bn.running_mean"]),
            "var": _np(state[f"{prefix}.bn.running_var"]),
        },
    }


def _inception_a(state, prefix: str) -> dict:
    return {
        "branch1x1": _basic_conv(state, f"{prefix}.branch1x1"),
        "branch5x5_1": _basic_conv(state, f"{prefix}.branch5x5_1"),
        "branch5x5_2": _basic_conv(state, f"{prefix}.branch5x5_2"),
        "branch3x3dbl_1": _basic_conv(state, f"{prefix}.branch3x3dbl_1"),
        "branch3x3dbl_2": _basic_conv(state, f"{prefix}.branch3x3dbl_2"),
        "branch3x3dbl_3": _basic_conv(state, f"{prefix}.branch3x3dbl_3"),
        "branch_pool": _basic_conv(state, f"{prefix}.branch_pool"),
    }


def _inception_b(state, prefix: str) -> dict:
    return {
        "branch3x3": _basic_conv(state, f"{prefix}.branch3x3"),
        "branch3x3dbl_1": _basic_conv(state, f"{prefix}.branch3x3dbl_1"),
        "branch3x3dbl_2": _basic_conv(state, f"{prefix}.branch3x3dbl_2"),
        "branch3x3dbl_3": _basic_conv(state, f"{prefix}.branch3x3dbl_3"),
    }


def _inception_c(state, prefix: str) -> dict:
    return {
        "branch1x1": _basic_conv(state, f"{prefix}.branch1x1"),
        "branch7x7_1": _basic_conv(state, f"{prefix}.branch7x7_1"),
        "branch7x7_2": _basic_conv(state, f"{prefix}.branch7x7_2"),
        "branch7x7_3": _basic_conv(state, f"{prefix}.branch7x7_3"),
        "branch7x7dbl_1": _basic_conv(state, f"{prefix}.branch7x7dbl_1"),
        "branch7x7dbl_2": _basic_conv(state, f"{prefix}.branch7x7dbl_2"),
        "branch7x7dbl_3": _basic_conv(state, f"{prefix}.branch7x7dbl_3"),
        "branch7x7dbl_4": _basic_conv(state, f"{prefix}.branch7x7dbl_4"),
        "branch7x7dbl_5": _basic_conv(state, f"{prefix}.branch7x7dbl_5"),
        "branch_pool": _basic_conv(state, f"{prefix}.branch_pool"),
    }


def _inception_d(state, prefix: str) -> dict:
    return {
        "branch3x3_1": _basic_conv(state, f"{prefix}.branch3x3_1"),
        "branch3x3_2": _basic_conv(state, f"{prefix}.branch3x3_2"),
        "branch7x7x3_1": _basic_conv(state, f"{prefix}.branch7x7x3_1"),
        "branch7x7x3_2": _basic_conv(state, f"{prefix}.branch7x7x3_2"),
        "branch7x7x3_3": _basic_conv(state, f"{prefix}.branch7x7x3_3"),
        "branch7x7x3_4": _basic_conv(state, f"{prefix}.branch7x7x3_4"),
    }


def _inception_e(state, prefix: str) -> dict:
    return {
        "branch1x1": _basic_conv(state, f"{prefix}.branch1x1"),
        "branch3x3_1": _basic_conv(state, f"{prefix}.branch3x3_1"),
        "branch3x3_2a": _basic_conv(state, f"{prefix}.branch3x3_2a"),
        "branch3x3_2b": _basic_conv(state, f"{prefix}.branch3x3_2b"),
        "branch3x3dbl_1": _basic_conv(state, f"{prefix}.branch3x3dbl_1"),
        "branch3x3dbl_2": _basic_conv(state, f"{prefix}.branch3x3dbl_2"),
        "branch3x3dbl_3a": _basic_conv(state, f"{prefix}.branch3x3dbl_3a"),
        "branch3x3dbl_3b": _basic_conv(state, f"{prefix}.branch3x3dbl_3b"),
        "branch_pool": _basic_conv(state, f"{prefix}.branch_pool"),
    }


def convert(input_path: Path, output_path: Path) -> None:
    state = torch.load(input_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}

    out = {
        "Conv2d_1a_3x3": _basic_conv(state, "Conv2d_1a_3x3"),
        "Conv2d_2a_3x3": _basic_conv(state, "Conv2d_2a_3x3"),
        "Conv2d_2b_3x3": _basic_conv(state, "Conv2d_2b_3x3"),
        "Conv2d_3b_1x1": _basic_conv(state, "Conv2d_3b_1x1"),
        "Conv2d_4a_3x3": _basic_conv(state, "Conv2d_4a_3x3"),
        "Mixed_5b": _inception_a(state, "Mixed_5b"),
        "Mixed_5c": _inception_a(state, "Mixed_5c"),
        "Mixed_5d": _inception_a(state, "Mixed_5d"),
        "Mixed_6a": _inception_b(state, "Mixed_6a"),
        "Mixed_6b": _inception_c(state, "Mixed_6b"),
        "Mixed_6c": _inception_c(state, "Mixed_6c"),
        "Mixed_6d": _inception_c(state, "Mixed_6d"),
        "Mixed_6e": _inception_c(state, "Mixed_6e"),
        "Mixed_7a": _inception_d(state, "Mixed_7a"),
        "Mixed_7b": _inception_e(state, "Mixed_7b"),
        "Mixed_7c": _inception_e(state, "Mixed_7c"),
        "fc": {
            "kernel": np.transpose(_np(state["fc.weight"]), (1, 0)),
            "bias": _np(state["fc.bias"]),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
