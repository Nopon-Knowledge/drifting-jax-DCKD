#!/usr/bin/env python3
"""Validate and hash the frozen ImageNet FID/PRDC reference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = path.expanduser().resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_fid(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        mu_key = "ref_mu" if "ref_mu" in data else "mu"
        sigma_key = "ref_sigma" if "ref_sigma" in data else "sigma"
        return np.asarray(data[mu_key]), np.asarray(data[sigma_key])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fid", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-samples", type=int, default=50000)
    parser.add_argument("--expected-dim", type=int, default=2048)
    parser.add_argument("--legacy-fid", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    fid_path = Path(args.fid).expanduser().resolve()
    feature_path = Path(args.features).expanduser().resolve()
    if not fid_path.is_file() or not feature_path.is_file():
        raise FileNotFoundError(
            f"Missing FID/features reference: {fid_path}, {feature_path}"
        )

    with np.load(feature_path, allow_pickle=False) as data:
        required = {"features", "labels", "relative_paths", "manifest_json"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Feature reference is missing arrays: {sorted(missing)}")
        features = np.asarray(data["features"])
        labels = np.asarray(data["labels"])
        paths = np.asarray(data["relative_paths"])
        embedded = json.loads(data["manifest_json"].item())

    expected_shape = (args.expected_samples, args.expected_dim)
    if features.shape != expected_shape:
        raise ValueError(
            f"Expected feature shape {expected_shape}, got {features.shape}."
        )
    if labels.shape != (args.expected_samples,):
        raise ValueError(f"Unexpected labels shape: {labels.shape}.")
    if paths.shape != (args.expected_samples,):
        raise ValueError(f"Unexpected relative_paths shape: {paths.shape}.")
    if not np.isfinite(features).all():
        raise ValueError("Reference features contain NaN or infinite values.")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError(f"Reference labels must be integral, got {labels.dtype}.")
    if labels.min() != 0 or labels.max() != 999:
        raise ValueError(
            f"Expected ImageNet label range 0..999, got {labels.min()}..{labels.max()}."
        )
    class_counts = np.bincount(labels.astype(np.int64), minlength=1000)
    if not np.all(class_counts == 50):
        raise ValueError(
            "Formal ImageNet validation reference must contain exactly 50 "
            "examples for every one of 1000 classes."
        )
    if np.unique(paths).shape[0] != args.expected_samples:
        raise ValueError("Reference relative paths are not unique.")
    if embedded.get("metadata", {}).get("full_reference") is not True:
        raise ValueError("Embedded artifact manifest does not mark a full reference.")

    ref_mu, ref_sigma = load_fid(fid_path)
    features64 = features.astype(np.float64)
    recomputed_mu = np.mean(features64, axis=0)
    recomputed_sigma = np.cov(features64, rowvar=False)
    fid_mu_max_abs = float(np.max(np.abs(ref_mu - recomputed_mu)))
    fid_sigma_max_abs = float(np.max(np.abs(ref_sigma - recomputed_sigma)))
    if fid_mu_max_abs > 1e-12 or fid_sigma_max_abs > 1e-12:
        raise ValueError(
            "FID stats are not derived from the frozen feature artifact: "
            f"mu max abs={fid_mu_max_abs}, sigma max abs={fid_sigma_max_abs}."
        )

    legacy_comparison = None
    if args.legacy_fid:
        legacy_path = Path(args.legacy_fid).expanduser().resolve()
        if legacy_path.is_file() and legacy_path != fid_path:
            legacy_mu, legacy_sigma = load_fid(legacy_path)
            legacy_comparison = {
                "artifact": file_record(legacy_path),
                "mu_max_abs": float(np.max(np.abs(legacy_mu - ref_mu))),
                "sigma_max_abs": float(np.max(np.abs(legacy_sigma - ref_sigma))),
                "interpretation": (
                    "Diagnostic only. Formal PR-DCKD evaluations always use the "
                    "new paired FID/features reference, so old and new stats are "
                    "never mixed."
                ),
            }

    report = {
        "schema_version": "drifting-pr-reference-validation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "expected_samples": args.expected_samples,
        "expected_feature_dim": args.expected_dim,
        "fid": file_record(fid_path),
        "features_archive": file_record(feature_path),
        "arrays": {
            "features": {
                "shape": list(features.shape),
                "dtype": str(features.dtype),
                "sha256": sha256_array(features),
            },
            "labels": {
                "shape": list(labels.shape),
                "dtype": str(labels.dtype),
                "sha256": sha256_array(labels),
                "class_count_min": int(class_counts.min()),
                "class_count_max": int(class_counts.max()),
            },
            "relative_paths": {
                "shape": list(paths.shape),
                "dtype": str(paths.dtype),
                "sha256": sha256_array(paths),
                "unique": int(np.unique(paths).shape[0]),
            },
        },
        "fid_feature_alignment": {
            "mu_max_abs": fid_mu_max_abs,
            "sigma_max_abs": fid_sigma_max_abs,
        },
        "embedded_manifest": embedded,
        "legacy_comparison": legacy_comparison,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, out)
    Path(f"{out}.sha256").write_text(
        f"{sha256_file(out)}  {out.name}\n", encoding="utf-8"
    )
    print(json.dumps({"status": "pass", "report": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
