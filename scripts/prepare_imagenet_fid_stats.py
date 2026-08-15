#!/usr/bin/env python3
"""Build auditable ImageNet-256 FID and feature-level PRDC references."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm

from dataset.dataset import center_crop_arr
from dataset.latent import (
    ImageNetFlatValFolder,
    OriginalImageFolder,
    _imagenet_val_annotation_root,
    _is_imagefolder_split,
    _resolve_imagenet_data_root,
)
from utils.env import (
    IMAGENET_FID_NPZ,
    IMAGENET_PATH,
    IMAGENET_PRDC_NPZ,
    INCEPTION_PARAMS_PATH,
)
from utils.fid_util import (
    _compute_stats,
    _sha256_file,
    save_inception_artifacts,
)
from utils.misc import run_init


def _to_uint8_array(image: Image.Image, resolution: int) -> np.ndarray:
    image = center_crop_arr(image.convert("RGB"), resolution)
    return np.asarray(image, dtype=np.uint8)


def _build_val_dataset(data_path: str, resolution: int):
    data_root = _resolve_imagenet_data_root(data_path)
    split_root = data_root / "val"
    transform = transforms.Lambda(lambda img: _to_uint8_array(img, resolution))
    if not _is_imagefolder_split(split_root):
        train_root = data_root / "train"
        classes = sorted(child.name for child in train_root.iterdir() if child.is_dir())
        return ImageNetFlatValFolder(
            root=split_root,
            annotation_root=_imagenet_val_annotation_root(data_root),
            classes=classes,
            transform=transform,
            return_rel_path=True,
        )
    return OriginalImageFolder(root=str(split_root), transform=transform)


def _as_numpy_images(batch) -> np.ndarray:
    if isinstance(batch, torch.Tensor):
        return batch.numpy()
    return np.asarray(batch)


def _file_record(path: str | os.PathLike[str]) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def build_stats(args: argparse.Namespace) -> None:
    run_init()
    dataset = _build_val_dataset(args.data_path, args.resolution)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
        persistent_workers=(args.num_workers > 0),
    )

    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    relative_paths: list[str] = []
    legacy_image_chunks: list[np.ndarray] = []
    processed_images_digest = hashlib.sha256()
    total = 0
    for images, labels, batch_relative_paths in tqdm(
        loader, desc="imagenet-inception-ref"
    ):
        arr = _as_numpy_images(images)
        labels = np.asarray(labels)
        batch_relative_paths = list(batch_relative_paths)
        if args.max_samples:
            remaining = args.max_samples - total
            if remaining <= 0:
                break
            arr = arr[:remaining]
            labels = labels[:remaining]
            batch_relative_paths = batch_relative_paths[:remaining]
        if arr.ndim != 4 or arr.shape[-1] != 3:
            raise ValueError(
                f"Expected NHWC uint8 images, got shape={arr.shape} dtype={arr.dtype}"
            )
        arr = arr.astype(np.uint8, copy=False)

        chunk_stats = _compute_stats(
            arr,
            arr.shape[0],
            compute_logits=False,
            compute_features=True,
        )
        feature_chunks.append(
            np.asarray(chunk_stats["features"], dtype=np.float32)
        )
        label_chunks.append(np.asarray(labels, dtype=np.int64))
        relative_paths.extend(str(path) for path in batch_relative_paths)
        processed_images_digest.update(memoryview(np.ascontiguousarray(arr)).cast("B"))
        if args.pr_out and sum(len(chunk) for chunk in legacy_image_chunks) < args.pr_samples:
            legacy_remaining = args.pr_samples - sum(
                len(chunk) for chunk in legacy_image_chunks
            )
            legacy_image_chunks.append(arr[:legacy_remaining].copy())
        total += arr.shape[0]
        if args.max_samples and total >= args.max_samples:
            break

    if not feature_chunks:
        raise RuntimeError("No ImageNet validation images were processed.")
    features = np.concatenate(feature_chunks, axis=0)
    labels = np.concatenate(label_chunks, axis=0)
    relative_paths_array = np.asarray(relative_paths, dtype=np.str_)
    if not (
        features.shape[0]
        == labels.shape[0]
        == relative_paths_array.shape[0]
        == total
    ):
        raise RuntimeError(
            "Reference ordering mismatch: "
            f"features={features.shape[0]}, labels={labels.shape[0]}, "
            f"paths={relative_paths_array.shape[0]}, processed={total}."
        )
    if (
        args.expected_samples
        and total != args.expected_samples
        and not args.allow_partial
    ):
        raise RuntimeError(
            f"Expected exactly {args.expected_samples} reference samples, got {total}. "
            "Use --allow-partial only for smoke tests; partial references are not "
            "valid for formal 50k PRDC evaluation."
        )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    features64 = features.astype(np.float64)
    ref_mu = np.mean(features64, axis=0)
    ref_sigma = np.cov(features64, rowvar=False)
    np.savez(out, ref_mu=ref_mu, ref_sigma=ref_sigma)
    fid_record = _file_record(out)
    print(f"Wrote FID stats: {out} ({total} images)")

    if args.pr_out:
        legacy_images = np.concatenate(legacy_image_chunks, axis=0)
        pr_count = min(args.pr_samples, legacy_images.shape[0])
        pr_out = Path(args.pr_out).expanduser().resolve()
        pr_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pr_out, legacy_images[:pr_count])
        print(
            "Wrote legacy improved-P/R image reference: "
            f"{pr_out} ({pr_count} images)"
        )

    if args.feature_out:
        code_paths = (
            Path(__file__).resolve(),
            REPO_ROOT / "utils" / "fid_util.py",
            REPO_ROOT / "utils" / "jax_fid" / "inception.py",
            REPO_ROOT / "utils" / "jax_fid" / "resize.py",
            REPO_ROOT / "utils" / "jax_fid" / "cvt.py",
        )
        metadata = {
            "artifact_role": "imagenet_validation_inception_reference",
            "dataset": "ImageNet-1k",
            "split": "val",
            "resolution": args.resolution,
            "sample_count": total,
            "expected_sample_count": args.expected_samples,
            "full_reference": bool(
                args.expected_samples and total == args.expected_samples
            ),
            "ordering": "dataset order; DataLoader shuffle=False",
            "preprocessing": (
                f"ADM-style center crop to {args.resolution}; repository "
                "jax_fid resize to 299x299 before InceptionV3"
            ),
            "feature_layer": "InceptionV3 global-average-pooled 2048-D features",
            "distance_protocol": "canonical PRDC uses squared Euclidean features",
            "data_path": str(Path(args.data_path).expanduser().resolve()),
            "processed_images_sha256": processed_images_digest.hexdigest(),
            "fid_stats": fid_record,
            "inception_params": _file_record(INCEPTION_PARAMS_PATH),
            "code": {
                str(path.relative_to(REPO_ROOT)): _file_record(path)
                for path in code_paths
            },
        }
        manifest = save_inception_artifacts(
            args.feature_out,
            features=features,
            labels=labels,
            relative_paths=relative_paths_array,
            metadata=metadata,
        )
        archive = manifest["archive"]
        print(
            "Wrote feature-level PRDC reference: "
            f"{archive['path']} ({total} x {features.shape[1]}, "
            f"sha256={archive['sha256']})"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ImageNet-256 FID stats and a full feature-level PRDC reference."
    )
    parser.add_argument("--data-path", default=IMAGENET_PATH)
    parser.add_argument("--out", default=IMAGENET_FID_NPZ)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument(
        "--expected-samples",
        type=int,
        default=50000,
        help="Required formal-reference count (default: 50000).",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Permit a non-50k reference for smoke tests only.",
    )
    parser.add_argument(
        "--feature-out",
        default=IMAGENET_PRDC_NPZ,
        help="Output NPZ for raw Inception features, labels, paths, and manifest.",
    )
    parser.add_argument("--pr-out", default="")
    parser.add_argument("--pr-samples", type=int, default=10000)
    return parser


def main() -> None:
    build_stats(build_parser().parse_args())


if __name__ == "__main__":
    main()
