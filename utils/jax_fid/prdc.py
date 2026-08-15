"""Canonical precision, recall, density, and coverage on feature vectors.

This module implements the PRDC definitions from Naeem et al. using exact,
blockwise squared-Euclidean distances.  It deliberately lives next to, rather
than replacing, :mod:`precision_recall`: that module implements the older
improved-precision/recall manifold metric used by the original repository.

The implementation never materializes an ``N x N`` distance matrix.  It is
therefore memory bounded by ``row_batch_size * col_batch_size`` while retaining
the exact result (up to floating-point arithmetic).  Runtime remains quadratic
in the number of reference/generated features, as required by exact PRDC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class PRDCConfig:
    """Configuration for exact blockwise PRDC."""

    nearest_k: int = 5
    row_batch_size: int = 1024
    col_batch_size: int = 1024
    distance_dtype: str = "float64"


def _validate_features(features: np.ndarray, name: str, nearest_k: int) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"{name} must have shape [N, D], got {features.shape}.")
    if features.shape[0] <= nearest_k:
        raise ValueError(
            f"{name} needs more than nearest_k={nearest_k} rows, got {features.shape[0]}."
        )
    if features.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature dimension.")
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype={features.dtype}.")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return features


def _validate_config(
    nearest_k: int,
    row_batch_size: int,
    col_batch_size: int,
    distance_dtype,
) -> np.dtype:
    if not isinstance(nearest_k, (int, np.integer)) or nearest_k < 1:
        raise ValueError(f"nearest_k must be a positive integer, got {nearest_k!r}.")
    if row_batch_size < 1 or col_batch_size < 1:
        raise ValueError(
            "row_batch_size and col_batch_size must be positive, got "
            f"{row_batch_size} and {col_batch_size}."
        )
    dtype = np.dtype(distance_dtype)
    if dtype.kind != "f":
        raise TypeError(f"distance_dtype must be floating point, got {dtype}.")
    return dtype


def pairwise_squared_distances(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    distance_dtype=np.float64,
) -> np.ndarray:
    """Return an exact dense block of squared-Euclidean distances."""

    dtype = np.dtype(distance_dtype)
    rows = np.asarray(rows, dtype=dtype)
    cols = np.asarray(cols, dtype=dtype)
    row_norms = np.einsum("ij,ij->i", rows, rows)[:, None]
    col_norms = np.einsum("ij,ij->i", cols, cols)[None, :]
    distances = row_norms - 2.0 * (rows @ cols.T) + col_norms
    # Round-off can make identical vectors slightly negative.
    np.maximum(distances, 0.0, out=distances)
    return distances


def compute_nearest_neighbour_distances(
    features: np.ndarray,
    *,
    nearest_k: int = 5,
    row_batch_size: int = 1024,
    col_batch_size: int = 1024,
    distance_dtype=np.float64,
) -> np.ndarray:
    """Compute each feature's squared distance to its k-th non-self neighbor.

    The diagonal identity match is explicitly excluded.  Other duplicate
    vectors remain valid neighbors with distance zero, matching the canonical
    dense implementation.
    """

    dtype = _validate_config(
        nearest_k, row_batch_size, col_batch_size, distance_dtype
    )
    features = _validate_features(features, "features", nearest_k)
    num_features = features.shape[0]
    radii = np.empty(num_features, dtype=dtype)

    for row_start in range(0, num_features, row_batch_size):
        row_end = min(row_start + row_batch_size, num_features)
        rows = features[row_start:row_end]
        best = np.full((row_end - row_start, nearest_k), np.inf, dtype=dtype)

        for col_start in range(0, num_features, col_batch_size):
            col_end = min(col_start + col_batch_size, num_features)
            distances = pairwise_squared_distances(
                rows,
                features[col_start:col_end],
                distance_dtype=dtype,
            )

            # Exclude only the identity/self pair, even when duplicate feature
            # vectors occur elsewhere in the dataset.
            overlap_start = max(row_start, col_start)
            overlap_end = min(row_end, col_end)
            if overlap_start < overlap_end:
                global_indices = np.arange(overlap_start, overlap_end)
                distances[
                    global_indices - row_start,
                    global_indices - col_start,
                ] = np.inf

            candidates = np.concatenate((best, distances), axis=1)
            best = np.partition(candidates, nearest_k - 1, axis=1)[:, :nearest_k]

        radii[row_start:row_end] = np.max(best, axis=1)

    if not np.isfinite(radii).all():
        raise RuntimeError("Failed to find finite k-nearest-neighbor radii.")
    return radii


def compute_prdc(
    real_features: np.ndarray,
    fake_features: np.ndarray,
    *,
    nearest_k: int = 5,
    row_batch_size: int = 1024,
    col_batch_size: int = 1024,
    distance_dtype=np.float64,
) -> Dict[str, float]:
    """Compute canonical PRDC using exact blockwise distances.

    Definitions:

    * precision: generated samples covered by at least one real k-NN sphere;
    * recall: real samples covered by at least one generated k-NN sphere;
    * density: average number of real spheres covering a generated sample,
      normalized by ``nearest_k``;
    * coverage: real samples whose nearest generated sample lies within their
      real k-NN radius.

    Returned distances are not exposed; all four scalar metrics are Python
    floats and ``nearest_k`` is included for an auditable result record.
    """

    dtype = _validate_config(
        nearest_k, row_batch_size, col_batch_size, distance_dtype
    )
    real_features = _validate_features(real_features, "real_features", nearest_k)
    fake_features = _validate_features(fake_features, "fake_features", nearest_k)
    if real_features.shape[1] != fake_features.shape[1]:
        raise ValueError(
            "real_features and fake_features must have the same feature dimension, "
            f"got {real_features.shape[1]} and {fake_features.shape[1]}."
        )

    real_radii = compute_nearest_neighbour_distances(
        real_features,
        nearest_k=nearest_k,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        distance_dtype=dtype,
    )
    fake_radii = compute_nearest_neighbour_distances(
        fake_features,
        nearest_k=nearest_k,
        row_batch_size=row_batch_size,
        col_batch_size=col_batch_size,
        distance_dtype=dtype,
    )

    num_real = real_features.shape[0]
    num_fake = fake_features.shape[0]
    precision_hits = np.zeros(num_fake, dtype=bool)
    density_counts = np.zeros(num_fake, dtype=np.int64)
    recall_hits = np.zeros(num_real, dtype=bool)
    nearest_fake_distances = np.full(num_real, np.inf, dtype=dtype)

    for real_start in range(0, num_real, row_batch_size):
        real_end = min(real_start + row_batch_size, num_real)
        real_batch = real_features[real_start:real_end]
        real_batch_radii = real_radii[real_start:real_end, None]

        for fake_start in range(0, num_fake, col_batch_size):
            fake_end = min(fake_start + col_batch_size, num_fake)
            distances = pairwise_squared_distances(
                real_batch,
                fake_features[fake_start:fake_end],
                distance_dtype=dtype,
            )

            inside_real_spheres = distances <= real_batch_radii
            precision_hits[fake_start:fake_end] |= np.any(
                inside_real_spheres, axis=0
            )
            density_counts[fake_start:fake_end] += np.sum(
                inside_real_spheres, axis=0, dtype=np.int64
            )

            inside_fake_spheres = distances <= fake_radii[None, fake_start:fake_end]
            recall_hits[real_start:real_end] |= np.any(
                inside_fake_spheres, axis=1
            )
            nearest_fake_distances[real_start:real_end] = np.minimum(
                nearest_fake_distances[real_start:real_end],
                np.min(distances, axis=1),
            )

    return {
        "precision": float(np.mean(precision_hits, dtype=np.float64)),
        "recall": float(np.mean(recall_hits, dtype=np.float64)),
        "density": float(
            np.mean(density_counts, dtype=np.float64) / float(nearest_k)
        ),
        "coverage": float(
            np.mean(nearest_fake_distances <= real_radii, dtype=np.float64)
        ),
        "nearest_k": int(nearest_k),
    }


__all__ = [
    "PRDCConfig",
    "compute_nearest_neighbour_distances",
    "compute_prdc",
    "pairwise_squared_distances",
]
