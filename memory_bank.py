from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np


class ArrayMemoryBank:
    """Class-wise ring buffer for feature/image samples used by generator training."""

    def __init__(self, num_classes: int = 1000, max_size: int = 64, dtype=np.float32):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.bank: Optional[np.ndarray] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)

    def _init_bank(self, sample_shape: Tuple[int, ...]) -> None:
        self.feature_shape = tuple(sample_shape)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.feature_shape), dtype=self.dtype)

    def add(self, samples, labels) -> None:
        """Insert samples into per-class ring buffers.

        Args:
            samples: array of shape ``(N, *feature_shape)`` to store.
            labels: integer class labels of shape ``(N,)``.
        """
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        if self.bank is None:
            self._init_bank(samples.shape[1:])

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def sample(self, labels, n_samples: int, rng: np.random.Generator | None = None):
        """Sample stored entries for each label.

        Args:
            labels: integer class labels of shape ``(B,)``.
            n_samples: number of samples to draw per label.

        Returns:
            jnp.ndarray of shape ``(B, n_samples, *feature_shape)``.
        """
        if self.bank is None or self.feature_shape is None:
            raise RuntimeError("MemoryBank is empty. Call add() before sample().")

        rng = rng or np.random.default_rng()
        labels = np.asarray(labels)
        bsz = labels.shape[0]
        sample_indices = np.empty((bsz, n_samples), dtype=np.int32)
        for i in range(bsz):
            lbl = int(labels[i])
            valid = int(self.count[lbl])
            if valid <= 0:
                sample_indices[i] = np.zeros((n_samples,), dtype=np.int32)
            else:
                sample_indices[i] = rng.choice(valid, n_samples, replace=(valid < n_samples))

        out = self.bank[labels[:, None], sample_indices]
        return jnp.asarray(out)


class RecencyGeneratedNegativeBank:
    """Generated-sample replay bank with recency, stale-distance, and hard sampling.

    The bank stores generated samples plus compact feature summaries. Sampling is
    class-aware when possible and falls back to the global generated pool when a
    class is still cold. Scores combine recency and feature-distance correction:

    score = exp(-age / half_life) * exp(-distance(query, stored) / distance_scale)

    A configurable fraction of candidates is selected as hard-but-fresh top
    scores; the rest is sampled stochastically from the same corrected scores.
    """

    def __init__(
        self,
        *,
        num_classes: int = 1000,
        max_size: int = 256,
        dtype=np.float32,
        half_life: float = 500.0,
        distance_scale: float = 1.0,
        hard_fraction: float = 0.5,
        hard_fraction_min: float = 0.1,
        candidate_multiplier: int = 4,
        min_weight: float = 0.05,
        max_weight: float = 4.0,
    ):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)
        self.dtype = dtype
        self.half_life = float(max(1e-6, half_life))
        self.distance_scale = float(max(1e-6, distance_scale))
        self.hard_fraction = float(np.clip(hard_fraction, 0.0, 1.0))
        self.hard_fraction_min = float(np.clip(hard_fraction_min, 0.0, self.hard_fraction))
        self.candidate_multiplier = int(max(1, candidate_multiplier))
        self.min_weight = float(max(0.0, min_weight))
        self.max_weight = float(max(self.min_weight, max_weight))

        self.bank: Optional[np.ndarray] = None
        self.summaries: Optional[np.ndarray] = None
        self.steps = np.zeros((self.num_classes, self.max_size), dtype=np.int32)
        self.ptr = np.zeros(self.num_classes, dtype=np.int32)
        self.count = np.zeros(self.num_classes, dtype=np.int32)
        self.sample_shape: Optional[Tuple[int, ...]] = None
        self.summary_dim: Optional[int] = None

    def _init_bank(self, sample_shape: Tuple[int, ...], summary_dim: int) -> None:
        self.sample_shape = tuple(sample_shape)
        self.summary_dim = int(summary_dim)
        self.bank = np.zeros((self.num_classes, self.max_size, *self.sample_shape), dtype=self.dtype)
        self.summaries = np.zeros((self.num_classes, self.max_size, self.summary_dim), dtype=np.float32)

    @property
    def total_count(self) -> int:
        return int(self.count.sum())

    def add(self, samples, labels, *, step: int, summaries) -> None:
        samples = np.asarray(samples)
        labels = np.asarray(labels)
        summaries = np.asarray(summaries, dtype=np.float32)
        if samples.shape[0] != labels.shape[0] or samples.shape[0] != summaries.shape[0]:
            raise ValueError(
                f"Mismatched add batch shapes: samples={samples.shape}, labels={labels.shape}, summaries={summaries.shape}"
            )
        if summaries.ndim != 2:
            summaries = summaries.reshape((summaries.shape[0], -1))
        norms = np.linalg.norm(summaries, axis=1, keepdims=True)
        summaries = summaries / np.clip(norms, 1e-6, None)

        if self.bank is None or self.summaries is None:
            self._init_bank(samples.shape[1:], summaries.shape[-1])
        if samples.shape[1:] != self.sample_shape:
            raise ValueError(f"Expected sample shape {self.sample_shape}, got {samples.shape[1:]}.")
        if summaries.shape[-1] != self.summary_dim:
            raise ValueError(f"Expected summary dim {self.summary_dim}, got {summaries.shape[-1]}.")

        for i in range(labels.shape[0]):
            lbl = int(labels[i])
            if not 0 <= lbl < self.num_classes:
                continue
            idx = self.ptr[lbl]
            self.bank[lbl, idx] = samples[i]
            self.summaries[lbl, idx] = summaries[i]
            self.steps[lbl, idx] = int(step)
            self.ptr[lbl] = (idx + 1) % self.max_size
            if self.count[lbl] < self.max_size:
                self.count[lbl] += 1

    def _entries_for_label(self, label: int) -> tuple[np.ndarray, np.ndarray]:
        valid = int(self.count[label]) if 0 <= label < self.num_classes else 0
        if valid > 0:
            return np.full(valid, label, dtype=np.int32), np.arange(valid, dtype=np.int32)

        classes = np.nonzero(self.count > 0)[0]
        if classes.size == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)
        cls_list = []
        idx_list = []
        for cls in classes:
            n = int(self.count[cls])
            cls_list.append(np.full(n, cls, dtype=np.int32))
            idx_list.append(np.arange(n, dtype=np.int32))
        return np.concatenate(cls_list), np.concatenate(idx_list)

    def sample(
        self,
        labels,
        *,
        n_samples: int,
        current_step: int,
        query_summaries=None,
        rng: np.random.Generator | None = None,
        return_info: bool = False,
    ):
        if self.bank is None or self.summaries is None or self.sample_shape is None:
            return None

        rng = rng or np.random.default_rng()
        labels = np.asarray(labels)
        query_summaries = None if query_summaries is None else np.asarray(query_summaries, dtype=np.float32)
        if query_summaries is not None:
            query_summaries = query_summaries.reshape((query_summaries.shape[0], -1))
            query_summaries = query_summaries / np.clip(
                np.linalg.norm(query_summaries, axis=1, keepdims=True),
                1e-6,
                None,
            )

        bsz = labels.shape[0]
        out = np.zeros((bsz, n_samples, *self.sample_shape), dtype=self.dtype)
        weights = np.zeros((bsz, n_samples), dtype=np.float32)
        ages_out = np.zeros((bsz, n_samples), dtype=np.float32)
        distances_out = np.zeros((bsz, n_samples), dtype=np.float32)
        hard_fracs = np.zeros((bsz,), dtype=np.float32)

        for i in range(bsz):
            cls_idx, entry_idx = self._entries_for_label(int(labels[i]))
            valid = cls_idx.shape[0]
            if valid <= 0:
                continue

            cand_n = min(valid, max(n_samples, n_samples * self.candidate_multiplier))
            cand_pos = rng.choice(valid, cand_n, replace=(valid < cand_n))
            cand_cls = cls_idx[cand_pos]
            cand_entry = entry_idx[cand_pos]
            cand_summaries = self.summaries[cand_cls, cand_entry]
            ages = np.maximum(int(current_step) - self.steps[cand_cls, cand_entry], 0).astype(np.float32)
            recency = np.exp(-ages / self.half_life)

            if query_summaries is not None:
                diff = cand_summaries - query_summaries[i][None, :]
                distances = np.linalg.norm(diff, axis=1).astype(np.float32)
                stale = np.exp(-distances / self.distance_scale)
            else:
                distances = np.zeros((cand_n,), dtype=np.float32)
                stale = np.ones((cand_n,), dtype=np.float32)

            scores = np.clip(recency * stale, 1e-8, None)
            score_cv = float(scores.std() / np.clip(scores.mean(), 1e-6, None))
            hard_mix = float(np.clip(score_cv, 0.0, 1.0))
            hard_fraction = self.hard_fraction_min + (self.hard_fraction - self.hard_fraction_min) * hard_mix
            hard_fracs[i] = hard_fraction
            hard_n = min(n_samples, int(round(n_samples * hard_fraction)))
            selected: list[int] = []
            if hard_n > 0:
                selected.extend(np.argsort(-scores)[:hard_n].tolist())
            rest_n = n_samples - len(selected)
            if rest_n > 0:
                probs = scores / scores.sum()
                selected.extend(rng.choice(cand_n, rest_n, replace=(cand_n < rest_n), p=probs).tolist())
            selected_arr = np.asarray(selected[:n_samples], dtype=np.int32)

            out[i] = self.bank[cand_cls[selected_arr], cand_entry[selected_arr]]
            raw_weights = scores[selected_arr].astype(np.float32)
            raw_weights = raw_weights / np.clip(raw_weights.mean(), 1e-6, None)
            weights[i] = np.clip(raw_weights, self.min_weight, self.max_weight)
            ages_out[i] = ages[selected_arr]
            distances_out[i] = distances[selected_arr]

        result = (jnp.asarray(out), jnp.asarray(weights))
        if not return_info:
            return result
        info = {
            "replay/total_count": float(self.total_count),
            "replay/weight_mean": float(weights.mean()) if weights.size else 0.0,
            "replay/age_mean": float(ages_out[weights > 0].mean()) if np.any(weights > 0) else 0.0,
            "replay/distance_mean": float(distances_out[weights > 0].mean()) if np.any(weights > 0) else 0.0,
            "replay/nonzero_frac": float((weights > 0).mean()) if weights.size else 0.0,
            "replay/hard_fraction_mean": float(hard_fracs[hard_fracs > 0].mean()) if np.any(hard_fracs > 0) else 0.0,
        }
        return (*result, info)
