from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.jax_utils import replicate as R
from jax.experimental import multihost_utils
import jax.experimental.multihost_utils as mu

from utils.logging import log_for_0
from dataset.dataset import epoch0_sampler
from utils.hsdp_util import pad_and_merge, ddp_shard
from utils.env import IMAGENET_FID_NPZ, IMAGENET_PR_NPZ, IMAGENET_PRDC_NPZ


INCEPTION_NET = None
_DATASET_STATS = {
    "imagenet256": IMAGENET_FID_NPZ,
}
_PR_REF_PATH = IMAGENET_PR_NPZ
_PRDC_REF_PATH = IMAGENET_PRDC_NPZ


def _canonical_dataset_name(name: str) -> str:
    n = name.lower()
    if "imagenet256" in n:
        return "imagenet256"
    raise ValueError(f"Only ImageNet is supported now, got: {name}")


def _build_jax_inception(batch_size=200):
    """Create the pmap-compiled Inception network used for FID/IS features."""
    # Delay these imports until after distributed init. Several upstream Flax/JAX
    # helpers construct PRNG keys at import time, which counts as a JAX
    # computation and breaks multihost initialization ordering.
    from .jax_fid import inception, resize
    from .jax_fid.cvt import load_all as load_inception_params

    model = inception.InceptionV3(pretrained=True, include_head=True, transform_input=False)
    params = R(load_inception_params())

    def apply_fn(p, x):
        return model.apply(p, x, train=False)

    fake_x = jnp.zeros((jax.local_device_count(), batch_size, 299, 299, 3), dtype=jnp.float32)
    fn = jax.pmap(apply_fn).lower(params, fake_x).compile()
    return {"params": params, "fn": fn}


def _to_local_cpu(jax_array):
    """Gather addressable shards of a JAX array into a single numpy array on CPU.

    Returns:
        np.ndarray with shape ``(local_devices * per_device_batch, ...)``.
    """
    local_shards = jax_array.addressable_shards
    local_arrays = [np.array(s.data) for s in local_shards]
    return np.concatenate(local_arrays, axis=0)


def _to_uint8(samples):
    """Convert float ``[0, 1]`` samples to ``uint8 [0, 255]``."""
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=0.0)
    return (samples * 255).clip(0, 255).astype(np.uint8)


def _revert_pmap_shape(x):
    """Flatten pmap leading dims ``(devices, batch, ...)`` to ``(devices*batch, ...)``."""
    return x.reshape((-1, *x.shape[2:]))


def _compute_stats(samples_uint8: np.ndarray, num_samples: int, *, compute_logits: bool, compute_features: bool, masks=None):
    """Run Inception over generated samples and compute dataset statistics.

    Args:
        samples_uint8: generated images as `NHWC` or `NCHW` uint8 arrays.
        num_samples: target number of valid samples after removing padding.
        compute_logits: whether to keep classifier logits for IS.
        compute_features: whether to keep raw pool features for PR.
        masks: optional validity mask with shape `(N,)`; padded samples should be `0`.
    """
    global INCEPTION_NET
    if INCEPTION_NET is None:
        INCEPTION_NET = _build_jax_inception()

    if samples_uint8.shape[-1] != 3:
        samples_uint8 = samples_uint8.transpose(0, 2, 3, 1)

    if masks is None:
        masks = np.ones((len(samples_uint8),), dtype=np.float32)

    ldc = jax.local_device_count()
    batch_size = 200
    full_batch = batch_size * ldc
    pad = int(np.ceil(len(samples_uint8) / full_batch)) * full_batch - len(samples_uint8)
    if pad > 0:
        samples_uint8 = np.concatenate([samples_uint8, np.zeros((pad, *samples_uint8.shape[1:]), dtype=np.uint8)], axis=0)
        masks = np.concatenate([masks, np.zeros(pad, dtype=masks.dtype)])

    feats_list = []
    logits_list = []
    for i in range(0, len(samples_uint8), full_batch):
        # Inception expects NHWC float input in [0, 255]; resize helper consumes BCHW.
        from .jax_fid import resize

        x = torch.from_numpy(samples_uint8[i : i + full_batch].astype(np.float32).transpose(0, 3, 1, 2))
        x = resize.forward(x).numpy().transpose(0, 2, 3, 1)
        x = x.reshape((ldc, -1, *x.shape[1:]))
        pooled, _, logits = INCEPTION_NET["fn"](INCEPTION_NET["params"], jax.lax.stop_gradient(x))
        feats_list.append(_revert_pmap_shape(pooled))
        if compute_logits and logits is not None:
            logits_list.append(_revert_pmap_shape(logits))

    feats = jnp.concatenate(feats_list)
    all_feats = multihost_utils.process_allgather(feats).reshape(-1, feats.shape[-1])
    all_feats = jax.device_get(all_feats)

    np_mask = jnp.array(masks)
    all_masks = multihost_utils.process_allgather(np_mask).reshape(-1)
    all_masks = jax.device_get(all_masks)
    valid_len = min(all_feats.shape[0], all_masks.shape[0])
    all_feats = all_feats[:valid_len]
    all_masks = all_masks[:valid_len]
    all_feats = all_feats[all_masks > 0.5][:num_samples]

    feats64 = all_feats.astype(np.float64)
    out = {
        "mu": np.mean(feats64, axis=0),
        "sigma": np.cov(feats64, rowvar=False),
    }

    if compute_features:
        out["features"] = all_feats

    if compute_logits and logits_list:
        logits = jnp.concatenate(logits_list)
        all_logits = multihost_utils.process_allgather(logits).reshape(-1, logits.shape[-1])
        all_logits = jax.device_get(all_logits)
        all_logits = all_logits[:valid_len]
        all_logits = all_logits[all_masks > 0.5][:num_samples]
        out["logits"] = all_logits

    return out


def _compute_inception_score(logits, splits=10):
    rng = np.random.RandomState(2020)
    logits = logits[rng.permutation(logits.shape[0]), :]
    probs = jax.nn.softmax(logits, axis=-1)
    probs = np.asarray(probs, dtype=np.float64)

    n = probs.shape[0]
    split_size = n // splits
    probs = probs[: split_size * splits]
    scores = []
    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    scores = np.asarray(scores, dtype=np.float64)
    return float(np.mean(scores)), float(np.std(scores))


def _sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def save_inception_artifacts(
    output_path: str | os.PathLike[str],
    *,
    features: np.ndarray,
    logits: np.ndarray | None = None,
    labels: np.ndarray | None = None,
    relative_paths: np.ndarray | list[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Atomically save auditable Inception features and optional companions.

    The uncompressed NPZ contains a JSON manifest plus the requested arrays.
    A ``.manifest.json`` sidecar additionally records the final archive's
    SHA-256 and byte size.  Uncompressed storage avoids the substantial CPU and
    temporary-memory cost of compressing 50k x 2048 reference features.
    """

    output = Path(output_path).expanduser().resolve()
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(f"features must have shape [N, D], got {features.shape}.")
    if not np.issubdtype(features.dtype, np.floating):
        raise TypeError(f"features must be floating point, got {features.dtype}.")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinite values.")

    arrays: Dict[str, np.ndarray] = {
        "features": np.ascontiguousarray(features),
    }
    for name, value in (("logits", logits), ("labels", labels)):
        if value is None:
            continue
        value = np.asarray(value)
        if value.shape[0] != features.shape[0]:
            raise ValueError(
                f"{name} first dimension must equal features ({features.shape[0]}), "
                f"got {value.shape}."
            )
        arrays[name] = np.ascontiguousarray(value)

    if relative_paths is not None:
        paths = np.asarray(relative_paths, dtype=np.str_)
        if paths.ndim != 1 or paths.shape[0] != features.shape[0]:
            raise ValueError(
                "relative_paths must have one entry per feature, got "
                f"{paths.shape} for {features.shape[0]} features."
            )
        arrays["relative_paths"] = np.ascontiguousarray(paths)

    manifest: Dict[str, Any] = {
        "schema_version": "drifting-inception-artifacts-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arrays": {
            name: {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": _sha256_array(array),
            }
            for name, array in arrays.items()
        },
        "metadata": dict(metadata or {}),
    }
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        with open(tmp_output, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp_output, output)
    finally:
        if tmp_output.exists():
            tmp_output.unlink()

    manifest["archive"] = {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256_file(output),
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    tmp_manifest = manifest_path.with_name(
        f".{manifest_path.name}.tmp.{os.getpid()}"
    )
    try:
        tmp_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_manifest, manifest_path)
    finally:
        if tmp_manifest.exists():
            tmp_manifest.unlink()
    return manifest


def load_inception_feature_reference(
    reference_path: str | os.PathLike[str] = _PRDC_REF_PATH,
) -> Dict[str, np.ndarray]:
    """Load a feature-level reference produced by the preparation script."""

    path = Path(reference_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Inception feature reference not found: {path}. "
            "Run scripts/prepare_imagenet_fid_stats.py first."
        )
    with np.load(path, allow_pickle=False) as data:
        if "features" not in data:
            raise ValueError(
                f"{path} is not a feature-level reference (missing 'features'). "
                "Legacy image archives cannot be used for canonical PRDC."
            )
        result = {"features": np.asarray(data["features"])}
        for name in ("labels", "relative_paths", "logits", "manifest_json"):
            if name in data:
                result[name] = np.asarray(data[name])
    if result["features"].ndim != 2:
        raise ValueError(
            f"Reference features must have shape [N, D], got {result['features'].shape}."
        )
    return result


def _load_improved_pr_reference_features(
    reference_path: str | os.PathLike[str] | None = None,
) -> np.ndarray:
    """Load features for the legacy improved-P/R metric.

    A new feature-level reference is preferred.  The old ``arr_0`` image
    archive remains supported so existing workflows are not silently broken.
    """

    if reference_path is None:
        reference_path = _PRDC_REF_PATH if Path(_PRDC_REF_PATH).is_file() else _PR_REF_PATH
    path = Path(reference_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Improved-P/R reference not found: {path}.")
    with np.load(path, allow_pickle=False) as data:
        if "features" in data:
            return np.asarray(data["features"])
        if "arr_0" not in data:
            raise ValueError(
                f"Unsupported improved-P/R reference format in {path}; "
                "expected 'features' or legacy 'arr_0'."
            )
        ref_images = np.asarray(data["arr_0"], dtype=np.uint8)
    return _compute_stats(
        ref_images,
        ref_images.shape[0],
        compute_logits=False,
        compute_features=True,
    )["features"]


def _gather_valid_labels(
    labels: np.ndarray,
    masks: np.ndarray,
    num_samples: int,
) -> np.ndarray:
    """Mirror feature gathering/filtering for numeric generated labels."""

    labels = np.asarray(labels)
    masks = np.asarray(masks)
    if labels.shape[0] != masks.shape[0]:
        raise ValueError(
            f"sample_labels and masks must have equal length, got "
            f"{labels.shape[0]} and {masks.shape[0]}."
        )
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError(f"sample_labels must be numeric, got {labels.dtype}.")
    all_labels = multihost_utils.process_allgather(jnp.asarray(labels))
    all_labels = np.asarray(jax.device_get(all_labels)).reshape(
        (-1, *labels.shape[1:])
    )
    all_masks = multihost_utils.process_allgather(jnp.asarray(masks))
    all_masks = np.asarray(jax.device_get(all_masks)).reshape(-1)
    valid_len = min(all_labels.shape[0], all_masks.shape[0])
    return all_labels[:valid_len][all_masks[:valid_len] > 0.5][:num_samples]


def _load_ref_stats(dataset_name: str):
    canon = _canonical_dataset_name(dataset_name)
    path = _DATASET_STATS[canon]
    data = np.load(path)
    if "ref_mu" in data:
        return {"mu": data["ref_mu"], "sigma": data["ref_sigma"]}
    return {"mu": data["mu"], "sigma": data["sigma"]}


def evaluate_fid(
    dataset_name,
    gen_func,
    gen_params,
    eval_loader,
    logger,
    num_samples=5000,
    log_folder="fid",
    log_prefix="gen_model",
    eval_prc_recall=False,
    eval_prdc=False,
    prdc_nearest_k=5,
    prdc_reference_path=None,
    improved_pr_reference_path=None,
    feature_artifact_path=None,
    sample_labels=None,
    artifact_metadata=None,
    prdc_row_batch_size=1024,
    prdc_col_batch_size=1024,
    eval_isc=True,
    eval_fid=True,
    rng_eval=None,
):
    """Generate samples, run Inception statistics, and log release metrics.

    Args:
        dataset_name: Dataset identifier used to select reference statistics.
            Only ImageNet-256 is supported in this release.
        gen_func: Generation callable that accepts one merged eval batch plus the
            contents of ``gen_params`` and ``rng=...``. It must return samples in
            ``BCHW`` or ``BHWC`` format with values in ``[0, 1]``.
        gen_params: Keyword arguments forwarded into ``gen_func`` for every eval
            batch. This typically contains the EMA params and a fixed CFG scale.
        eval_loader: Iterable of ``(images, labels)`` batches. The labels are
            used to drive conditional generation; the image tensors are ignored.
        logger: Logger that receives scalar metrics via ``log_dict`` and a
            64-image preview grid via ``log_image``.
        num_samples: Number of valid generated samples to score after padding is
            removed across all hosts.
        log_folder: Top-level metric namespace written into the logger.
        log_prefix: Per-run metric prefix inside ``log_folder``.
        eval_prc_recall: Whether to compute the legacy improved P/R at k=3.
        eval_prdc: Whether to compute canonical P/R/D/C.
        prdc_nearest_k: Neighborhood size for canonical PRDC (default 5).
        eval_isc: Whether to compute Inception Score.
        eval_fid: Whether to compute FID.
        rng_eval: Base PRNGKey for deterministic evaluation sampling.

    Returns:
        Dict[str, float] containing the computed metrics. Keys may include
        ``fid``, ``isc_mean``, ``isc_std``, ``precision``, ``recall``, and
        ``fid_time`` depending on which evaluations are enabled.
    """
    samples, masks, generation_time = generate_samples(
        gen_func=gen_func,
        gen_params=gen_params,
        eval_loader=eval_loader,
        num_samples=num_samples,
        rng_eval=rng_eval,
    )
    return evaluate_generated_samples(
        dataset_name=dataset_name,
        samples=samples,
        masks=masks,
        logger=logger,
        num_samples=num_samples,
        log_folder=log_folder,
        log_prefix=log_prefix,
        eval_prc_recall=eval_prc_recall,
        eval_prdc=eval_prdc,
        prdc_nearest_k=prdc_nearest_k,
        prdc_reference_path=prdc_reference_path,
        improved_pr_reference_path=improved_pr_reference_path,
        feature_artifact_path=feature_artifact_path,
        sample_labels=sample_labels,
        artifact_metadata=artifact_metadata,
        prdc_row_batch_size=prdc_row_batch_size,
        prdc_col_batch_size=prdc_col_batch_size,
        eval_isc=eval_isc,
        eval_fid=eval_fid,
        generation_time=generation_time,
    )


def generate_samples(
    *,
    gen_func,
    gen_params,
    eval_loader,
    num_samples,
    rng_eval=None,
    return_metadata=False,
):
    """Generate local uint8 samples and validity masks without running Inception.

    When ``return_metadata`` is true, the fourth return value contains arrays
    aligned one-for-one with the saved sample rows.  They make the conditional
    label schedule and the ``fold_in(base_key, batch_index)`` RNG stream
    auditable without changing the generator call itself.
    """
    if rng_eval is None:
        rng_eval = jax.random.PRNGKey(0)

    start = time.time()
    eval_iter = epoch0_sampler(eval_loader)
    all_samples = []
    all_masks = []
    all_labels = []
    all_rng_batch_indices = []
    all_rng_positions = []
    cur = 0
    goal_bsz = None
    for i, batch in enumerate(eval_iter):
        if goal_bsz is None:
            goal_bsz = jax.tree.leaves(batch)[0].shape[0]
        # Pad the final batch so every host/device sees a static shape.
        batch, mask = pad_and_merge(batch, goal_bsz)
        rng_step = jax.random.fold_in(rng_eval, i)
        gen_samples = gen_func(batch, **gen_params, rng=rng_step)
        gen_samples = jax.device_put(gen_samples, ddp_shard())
        mask = jax.device_put(mask, ddp_shard())
        local_samples = _to_uint8(_to_local_cpu(gen_samples))
        local_masks = _to_local_cpu(mask)
        local_labels = _to_local_cpu(batch[1])
        local_count = int(local_samples.shape[0])
        if not (
            local_masks.shape[0] == local_labels.shape[0] == local_count
        ):
            raise RuntimeError(
                "Generated sample provenance arrays are misaligned: "
                f"samples={local_count}, masks={local_masks.shape[0]}, "
                f"labels={local_labels.shape[0]}."
            )
        all_samples.append(local_samples)
        all_masks.append(local_masks)
        all_labels.append(np.asarray(local_labels, dtype=np.int64))
        all_rng_batch_indices.append(
            np.full(local_count, i, dtype=np.int64)
        )
        all_rng_positions.append(
            np.arange(local_count, dtype=np.int64)
        )
        cur += gen_samples.shape[0]
        if i == 0 or (i + 1) % 100 == 0 or cur >= num_samples:
            log_for_0("FID generation: %d/%d local samples", min(cur, num_samples), num_samples)
        if cur >= num_samples:
            break

    samples = np.concatenate(all_samples, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    generation_time = float(time.time() - start)
    if not return_metadata:
        return samples, masks, generation_time
    metadata = {
        "labels": np.concatenate(all_labels, axis=0),
        "rng_batch_indices": np.concatenate(all_rng_batch_indices, axis=0),
        "rng_positions": np.concatenate(all_rng_positions, axis=0),
        "process_index": int(jax.process_index()),
        "process_count": int(jax.process_count()),
        "rng_scheme": "jax.random.fold_in(PRNGKey(generation_seed), batch_index)",
    }
    return samples, masks, generation_time, metadata


def evaluate_generated_samples(
    *,
    dataset_name,
    samples,
    masks,
    logger,
    num_samples,
    log_folder="fid",
    log_prefix="gen_model",
    eval_prc_recall=False,
    eval_prdc=False,
    prdc_nearest_k=5,
    prdc_reference_path=None,
    improved_pr_reference_path=None,
    feature_artifact_path=None,
    sample_labels=None,
    artifact_metadata=None,
    prdc_row_batch_size=1024,
    prdc_col_batch_size=1024,
    eval_isc=True,
    eval_fid=True,
    generation_time=0.0,
):
    """Compute FID/IS and optional improved-P/R or canonical PRDC metrics."""
    from .jax_fid.fid import compute_frechet_distance
    from .jax_fid.prdc import compute_prdc
    from .jax_fid.precision_recall import compute_improved_precision_recall

    start = time.time()
    save_artifacts = bool(feature_artifact_path)
    stats = _compute_stats(
        samples,
        num_samples,
        compute_logits=(eval_isc or save_artifacts),
        compute_features=(eval_prc_recall or eval_prdc or save_artifacts),
        masks=masks,
    )
    ref = _load_ref_stats(dataset_name)

    metrics: Dict[str, float] = {}
    if eval_fid:
        metrics["fid"] = float(compute_frechet_distance(ref["mu"], stats["mu"], ref["sigma"], stats["sigma"]))
    if eval_isc and "logits" in stats:
        mean, std = _compute_inception_score(stats["logits"])
        metrics["isc_mean"] = mean
        metrics["isc_std"] = std
    if eval_prc_recall and "features" in stats:
        ref_features = _load_improved_pr_reference_features(
            improved_pr_reference_path
        )
        precision, recall = compute_improved_precision_recall(
            ref_features, stats["features"], k=3
        )
        metrics["improved_precision_k3"] = float(precision)
        metrics["improved_recall_k3"] = float(recall)
    if eval_prdc and "features" in stats:
        reference = load_inception_feature_reference(
            prdc_reference_path or _PRDC_REF_PATH
        )
        prdc = compute_prdc(
            reference["features"],
            stats["features"],
            nearest_k=prdc_nearest_k,
            row_batch_size=prdc_row_batch_size,
            col_batch_size=prdc_col_batch_size,
        )
        metrics.update(
            {
                "prdc_precision": prdc["precision"],
                "prdc_recall": prdc["recall"],
                "prdc_density": prdc["density"],
                "prdc_coverage": prdc["coverage"],
                "prdc_nearest_k": prdc["nearest_k"],
            }
        )
    if save_artifacts:
        artifact_labels = None
        if sample_labels is not None:
            artifact_labels = _gather_valid_labels(
                sample_labels, masks, num_samples
            )
        save_inception_artifacts(
            feature_artifact_path,
            features=stats["features"],
            logits=stats.get("logits"),
            labels=artifact_labels,
            metadata=artifact_metadata,
        )

    metrics["fid_time"] = float(generation_time + time.time() - start)
    logger.log_dict({f"{log_folder}/{log_prefix}_{k}": v for k, v in metrics.items()})
    logger.log_image(f"{log_folder}/{log_prefix}_viz", samples[:64])
    mu.sync_global_devices("fid evaluation finished")
    return metrics
