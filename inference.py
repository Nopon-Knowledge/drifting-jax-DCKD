"""Auditable ImageNet generation and FID/IS/PRDC evaluation entrypoint.

Usage:
    python inference.py --init-from /path/to/run --generation-seed 271828 \
      --workdir runs/fid --generate-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from dataset.dataset import create_imagenet_split
from utils.env import (
    HF_ROOT,
    IMAGENET_FID_NPZ,
    IMAGENET_PRDC_NPZ,
    INCEPTION_PARAMS_PATH,
)
from utils.fid_util import evaluate_fid, evaluate_generated_samples, generate_samples
from utils.hsdp_util import data_shard, ddp_shard, set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.logging import WandbLogger
from utils.misc import prepare_rng, run_init

run_init()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _file_record(path: str | os.PathLike[str]) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    return {
        "path": str(resolved),
        "exists": True,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    Path(f"{path}.sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _atomic_save_npy(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return _file_record(path)


def _checkpoint_records(init_from: str) -> list[dict[str, Any]]:
    """Hash the local EMA payload and nearby provenance files when available."""

    if init_from.startswith("hf://"):
        return [{"path": init_from, "exists": None, "kind": "hf_alias"}]
    source = Path(init_from).expanduser().resolve()
    candidates: list[Path] = []
    if source.is_file():
        candidates.append(source)
    elif source.is_dir():
        for relative in (
            "params_ema/ema_params.msgpack",
            "ema_params.msgpack",
            "run_manifest.json",
            "config_snapshot.yaml",
            "metadata.json",
        ):
            candidate = source / relative
            if candidate.is_file():
                candidates.append(candidate)
    if not candidates:
        return [{"path": str(source), "exists": False}]
    records = []
    for candidate in candidates:
        record = _file_record(candidate)
        record["relative_to_init"] = (
            candidate.relative_to(source).as_posix()
            if source.is_dir()
            else candidate.name
        )
        records.append(record)
    return records


def _source_snapshot() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    paths = [
        root / "inference.py",
        root / "utils" / "fid_util.py",
        root / "utils" / "env.py",
        root / "utils" / "jax_fid" / "fid.py",
        root / "utils" / "jax_fid" / "inception.py",
        root / "utils" / "jax_fid" / "prdc.py",
        root / "utils" / "jax_fid" / "resize.py",
        root / "utils" / "jax_fid" / "cvt.py",
        root / "dataset" / "dataset.py",
    ]
    records = [_file_record(path) for path in paths]
    aggregate = hashlib.sha256()
    for record in records:
        relative = Path(record["path"]).relative_to(root).as_posix()
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(record.get("sha256", "")).encode("ascii"))
        aggregate.update(b"\n")
        record["relative_path"] = relative
    return {
        "project_root": str(root),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": records,
    }


def _optional_file_record(path: str) -> dict[str, Any] | None:
    return _file_record(path) if path else None


def _verify_record(record: dict[str, Any], *, role: str) -> None:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing {role}: {path}")
    actual_size = path.stat().st_size
    actual_hash = _sha256_file(path)
    if actual_size != int(record["bytes"]) or actual_hash != record["sha256"]:
        raise ValueError(
            f"{role} changed after generation: {path}; "
            f"expected bytes/hash={record['bytes']}/{record['sha256']}, "
            f"got {actual_size}/{actual_hash}."
        )


def _valid_rows(array: np.ndarray, masks: np.ndarray, num_samples: int) -> np.ndarray:
    return np.asarray(array)[np.asarray(masks) > 0.5][:num_samples]


def _is_latent(metadata: dict) -> bool:
    """Determine if the model operates in latent space from its metadata."""
    model_cfg = metadata.get("model_config", {})
    return model_cfg.get("in_channels", 3) == 4


def _load_model(
    init_from: str,
    *,
    vae_backend: str | None = None,
    vae_decode_batch_size: int = 2,
):
    """Build the generator and a bounded-memory inference callable."""
    model, params, metadata = load_generator_model_and_params(
        init_from,
        hf_cache_dir=HF_ROOT,
    )
    latent = _is_latent(metadata)
    vae_params = {}
    if latent:
        from dataset.vae import load_vae_module_and_params

        vae, vae_params = load_vae_module_and_params(
            replicate_params=True,
            backend=vae_backend,
        )
        gen_step_jit = _build_staged_latent_generator(
            apply_fn=model.apply,
            vae=vae,
            decode_batch_size=vae_decode_batch_size,
        )
    else:
        gen_step_jit = jax.jit(partial(generate_pixel_step, apply_fn=model.apply))
    return gen_step_jit, params, vae_params, metadata


def generate_model_step(batch, params, rng, apply_fn, cfg_scale=1.0):
    """Generate raw model samples without pixel-space postprocessing."""
    _, labels = batch
    labels = jax.lax.with_sharding_constraint(labels, data_shard())
    samples = apply_fn(
        {"params": params},
        train=False,
        rngs=prepare_rng(rng, ["noise"]),
        c=labels,
        cfg_scale=cfg_scale,
    )["samples"]
    samples = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        samples,
    )
    return samples


def generate_pixel_step(batch, params, vae_params, rng, apply_fn, cfg_scale=1.0):
    """Generate and postprocess samples for a pixel-space model."""
    del vae_params
    samples = generate_model_step(batch, params, rng, apply_fn, cfg_scale)
    return jnp.clip((samples + 1) / 2, 0, 1).transpose(0, 3, 1, 2)


def decode_latents(latents, vae_params, vae):
    """Decode one small latent block independently of the generator graph."""
    from diffusers.models import FlaxAutoencoderKL

    decoded = vae.apply(
        {"params": vae_params},
        latents / 0.18215,
        method=FlaxAutoencoderKL.decode,
    ).sample
    return jnp.clip((decoded + 1) / 2, 0, 1)


def _build_staged_latent_generator(*, apply_fn, vae, decode_batch_size: int):
    """Keep generator and VAE compilation isolated and bound VAE memory use."""
    if decode_batch_size < 1:
        raise ValueError(f"vae_decode_batch_size must be >= 1, got {decode_batch_size}")

    generate_latents_jit = jax.jit(partial(generate_model_step, apply_fn=apply_fn))
    decode_latents_jit = jax.jit(partial(decode_latents, vae=vae))
    warmed_up = False

    def staged_generate(batch, params, vae_params, rng, cfg_scale=1.0):
        nonlocal warmed_up
        if not warmed_up and jax.process_index() == 0:
            print("[INFO] Compile staged generator (VAE excluded)", flush=True)
        latents = generate_latents_jit(
            batch,
            params=params,
            rng=rng,
            cfg_scale=cfg_scale,
        )
        jax.block_until_ready(latents)
        if not warmed_up and jax.process_index() == 0:
            print(
                f"[INFO] Compile staged VAE decoder (batch={decode_batch_size})",
                flush=True,
            )

        decoded_blocks = []
        total = int(latents.shape[0])
        for start in range(0, total, decode_batch_size):
            stop = min(start + decode_batch_size, total)
            block = latents[start:stop]
            valid = stop - start
            if valid < decode_batch_size:
                pad_width = [(0, decode_batch_size - valid)] + [(0, 0)] * (block.ndim - 1)
                block = jnp.pad(block, pad_width)

            decoded = decode_latents_jit(block, vae_params=vae_params)
            jax.block_until_ready(decoded)
            decoded_blocks.append(np.asarray(jax.device_get(decoded))[:valid])

        if not warmed_up and jax.process_index() == 0:
            print("[INFO] Staged generator warmup complete", flush=True)
        warmed_up = True
        return np.concatenate(decoded_blocks, axis=0)

    return staged_generate


def _create_eval_loader(eval_batch_size: int):
    eval_loader, _, _ = create_imagenet_split(
        resolution=256,
        split="val",
        batch_size=eval_batch_size // jax.process_count(),
        num_workers=0,
    )
    return eval_loader


def _create_logger(
    *,
    init_from: str,
    workdir: str,
    use_wandb: bool,
    wandb_entity: str | None,
    wandb_project: str,
    wandb_name: str | None,
):
    logger = WandbLogger()
    logger.set_logging(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"{Path(init_from).name}_fid",
        use_wandb=use_wandb,
        workdir=str(Path(workdir).resolve()),
        log_every_k=1,
    )
    return logger

# ---------------------------------------------------------------------------
# eval_fid
# ---------------------------------------------------------------------------

def run_eval_fid(
    gen_step_jit, params, vae_params, metadata, init_from: str, workdir: str,
    *, num_samples: int, cfg_scale: float, eval_batch_size: int, eval_prc_recall: bool,
    generation_seed: int, eval_prdc: bool, prdc_nearest_k: int,
    prdc_reference_path: str, feature_artifact_path: str,
    use_wandb: bool, wandb_entity: str | None, wandb_project: str, wandb_name: str | None,
) -> dict:
    eval_loader = _create_eval_loader(eval_batch_size)
    logger = _create_logger(
        init_from=init_from,
        workdir=workdir,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )

    metrics = evaluate_fid(
        dataset_name="imagenet256",
        gen_func=gen_step_jit,
        gen_params={"params": params, "vae_params": vae_params, "cfg_scale": cfg_scale},
        eval_loader=eval_loader,
        logger=logger,
        num_samples=num_samples,
        log_folder="fid_eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=eval_prc_recall,
        eval_prdc=eval_prdc,
        prdc_nearest_k=prdc_nearest_k,
        prdc_reference_path=prdc_reference_path or None,
        feature_artifact_path=feature_artifact_path or None,
        artifact_metadata={
            "generation_seed": int(generation_seed),
            "cfg_scale": float(cfg_scale),
            "num_samples": int(num_samples),
            "init_from": init_from,
        },
        eval_isc=True,
        eval_fid=True,
        rng_eval=jax.random.PRNGKey(generation_seed),
    )
    logger.finish()
    return {
        "init_from": init_from,
        "cfg_scale": cfg_scale,
        "generation_seed": generation_seed,
        "metadata": metadata,
        **metrics,
    }


def run_generate_only(
    gen_step_jit,
    params,
    vae_params,
    metadata,
    init_from: str,
    *,
    workdir: str,
    samples_dir: str,
    num_samples: int,
    cfg_scale: float,
    eval_batch_size: int,
    vae_decode_batch_size: int,
    generation_seed: int,
    train_seed: str,
    config_path: str,
    protocol_id: str,
    protocol_path: str,
    source_snapshot_id: str,
    allow_overwrite: bool,
) -> dict:
    if jax.process_count() != 1:
        raise RuntimeError(
            "The auditable on-disk generation workflow currently requires one "
            "JAX process; use one gpu02 MIG allocation for formal evaluation."
        )
    eval_loader = _create_eval_loader(eval_batch_size)
    samples, masks, generation_time, generation_metadata = generate_samples(
        gen_func=gen_step_jit,
        gen_params={"params": params, "vae_params": vae_params, "cfg_scale": cfg_scale},
        eval_loader=eval_loader,
        num_samples=num_samples,
        rng_eval=jax.random.PRNGKey(generation_seed),
        return_metadata=True,
    )

    outdir = Path(samples_dir).resolve()
    workdir_path = Path(workdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    workdir_path.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "samples": outdir / "samples.npy",
        "masks": outdir / "masks.npy",
        "labels": outdir / "labels.npy",
        "rng_batch_indices": outdir / "rng_batch_indices.npy",
        "rng_positions": outdir / "rng_positions.npy",
    }
    manifest_paths = (
        outdir / "manifest.json",
        workdir_path / "generation_manifest.json",
    )
    existing = [
        path
        for path in (*artifact_paths.values(), *manifest_paths)
        if path.exists()
    ]
    if existing and not allow_overwrite:
        raise FileExistsError(
            "Refusing to overwrite generation evidence: "
            + ", ".join(str(path) for path in existing)
        )

    labels = np.asarray(generation_metadata["labels"], dtype=np.int64)
    rng_batch_indices = np.asarray(
        generation_metadata["rng_batch_indices"], dtype=np.int64
    )
    rng_positions = np.asarray(
        generation_metadata["rng_positions"], dtype=np.int64
    )
    row_count = int(samples.shape[0])
    if not (
        masks.shape[0]
        == labels.shape[0]
        == rng_batch_indices.shape[0]
        == rng_positions.shape[0]
        == row_count
    ):
        raise RuntimeError(
            "Generation evidence arrays are not aligned: "
            f"samples={row_count}, masks={masks.shape[0]}, labels={labels.shape[0]}, "
            f"batch_indices={rng_batch_indices.shape[0]}, "
            f"positions={rng_positions.shape[0]}."
        )
    valid_mask = np.asarray(masks) > 0.5
    valid_available = int(valid_mask.sum())
    if valid_available < num_samples:
        raise RuntimeError(
            f"Generated only {valid_available} valid rows for {num_samples} requested."
        )
    valid_slice = np.flatnonzero(valid_mask)[:num_samples]
    rng_tuples = np.stack(
        (
            rng_batch_indices[valid_slice],
            rng_positions[valid_slice],
        ),
        axis=1,
    )
    unique_rng_tuples = int(np.unique(rng_tuples, axis=0).shape[0])
    if unique_rng_tuples != num_samples:
        raise RuntimeError(
            "Duplicate generated RNG tuple detected: "
            f"{unique_rng_tuples}/{num_samples} unique."
        )

    arrays = {
        "samples": _atomic_save_npy(artifact_paths["samples"], samples),
        "masks": _atomic_save_npy(artifact_paths["masks"], masks),
        "labels": _atomic_save_npy(artifact_paths["labels"], labels),
        "rng_batch_indices": _atomic_save_npy(
            artifact_paths["rng_batch_indices"], rng_batch_indices
        ),
        "rng_positions": _atomic_save_npy(
            artifact_paths["rng_positions"], rng_positions
        ),
    }
    for name, record in arrays.items():
        record["shape"] = list(
            {
                "samples": samples,
                "masks": masks,
                "labels": labels,
                "rng_batch_indices": rng_batch_indices,
                "rng_positions": rng_positions,
            }[name].shape
        )
        record["dtype"] = str(
            {
                "samples": samples,
                "masks": masks,
                "labels": labels,
                "rng_batch_indices": rng_batch_indices,
                "rng_positions": rng_positions,
            }[name].dtype
        )

    manifest = {
        "schema_version": "drifting-generation-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "generated_and_hashed",
        "protocol_id": protocol_id,
        "protocol": _optional_file_record(protocol_path),
        "source_snapshot_id": source_snapshot_id,
        "source": _source_snapshot(),
        "init_from": init_from,
        "checkpoint_artifacts": _checkpoint_records(init_from),
        "train_seed": train_seed,
        "config": _optional_file_record(config_path),
        "cfg_scale": float(cfg_scale),
        "generation_seed": int(generation_seed),
        "rng": {
            "base": f"jax.random.PRNGKey({int(generation_seed)})",
            "batch": "jax.random.fold_in(base, batch_index)",
            "row_identity": (
                "(generation_seed, process_index, batch_index, position); "
                "position indexes output drawn from the batch key"
            ),
            "process_index": generation_metadata["process_index"],
            "process_count": generation_metadata["process_count"],
            "unique_valid_row_tuples": unique_rng_tuples,
        },
        "num_samples_requested": int(num_samples),
        "rows_saved": row_count,
        "valid_rows_available": valid_available,
        "eval_batch_size": int(eval_batch_size),
        "vae_decode_batch_size": int(vae_decode_batch_size),
        "generation_time_seconds": float(generation_time),
        "valid_label_schedule": {
            "count": int(num_samples),
            "sha256": _sha256_array(labels[valid_slice]),
            "ordering": "ImageNet validation DistributedSampler epoch=0",
        },
        "arrays": arrays,
        "metadata": metadata,
    }
    for manifest_path in manifest_paths:
        _atomic_write_json(manifest_path, manifest)
    return {
        "init_from": init_from,
        "cfg_scale": float(cfg_scale),
        "generation_seed": int(generation_seed),
        "num_samples": int(num_samples),
        "generation_time": float(generation_time),
        "generation_manifest": str(manifest_paths[1]),
        "generation_manifest_sha256": _sha256_file(manifest_paths[1]),
        "samples_dir": str(outdir),
    }


def run_metrics_only(
    init_from: str,
    workdir: str,
    *,
    samples_dir: str,
    num_samples: int,
    cfg_scale: float,
    eval_prc_recall: bool,
    eval_prdc: bool,
    prdc_nearest_k: int,
    prdc_reference_path: str,
    prdc_row_batch_size: int,
    prdc_col_batch_size: int,
    feature_artifact_path: str,
    generation_seed: int,
    train_seed: str,
    eval_batch_size: int,
    vae_decode_batch_size: int,
    source_snapshot_id: str,
    allow_overwrite: bool,
    use_wandb: bool,
    wandb_entity: str | None,
    wandb_project: str,
    wandb_name: str | None,
) -> dict:
    if jax.process_count() != 1:
        raise RuntimeError(
            "The auditable on-disk metrics workflow currently requires one JAX process."
        )
    indir = Path(samples_dir).resolve()
    generation_manifest_path = indir / "manifest.json"
    manifest = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "drifting-generation-manifest-v1":
        raise ValueError(
            f"Unsupported or legacy generation manifest: {generation_manifest_path}"
        )
    if int(manifest["num_samples_requested"]) != num_samples:
        raise ValueError(
            "Metrics must use the exact frozen generation sample count: "
            f"manifest={manifest['num_samples_requested']}, requested={num_samples}."
        )
    if float(manifest["cfg_scale"]) != float(cfg_scale):
        raise ValueError(
            f"CFG mismatch: manifest={manifest['cfg_scale']}, requested={cfg_scale}."
        )
    if int(manifest["generation_seed"]) != int(generation_seed):
        raise ValueError(
            "Generation-seed mismatch: "
            f"manifest={manifest['generation_seed']}, requested={generation_seed}."
        )
    if manifest["init_from"] != init_from:
        raise ValueError(
            f"Checkpoint input mismatch: manifest={manifest['init_from']!r}, "
            f"requested={init_from!r}."
        )
    if train_seed and str(manifest.get("train_seed", "")) != str(train_seed):
        raise ValueError(
            f"Train-seed mismatch: manifest={manifest.get('train_seed')!r}, "
            f"requested={train_seed!r}."
        )
    if int(manifest["eval_batch_size"]) != int(eval_batch_size):
        raise ValueError(
            "Evaluation-batch mismatch: "
            f"manifest={manifest['eval_batch_size']}, requested={eval_batch_size}."
        )
    if int(manifest["vae_decode_batch_size"]) != int(vae_decode_batch_size):
        raise ValueError(
            "VAE decode-batch mismatch: "
            f"manifest={manifest['vae_decode_batch_size']}, "
            f"requested={vae_decode_batch_size}."
        )
    if (
        source_snapshot_id
        and manifest.get("source_snapshot_id", "") != source_snapshot_id
    ):
        raise ValueError(
            "Source snapshot mismatch: "
            f"manifest={manifest.get('source_snapshot_id')!r}, "
            f"requested={source_snapshot_id!r}."
        )

    for role, record in manifest["arrays"].items():
        _verify_record(record, role=role)

    samples = np.load(indir / "samples.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(indir / "masks.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(indir / "labels.npy", mmap_mode="r", allow_pickle=False)
    rng_batch_indices = np.load(
        indir / "rng_batch_indices.npy", mmap_mode="r", allow_pickle=False
    )
    rng_positions = np.load(
        indir / "rng_positions.npy", mmap_mode="r", allow_pickle=False
    )
    if not (
        samples.shape[0]
        == masks.shape[0]
        == labels.shape[0]
        == rng_batch_indices.shape[0]
        == rng_positions.shape[0]
    ):
        raise ValueError("Saved generation arrays have inconsistent row counts.")
    valid_indices = np.flatnonzero(np.asarray(masks) > 0.5)[:num_samples]
    if valid_indices.shape[0] != num_samples:
        raise ValueError(
            f"Expected {num_samples} valid rows, got {valid_indices.shape[0]}."
        )
    label_hash = _sha256_array(np.asarray(labels)[valid_indices])
    if label_hash != manifest["valid_label_schedule"]["sha256"]:
        raise ValueError(
            "Valid label schedule no longer matches the generation manifest."
        )
    rng_tuples = np.stack(
        (
            np.asarray(rng_batch_indices)[valid_indices],
            np.asarray(rng_positions)[valid_indices],
        ),
        axis=1,
    )
    if np.unique(rng_tuples, axis=0).shape[0] != num_samples:
        raise ValueError("Duplicate RNG row tuple found in saved generation evidence.")

    workdir_path = Path(workdir).resolve()
    workdir_path.mkdir(parents=True, exist_ok=True)
    feature_path = Path(
        feature_artifact_path
        or (workdir_path / "generated_inception_artifacts.npz")
    ).expanduser().resolve()
    evaluation_manifest_path = workdir_path / "evaluation_manifest.json"
    if (
        (feature_path.exists() or evaluation_manifest_path.exists())
        and not allow_overwrite
    ):
        raise FileExistsError(
            "Refusing to overwrite existing metric evidence; use a new workdir "
            "for technical retries."
        )

    logger = _create_logger(
        init_from=init_from,
        workdir=workdir,
        use_wandb=use_wandb,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    metrics = evaluate_generated_samples(
        dataset_name="imagenet256",
        samples=samples,
        masks=masks,
        logger=logger,
        num_samples=num_samples,
        log_folder="fid_eval",
        log_prefix=f"cfg_{cfg_scale:g}",
        eval_prc_recall=eval_prc_recall,
        eval_prdc=eval_prdc,
        prdc_nearest_k=prdc_nearest_k,
        prdc_reference_path=prdc_reference_path or None,
        feature_artifact_path=str(feature_path),
        sample_labels=labels,
        artifact_metadata={
            "protocol_id": manifest.get("protocol_id", ""),
            "source_snapshot_id": manifest.get("source_snapshot_id", ""),
            "train_seed": manifest.get("train_seed", ""),
            "generation_seed": int(generation_seed),
            "cfg_scale": float(cfg_scale),
            "num_samples": int(num_samples),
            "valid_label_schedule_sha256": label_hash,
            "generation_manifest_sha256": _sha256_file(
                generation_manifest_path
            ),
        },
        prdc_row_batch_size=prdc_row_batch_size,
        prdc_col_batch_size=prdc_col_batch_size,
        eval_isc=True,
        eval_fid=True,
        generation_time=float(manifest.get("generation_time_seconds", 0.0)),
    )
    logger.finish()
    feature_manifest_path = feature_path.with_suffix(
        feature_path.suffix + ".manifest.json"
    )
    if not feature_path.is_file() or not feature_manifest_path.is_file():
        raise RuntimeError("Inception feature artifact was not written completely.")
    feature_sidecar = json.loads(
        feature_manifest_path.read_text(encoding="utf-8")
    )
    _verify_record(feature_sidecar["archive"], role="generated Inception archive")

    references = {
        "fid_stats": _file_record(IMAGENET_FID_NPZ),
        "inception_params": _file_record(INCEPTION_PARAMS_PATH),
        "prdc_features": (
            _file_record(prdc_reference_path or IMAGENET_PRDC_NPZ)
            if eval_prdc
            else None
        ),
    }
    for role, record in references.items():
        if record is not None and not record.get("exists", False):
            raise FileNotFoundError(f"Missing {role}: {record['path']}")

    evaluation_manifest = {
        "schema_version": "drifting-evaluation-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "metrics_computed_and_hashed",
        "protocol_id": manifest.get("protocol_id", ""),
        "source_snapshot_id": manifest.get("source_snapshot_id", ""),
        "source": _source_snapshot(),
        "generation_manifest": _file_record(generation_manifest_path),
        "generation_arrays": manifest["arrays"],
        "checkpoint_artifacts": manifest.get("checkpoint_artifacts", []),
        "train_seed": manifest.get("train_seed", ""),
        "generation_seed": int(generation_seed),
        "cfg_scale": float(cfg_scale),
        "num_samples": int(num_samples),
        "eval_batch_size": int(eval_batch_size),
        "vae_decode_batch_size": int(vae_decode_batch_size),
        "valid_label_schedule": manifest["valid_label_schedule"],
        "rng": manifest["rng"],
        "metrics": metrics,
        "metric_protocol": {
            "fid": True,
            "inception_score": True,
            "legacy_improved_pr": bool(eval_prc_recall),
            "canonical_prdc": bool(eval_prdc),
            "prdc_nearest_k": int(prdc_nearest_k) if eval_prdc else None,
            "prdc_distance": "exact blockwise squared Euclidean",
        },
        "references": references,
        "generated_inception_artifact": feature_sidecar,
    }
    _atomic_write_json(evaluation_manifest_path, evaluation_manifest)
    return {
        "init_from": manifest["init_from"],
        "cfg_scale": float(manifest["cfg_scale"]),
        "train_seed": manifest.get("train_seed", ""),
        "generation_seed": int(manifest["generation_seed"]),
        "num_samples": int(num_samples),
        "metadata": manifest.get("metadata", {}),
        "generation_manifest": str(generation_manifest_path),
        "generation_manifest_sha256": _sha256_file(generation_manifest_path),
        "evaluation_manifest": str(evaluation_manifest_path),
        "evaluation_manifest_sha256": _sha256_file(evaluation_manifest_path),
        "feature_artifact": str(feature_path),
        "feature_artifact_sha256": _sha256_file(feature_path),
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference: FID evaluation.")
    parser.add_argument("--init-from", required=True,
                        help="hf://<name> or local checkpoint path.")
    parser.add_argument("--workdir", default="runs/infer", help="Output directory.")
    parser.add_argument("--cfg-scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=0,
        help="Base JAX generation seed. Formal PR-DCKD evaluation uses 271828.",
    )
    parser.add_argument(
        "--train-seed",
        default="",
        help="Training seed of the checkpoint, recorded for provenance.",
    )
    parser.add_argument(
        "--config-path",
        default="",
        help="Training config to hash into the generation manifest.",
    )
    parser.add_argument("--protocol-id", default="")
    parser.add_argument("--protocol-path", default="")
    parser.add_argument(
        "--source-snapshot-id",
        default="",
        help="Frozen source identifier expected in both generation stages.",
    )
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--samples-dir", type=str, default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--hsdp-dim", type=int, default=None)
    parser.add_argument("--vae-backend", default=None, help="Backend for latent VAE decode, e.g. cuda/gpu/cpu.")
    parser.add_argument(
        "--vae-decode-batch-size",
        type=int,
        default=32,
        help="Independent latent VAE decode batch size. Keep this small to bound XLA compile memory.",
    )
    parser.add_argument("--eval-prc-recall", action="store_true", help="Also compute precision/recall if IMAGENET_PR_NPZ is set.")
    parser.add_argument(
        "--eval-prdc",
        action="store_true",
        help="Compute canonical Precision/Recall/Density/Coverage.",
    )
    parser.add_argument("--prdc-nearest-k", type=int, default=5)
    parser.add_argument(
        "--prdc-reference-path",
        default="",
        help="Feature-level ImageNet reference NPZ.",
    )
    parser.add_argument("--prdc-row-batch-size", type=int, default=1024)
    parser.add_argument("--prdc-col-batch-size", type=int, default=1024)
    parser.add_argument(
        "--feature-artifact-path",
        default="",
        help="Output NPZ for generated Inception features/logits/labels.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Permit a deliberate technical rerun to overwrite evidence files.",
    )
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="release-fid")
    parser.add_argument("--wandb-name", type=str, default=None)
    return parser


def run_inference_from_args(args: argparse.Namespace) -> dict:
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive.")
    if args.eval_batch_size < 1 or args.vae_decode_batch_size < 1:
        raise ValueError("Evaluation and VAE decode batch sizes must be positive.")
    if args.generation_seed < 0 or args.generation_seed > 0xFFFFFFFF:
        raise ValueError("--generation-seed must fit in an unsigned 32-bit key.")
    hsdp = args.hsdp_dim or min(8, jax.local_device_count() * jax.process_count())
    set_global_mesh(hsdp)
    samples_dir = args.samples_dir or str(Path(args.workdir) / "generated_samples")
    feature_artifact_path = args.feature_artifact_path or str(
        Path(args.workdir) / "generated_inception_artifacts.npz"
    )
    if args.metrics_only:
        return run_metrics_only(
            args.init_from,
            args.workdir,
            samples_dir=samples_dir,
            num_samples=args.num_samples,
            cfg_scale=args.cfg_scale,
            eval_prc_recall=bool(args.eval_prc_recall),
            eval_prdc=bool(args.eval_prdc),
            prdc_nearest_k=args.prdc_nearest_k,
            prdc_reference_path=args.prdc_reference_path,
            prdc_row_batch_size=args.prdc_row_batch_size,
            prdc_col_batch_size=args.prdc_col_batch_size,
            feature_artifact_path=feature_artifact_path,
            generation_seed=args.generation_seed,
            train_seed=args.train_seed,
            eval_batch_size=args.eval_batch_size,
            vae_decode_batch_size=args.vae_decode_batch_size,
            source_snapshot_id=args.source_snapshot_id,
            allow_overwrite=bool(args.allow_overwrite),
            use_wandb=args.use_wandb,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            wandb_name=args.wandb_name,
        )

    gen_step_jit, params, vae_params, metadata = _load_model(
        args.init_from,
        vae_backend=args.vae_backend,
        vae_decode_batch_size=args.vae_decode_batch_size,
    )
    if args.generate_only:
        return run_generate_only(
            gen_step_jit,
            params,
            vae_params,
            metadata,
            args.init_from,
            workdir=args.workdir,
            samples_dir=samples_dir,
            num_samples=args.num_samples,
            cfg_scale=args.cfg_scale,
            eval_batch_size=args.eval_batch_size,
            vae_decode_batch_size=args.vae_decode_batch_size,
            generation_seed=args.generation_seed,
            train_seed=args.train_seed,
            config_path=args.config_path,
            protocol_id=args.protocol_id,
            protocol_path=args.protocol_path,
            source_snapshot_id=args.source_snapshot_id,
            allow_overwrite=bool(args.allow_overwrite),
        )

    result = run_eval_fid(
        gen_step_jit, params, vae_params, metadata, args.init_from, args.workdir,
        num_samples=args.num_samples,
        cfg_scale=args.cfg_scale,
        eval_batch_size=args.eval_batch_size,
        eval_prc_recall=bool(args.eval_prc_recall),
        generation_seed=args.generation_seed,
        eval_prdc=bool(args.eval_prdc),
        prdc_nearest_k=args.prdc_nearest_k,
        prdc_reference_path=args.prdc_reference_path,
        feature_artifact_path=feature_artifact_path,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
    )
    return result


def main() -> None:
    args = build_parser().parse_args()
    result = run_inference_from_args(args)
    print(json.dumps(result, indent=2))
    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
