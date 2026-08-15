#!/usr/bin/env python3
"""Reproducible generator-only efficiency benchmark for PR-DCKD-V1.

The benchmark intentionally measures the one-step generator only.  It excludes
the latent VAE decoder, data loading, FID/PRDC feature extraction, and training.
That scope is the right one for testing the claim that DCKD changes training
but adds no generator structure or inference-time model evaluations.

For a formal run, pass a local completed ``params_ema`` artifact (or its parent
work directory).  The script records a stable SHA-256 digest of every artifact
file, the model/config metadata, parameter counts, XLA cost analysis, compiler
memory analysis, wall-clock latency/throughput, and two complementary memory
measurements:

* JAX allocator statistics, when exposed by the backend.  Their peak is scoped
  to the current process lifetime and therefore includes loading/compilation.
* periodic ``nvidia-smi`` samples for the current PID.  This is an observed
  process-memory peak and may miss transients shorter than the sample interval.

Use ``--self-test`` to exercise compilation, timing, cost analysis, memory
collection, and atomic JSON output without loading a real checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from utils.hsdp_util import set_global_mesh
from utils.init_util import load_generator_model_and_params, resolve_artifact_dir


SCHEMA_VERSION = "pr-dckd-efficiency-v1"
MEASUREMENT_SCOPE = (
    "generator model forward only; excludes VAE decode, input pipeline, "
    "FID/IS/PRDC, checkpoint loading, and training"
)
NFE_DEFINITION = (
    "One invocation of the one-step DitGen/LightningDiT generator per sample. "
    "CFG is an embedded conditioning scalar and does not invoke a second model pass."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file_stable(path: Path, chunk_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    """Hash one file and fail if its identity/size/mtime changes while reading."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if signature_before != signature_after:
        raise RuntimeError(f"Artifact file changed while hashing: {path}")
    return digest.hexdigest(), int(after.st_size)


def hash_artifact(path: Path) -> dict[str, Any]:
    """Return a content-addressed manifest for a file or directory artifact."""
    root = path.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint artifact does not exist: {root}")

    if root.is_file():
        files_before = [root]
        relative = lambda _: root.name
    else:
        files_before = sorted(
            (candidate for candidate in root.rglob("*") if candidate.is_file()),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
        relative = lambda candidate: candidate.relative_to(root).as_posix()
    if not files_before:
        raise ValueError(f"Checkpoint artifact contains no files: {root}")

    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    total_bytes = 0
    for file_path in files_before:
        file_sha, file_bytes = _sha256_file_stable(file_path)
        rel = relative(file_path)
        record = {"path": rel, "bytes": file_bytes, "sha256": file_sha}
        records.append(record)
        total_bytes += file_bytes
        aggregate.update(rel.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(file_bytes).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(file_sha.encode("ascii"))
        aggregate.update(b"\n")

    if root.is_dir():
        files_after = sorted(
            candidate.relative_to(root).as_posix()
            for candidate in root.rglob("*")
            if candidate.is_file()
        )
        if files_after != [record["path"] for record in records]:
            raise RuntimeError(f"Artifact file set changed while hashing: {root}")

    return {
        "path": str(root),
        "hash_scope": "all regular files recursively; aggregate hashes path, size, and file SHA-256",
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": records,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    """Linear-interpolated percentile with no optional dependency."""
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentile must be in [0, 100], got {percentile}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parameter_statistics(params: Any) -> dict[str, Any]:
    """Count loaded parameter leaves without copying their values to the host."""
    leaves = jax.tree_util.tree_leaves(params)
    dtype_elements: Counter[str] = Counter()
    dtype_bytes: Counter[str] = Counter()
    total = 0
    total_bytes = 0
    array_leaves = 0
    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            continue
        elements = math.prod(int(dim) for dim in leaf.shape)
        itemsize = int(leaf.dtype.itemsize)
        dtype_name = str(leaf.dtype)
        total += elements
        total_bytes += elements * itemsize
        dtype_elements[dtype_name] += elements
        dtype_bytes[dtype_name] += elements * itemsize
        array_leaves += 1

    return {
        "total_parameters": total,
        "trainable_parameters": total,
        "trainable_count_basis": (
            "All leaves in the loaded Flax 'params' collection are trainable model "
            "parameters; no frozen/non-params collection is loaded for inference."
        ),
        "parameter_bytes_from_dtype": total_bytes,
        "array_leaf_count": array_leaves,
        "by_dtype": {
            dtype: {
                "parameters": dtype_elements[dtype],
                "bytes": dtype_bytes[dtype],
            }
            for dtype in sorted(dtype_elements)
        },
    }


def _to_json_number(value: Any) -> int | float | str | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else str(numeric)
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _normalize_cost_analysis(raw: Any) -> dict[str, Any]:
    if isinstance(raw, (list, tuple)):
        combined: dict[str, Any] = {}
        for index, entry in enumerate(raw):
            if isinstance(entry, Mapping):
                for key, value in entry.items():
                    combined[f"module_{index}/{key}"] = _to_json_number(value)
        return combined
    if isinstance(raw, Mapping):
        return {str(key): _to_json_number(value) for key, value in raw.items()}
    return {"unavailable": str(raw)}


def _cost_metric_total(cost: Mapping[str, Any], metric_name: str) -> float | None:
    """Sum a named metric across one or more XLA module records."""
    values: list[float] = []
    for key, value in cost.items():
        if key == metric_name or key.endswith(f"/{metric_name}"):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return sum(values) if values else None


def _compiler_memory_analysis(compiled: Any) -> dict[str, Any]:
    try:
        analysis = compiled.memory_analysis()
    except (AttributeError, RuntimeError, NotImplementedError) as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    if analysis is None:
        return {"available": False, "reason": "backend returned None"}

    fields = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "host_argument_size_in_bytes",
        "host_output_size_in_bytes",
        "host_alias_size_in_bytes",
        "host_temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    payload = {
        field: int(getattr(analysis, field))
        for field in fields
        if getattr(analysis, field, None) is not None
    }
    if {
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
    }.issubset(payload):
        payload["estimated_peak_bytes"] = (
            payload["argument_size_in_bytes"]
            + payload["output_size_in_bytes"]
            - payload["alias_size_in_bytes"]
            + payload["temp_size_in_bytes"]
        )
    return {
        "available": True,
        "scope": (
            "XLA compiler estimate for this executable; estimated_peak_bytes is "
            "arguments + outputs - aliases + temporary buffers"
        ),
        **payload,
    }


def _block_until_ready(value: Any) -> None:
    jax.block_until_ready(value)


def compile_and_measure(
    jitted_fn: Any,
    args: tuple[Any, ...],
    *,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Lower/compile a JAX callable and return static plus runtime metrics."""
    lower_started = time.perf_counter()
    lowered = jitted_fn.lower(*args)
    lowering_seconds = time.perf_counter() - lower_started

    compile_started = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_started

    try:
        cost = _normalize_cost_analysis(compiled.cost_analysis())
    except (AttributeError, RuntimeError, NotImplementedError) as exc:
        cost = {"unavailable": f"{type(exc).__name__}: {exc}"}
    compiler_memory = _compiler_memory_analysis(compiled)
    flops_per_call = _cost_metric_total(cost, "flops")
    bytes_per_call = _cost_metric_total(cost, "bytes accessed")
    transcendentals_per_call = _cost_metric_total(cost, "transcendentals")

    for _ in range(warmup_iterations):
        output = compiled(*args)
        _block_until_ready(output)

    durations_seconds: list[float] = []
    for _ in range(measured_iterations):
        started = time.perf_counter()
        output = compiled(*args)
        _block_until_ready(output)
        durations_seconds.append(time.perf_counter() - started)

    durations_ms = [duration * 1000.0 for duration in durations_seconds]
    elapsed = sum(durations_seconds)
    return {
        "batch_size": batch_size,
        "lowering_seconds": lowering_seconds,
        "compile_seconds_observed": compile_seconds,
        "lowering_plus_compile_seconds_observed": lowering_seconds + compile_seconds,
        "compile_time_caveat": (
            "Observed wall time for lowered.compile(); a configured persistent "
            "compilation cache can turn this into a cache-hit measurement."
        ),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "timing_scope": "Python dispatch + device execution + explicit synchronization per call",
        "latency_ms": {
            "median": statistics.median(durations_ms),
            "p95": _percentile(durations_ms, 95.0),
            "mean": statistics.fmean(durations_ms),
            "min": min(durations_ms),
            "max": max(durations_ms),
            "per_iteration": durations_ms,
        },
        "throughput_samples_per_second": (
            batch_size * measured_iterations / elapsed if elapsed > 0.0 else None
        ),
        "xla_cost_analysis": {
            "scope": (
                "Backend/XLA static estimate for one generator call at this batch "
                "shape; FLOPs are not a hardware-counter measurement."
            ),
            "flops_per_call": flops_per_call,
            "flops_per_sample": (
                flops_per_call / batch_size if flops_per_call is not None else None
            ),
            "bytes_accessed_per_call": bytes_per_call,
            "transcendentals_per_call": transcendentals_per_call,
            "raw": cost,
        },
        "xla_compiler_memory_analysis": compiler_memory,
    }


def _device_memory_stats() -> list[dict[str, Any]]:
    snapshots: list[dict[str, Any]] = []
    for device in jax.local_devices():
        record: dict[str, Any] = {
            "device_id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        try:
            raw = device.memory_stats()
        except (AttributeError, RuntimeError, NotImplementedError) as exc:
            record.update(
                {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
            )
        else:
            if raw is None:
                record.update({"available": False, "reason": "backend returned None"})
            else:
                record.update(
                    {
                        "available": True,
                        "raw": {
                            str(key): _to_json_number(value)
                            for key, value in raw.items()
                        },
                    }
                )
        snapshots.append(record)
    return snapshots


class NvidiaSmiProcessSampler:
    """Periodically sample current-process GPU memory with ``nvidia-smi``."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = float(interval_seconds)
        self.pid = os.getpid()
        self.samples_mib: list[float] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.interval_seconds <= 0.0:
            return
        if shutil.which("nvidia-smi") is None:
            self._record_error("nvidia-smi not found")
            return
        self._thread = threading.Thread(target=self._run, name="nvidia-smi-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 3.0))
        return {
            "available": bool(self.samples_mib),
            "scope": (
                "Maximum summed nvidia-smi compute-process memory across visible GPUs "
                "for this PID during the benchmark process."
            ),
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": len(self.samples_mib),
            "peak_used_memory_mib": max(self.samples_mib) if self.samples_mib else None,
            "first_used_memory_mib": self.samples_mib[0] if self.samples_mib else None,
            "last_used_memory_mib": self.samples_mib[-1] if self.samples_mib else None,
            "caveat": (
                "Polling measurement; it may miss transients shorter than the "
                "sampling interval and includes model parameters/compiler/runtime allocations."
            ),
            "errors": self.errors[:10],
        }

    def _sample_once(self) -> None:
        command = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(2.0, self.interval_seconds * 2.0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            self._record_error(f"{type(exc).__name__}: {exc}")
            return
        if completed.returncode != 0:
            error = completed.stderr.strip() or f"exit={completed.returncode}"
            self._record_error(error)
            return
        current_process_total_mib = 0.0
        found_current_process = False
        for line in completed.stdout.splitlines():
            pieces = [piece.strip() for piece in line.split(",")]
            if len(pieces) < 2:
                continue
            try:
                pid = int(pieces[0])
                used_mib = float(pieces[1])
            except ValueError:
                continue
            if pid == self.pid:
                current_process_total_mib += used_mib
                found_current_process = True
        if found_current_process:
            self.samples_mib.append(current_process_total_mib)

    def _record_error(self, message: str) -> None:
        if len(self.errors) < 10:
            self.errors.append(message)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)
        self._sample_once()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _run_command(command: Sequence[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def environment_manifest(project_root: Path) -> dict[str, Any]:
    devices = [
        {
            "id": int(device.id),
            "process_index": int(device.process_index),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]
    status = _run_command(["git", "status", "--short"], cwd=project_root)
    nvidia_query = "index,name,uuid,driver_version,memory.total"
    return {
        "timestamp_utc": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: _package_version(name)
            for name in (
                "jax",
                "jaxlib",
                "jax-cuda12-plugin",
                "jax-cuda12-pjrt",
                "flax",
                "numpy",
                "ml-dtypes",
            )
        },
        "jax_backend": jax.default_backend(),
        "jax_process_index": jax.process_index(),
        "jax_process_count": jax.process_count(),
        "devices": devices,
        "nvidia_smi": {
            "query": nvidia_query,
            "csv_noheader_nounits": _run_command(
                [
                    "nvidia-smi",
                    f"--query-gpu={nvidia_query}",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
        "invocation": {
            "argv": sys.argv,
            "working_directory": str(Path.cwd().resolve()),
        },
        "selected_environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "JAX_PLATFORMS",
                "JAX_ENABLE_COMPILATION_CACHE",
                "JAX_COMPILATION_CACHE_DIR",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "LSB_JOBID",
                "LSB_JOB_HOSTS",
                "LSB_QUEUE",
            )
        },
        "git": {
            "head": _run_command(["git", "rev-parse", "HEAD"], cwd=project_root),
            "status_short": status,
            "status_short_sha256": (
                hashlib.sha256(status.encode("utf-8")).hexdigest()
                if status is not None
                else None
            ),
        },
    }


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    digest, _ = _sha256_file_stable(path)
    sidecar = path.with_name(f"{path.name}.sha256")
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="ascii")


def implementation_manifest(project_root: Path) -> dict[str, Any]:
    """Hash every local source file that directly defines this benchmark path."""
    paths = [
        Path(__file__).resolve(),
        project_root / "models" / "generator.py",
        project_root / "utils" / "init_util.py",
        project_root / "utils" / "hsdp_util.py",
        project_root / "requirements-a800.txt",
    ]
    wrapper_value = os.environ.get("EFFICIENCY_BENCHMARK_WRAPPER")
    if wrapper_value:
        paths.append(Path(wrapper_value).resolve())

    records: list[dict[str, Any]] = []
    for path in paths:
        if path.is_file():
            digest, file_bytes = _sha256_file_stable(path)
            records.append(
                {"path": str(path), "bytes": file_bytes, "sha256": digest}
            )
    aggregate = hashlib.sha256()
    for record in sorted(records, key=lambda item: item["path"]):
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "hash_scope": "path and SHA-256 of direct benchmark/model/load/sharding sources",
        "files": records,
    }


def _validate_iterations(args: argparse.Namespace) -> None:
    for name in ("warmup_iterations", "latency_iterations", "throughput_iterations"):
        value = int(getattr(args, name))
        if value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 1, got {value}")
    if args.latency_batch_size < 1 or args.throughput_batch_size < 1:
        raise ValueError("Batch sizes must be positive.")
    if args.num_classes < 1:
        raise ValueError("--num-classes must be positive.")
    if args.nvidia_smi_interval < 0:
        raise ValueError("--nvidia-smi-interval must be >= 0.")


def _make_generator_forward(model: Any, cfg_scale: float) -> Any:
    def forward(params: Any, labels: jax.Array, rng: jax.Array) -> jax.Array:
        return model.apply(
            {"params": params},
            train=False,
            rngs={"noise": rng},
            c=labels,
            cfg_scale=cfg_scale,
        )["samples"]

    return jax.jit(forward)


def _benchmark_shape(
    model: Any,
    params: Any,
    *,
    cfg_scale: float,
    batch_size: int,
    num_classes: int,
    seed: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    labels = jnp.arange(batch_size, dtype=jnp.int32) % num_classes
    rng = jax.random.PRNGKey(seed)
    jitted = _make_generator_forward(model, cfg_scale)
    return compile_and_measure(
        jitted,
        (params, labels, rng),
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )


def run_formal_benchmark(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    if not args.init_from:
        raise ValueError("--init-from is required unless --self-test is used.")
    if not args.method:
        raise ValueError("--method is required unless --self-test is used.")
    if args.init_from.startswith("hf://"):
        raise ValueError(
            "Formal efficiency runs require a local content-hashable checkpoint; "
            "materialize the HF artifact first."
        )

    checkpoint_root = resolve_artifact_dir(args.init_from)
    checkpoint_integrity = hash_artifact(checkpoint_root)
    metadata = _read_json_if_present(checkpoint_root / "metadata.json")
    if args.expected_step >= 0:
        if metadata is None or "step" not in metadata:
            raise ValueError(
                f"Cannot verify expected step {args.expected_step}: "
                f"{checkpoint_root / 'metadata.json'} has no step."
            )
        actual_step = int(metadata["step"])
        if actual_step != args.expected_step:
            raise ValueError(
                f"Checkpoint is not the required final step: "
                f"expected {args.expected_step}, got {actual_step}."
            )

    set_global_mesh(args.hsdp_dim)
    model, params, loaded_metadata = load_generator_model_and_params(args.init_from)
    model_num_classes = int(getattr(model, "num_classes", args.num_classes))
    if args.num_classes > model_num_classes:
        raise ValueError(
            f"--num-classes={args.num_classes} exceeds model.num_classes={model_num_classes}"
        )

    params_stats = parameter_statistics(params)
    memory_after_load = _device_memory_stats()
    latency = _benchmark_shape(
        model,
        params,
        cfg_scale=args.cfg_scale,
        batch_size=args.latency_batch_size,
        num_classes=args.num_classes,
        seed=args.seed,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.latency_iterations,
    )
    memory_after_latency = _device_memory_stats()
    throughput = _benchmark_shape(
        model,
        params,
        cfg_scale=args.cfg_scale,
        batch_size=args.throughput_batch_size,
        num_classes=args.num_classes,
        seed=args.seed + 1,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.throughput_iterations,
    )
    memory_after_throughput = _device_memory_stats()

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "measurement_scope": MEASUREMENT_SCOPE,
        "method": args.method,
        "input": {
            "init_from": str(Path(args.init_from).resolve()),
            "cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "num_classes": args.num_classes,
            "class_label_rule": "arange(batch_size) modulo num_classes",
            "latency_batch_size": args.latency_batch_size,
            "throughput_batch_size": args.throughput_batch_size,
            "expected_checkpoint_step": args.expected_step,
        },
        "checkpoint_integrity": checkpoint_integrity,
        "checkpoint_metadata": metadata,
        "loaded_metadata": loaded_metadata,
        "parameters": params_stats,
        "nfe": {
            "generator_model_forward_evaluations_per_sample": 1,
            "definition": NFE_DEFINITION,
        },
        "generator_inference": {"latency_protocol": latency, "throughput_protocol": throughput},
        "memory": {
            "jax_allocator_scope": (
                "Backend allocator statistics for the current process lifetime; "
                "peak fields, when present, include load, compilation, warmup, and measurement."
            ),
            "after_checkpoint_load": memory_after_load,
            "after_batch_1": memory_after_latency,
            "after_batch_128": memory_after_throughput,
        },
        "known_limitations": [
            "Training-step time, training samples/s, 30k wall time, and GPU-hours are not measured here.",
            "Latent VAE decode is excluded; measure it separately if end-to-end image latency is claimed.",
            "XLA FLOPs/cost_analysis and compiler memory are static estimates, not hardware counters.",
        ],
        "environment": environment_manifest(project_root),
        "benchmark_implementation": implementation_manifest(project_root),
    }


def run_self_test(args: argparse.Namespace, project_root: Path) -> dict[str, Any]:
    """Exercise the measurement path with a small CPU/GPU-independent function."""
    set_global_mesh(1)

    def tiny_forward(weights: jax.Array, inputs: jax.Array) -> jax.Array:
        return jnp.tanh(inputs @ weights)

    jitted = jax.jit(tiny_forward)
    weights = jnp.arange(16, dtype=jnp.float32).reshape(4, 4) / 16.0
    inputs = jnp.ones((args.latency_batch_size, 4), dtype=jnp.float32)
    result = compile_and_measure(
        jitted,
        (weights, inputs),
        batch_size=args.latency_batch_size,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.latency_iterations,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "self-test-complete",
        "measurement_scope": "tiny synthetic JAX matmul/tanh; not a scientific result",
        "result": result,
        "parameter_helper_check": parameter_statistics({"weight": weights}),
        "memory": {"jax_allocator": _device_memory_stats()},
        "environment": environment_manifest(project_root),
        "benchmark_implementation": implementation_manifest(project_root),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="PR-DCKD-V1 generator-only efficiency benchmark."
    )
    parser.add_argument(
        "--init-from",
        default="",
        help="Local completed training workdir or params_ema artifact.",
    )
    parser.add_argument("--method", default="", help="Evidence label, e.g. baseline or v4.3.")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--latency-batch-size", type=int, default=1)
    parser.add_argument("--throughput-batch-size", type=int, default=128)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--latency-iterations", type=int, default=30)
    parser.add_argument("--throughput-iterations", type=int, default=20)
    parser.add_argument(
        "--expected-step",
        type=int,
        default=30000,
        help="Required metadata step; use -1 only for non-formal diagnostics.",
    )
    parser.add_argument("--hsdp-dim", type=int, default=8)
    parser.add_argument(
        "--nvidia-smi-interval",
        type=float,
        default=0.1,
        help="Seconds between current-process memory samples; 0 disables polling.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a tiny synthetic compile/timing test without a checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _validate_iterations(args)
    project_root = REPO_ROOT
    sampler = NvidiaSmiProcessSampler(args.nvidia_smi_interval)
    sampler.start()
    started = time.perf_counter()
    payload: dict[str, Any]
    try:
        if args.self_test:
            payload = run_self_test(args, project_root)
        else:
            payload = run_formal_benchmark(args, project_root)
    finally:
        nvidia_memory = sampler.stop()
    payload["memory"]["nvidia_smi_process_sampling"] = nvidia_memory
    payload["total_script_wall_seconds"] = time.perf_counter() - started
    payload["completed_at_utc"] = _utc_now()
    output = Path(args.json_out).resolve()
    _atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"[OK] wrote {output}", flush=True)
    print(f"[OK] wrote {output}.sha256", flush=True)


if __name__ == "__main__":
    main()
