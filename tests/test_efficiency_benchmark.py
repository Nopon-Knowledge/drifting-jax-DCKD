from __future__ import annotations

import hashlib
from pathlib import Path

import jax.numpy as jnp
import pytest

from scripts.benchmark_generator_efficiency import (
    _percentile,
    hash_artifact,
    parameter_statistics,
)


def test_percentile_uses_linear_interpolation() -> None:
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)
    assert _percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)


def test_parameter_statistics_counts_shapes_and_dtypes() -> None:
    params = {
        "a": jnp.zeros((2, 3), dtype=jnp.float32),
        "b": jnp.zeros((4,), dtype=jnp.float16),
    }
    stats = parameter_statistics(params)
    assert stats["total_parameters"] == 10
    assert stats["trainable_parameters"] == 10
    assert stats["parameter_bytes_from_dtype"] == 6 * 4 + 4 * 2
    assert stats["by_dtype"]["float32"]["parameters"] == 6
    assert stats["by_dtype"]["float16"]["parameters"] == 4


def test_hash_artifact_is_content_and_path_addressed(tmp_path: Path) -> None:
    artifact = tmp_path / "params_ema"
    artifact.mkdir()
    (artifact / "metadata.json").write_text('{"step": 30000}\\n', encoding="utf-8")
    (artifact / "ema_params.msgpack").write_bytes(b"checkpoint")

    manifest = hash_artifact(artifact)
    records = {record["path"]: record for record in manifest["files"]}
    assert manifest["file_count"] == 2
    assert records["ema_params.msgpack"]["sha256"] == hashlib.sha256(
        b"checkpoint"
    ).hexdigest()

    original_aggregate = manifest["aggregate_sha256"]
    (artifact / "metadata.json").write_text('{"step": 29999}\\n', encoding="utf-8")
    assert hash_artifact(artifact)["aggregate_sha256"] != original_aggregate
