import json
from pathlib import Path

import jax
import numpy as np
import pytest

import inference


def _fake_generation(captured):
    def generate_samples(**kwargs):
        captured["rng_eval"] = np.asarray(kwargs["rng_eval"])
        assert kwargs["return_metadata"] is True
        samples = np.arange(48, dtype=np.uint8).reshape(4, 2, 2, 3)
        masks = np.ones(4, dtype=np.int32)
        metadata = {
            "labels": np.asarray([4, 3, 2, 1], dtype=np.int64),
            "rng_batch_indices": np.asarray([0, 0, 1, 1], dtype=np.int64),
            "rng_positions": np.asarray([0, 1, 0, 1], dtype=np.int64),
            "process_index": 0,
            "process_count": 1,
            "rng_scheme": "test",
        }
        return samples, masks, 1.25, metadata

    return generate_samples


def test_generation_seed_and_evidence_manifest_are_bound(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(inference, "_create_eval_loader", lambda _: object())
    monkeypatch.setattr(
        inference, "generate_samples", _fake_generation(captured)
    )

    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params_ema").mkdir(parents=True)
    (checkpoint / "params_ema" / "ema_params.msgpack").write_bytes(b"ema")
    config = tmp_path / "config.yaml"
    config.write_text("train:\n  seed: 123\n", encoding="utf-8")
    protocol = tmp_path / "protocol.yaml"
    protocol.write_text("protocol_id: TEST\n", encoding="utf-8")
    workdir = tmp_path / "eval"
    samples_dir = workdir / "generated"

    result = inference.run_generate_only(
        None,
        None,
        None,
        {"model_config": {"in_channels": 4}},
        str(checkpoint),
        workdir=str(workdir),
        samples_dir=str(samples_dir),
        num_samples=4,
        cfg_scale=2.5,
        eval_batch_size=2,
        vae_decode_batch_size=1,
        generation_seed=271828,
        train_seed="123",
        config_path=str(config),
        protocol_id="TEST",
        protocol_path=str(protocol),
        source_snapshot_id="snapshot-abc",
        allow_overwrite=False,
    )

    np.testing.assert_array_equal(
        captured["rng_eval"], np.asarray(jax.random.PRNGKey(271828))
    )
    manifest = json.loads(
        (samples_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["generation_seed"] == 271828
    assert manifest["train_seed"] == "123"
    assert manifest["valid_label_schedule"]["count"] == 4
    assert manifest["rng"]["unique_valid_row_tuples"] == 4
    assert manifest["checkpoint_artifacts"][0]["sha256"]
    assert manifest["config"]["sha256"]
    assert manifest["protocol"]["sha256"]
    assert result["generation_manifest_sha256"]

    for role, record in manifest["arrays"].items():
        inference._verify_record(record, role=role)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        inference.run_generate_only(
            None,
            None,
            None,
            {},
            str(checkpoint),
            workdir=str(workdir),
            samples_dir=str(samples_dir),
            num_samples=4,
            cfg_scale=2.5,
            eval_batch_size=2,
            vae_decode_batch_size=1,
            generation_seed=271828,
            train_seed="123",
            config_path=str(config),
            protocol_id="TEST",
            protocol_path=str(protocol),
            source_snapshot_id="snapshot-abc",
            allow_overwrite=False,
        )


def test_metrics_only_rejects_generation_seed_mismatch_before_scoring(
    monkeypatch, tmp_path
):
    captured = {}
    monkeypatch.setattr(inference, "_create_eval_loader", lambda _: object())
    monkeypatch.setattr(
        inference, "generate_samples", _fake_generation(captured)
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "ema_params.msgpack").write_bytes(b"ema")
    workdir = tmp_path / "eval"
    samples_dir = workdir / "generated"
    inference.run_generate_only(
        None,
        None,
        None,
        {},
        str(checkpoint),
        workdir=str(workdir),
        samples_dir=str(samples_dir),
        num_samples=4,
        cfg_scale=2.5,
        eval_batch_size=2,
        vae_decode_batch_size=1,
        generation_seed=104729,
        train_seed="42",
        config_path="",
        protocol_id="TEST",
        protocol_path="",
        source_snapshot_id="snapshot-abc",
        allow_overwrite=False,
    )

    with pytest.raises(ValueError, match="Generation-seed mismatch"):
        inference.run_metrics_only(
            str(checkpoint),
            str(workdir),
            samples_dir=str(samples_dir),
            num_samples=4,
            cfg_scale=2.5,
            eval_prc_recall=False,
            eval_prdc=False,
            prdc_nearest_k=5,
            prdc_reference_path="",
            prdc_row_batch_size=2,
            prdc_col_batch_size=2,
            feature_artifact_path=str(workdir / "features.npz"),
            generation_seed=130363,
            train_seed="42",
            eval_batch_size=2,
            vae_decode_batch_size=1,
            source_snapshot_id="snapshot-abc",
            allow_overwrite=False,
            use_wandb=False,
            wandb_entity=None,
            wandb_project="test",
            wandb_name=None,
        )


def test_generation_seed_parser_accepts_frozen_stream():
    args = inference.build_parser().parse_args(
        [
            "--init-from",
            "/tmp/checkpoint",
            "--generation-seed",
            "271828",
            "--generate-only",
        ]
    )
    assert args.generation_seed == 271828
