import hashlib
import json

import numpy as np

from utils.fid_util import (
    load_inception_feature_reference,
    save_inception_artifacts,
)


def _file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inception_artifact_round_trip_and_manifest_hash(tmp_path):
    output = tmp_path / "features.npz"
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    logits = np.arange(18, dtype=np.float32).reshape(6, 3)
    labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    relative_paths = [f"n{i:08d}/sample.JPEG" for i in range(6)]

    manifest = save_inception_artifacts(
        output,
        features=features,
        logits=logits,
        labels=labels,
        relative_paths=relative_paths,
        metadata={"generation_seed": 20260728},
    )
    loaded = load_inception_feature_reference(output)

    np.testing.assert_array_equal(loaded["features"], features)
    np.testing.assert_array_equal(loaded["logits"], logits)
    np.testing.assert_array_equal(loaded["labels"], labels)
    np.testing.assert_array_equal(
        loaded["relative_paths"], np.asarray(relative_paths)
    )
    embedded = json.loads(loaded["manifest_json"].item())
    assert embedded["metadata"]["generation_seed"] == 20260728

    manifest_path = output.with_suffix(".npz.manifest.json")
    sidecar = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["archive"]["sha256"] == _file_sha256(output)
    assert sidecar["archive"]["sha256"] == _file_sha256(output)
    assert sidecar["arrays"]["features"]["shape"] == [6, 4]


def test_inception_artifact_rejects_misaligned_labels(tmp_path):
    with np.testing.assert_raises_regex(ValueError, "first dimension"):
        save_inception_artifacts(
            tmp_path / "bad.npz",
            features=np.ones((3, 2), dtype=np.float32),
            labels=np.ones((2,), dtype=np.int64),
        )
