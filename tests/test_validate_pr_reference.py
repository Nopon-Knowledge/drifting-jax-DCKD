import json

import numpy as np

from scripts import validate_pr_reference


def test_array_hash_binds_dtype_shape_and_values():
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert validate_pr_reference.sha256_array(values) == (
        validate_pr_reference.sha256_array(values.copy())
    )
    assert validate_pr_reference.sha256_array(values) != (
        validate_pr_reference.sha256_array(values.astype(np.float64))
    )
    assert validate_pr_reference.sha256_array(values) != (
        validate_pr_reference.sha256_array(values.reshape(2, 6))
    )


def test_load_fid_supports_both_key_conventions(tmp_path):
    mu = np.arange(3, dtype=np.float64)
    sigma = np.eye(3)
    modern = tmp_path / "modern.npz"
    legacy = tmp_path / "legacy.npz"
    np.savez(modern, ref_mu=mu, ref_sigma=sigma)
    np.savez(legacy, mu=mu, sigma=sigma)
    for path in (modern, legacy):
        loaded_mu, loaded_sigma = validate_pr_reference.load_fid(path)
        np.testing.assert_array_equal(loaded_mu, mu)
        np.testing.assert_array_equal(loaded_sigma, sigma)
        assert json.loads(json.dumps(validate_pr_reference.file_record(path)))[
            "sha256"
        ]
