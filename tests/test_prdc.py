import numpy as np

from utils.jax_fid.prdc import (
    compute_nearest_neighbour_distances,
    compute_prdc,
    pairwise_squared_distances,
)


def _dense_prdc(real, fake, nearest_k):
    real_distances = pairwise_squared_distances(real, real)
    fake_distances = pairwise_squared_distances(fake, fake)
    np.fill_diagonal(real_distances, np.inf)
    np.fill_diagonal(fake_distances, np.inf)
    real_radii = np.partition(
        real_distances, nearest_k - 1, axis=1
    )[:, nearest_k - 1]
    fake_radii = np.partition(
        fake_distances, nearest_k - 1, axis=1
    )[:, nearest_k - 1]
    cross = pairwise_squared_distances(real, fake)
    return {
        "precision": np.mean(np.any(cross <= real_radii[:, None], axis=0)),
        "recall": np.mean(np.any(cross <= fake_radii[None, :], axis=1)),
        "density": np.mean(
            np.sum(cross <= real_radii[:, None], axis=0) / nearest_k
        ),
        "coverage": np.mean(np.min(cross, axis=1) <= real_radii),
    }


def test_kth_neighbour_explicitly_excludes_identity_self_match():
    features = np.asarray([[0.0], [1.0], [3.0]], dtype=np.float64)
    radii = compute_nearest_neighbour_distances(
        features,
        nearest_k=1,
        row_batch_size=1,
        col_batch_size=2,
    )
    np.testing.assert_allclose(radii, [1.0, 1.0, 4.0])


def test_prdc_matches_hand_computed_artificial_example():
    real = np.asarray([[0.0], [10.0]], dtype=np.float64)
    fake = np.asarray([[0.0], [100.0]], dtype=np.float64)
    result = compute_prdc(
        real,
        fake,
        nearest_k=1,
        row_batch_size=1,
        col_batch_size=1,
    )
    assert result == {
        "precision": 0.5,
        "recall": 1.0,
        "density": 1.0,
        "coverage": 1.0,
        "nearest_k": 1,
    }


def test_identical_feature_sets_have_perfect_prdc():
    features = np.asarray(
        [[0.0, 0.0], [1.0, 0.5], [0.5, 2.0], [3.0, 1.0]],
        dtype=np.float32,
    )
    result = compute_prdc(
        features,
        features,
        nearest_k=2,
        row_batch_size=2,
        col_batch_size=3,
    )
    for metric in ("precision", "recall", "coverage"):
        np.testing.assert_allclose(result[metric], 1.0)
    # Canonical density is not bounded by one: overlapping real k-NN spheres
    # can cover one generated point more than k times.
    expected = _dense_prdc(features, features, nearest_k=2)
    np.testing.assert_allclose(result["density"], expected["density"])


def test_blockwise_prdc_matches_dense_and_is_batch_size_invariant():
    rng = np.random.default_rng(20260728)
    real = rng.normal(size=(13, 7)).astype(np.float32)
    fake = rng.normal(loc=0.2, scale=1.1, size=(11, 7)).astype(np.float32)
    expected = _dense_prdc(real, fake, nearest_k=3)

    for row_batch_size, col_batch_size in ((1, 1), (4, 3), (20, 20)):
        actual = compute_prdc(
            real,
            fake,
            nearest_k=3,
            row_batch_size=row_batch_size,
            col_batch_size=col_batch_size,
        )
        for metric, value in expected.items():
            np.testing.assert_allclose(
                actual[metric], value, rtol=0.0, atol=np.finfo(np.float64).eps
            )


def test_duplicate_nonself_vectors_remain_valid_zero_distance_neighbours():
    features = np.asarray([[0.0], [0.0], [2.0]], dtype=np.float64)
    radii = compute_nearest_neighbour_distances(
        features,
        nearest_k=1,
        row_batch_size=2,
        col_batch_size=2,
    )
    np.testing.assert_allclose(radii, [0.0, 0.0, 4.0])
