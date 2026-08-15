import jax
import jax.numpy as jnp
import numpy as np

from drift_loss import drift_loss


def _inputs():
    gen = jnp.linspace(-1.0, 1.0, 2 * 4 * 6, dtype=jnp.float32).reshape(2, 4, 6)
    fixed_pos = jnp.flip(gen, axis=1) + 0.1
    fixed_neg = jnp.concatenate([gen[:, :2] - 0.4, gen[:, 2:] + 0.4], axis=1)
    return gen, fixed_pos, fixed_neg


def test_fixed_kernel_path_is_unchanged_when_adaptive_is_disabled():
    gen, fixed_pos, fixed_neg = _inputs()
    expected, expected_info = drift_loss(gen, fixed_pos, fixed_neg)
    actual, actual_info = drift_loss(
        gen,
        fixed_pos,
        fixed_neg,
        adaptive_kernel=False,
    )

    np.testing.assert_allclose(actual, expected)
    assert actual_info.keys() == expected_info.keys()


def test_adaptive_kernel_has_finite_loss_metrics_and_gradients():
    gen, fixed_pos, fixed_neg = _inputs()
    kwargs = dict(
        adaptive_kernel=True,
        adaptive_radius_mode="local",
        adaptive_rank=2,
        adaptive_reference_weight=0.1,
        adaptive_multipliers=(0.5, 1.0, 2.0),
        adaptive_min_radius=0.01,
        adaptive_max_radius=0.5,
    )

    loss, info = drift_loss(gen, fixed_pos, fixed_neg, **kwargs)
    grads = jax.grad(lambda x: drift_loss(x, fixed_pos, fixed_neg, **kwargs)[0].sum())(gen)

    assert jnp.isfinite(loss).all()
    assert jnp.isfinite(grads).all()
    assert 0.01 <= float(info["adaptive_radius_base"]) <= 0.5
    assert float(info["adaptive_neighbor_distance"]) > 0.0
    assert float(info["adaptive_local_radius_std"]) >= 0.0
    assert float(info["adaptive_radius_std"]) >= 0.0
    for multiplier in (0.5, 1.0, 2.0):
        suffix = f"adaptive_{multiplier:g}"
        assert jnp.isfinite(info[f"loss_{suffix}"])
        assert 0.0 <= float(info[f"kernel_entropy_{suffix}"]) <= 1.0
        assert float(info[f"effective_neighbors_{suffix}"]) >= 1.0


def test_global_radius_control_removes_per_sample_radius_variation():
    gen, fixed_pos, fixed_neg = _inputs()
    common = dict(
        adaptive_kernel=True,
        adaptive_rank=2,
        adaptive_reference_weight=0.1,
        adaptive_multipliers=(1.0,),
        adaptive_min_radius=0.01,
        adaptive_max_radius=0.5,
    )

    _, local_info = drift_loss(
        gen,
        fixed_pos,
        fixed_neg,
        adaptive_radius_mode="local",
        **common,
    )
    _, global_info = drift_loss(
        gen,
        fixed_pos,
        fixed_neg,
        adaptive_radius_mode="global",
        **common,
    )

    np.testing.assert_allclose(
        global_info["adaptive_radius_base"],
        local_info["adaptive_radius_base"],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(global_info["adaptive_radius_std"], 0.0, atol=1e-7)
    assert float(local_info["adaptive_local_radius_std"]) > 0.0


def test_adaptive_diagnostics_do_not_change_loss_and_report_clamps():
    gen, fixed_pos, fixed_neg = _inputs()
    kwargs = dict(
        adaptive_kernel=True,
        adaptive_radius_mode="global",
        adaptive_rank=2,
        adaptive_reference_weight=0.1,
        adaptive_multipliers=(0.5, 1.0, 2.0),
        adaptive_min_radius=0.01,
        adaptive_max_radius=0.05,
    )

    loss_without, info_without = drift_loss(
        gen,
        fixed_pos,
        fixed_neg,
        adaptive_diagnostics=False,
        **kwargs,
    )
    loss_with, info_with = drift_loss(
        gen,
        fixed_pos,
        fixed_neg,
        adaptive_diagnostics=True,
        **kwargs,
    )

    np.testing.assert_allclose(loss_with, loss_without, rtol=0.0, atol=0.0)
    assert "adaptive_unclipped_radius_q50" not in info_without
    for name in (
        "adaptive_neighbor_distance_q05",
        "adaptive_neighbor_distance_q95",
        "adaptive_unclipped_radius_q50",
        "adaptive_clipped_base_radius_q50",
        "adaptive_base_clamp_low_frac",
        "adaptive_base_clamp_high_frac",
        "adaptive_2_clamp_high_frac",
        "adaptive_2_radius_q95",
    ):
        assert name in info_with
        assert jnp.isfinite(info_with[name])

    for name in (
        "adaptive_base_clamp_low_frac",
        "adaptive_base_clamp_high_frac",
        "adaptive_0.5_clamp_low_frac",
        "adaptive_0.5_clamp_high_frac",
        "adaptive_1_clamp_low_frac",
        "adaptive_1_clamp_high_frac",
        "adaptive_2_clamp_low_frac",
        "adaptive_2_clamp_high_frac",
    ):
        assert 0.0 <= float(info_with[name]) <= 1.0
