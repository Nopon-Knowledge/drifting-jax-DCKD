import jax
import jax.numpy as jnp
from functools import partial
from einops import repeat

def cdist(x, y, eps=1e-8):
    # [B, N, D] x [B, M, D] -> [B, N, M]
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))

@partial(
    jax.jit,
    static_argnames=(
        "R_list",
        "adaptive_kernel",
        "adaptive_radius_mode",
        "adaptive_rank",
        "adaptive_reference_weight",
        "adaptive_multipliers",
        "adaptive_min_radius",
        "adaptive_max_radius",
        "adaptive_diagnostics",
    ),
)
def drift_loss(
    gen,
    fixed_pos,
    fixed_neg=None,
    weight_gen=None,
    weight_pos=None,
    weight_neg=None,
    R_list=(0.02, 0.05, 0.2),
    adaptive_kernel=False,
    adaptive_radius_mode="local",
    adaptive_rank=4,
    adaptive_reference_weight=0.1,
    adaptive_multipliers=(0.5, 1.0, 2.0),
    adaptive_min_radius=0.01,
    adaptive_max_radius=0.5,
    adaptive_diagnostics=False,
):
    '''
    Args:
        gen: [B, C_g, S]
        fixed_pos: [B, C_p, S]
        fixed_neg: [B, C_n, S] (optional, can be None)
        weight_gen: [B, C_g] (optional; if None: weight is 1)
        weight_pos: [B, C_p] (optional; if None: weight is 1)
        weight_neg: [B, C_n] (optional; if None: weight is 1)
        R_list: a list of R values to use for the kernel function
        adaptive_kernel: calibrate the kernel radius from local feature density.
        adaptive_radius_mode: use per-sample ("local") or batch-mean ("global") radius.
        adaptive_rank: neighbor rank used to estimate local feature spacing.
        adaptive_reference_weight: desired unnormalized affinity at that rank.
        adaptive_multipliers: scales applied around the calibrated base radius.
        adaptive_min_radius: minimum calibrated radius.
        adaptive_max_radius: maximum calibrated radius.
        adaptive_diagnostics: record distribution and clamp diagnostics.
    Returns:
        loss: [batch_size]
        (optional) info: a dict with entries:
            scale: the scale of the loss 
            loss_R: the loss for each R value
    '''
    
    # 1. Defaults & Casting
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]
    
    if fixed_neg is None:
        fixed_neg = jnp.zeros_like(gen[:, :0, :])
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = jnp.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = jnp.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = jnp.ones_like(fixed_neg[:, :, 0])
    gen = gen.astype(jnp.float32)
    fixed_pos = fixed_pos.astype(jnp.float32)
    fixed_neg = fixed_neg.astype(jnp.float32)
    weight_gen = weight_gen.astype(jnp.float32)
    weight_pos = weight_pos.astype(jnp.float32)
    weight_neg = weight_neg.astype(jnp.float32)
    old_gen = jax.lax.stop_gradient(gen)
    targets = jnp.concatenate([old_gen, fixed_neg, fixed_pos], axis=1)
    targets_w = jnp.concatenate([weight_gen, weight_neg, weight_pos], axis=1)

    # 2. Core Logic (Wrapped for stop_gradient)
    def calculate_scaled_goal_and_factor(old_gen_in, targets_in, targets_w_in):
        # --- Scaling ---
        info = {}
        dist = cdist(old_gen_in, targets_in)
        weighted_dist = dist * targets_w_in[:, None, :] # [B, C_g, C_g + C_n + C_p]
        scale = weighted_dist.mean() / targets_w_in.mean() # [B]
        info["scale"] = scale

        scale_inputs = jnp.clip(scale / jnp.sqrt(S), a_min=1e-3) # Normalize coords to have order 1
        old_gen_scaled = old_gen_in / scale_inputs
        targets_scaled = targets_in / scale_inputs
        
        # Normalize distance for kernel
        dist_normed = dist / jnp.clip(scale, a_min=1e-3)
        
        # --- Masking ---
        mask_val = 100.0
        diag_mask = jnp.eye(C_g, dtype=jnp.float32)
        block_mask = jnp.pad(diag_mask, ((0, 0), (0, C_n + C_p))) 
        block_mask = jnp.expand_dims(block_mask, 0)
        dist_normed = dist_normed + block_mask * mask_val

        if adaptive_kernel:
            total_targets = C_g + C_n + C_p
            neighbor_rank = max(1, min(int(adaptive_rank), total_targets - 1))
            nearest_values, _ = jax.lax.top_k(-dist_normed, neighbor_rank)
            neighbor_distance = -nearest_values[..., -1]
            neighbor_distance = jnp.mean(neighbor_distance, axis=-1)
            reference_log_weight = -jnp.log(
                jnp.clip(
                    jnp.asarray(adaptive_reference_weight, dtype=jnp.float32),
                    a_min=1e-4,
                    a_max=1.0 - 1e-4,
                )
            )
            unclipped_local_radius_base = neighbor_distance / reference_log_weight
            local_radius_base = jnp.clip(
                unclipped_local_radius_base,
                a_min=adaptive_min_radius,
                a_max=adaptive_max_radius,
            )
            if adaptive_radius_mode == "local":
                radius_base = local_radius_base
            elif adaptive_radius_mode == "global":
                radius_base = jnp.full_like(local_radius_base, local_radius_base.mean())
            else:
                raise ValueError(
                    "adaptive_radius_mode must be 'local' or 'global', "
                    f"got {adaptive_radius_mode!r}"
                )
            radius_base = jax.lax.stop_gradient(radius_base)
            info["adaptive_neighbor_distance"] = neighbor_distance.mean()
            info["adaptive_local_radius_std"] = local_radius_base.std()
            info["adaptive_radius_base"] = radius_base.mean()
            info["adaptive_radius_std"] = radius_base.std()
            if adaptive_diagnostics:
                quantiles = (
                    ("q05", 0.05),
                    ("q25", 0.25),
                    ("q50", 0.50),
                    ("q75", 0.75),
                    ("q95", 0.95),
                )
                info["adaptive_neighbor_distance_std"] = neighbor_distance.std()
                info["adaptive_neighbor_distance_min"] = neighbor_distance.min()
                info["adaptive_neighbor_distance_max"] = neighbor_distance.max()
                info["adaptive_unclipped_radius_mean"] = unclipped_local_radius_base.mean()
                info["adaptive_unclipped_radius_std"] = unclipped_local_radius_base.std()
                info["adaptive_unclipped_radius_min"] = unclipped_local_radius_base.min()
                info["adaptive_unclipped_radius_max"] = unclipped_local_radius_base.max()
                info["adaptive_base_clamp_low_frac"] = jnp.mean(
                    unclipped_local_radius_base <= adaptive_min_radius
                )
                info["adaptive_base_clamp_high_frac"] = jnp.mean(
                    unclipped_local_radius_base >= adaptive_max_radius
                )
                for quantile_name, quantile_value in quantiles:
                    info[f"adaptive_neighbor_distance_{quantile_name}"] = jnp.quantile(
                        neighbor_distance, quantile_value
                    )
                    info[f"adaptive_unclipped_radius_{quantile_name}"] = jnp.quantile(
                        unclipped_local_radius_base, quantile_value
                    )
                    info[f"adaptive_clipped_base_radius_{quantile_name}"] = jnp.quantile(
                        radius_base, quantile_value
                    )
            kernel_scales = adaptive_multipliers
        else:
            radius_base = None
            kernel_scales = R_list

        # --- Force Loop ---
        force_across_R = jnp.zeros_like(old_gen_scaled)
        
        for kernel_scale in kernel_scales:
            if adaptive_kernel:
                unclipped_scaled_radius = radius_base * kernel_scale
                radius = jnp.clip(
                    unclipped_scaled_radius,
                    a_min=adaptive_min_radius,
                    a_max=adaptive_max_radius,
                )
                logits = -dist_normed / radius[:, None, None]
                metric_suffix = f"adaptive_{kernel_scale:g}"
                info[f"adaptive_radius_{kernel_scale:g}"] = radius.mean()
                if adaptive_diagnostics:
                    info[f"{metric_suffix}_clamp_low_frac"] = jnp.mean(
                        unclipped_scaled_radius <= adaptive_min_radius
                    )
                    info[f"{metric_suffix}_clamp_high_frac"] = jnp.mean(
                        unclipped_scaled_radius >= adaptive_max_radius
                    )
                    for quantile_name, quantile_value in (
                        ("q05", 0.05),
                        ("q25", 0.25),
                        ("q50", 0.50),
                        ("q75", 0.75),
                        ("q95", 0.95),
                    ):
                        info[f"{metric_suffix}_radius_{quantile_name}"] = jnp.quantile(
                            radius, quantile_value
                        )
            else:
                logits = -dist_normed / kernel_scale
                metric_suffix = f"{kernel_scale:g}"

            affinity = jax.nn.softmax(logits, axis=-1)
            aff_transpose = jax.nn.softmax(logits, axis=-2)
            affinity_entropy = -jnp.sum(
                affinity * jnp.log(jnp.clip(affinity, a_min=1e-8)),
                axis=-1,
            )
            info[f"kernel_entropy_{metric_suffix}"] = (
                affinity_entropy / jnp.log(jnp.asarray(targets.shape[1], dtype=jnp.float32))
            ).mean()
            info[f"effective_neighbors_{metric_suffix}"] = jnp.exp(affinity_entropy).mean()
            affinity = jnp.sqrt(jnp.clip(affinity * aff_transpose, a_min=1e-6))

            affinity = affinity * targets_w_in[:, None, :]

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]
            aff_pos = affinity[:, :, split_idx:]
            
            sum_pos = jnp.sum(aff_pos, axis=-1, keepdims=True)
            r_coeff_neg = -aff_neg * sum_pos 
            sum_neg = jnp.sum(aff_neg, axis=-1, keepdims=True)
            r_coeff_pos = aff_pos * sum_neg 
            
            R_coeff = jnp.concatenate([r_coeff_neg, r_coeff_pos], axis=2)

            total_force_R = jnp.einsum("biy,byx->bix", R_coeff, targets_scaled)

            total_coeffs = R_coeff.sum(axis=-1) # guaranteed to be 0, in no_repulsion case
            total_force_R = total_force_R - total_coeffs[..., None] * old_gen_scaled
            f_norm_val = (total_force_R ** 2).mean() # [B]

            info[f"loss_{metric_suffix}"] = f_norm_val

            force_scale = jnp.sqrt(jnp.clip(f_norm_val, a_min=1e-8)) # normalize force of each temperature
            force_across_R = force_across_R + total_force_R / force_scale

        goal_scaled = old_gen_scaled + force_across_R
        
        return goal_scaled, scale_inputs, info

    # 3. Compute Goal (No Gradients)
    goal_scaled, scale_inputs, info = jax.lax.stop_gradient(
        calculate_scaled_goal_and_factor(old_gen, targets, targets_w)
    )
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = jnp.mean(diff ** 2, axis=(-1, -2))
    info = jax.tree.map(lambda x: x.mean(), info)

    return loss, info
