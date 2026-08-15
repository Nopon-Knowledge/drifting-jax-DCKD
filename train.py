import argparse
import gc
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
import numpy as np
import optax
from flax.training import train_state
from tqdm import tqdm
from einops import repeat, rearrange

from dataset.dataset import infinite_sampler, get_postprocess_fn
from drift_loss import drift_loss
from memory_bank import ArrayMemoryBank, RecencyGeneratedNegativeBank
from models.mae_model import build_activation_function
from utils.ckpt_util import save_checkpoint, restore_checkpoint, save_params_ema_artifact
from utils.env import HF_ROOT
from utils.fid_util import evaluate_fid
from utils.hsdp_util import (
    map_to_sharding, data_shard, merge_data, pad_and_merge,
    init_state_from_dummy_input, ddp_shard, set_global_mesh, enforce_ddp,
)
from utils.init_util import maybe_init_state_params
from utils.logging import log_for_0, is_rank_zero
from utils.misc import load_config, prepare_rng, profile_func, run_init
from utils.model_builder import build_model_dict
run_init()

class TrainState(train_state.TrainState):
    ema_params: Optional[Any] = None
    ema_decay: float = 0.999


def _generator_model_config(model) -> dict:
    return {
        name: value
        for name, value in vars(model).items()
        if name not in {"parent", "name"} and not name.startswith("_")
    }


def _linear_ramp(step, warmup_steps: int = 0, ramp_steps: int = 1):
    step_f = jnp.asarray(step + 1, dtype=jnp.float32)
    warmup_f = jnp.asarray(float(warmup_steps), dtype=jnp.float32)
    ramp_f = jnp.asarray(max(1, int(ramp_steps)), dtype=jnp.float32)
    return jnp.clip((step_f - warmup_f) / ramp_f, 0.0, 1.0)


def _pairwise_distance(x, y, eps=1e-8):
    xydot = jnp.einsum("bnd,bmd->bnm", x, y)
    xnorms = jnp.einsum("bnd,bnd->bn", x, x)
    ynorms = jnp.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms[:, :, None] + ynorms[:, None, :] - 2 * xydot
    return jnp.sqrt(jnp.clip(sq_dist, a_min=eps))


def _positive_coupling_weight(feature_pos, feature_gen, step, alpha=0.0, tau=0.5, warmup_steps=0, ramp_steps=1):
    """Soft-OT positive reweighting that favors real samples near current generated samples."""
    base_weight = jnp.ones_like(feature_pos[:, :, 0])
    alpha_eff = float(alpha) * _linear_ramp(step, warmup_steps, ramp_steps)
    distances = _pairwise_distance(
        jax.lax.stop_gradient(feature_gen.astype(jnp.float32)),
        jax.lax.stop_gradient(feature_pos.astype(jnp.float32)),
    )
    scores = -distances.mean(axis=1) / max(float(tau), 1e-6)
    soft_weight = jax.nn.softmax(scores, axis=-1) * feature_pos.shape[1]
    weight = (1.0 - alpha_eff) * base_weight + alpha_eff * soft_weight
    probs = weight / jnp.clip(weight.sum(axis=-1, keepdims=True), a_min=1e-6)
    entropy = -jnp.sum(probs * jnp.log(jnp.clip(probs, a_min=1e-6)), axis=-1)
    entropy = entropy / jnp.log(jnp.asarray(feature_pos.shape[1], dtype=jnp.float32))
    return weight, {
        "pos_coupling_alpha": alpha_eff,
        "pos_coupling_entropy": entropy.mean(),
        "pos_coupling_weight_std": weight.std(),
        "pos_coupling_distance": distances.mean(),
    }


def _radial_log_power(samples, radial_bins: int = 16, eps: float = 1e-6, normalize_total_power: bool = True):
    samples = samples.astype(jnp.float32)
    samples = samples - samples.mean(axis=(1, 2), keepdims=True)
    fft = jnp.fft.fft2(samples, axes=(1, 2))
    power = (jnp.real(fft) ** 2 + jnp.imag(fft) ** 2).mean(axis=-1)

    h, w = samples.shape[1], samples.shape[2]
    fy = jnp.fft.fftfreq(h)
    fx = jnp.fft.fftfreq(w)
    yy, xx = jnp.meshgrid(fy, fx, indexing="ij")
    radius = jnp.sqrt(yy ** 2 + xx ** 2)
    radius = radius / jnp.clip(radius.max(), a_min=eps)
    bin_idx = jnp.clip((radius * radial_bins).astype(jnp.int32), 0, radial_bins - 1)
    bin_one_hot = jax.nn.one_hot(bin_idx.reshape(-1), radial_bins, dtype=jnp.float32)

    power_flat = power.reshape((power.shape[0], -1))
    radial_power = power_flat @ bin_one_hot
    bin_count = jnp.clip(bin_one_hot.sum(axis=0), a_min=1.0)
    radial_power = radial_power / bin_count[None, :]
    if normalize_total_power:
        radial_power = radial_power / jnp.clip(radial_power.mean(axis=-1, keepdims=True), a_min=eps)
    return jnp.log(jnp.clip(radial_power, a_min=eps))


def _spectral_prior_loss(
    gen_samples,
    positive_samples,
    gen_per_label: int,
    radial_bins: int = 16,
    loss_type: str = "l2",
    normalize_total_power: bool = True,
):
    gen_power = _radial_log_power(
        gen_samples,
        radial_bins=radial_bins,
        normalize_total_power=normalize_total_power,
    )
    gen_power = rearrange(gen_power, "(b g) k -> b g k", g=gen_per_label).mean(axis=1)

    pos_flat = rearrange(positive_samples, "b p h w c -> (b p) h w c")
    pos_power = _radial_log_power(
        pos_flat,
        radial_bins=radial_bins,
        normalize_total_power=normalize_total_power,
    )
    pos_power = rearrange(pos_power, "(b p) k -> b p k", p=positive_samples.shape[1]).mean(axis=1)
    pos_power = jax.lax.stop_gradient(pos_power)

    diff = gen_power - pos_power
    if loss_type == "l1":
        loss = jnp.abs(diff).mean()
    else:
        loss = (diff ** 2).mean()
    return loss, {
        "spectral_prior_loss": loss,
        "spectral_prior_diff": jnp.abs(diff).mean(),
    }


def train_step(state: TrainState, labels, samples, negative_samples, negative_weights, feature_params, feature_apply, rng_init: jax.random.PRNGKey, learning_rate_fn: Any = None, cfg_min=1.0, cfg_max=4.0, neg_cfg_pw=1.0, no_cfg_frac=0.0, gen_per_label=8, activation_kwargs=dict(), loss_kwargs=dict(R_list=[0.02, 0.05, 0.2]), positive_coupling=dict(), spectral_prior=dict(), max_grad_norm=2.0):
    """Run one generator optimization step.

    Args:
        state: generator TrainState.
        labels: class labels with shape `(B,)`.
        samples: positive memory-bank samples with shape `(B, P, H, W, C)`.
        negative_samples: negative memory-bank samples with shape `(B, N, H, W, C)`.
        negative_weights: negative weights with shape `(B, N)`.
        feature_params: feature-model variable tree consumed by `feature_apply`.
        feature_apply: callable returning activation dict for input batch of shape `(B', H, W, C)`.
        rng_init: base PRNGKey for this train loop.
        learning_rate_fn: schedule mapping step -> scalar lr.
        cfg_min: lower bound for sampled CFG scale.
        cfg_max: upper bound for sampled CFG scale.
        neg_cfg_pw: power-law exponent for negative CFG sampling weights.
        no_cfg_frac: probability of replacing sampled CFG with `1.0`.
        gen_per_label: number of generator samples drawn per label, output shape `(B * gen_per_label, H, W, C)`.
        activation_kwargs: keyword args forwarded to feature activation extraction.
        loss_kwargs: keyword args forwarded to `drift_loss`.
        positive_coupling: soft-OT positive reweighting settings.
        spectral_prior: latent/image radial-spectrum prior settings.
        max_grad_norm: gradient clipping norm.
    """
    positive_coupling = positive_coupling or {}
    pos_coupling_enabled = bool(positive_coupling.get("enabled", False))
    pos_coupling_alpha = float(positive_coupling.get("alpha", 0.0))
    pos_coupling_tau = float(positive_coupling.get("tau", 0.5))
    pos_coupling_warmup = int(positive_coupling.get("warmup_steps", 0))
    pos_coupling_ramp = int(positive_coupling.get("ramp_steps", 1))

    spectral_prior = spectral_prior or {}
    spectral_prior_enabled = bool(spectral_prior.get("enabled", False))
    spectral_lambda = float(spectral_prior.get("lambda_spec", 0.0))
    spectral_warmup = int(spectral_prior.get("warmup_steps", 0))
    spectral_ramp = int(spectral_prior.get("ramp_steps", 1))
    spectral_bins = int(spectral_prior.get("radial_bins", 16))
    spectral_loss_type = str(spectral_prior.get("loss_type", "l2")).lower()
    spectral_normalize = bool(spectral_prior.get("normalize_total_power", True))

    rng_step = jax.random.fold_in(rng_init, state.step)


    # first: compute cfg
    cfg_seed, rng_step = jax.random.split(rng_step) # [B]
    cfg_seed1, cfg_seed2 = jax.random.split(cfg_seed)
    frac = jax.random.uniform(cfg_seed1, (samples.shape[0],))
    pw = 1 - neg_cfg_pw
    if abs(pw) < 1e-6:
        cfg = jnp.exp(jnp.log(cfg_min) + frac * (jnp.log(cfg_max) - jnp.log(cfg_min)))
    else:
        cfg = (cfg_min ** pw + frac * (cfg_max ** pw - cfg_min ** pw)) ** (1/pw)
    
    frac2 = jax.random.uniform(cfg_seed2, (samples.shape[0],))
    cfg = jnp.where(frac2 < no_cfg_frac, 1.0, cfg)

    def loss_grad_info(labels, samples, negative_samples, negative_weights, cfg, rng_step):
        labels = enforce_ddp(labels)
        samples = enforce_ddp(samples)
        negative_samples = enforce_ddp(negative_samples)
        negative_weights = enforce_ddp(negative_weights)
        cfg = enforce_ddp(cfg)
        bsz = labels.shape[0]
        
        uncond_w = (cfg - 1) * (gen_per_label - 1) / max(1, negative_samples.shape[1]) # [B]
        n_pos, n_gen, n_uncond = samples.shape[1], gen_per_label, negative_samples.shape[1]
        neg_samples_input = rearrange(jnp.concatenate([samples, negative_samples], axis=1), 'b x ... -> (b x) ...')
        neg_samples_input = enforce_ddp(neg_samples_input)
        sg_features = jax.lax.stop_gradient(feature_apply(feature_params, neg_samples_input, **activation_kwargs))
        if bsz % jax.device_count() == 0:
            sg_features = jax.tree.map(lambda u: rearrange(u, '(b x) ... -> b x ...', x=n_pos + n_uncond), sg_features) 
        else:
            sg_features = jax.tree.map(lambda u: rearrange(enforce_ddp(u), '(b x) ... -> b x ...', x=n_pos + n_uncond), sg_features) 
        sg_features = enforce_ddp(sg_features)

        def loss_fn(params):
            input_labels = enforce_ddp(repeat(labels, 'b -> (b g)', g=gen_per_label))
            input_cfg = enforce_ddp(repeat(cfg, 'b -> (b g)', g=gen_per_label))
            gen_samples = state.apply_fn(
                {'params': params},
                train=True,
                rngs=prepare_rng(rng_step, ['noise']),
                c=input_labels,
                cfg_scale=input_cfg,
            )['samples'] 
            gen_features = feature_apply(feature_params, gen_samples, **activation_kwargs)
            if bsz % jax.device_count() == 0:
                gen_features = jax.tree.map(lambda u: rearrange(u, '(b g) ... -> b g ...', g=n_gen), gen_features) # [B, G, F, D]
            else:
                gen_features = jax.tree.map(lambda u: rearrange(enforce_ddp(u), '(b g) ... -> b g ...', g=n_gen), gen_features) # [B, G, F, D]
            gen_features = enforce_ddp(gen_features)

            def feature_loss(sg_features, gen_features):
                feature_pos, feature_gen, feature_uncond = sg_features[:, :n_pos], gen_features, sg_features[:, n_pos:]
                feature_pos = enforce_ddp(rearrange(feature_pos, 'b x f d -> (b f) x d'))
                feature_gen = enforce_ddp(rearrange(feature_gen, 'b x f d -> (b f) x d'))
                feature_uncond = enforce_ddp(rearrange(feature_uncond, 'b x f d -> (b f) x d'))
                B = feature_gen.shape[0]
                weighted_uncond = repeat(
                    uncond_w[:, None] * negative_weights,
                    'b k -> (b f) k',
                    f=B // uncond_w.shape[0],
                )
                weight_pos = jnp.ones_like(feature_pos[:, :, 0])
                coupling_info = {}
                if pos_coupling_enabled and pos_coupling_alpha > 0.0:
                    weight_pos, coupling_info = _positive_coupling_weight(
                        feature_pos=feature_pos,
                        feature_gen=feature_gen,
                        step=state.step,
                        alpha=pos_coupling_alpha,
                        tau=pos_coupling_tau,
                        warmup_steps=pos_coupling_warmup,
                        ramp_steps=pos_coupling_ramp,
                    )
                loss, info = drift_loss(
                    gen=feature_gen,
                    fixed_pos=feature_pos,
                    fixed_neg=feature_uncond,
                    weight_gen=jnp.ones_like(feature_gen[:, :, 0]),
                    weight_pos=weight_pos,
                    weight_neg=weighted_uncond,
                    **loss_kwargs,
                )
                info.update(coupling_info)
                return loss, info
            
            loss_per_feature = jax.tree.map(feature_loss, sg_features, gen_features)
            total_loss = 0
            total_info = dict()
            for k, v in loss_per_feature.items():
                total_loss = total_loss + v[0].mean()
                for k2, v2 in v[1].items():
                    total_info[f'{k2}/{k}'] = v2
            total_loss = total_loss.mean()
            total_info = jax.tree.map(lambda x: x.mean(), total_info)
            if spectral_prior_enabled and spectral_lambda > 0.0:
                spectral_loss, spectral_info = _spectral_prior_loss(
                    gen_samples=gen_samples,
                    positive_samples=samples,
                    gen_per_label=n_gen,
                    radial_bins=spectral_bins,
                    loss_type=spectral_loss_type,
                    normalize_total_power=spectral_normalize,
                )
                spectral_lambda_eff = spectral_lambda * _linear_ramp(
                    state.step,
                    warmup_steps=spectral_warmup,
                    ramp_steps=spectral_ramp,
                )
                total_loss = total_loss + spectral_lambda_eff * spectral_loss
                total_info.update(spectral_info)
                total_info["spectral_prior_lambda"] = spectral_lambda_eff

            return total_loss, total_info

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metric), grads = grad_fn(state.params)
        return loss, metric, grads

    loss, metric, grads = loss_grad_info(labels, samples, negative_samples, negative_weights, cfg, rng_step)

    g_norm = optax.global_norm(grads)
    clipper = optax.clip_by_global_norm(max_grad_norm)
    updates, _ = clipper.update(grads, None)
    
    new_state = state.apply_gradients(grads=updates)

    new_ema_params = jax.tree.map(
        lambda ema, p: ema * state.ema_decay + p * (1.0 - state.ema_decay),
        state.ema_params,
        new_state.params,
    )
    new_state = new_state.replace(ema_params=new_ema_params)
    
    metric['loss'] = loss
    metric['g_norm'] = g_norm
    metric['lr'] = learning_rate_fn(state.step)
    metric = jax.tree.map(lambda x: x.mean(), metric)
    return new_state, metric

def generate_step(batch, params, rng, apply_fn, postprocess_fn, cfg_scale=1.0):
    """Generate samples from a batch of labels for FID evaluation.

    Args:
        batch: tuple ``(images, labels)`` from ``epoch0_sampler``; only labels are used.
        params: generator parameter tree.
        rng: PRNGKey for noise sampling.
        apply_fn: model ``apply`` callable.
        postprocess_fn: maps raw model output ``(B, H, W, C)`` to uint8 ``(B, C, H, W)`` or ``(B, H, W, C)``.
        cfg_scale: classifier-free guidance scale.

    Returns:
        Postprocessed samples with shape ``(B, ...)``.
    """
    _, labels = batch
    labels = jax.lax.with_sharding_constraint(labels, data_shard())
    latent_samples = apply_fn(
        {'params': params},
        train=False,
        rngs=prepare_rng(rng, ['noise']),
        c=labels,
        cfg_scale=cfg_scale,
    )['samples']
    latent_samples = jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        latent_samples
    )
    return postprocess_fn(latent_samples)


def _select_summary_leaf(features, summary_key: str = ""):
    """Pick one activation leaf for compact replay-bank distance summaries."""
    if summary_key and summary_key in features:
        return features[summary_key]
    for key in ("layer4_mean", "layer4", "layer3_mean", "layer3", "norm_x"):
        if key in features:
            return features[key]
    first_key = sorted(features.keys())[0]
    return features[first_key]


def _summarize_features(features, summary_key: str = ""):
    leaf = _select_summary_leaf(features, summary_key).astype(jnp.float32)
    if leaf.ndim > 2:
        leaf = leaf.mean(axis=tuple(range(1, leaf.ndim - 1)))
    leaf = leaf.reshape((leaf.shape[0], -1))
    norm = jnp.linalg.norm(leaf, axis=1, keepdims=True)
    return leaf / jnp.clip(norm, a_min=1e-6)


def feature_summary_step(samples, feature_params, feature_apply, activation_kwargs=dict(), summary_key: str = ""):
    features = feature_apply(feature_params, samples, **activation_kwargs)
    return _summarize_features(features, summary_key)


def generate_replay_step(labels, params, rng, apply_fn, cfg_scale=1.0):
    labels = jax.lax.with_sharding_constraint(labels, data_shard())
    samples = apply_fn(
        {'params': params},
        train=False,
        rngs=prepare_rng(rng, ['noise']),
        c=labels,
        cfg_scale=cfg_scale,
    )['samples']
    return jax.tree_util.tree_map(
        lambda x: jax.lax.with_sharding_constraint(x, ddp_shard()),
        samples,
    )
def train_gen(
    model,  # DitGen model instance
    optimizer,  # Optax optimizer transform
    logger,  # logger with log_dict / finish
    eval_loader,  # evaluation dataloader iterator source
    train_loader,  # training dataloader iterator source
    learning_rate_fn,  # callable(step) -> lr
    preprocess_fn,  # preprocessing function for dataloader batches
    postprocess_fn,  # generated sample postprocess function
    dataset_name="imagenet256",  # dataset name for eval logging
    train_batch_size=0,  # override per-host train batch if > 0
    total_steps=100000,  # max optimization steps
    save_per_step=10000,  # checkpoint save interval
    eval_per_step=5000,  # evaluation interval
    eval_samples=50000,  # number of generated samples for FID evaluation
    activation_fn=None,  # feature function used by drift loss
    feature_params=None,  # params bundle consumed by activation_fn
    ema_decay=0.999,  # single EMA decay
    seed=42,  # global RNG seed
    pos_per_sample=32,  # positive samples from memory bank
    neg_per_sample=16,  # negative samples from memory bank
    forward_dict=dict(
        gen_per_label=16,
        cfg_min=1.0,
        cfg_max=4.0,
        neg_cfg_pw=1.0,
        no_cfg_frac=0.0,
    ), 
    positive_bank_size=64,
    negative_bank_size=512,
    cfg_list=(1.0,),
    activation_kwargs=dict(
        patch_mean_size=[2,4],
        patch_std_size=[2,4],
        use_std=True,
        use_mean=True,
        every_k_block=2,
    ),
    max_grad_norm=2.0,
    loss_kwargs=dict(R_list=(0.02, 0.05, 0.2)),
    keep_every=500000,  # long-term checkpoint retention interval
    keep_last=2,  # number of latest checkpoints to keep
    init_from="",  # `hf://<name>` or local dir of model
    push_per_step=0,  # memory-bank fill factor per train step
    push_at_resume=3000,  # extra fill multiplier when resuming
    generated_replay=dict(),  # recency-aware generated negative replay settings
    positive_coupling=dict(),  # soft-OT positive reweighting settings
    spectral_prior=dict(),  # latent/image radial-spectrum prior settings
    workdir="runs",  # run root containing checkpoints/logs
):
    """
    Main training loop.
    """
    eval_enabled = int(eval_per_step) > 0 and int(eval_samples) > 0
    if isinstance(ema_decay, (list, tuple)):
        if len(ema_decay) != 1:
            raise ValueError(f"Expected a single ema_decay value, got {ema_decay}")
        ema_decay = float(ema_decay[0])
    else:
        ema_decay = float(ema_decay)
    if cfg_list is None:
        cfg_list = [1.0]
    elif isinstance(cfg_list, (list, tuple)):
        cfg_list = [float(cfg) for cfg in cfg_list]
    else:
        cfg_list = [float(cfg_list)]

    generated_replay = generated_replay or {}
    replay_enabled = bool(generated_replay.get("enabled", False))
    replay_neg_per_sample = int(generated_replay.get("neg_per_sample", 0))
    if replay_enabled and replay_neg_per_sample <= 0:
        raise ValueError("train.generated_replay.neg_per_sample must be > 0 when generated replay is enabled.")
    replay_bank_size = int(generated_replay.get("bank_size", 256))
    replay_warmup_steps = int(generated_replay.get("warmup_steps", 100))
    replay_push_interval = int(generated_replay.get("push_interval", 1))
    replay_push_per_label = int(generated_replay.get("push_per_label", 1))
    replay_cfg_scale = float(generated_replay.get("cfg_scale", 1.0))
    replay_half_life = float(generated_replay.get("half_life", 500.0))
    replay_distance_scale = float(generated_replay.get("distance_scale", 1.0))
    replay_hard_fraction = float(generated_replay.get("hard_fraction", 0.5))
    replay_hard_fraction_min = float(generated_replay.get("hard_fraction_min", 0.1))
    replay_candidate_multiplier = int(generated_replay.get("candidate_multiplier", 4))
    replay_min_weight = float(generated_replay.get("min_weight", 0.05))
    replay_max_weight = float(generated_replay.get("max_weight", 4.0))
    replay_weight_scale = float(generated_replay.get("weight_scale", 0.5))
    replay_summary_key = str(generated_replay.get("summary_key", "layer4_mean"))

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    rng_train, rng_eval = jax.random.split(rng)
    state = init_state_from_dummy_input(model, optimizer, TrainState, rng, model.dummy_input(), model.rng_keys(), ema_decay=ema_decay)
    state = restore_checkpoint(state=state, workdir=workdir)
    if int(jax.device_get(state.step)) == 0 and init_from:
        log_for_0("Initializing generator params from init_from=%s", init_from)
        state = maybe_init_state_params(
            state,
            model_type="generator",
            init_from=init_from,
            hf_cache_dir=HF_ROOT,
        )
    gen_step_jit = jax.jit(partial(generate_step, apply_fn=state.apply_fn, postprocess_fn=postprocess_fn))
    assert feature_params is not None, "feature_params must be provided for multi-host safe feature extraction"
    loss_kwargs['R_list'] = tuple(loss_kwargs['R_list'])
    if "adaptive_multipliers" in loss_kwargs:
        loss_kwargs["adaptive_multipliers"] = tuple(loss_kwargs["adaptive_multipliers"])
    state_sharding = jax.tree.map(lambda x: x.sharding, state)
    train_step_jit = jax.jit(
        partial(
            train_step,
            rng_init=rng_train,
            learning_rate_fn=learning_rate_fn,
            feature_apply=activation_fn,
            activation_kwargs=activation_kwargs,
            loss_kwargs=loss_kwargs,
            positive_coupling=positive_coupling,
            spectral_prior=spectral_prior,
            **forward_dict,
            max_grad_norm=max_grad_norm,
        ),
        out_shardings=(state_sharding, None),
    )
    replay_feature_summary_jit = None
    replay_generate_jit = None
    if replay_enabled:
        replay_feature_summary_jit = jax.jit(
            partial(
                feature_summary_step,
                feature_apply=activation_fn,
                activation_kwargs=activation_kwargs,
                summary_key=replay_summary_key,
            )
        )
        replay_generate_jit = jax.jit(
            partial(
                generate_replay_step,
                apply_fn=state.apply_fn,
                cfg_scale=replay_cfg_scale,
            )
        )

    ema_to_params_func = map_to_sharding(state.params)
    
    log_for_0("Starting training loop...")
    step = int(state.step)
    initial_step = step
    pbar = tqdm(range(step, total_steps), initial=step, total=total_steps) if is_rank_zero() else range(step, total_steps)
    memory_bank_positive = ArrayMemoryBank(num_classes=1000, max_size=positive_bank_size)
    memory_bank_negative = ArrayMemoryBank(num_classes=1, max_size=negative_bank_size)
    replay_rng = np.random.default_rng(int(seed) + 7919 + jax.process_index())
    generated_negative_bank = None
    if replay_enabled:
        generated_negative_bank = RecencyGeneratedNegativeBank(
            num_classes=1000,
            max_size=replay_bank_size,
            half_life=replay_half_life,
            distance_scale=replay_distance_scale,
            hard_fraction=replay_hard_fraction,
            hard_fraction_min=replay_hard_fraction_min,
            candidate_multiplier=replay_candidate_multiplier,
            min_weight=replay_min_weight,
            max_weight=replay_max_weight,
        )
    mu.sync_global_devices("train loop started")
    train_iter = infinite_sampler(train_loader, step)

    print(f"process_count={jax.process_count()} "
            f"local_device_count={jax.local_device_count()} "
            f"device_count={jax.device_count()}")

    for step in pbar:
        start_time = time.time()
        n_push = 0
        logger.set_step(step)

        # do push to memory bank; per host 
        goal = push_per_step
        if initial_step > 0 and step == initial_step:
            goal = push_at_resume * push_per_step
            print(f"pushing at resume: {goal}")
        while True:
            batch = next(train_iter)
            # Preprocess batch: converts (images, labels) tuple to {'images': BHWC, 'labels': ...}
            processed_batch = preprocess_fn(batch)
            images = processed_batch['images']  # BHWC format
            labels = processed_batch['labels']
            memory_bank_positive.add(images, labels)
            memory_bank_negative.add(images, labels * 0)
            n_push += images.shape[0]
            if n_push >= goal:
                break
        
        bsz_per_host = train_batch_size // jax.process_count()
        assert labels.shape[0] >= bsz_per_host, f"Labels shape {labels.shape[0]} < bsz_per_host {bsz_per_host}"
        select_indices = jax.random.choice(jax.random.fold_in(rng_train, step), jnp.arange(labels.shape[0]), (bsz_per_host,), replace=False)
        labels = labels[select_indices]
        images = images[select_indices]

        labels_host = np.asarray(jax.device_get(labels), dtype=np.int32)
        positive_samples = memory_bank_positive.sample(labels_host, n_samples=pos_per_sample, rng=replay_rng)
        real_negative_samples = memory_bank_negative.sample(
            np.zeros_like(labels_host),
            n_samples=neg_per_sample,
            rng=replay_rng,
        )
        real_negative_weights = jnp.ones(
            (real_negative_samples.shape[0], real_negative_samples.shape[1]),
            dtype=jnp.float32,
        )
        negative_samples = real_negative_samples
        negative_weights = real_negative_weights
        replay_metrics = {}
        if replay_enabled:
            replay_result = None
            if generated_negative_bank is not None and generated_negative_bank.total_count > 0:
                query_summaries = replay_feature_summary_jit(images, feature_params)
                replay_result = generated_negative_bank.sample(
                    labels_host,
                    n_samples=replay_neg_per_sample,
                    current_step=step,
                    query_summaries=np.asarray(jax.device_get(query_summaries)),
                    rng=replay_rng,
                    return_info=True,
                )
            if replay_result is None:
                replay_negative_samples = jnp.zeros(
                    (
                        real_negative_samples.shape[0],
                        replay_neg_per_sample,
                        *real_negative_samples.shape[2:],
                    ),
                    dtype=real_negative_samples.dtype,
                )
                replay_negative_weights = jnp.zeros(
                    (real_negative_samples.shape[0], replay_neg_per_sample),
                    dtype=jnp.float32,
                )
                replay_metrics = {
                    "replay/total_count": 0.0,
                    "replay/nonzero_frac": 0.0,
                    "replay/weight_mean": 0.0,
                    "replay/age_mean": 0.0,
                    "replay/distance_mean": 0.0,
                    "replay/hard_fraction_mean": 0.0,
                }
            else:
                replay_negative_samples, replay_negative_weights, replay_metrics = replay_result
                replay_negative_weights = replay_negative_weights * replay_weight_scale
            negative_samples = jnp.concatenate([real_negative_samples, replay_negative_samples], axis=1)
            negative_weights = jnp.concatenate([real_negative_weights, replay_negative_weights], axis=1)

        merged_positive, merged_negative, merged_negative_weights, merged_labels = merge_data(
            (positive_samples, negative_samples, negative_weights, labels)
        )

        process_time = time.time() - start_time

        profile_metrics = dict()
        if (step == initial_step):
            profile_metrics = profile_func(
                train_step_jit,
                (state, merged_labels, merged_positive, merged_negative, merged_negative_weights, feature_params),
                name="train_step",
            )

        new_state, metrics = train_step_jit(
            state,
            merged_labels,
            merged_positive,
            merged_negative,
            merged_negative_weights,
            feature_params,
        )
        metrics = jax.tree.map(lambda x: x.mean(), metrics)
        state = new_state
        replay_pushed = 0
        if (
            replay_enabled
            and generated_negative_bank is not None
            and replay_generate_jit is not None
            and replay_feature_summary_jit is not None
            and replay_push_per_label > 0
            and replay_push_interval > 0
            and (step + 1) >= replay_warmup_steps
            and ((step + 1) % replay_push_interval == 0)
        ):
            replay_labels_host = np.repeat(labels_host, replay_push_per_label)
            replay_labels = jnp.asarray(replay_labels_host, dtype=jnp.int32)
            replay_samples = replay_generate_jit(
                replay_labels,
                params=state.params,
                rng=jax.random.fold_in(rng_train, step + 17017),
            )
            replay_summaries = replay_feature_summary_jit(replay_samples, feature_params)
            generated_negative_bank.add(
                np.asarray(jax.device_get(replay_samples)),
                replay_labels_host,
                step=step + 1,
                summaries=np.asarray(jax.device_get(replay_summaries)),
            )
            replay_pushed = replay_labels_host.shape[0]
        total_time = time.time() - start_time
        metrics['total_time'] = total_time
        metrics['process_time'] = process_time
        metrics['kimg'] = (step + 1) * merged_positive.shape[0] / 1000.0
        metrics['forward_kimg'] = (step + 1) * merged_positive.shape[0] / 1000.0 * forward_dict['gen_per_label']
        metrics.update(profile_metrics)
        metrics.update(replay_metrics)
        if replay_enabled:
            metrics['replay/pushed'] = float(replay_pushed)
    
        logger.log_dict(metrics)
        step += 1

        if step % save_per_step == 0 or step == total_steps: 
            mu.sync_global_devices("save checkpoint started")
            save_checkpoint(state, keep=keep_last, keep_every=keep_every, workdir=workdir)
            save_params_ema_artifact(
                state,
                workdir=workdir,
                kind="gen",
                model_config=_generator_model_config(model),
            )
            mu.sync_global_devices("save checkpoint finished")

        if eval_enabled and ((step % eval_per_step == 0) or (step == 1) or (step == total_steps)):
            is_sanity = (step == 1)  # do a sanity check, to make sure FID env is working

            n_samples = 500 if is_sanity else eval_samples
            folder_prefix = "sanity" if is_sanity else "CFG"
            eval_params = ema_to_params_func(state.ema_params)
            round_best_fid = float("inf")
            round_best_cfg = cfg_list[0]
            eval_cfg_list = cfg_list if not is_sanity else [cfg_list[0]]

            for eval_cfg in eval_cfg_list:
                mu.sync_global_devices("eval started")
                result = evaluate_fid(
                    dataset_name=dataset_name,
                    gen_func=gen_step_jit,
                    gen_params={"params": eval_params, "cfg_scale": eval_cfg},
                    eval_loader=eval_loader,
                    logger=logger,
                    num_samples=n_samples,
                    log_folder=f"{folder_prefix}{eval_cfg}",
                    log_prefix=f"EMA_{state.ema_decay:g}",
                    rng_eval=rng_eval,
                )
                mu.sync_global_devices("eval finished")
                fid_val = result.get("fid", float("inf"))
                if fid_val < round_best_fid:
                    round_best_fid = fid_val
                    round_best_cfg = eval_cfg
            if not is_sanity:
                log_for_0("best_fid=%.4f best_cfg=%.1f (step=%d)", round_best_fid, round_best_cfg, step)
                logger.log_dict({"best_fid": round_best_fid, "best_cfg": round_best_cfg})

        if step % 100 == 0:
            mu.sync_global_devices(f"train step {step} finished")


    mu.sync_global_devices("train loop finished")
    logger.finish()
    del model, optimizer, eval_loader, train_loader, state    
    gc.collect()
    jax.clear_caches()
    mu.sync_global_devices("train loop finished")


def main_gen(config, output_dir="runs"):
    if "logging" not in config:
        config.logging = {}
    config.logging.name = Path(output_dir).resolve().name
        
    from models.generator import DitGen
    
    set_global_mesh(config.get("hsdp_dim", min(8, jax.local_device_count() * jax.process_count())))
    
    model_dict = build_model_dict(config, DitGen, workdir=output_dir)
    use_aug = bool(config.dataset.get("use_aug", False))
    use_latent = bool(config.dataset.get("use_latent", False))
    use_cache = bool(config.dataset.get("use_cache", False))
    postprocess_fn_noclip = get_postprocess_fn(
        use_aug=use_aug,
        use_latent=use_latent,
        use_cache=use_cache,
        has_clip=False,
    )
    feature_cfg = model_dict.feature
    mae_path = str(feature_cfg.get("mae_path", "")).strip()
    if not mae_path and bool(feature_cfg.get("use_mae", True)):
        load_dict = feature_cfg.get("load_dict", {})
        if str(load_dict.get("source", "hf")).strip().lower() == "local":
            mae_path = str(load_dict.get("path", "")).strip()
        else:
            model_name = str(load_dict.get("hf_model_name", "")).strip()
            if model_name:
                mae_path = f"hf://{model_name}"
    if bool(feature_cfg.get("use_mae", True)) and not mae_path:
        raise ValueError("feature.mae_path (or feature.load_dict.hf_model_name / feature.load_dict.path) is required when use_mae=true.")
    activation_fn, variables = build_activation_function(
        mae_path=mae_path,
        use_convnext=bool(feature_cfg.get("use_convnext", False)),
        convnext_bf16=bool(feature_cfg.get("convnext_bf16", False)),
        use_mae=bool(feature_cfg.get("use_mae", True)),
        postprocess_fn=postprocess_fn_noclip,
    )
    train_gen(
        model=model_dict.model,
        optimizer=model_dict.optimizer,
        logger=model_dict.logger,
        eval_loader=model_dict.eval_loader,
        train_loader=model_dict.train_loader,
        learning_rate_fn=model_dict.learning_rate_fn,
        preprocess_fn=model_dict.preprocess_fn,
        postprocess_fn=model_dict.postprocess_fn,
        dataset_name=model_dict.dataset_name,
        activation_fn=activation_fn,
        feature_params=variables,
        workdir=output_dir,
        **config.train
    )
    mu.sync_global_devices("main_gen finished")
    del model_dict
    gc.collect()
    jax.clear_caches()
    mu.sync_global_devices("main_gen finished")

def main(args):
    run_init()
    config = load_config(args.config)
    if getattr(args, "seed", None) is not None:
        config.train.seed = int(args.seed)
    main_gen(config, output_dir=args.workdir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gen/latent_ablation.yaml", help="Path to configuration file.")
    parser.add_argument("--workdir", type=str, default="runs", help="Local workdir root for checkpoints/logs.")
    parser.add_argument("--seed", type=int, default=None, help="Override train.seed from the YAML config.")
    args = parser.parse_args()
    args.output_dir = args.workdir

    main(args)
