from diffusers.models import FlaxAutoencoderKL
from functools import partial
import jax.numpy as jnp
import numpy as np
import jax

from utils.devices import local_devices_for_backend


# Module-level cache for VAE encode/decode functions
_vae_cache = {}


def _put_tree_on_local_device(tree, backend: str | None = None):
    local_devices = local_devices_for_backend(backend)
    target_device = local_devices[0]
    def _to_local_device(x):
        host_value = np.asarray(x) if isinstance(x, (np.ndarray, jax.Array)) else x
        return jax.device_put(host_value, target_device)
    return jax.tree.map(_to_local_device, tree)


def load_vae_module_and_params(replicate_params: bool = True, backend: str | None = None):
    """Load the Flax VAE and place params on the requested backend."""
    cache_key = ("vae_module_and_params", replicate_params, backend or "default")
    if cache_key in _vae_cache:
        return _vae_cache[cache_key]

    vae, vae_params = FlaxAutoencoderKL.from_pretrained("pcuenq/sd-vae-ft-mse-flax")

    # Replicate params across all devices/processes via the global mesh.
    if replicate_params:
        from jax.sharding import NamedSharding, PartitionSpec as P
        from utils.hsdp_util import get_global_mesh

        mesh = get_global_mesh()
        replicated_sharding = NamedSharding(mesh, P())

        def _replicate(x):
            x = jnp.asarray(x) if isinstance(x, np.ndarray) else x
            if jax.process_count() > 1:
                return jax.make_array_from_process_local_data(replicated_sharding, x)
            return jax.device_put(x, replicated_sharding)

        vae_params = jax.tree.map(_replicate, vae_params)
    else:
        vae_params = _put_tree_on_local_device(vae_params, backend=backend)

    result = (vae, vae_params)
    _vae_cache[cache_key] = result
    return result


def vae_enc_decode(replicate_params: bool = True, backend: str | None = None):
    '''
    Returns:
        encode_fn: a function that takes in (B, C, H, W) & rng and returns (B, H, W, C)
        decode_fn: a function that takes in (B, H, W, C) and returns (B, C, H, W)

    Args:
        replicate_params: If True, replicate VAE params across all devices in the global mesh.
        backend: Optional JAX backend used when replicate_params=False, e.g. "gpu", "cuda", "tpu", or "cpu".
    
    Note: Results are cached to avoid reloading the VAE model on repeated calls.
    '''
    cache_key = ('vae_enc_decode', replicate_params, backend or 'default')
    if cache_key in _vae_cache:
        return _vae_cache[cache_key]

    vae, vae_params = load_vae_module_and_params(replicate_params=replicate_params, backend=backend)

    def _encode_fn(images, rng, params):
        dist = vae.apply({'params': params}, images, method=FlaxAutoencoderKL.encode).latent_dist
        return dist.sample(key=rng) * 0.18215

    def _decode_fn(latents, params):
        return vae.apply({'params': params}, latents / 0.18215, method=FlaxAutoencoderKL.decode).sample

    result = (partial(_encode_fn, params=vae_params), partial(_decode_fn, params=vae_params))
    _vae_cache[cache_key] = result
    return result
