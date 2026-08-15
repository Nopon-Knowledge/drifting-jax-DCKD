#!/usr/bin/env python3
"""Generate ImageNet-class samples from a Drift generator artifact."""

from __future__ import annotations

import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from utils.env import HF_ROOT
from utils.hsdp_util import set_global_mesh
from utils.init_util import load_generator_model_and_params
from utils.misc import prepare_rng, run_init


def parse_class_ids(text: str) -> list[int]:
    ids = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not ids:
        raise ValueError("Provide at least one class id, for example --class-ids 95,22,88.")
    for class_id in ids:
        if not 0 <= class_id < 1000:
            raise ValueError(f"class id out of ImageNet-1k range [0,999]: {class_id}")
    return ids


def expand_class_ids(class_ids: list[int], samples_per_class: int) -> list[int]:
    if samples_per_class < 1:
        raise ValueError("--samples-per-class must be >= 1")
    return [class_id for class_id in class_ids for _ in range(samples_per_class)]


def is_latent_model(metadata: dict) -> bool:
    return metadata.get("model_config", {}).get("in_channels", 3) == 4


def generate_step(labels, params, vae_params, rng, *, apply_fn, vae, latent: bool, cfg_scale: float):
    samples = apply_fn(
        {"params": params},
        train=False,
        rngs=prepare_rng(rng, ["noise"]),
        c=labels,
        cfg_scale=cfg_scale,
    )["samples"]
    if latent:
        from diffusers.models import FlaxAutoencoderKL

        decoded = vae.apply(
            {"params": vae_params},
            samples / 0.18215,
            method=FlaxAutoencoderKL.decode,
        ).sample
        return jnp.clip((decoded + 1) / 2, 0, 1)
    return jnp.clip((samples + 1) / 2, 0, 1).transpose(0, 3, 1, 2)


def to_uint8_images(samples: np.ndarray) -> np.ndarray:
    images = np.asarray(samples)
    if images.ndim != 4:
        raise ValueError(f"Expected 4D samples, got shape={images.shape}")
    if images.shape[1] in (3, 4):
        images = images.transpose(0, 2, 3, 1)
    images = np.clip(images[..., :3], 0.0, 1.0)
    return (images * 255.0 + 0.5).astype(np.uint8)


def save_grid(images: np.ndarray, labels: list[int], path: Path, *, cols: int) -> None:
    if len(images) == 0:
        return
    h, w = images.shape[1:3]
    label_h = 18
    cols = max(1, min(cols, len(images)))
    rows = math.ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * w, rows * (h + label_h)), "white")
    draw = ImageDraw.Draw(grid)
    for i, image in enumerate(images):
        row, col = divmod(i, cols)
        x = col * w
        y = row * (h + label_h)
        grid.paste(Image.fromarray(image), (x, y))
        draw.text((x + 4, y + h + 2), f"class {labels[i]:03d}", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate image samples for selected ImageNet class ids.")
    parser.add_argument("--init-from", required=True, help="Local workdir/artifact path or hf://<model_id>.")
    parser.add_argument("--class-ids", required=True, help="Comma-separated ImageNet-1k class ids, e.g. 95,22,88.")
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--outdir", default="runs/inference_samples")
    parser.add_argument("--grid-cols", type=int, default=4)
    parser.add_argument("--vae-backend", default=None, help="Backend for latent VAE decode, e.g. cuda/gpu/cpu.")
    parser.add_argument("--hsdp-dim", type=int, default=None, help="Optional HSDP mesh dimension.")
    parser.add_argument("--metadata-out", default="")
    return parser


def main() -> None:
    run_init()
    args = build_parser().parse_args()
    hsdp = args.hsdp_dim or min(8, jax.local_device_count() * jax.process_count())
    set_global_mesh(hsdp)
    class_ids = expand_class_ids(parse_class_ids(args.class_ids), args.samples_per_class)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    model, params, metadata = load_generator_model_and_params(args.init_from, hf_cache_dir=HF_ROOT)
    latent = is_latent_model(metadata)
    vae = None
    vae_params = {}
    if latent:
        from dataset.vae import load_vae_module_and_params

        vae, vae_params = load_vae_module_and_params(replicate_params=False, backend=args.vae_backend)
    step = jax.jit(
        partial(generate_step, apply_fn=model.apply, vae=vae, latent=latent, cfg_scale=args.cfg_scale)
    )

    all_images: list[np.ndarray] = []
    written: list[dict] = []
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(class_ids), batch_size):
        labels_np = np.asarray(class_ids[start : start + batch_size], dtype=np.int32)
        labels = jax.device_put(labels_np)
        samples = step(labels, params=params, vae_params=vae_params, rng=jax.random.PRNGKey(args.seed + start))
        images = to_uint8_images(np.asarray(jax.device_get(samples)))
        all_images.append(images)
        for offset, image in enumerate(images):
            index = start + offset
            class_id = class_ids[index]
            image_path = outdir / f"sample_{index:04d}_class_{class_id:03d}.png"
            Image.fromarray(image).save(image_path)
            written.append({"index": index, "class_id": class_id, "path": str(image_path)})

    merged = np.concatenate(all_images, axis=0)
    grid_path = outdir / "grid.png"
    save_grid(merged, class_ids, grid_path, cols=args.grid_cols)

    summary = {
        "init_from": args.init_from,
        "class_ids": class_ids,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "latent": latent,
        "outdir": str(outdir),
        "grid": str(grid_path),
        "images": written,
        "metadata": metadata,
    }
    metadata_out = Path(args.metadata_out).resolve() if args.metadata_out else outdir / "summary.json"
    metadata_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("init_from", "class_ids", "cfg_scale", "seed", "grid")}, indent=2))


if __name__ == "__main__":
    main()
