"""Latent cache dataset and cache builder for ImageNet release workflows."""

from __future__ import annotations

import argparse
import os
import multiprocessing as mp
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as mu
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from tqdm import tqdm

from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH
from utils.devices import devices_for_backend, local_devices_for_backend


@dataclass(frozen=True)
class _CacheWriteItem:
    output_path: str
    moments: np.ndarray
    moments_flip: np.ndarray


def _write_cache_file(item: _CacheWriteItem) -> None:
    output_path = Path(item.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp.{os.getpid()}")
    torch.save(
        {
            "moments": item.moments,
            "moments_flip": item.moments_flip,
        },
        tmp_path,
    )
    os.replace(tmp_path, output_path)


class LatentDataset(datasets.DatasetFolder):
    """ImageFolder-style dataset for cached latent `.pt` files."""

    def __init__(self, root: str):
        super().__init__(root=root, loader=str, extensions=(".pt",))

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        data = torch.load(path, map_location="cpu", weights_only=False)
        moments = data["moments"] if torch.rand(1) < 0.5 else data["moments_flip"]
        return np.asarray(moments), target


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM-style center crop used before encoding to latent."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


class OriginalImageFolder(datasets.ImageFolder):
    """ImageFolder that also returns class/file relative path for cache writing."""

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        rel_path = os.path.join(*path.split(os.path.sep)[-2:])
        return sample, target, rel_path


class ImageNetFlatValFolder(torch.utils.data.Dataset):
    """ImageNet validation split with flat JPEG files and CLS-LOC XML labels."""

    def __init__(
        self,
        root: str | Path,
        annotation_root: str | Path,
        classes: list[str],
        transform=None,
        return_rel_path: bool = True,
    ):
        self.root = Path(root)
        self.annotation_root = Path(annotation_root)
        self.classes = list(classes)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.transform = transform
        self.return_rel_path = return_rel_path
        self.loader = datasets.folder.default_loader
        self.image_paths = self._load_image_paths()
        self.samples = [(str(path), -1) for path in self.image_paths]
        if not self.image_paths:
            raise RuntimeError(f"No ImageNet validation images found in {self.root}")

    def _label_for_image(self, image_path: Path) -> str:
        annotation_path = self.annotation_root / f"{image_path.stem}.xml"
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing ImageNet val annotation: {annotation_path}")
        root = ET.parse(annotation_path).getroot()
        label = root.findtext("object/name")
        if not label:
            raise ValueError(f"Could not read ImageNet class label from {annotation_path}")
        if label not in self.class_to_idx:
            raise ValueError(f"Validation label {label!r} from {annotation_path} is not present in train classes.")
        return label

    def _imageset_file(self) -> Path:
        ilsvrc_root = self.annotation_root.parent.parent.parent
        return ilsvrc_root / "ImageSets" / "CLS-LOC" / "val.txt"

    def _path_from_stem(self, stem: str) -> Path:
        for suffix in (".JPEG", ".jpg", ".jpeg", ".png"):
            path = self.root / f"{stem}{suffix}"
            if path.is_file():
                return path
        return self.root / f"{stem}.JPEG"

    def _load_image_paths(self) -> list[Path]:
        imageset = self._imageset_file()
        if imageset.is_file():
            paths = []
            for line in imageset.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                stem = line.split()[0]
                path = self._path_from_stem(stem)
                if path.is_file():
                    paths.append(path)
            return paths
        return [
            image_path
            for image_path in sorted(self.root.iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMG_EXTENSIONS
        ]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self._label_for_image(image_path)
        target = self.class_to_idx[label]
        rel_path = os.path.join(label, image_path.name)
        path = str(image_path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if not self.return_rel_path:
            return sample, target
        return sample, target, rel_path


def _is_imagefolder_split(root: Path) -> bool:
    return root.is_dir() and any(child.is_dir() for child in root.iterdir())


def _resolve_imagenet_data_root(data_path: str | Path) -> Path:
    """Return the directory that directly contains ImageNet train/ and val/."""
    root = Path(data_path).expanduser().resolve()
    if (root / "train").is_dir() and (root / "val").is_dir():
        return root

    candidates = [
        root / "ILSVRC" / "Data" / "CLS-LOC",
        root / "Data" / "CLS-LOC",
    ]
    candidates.extend(root.glob("*/ILSVRC/Data/CLS-LOC"))
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "val").is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not find ImageNet train/ and val/ under {root}. "
        "Pass the directory containing train/ and val/, or an ImageNet CLS-LOC extraction root."
    )


def _imagenet_val_annotation_root(data_root: Path) -> Path:
    candidates = [
        data_root.parent.parent / "Annotations" / "CLS-LOC" / "val",
        data_root.parent / "Annotations" / "CLS-LOC" / "val",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find ImageNet CLS-LOC validation annotations near {data_root}. "
        "Expected an Annotations/CLS-LOC/val directory."
    )


def _build_cache_split_dataset(data_root: Path, split: str, transform):
    split_root = data_root / split
    if split == "val" and not _is_imagefolder_split(split_root):
        train_root = data_root / "train"
        classes = sorted(child.name for child in train_root.iterdir() if child.is_dir())
        return ImageNetFlatValFolder(
            root=split_root,
            annotation_root=_imagenet_val_annotation_root(data_root),
            classes=classes,
            transform=transform,
        )
    return OriginalImageFolder(str(split_root), transform=transform)


def _prepare_batch_data(images: torch.Tensor) -> np.ndarray:
    """Convert `(B,C,H,W)` torch tensor to host numpy for local-device sharding."""
    return images.numpy()


def create_cached_dataset(
    local_batch_size: int,
    target_path: str,
    data_path: str,
    *,
    backend: str | None = None,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    save_workers: int = 0,
) -> None:
    """Encode ImageNet train/val images and write latent cache files."""
    from diffusers.models import FlaxAutoencoderKL

    from dataset.vae import load_vae_module_and_params
    from utils.hsdp_util import get_global_mesh, set_global_mesh

    data_root = _resolve_imagenet_data_root(data_path)
    local_devices = local_devices_for_backend(backend)
    global_devices = devices_for_backend(backend)
    n_local_devices = len(local_devices)

    if local_batch_size % n_local_devices != 0:
        raise ValueError(
            f"`local_batch_size` must be divisible by local device count={n_local_devices}, got {local_batch_size}."
        )

    set_global_mesh(min(8, len(global_devices)), devices=global_devices)
    Path(target_path, "train").mkdir(parents=True, exist_ok=True)
    Path(target_path, "val").mkdir(parents=True, exist_ok=True)
    if jax.process_count() > 1:
        mu.sync_global_devices("latent cache target dirs ready")

    # Keep VAE params as an explicit JIT input. Closing over device arrays can
    # make JAX lower them as stale constants after scheduler-launched startup.
    vae, vae_params = load_vae_module_and_params(replicate_params=True, backend=backend)

    local_mesh = Mesh(np.array(local_devices), axis_names=("data",))
    sample_sharding = NamedSharding(local_mesh, P("data", None, None, None))
    rng_sharding = NamedSharding(local_mesh, P("data", None))
    output_sharding = NamedSharding(local_mesh, P("data", None, None, None))
    param_sharding = jax.tree.map(lambda _: NamedSharding(get_global_mesh(), P()), vae_params)
    per_device_batch = local_batch_size // n_local_devices

    @partial(
        jax.jit,
        in_shardings=(param_sharding, sample_sharding, rng_sharding),
        out_shardings={
            "moments": output_sharding,
            "moments_flip": output_sharding,
        },
    )
    def encode(params, samples, rngs):
        # Data is sharded across local devices, while the VAE params stay
        # replicated. Reshape once inside the jitted region so each device
        # processes its local microbatch with the same encode_fn used in train.
        samples = samples.reshape((n_local_devices, per_device_batch, *samples.shape[1:]))

        def _encode_shard(sample_shard, rng_shard):
            def _encode_images(images):
                dist = vae.apply(
                    {"params": params},
                    images,
                    method=FlaxAutoencoderKL.encode,
                ).latent_dist
                return dist.sample(key=rng_shard) * 0.18215

            return {
                "moments": _encode_images(sample_shard),
                "moments_flip": _encode_images(jnp.flip(sample_shard, axis=3)),
            }

        encoded = jax.vmap(_encode_shard, in_axes=(0, 0), out_axes=0)(samples, rngs)
        return jax.tree_util.tree_map(
            lambda x: x.reshape((local_batch_size, *x.shape[2:])),
            encoded,
        )

    transform = transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    save_pool = None
    save_futures: list[Future] = []
    if save_workers > 0:
        save_pool = ProcessPoolExecutor(max_workers=save_workers, mp_context=mp.get_context("spawn"))

    global_batch_size = local_batch_size * max(1, jax.process_count())
    process_slice_start = jax.process_index() * local_batch_size
    process_slice_end = process_slice_start + local_batch_size

    for split in ("train", "val"):
        dataset = _build_cache_split_dataset(data_root, split, transform=transform)
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": global_batch_size,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
            loader_kwargs["multiprocessing_context"] = "spawn"
        loader = torch.utils.data.DataLoader(**loader_kwargs)

        base_rng = jax.random.PRNGKey(0)
        for step, (samples, _, rel_paths) in tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"cache:{split}:host{jax.process_index()}",
        ):
            step_rng = jax.random.fold_in(base_rng, step)
            step_rng = jax.random.split(step_rng, n_local_devices)

            n_valid_global = samples.shape[0]
            rel_paths = list(rel_paths)
            if n_valid_global != global_batch_size:
                pad = global_batch_size - n_valid_global
                samples = torch.cat([samples, torch.zeros((pad,) + samples.shape[1:], dtype=samples.dtype)], dim=0)
                rel_paths.extend([""] * pad)

            local_samples = samples[process_slice_start:process_slice_end]
            encoded_local = encode(
                vae_params,
                jax.device_put(_prepare_batch_data(local_samples), sample_sharding),
                jax.device_put(step_rng, rng_sharding),
            )
            encoded_local = jax.tree_util.tree_map(np.asarray, encoded_local)
            encoded = {
                "moments": mu.process_allgather(encoded_local["moments"], tiled=True),
                "moments_flip": mu.process_allgather(encoded_local["moments_flip"], tiled=True),
            }

            write_items = []
            for i, rel_path in enumerate(rel_paths[:n_valid_global]):
                if not rel_path:
                    continue
                output_path = str(Path(target_path, split, rel_path).with_suffix(".pt"))
                write_items.append(
                    _CacheWriteItem(
                        output_path=output_path,
                        moments=np.asarray(encoded["moments"][i]),
                        moments_flip=np.asarray(encoded["moments_flip"][i]),
                    )
                )
            if save_pool is None:
                for item in write_items:
                    _write_cache_file(item)
            else:
                save_futures.extend(save_pool.submit(_write_cache_file, item) for item in write_items)

        if jax.process_count() > 1:
            mu.sync_global_devices(f"latent cache split {split} encoded")

    if save_pool is not None:
        for future in tqdm(save_futures, desc="cache:flush", disable=jax.process_index() != 0):
            future.result()
        save_pool.shutdown()

    if jax.process_count() > 1:
        mu.sync_global_devices("latent cache files flushed")


def build_cache_from_args(args: argparse.Namespace) -> None:
    from utils.misc import run_init

    run_init()
    create_cached_dataset(
        local_batch_size=int(args.local_batch_size),
        target_path=args.target_path,
        data_path=args.data_path,
        backend=args.backend,
        num_workers=int(args.num_workers),
        prefetch_factor=int(args.prefetch_factor),
        pin_memory=bool(args.pin_memory),
        save_workers=int(args.save_workers),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ImageNet latent cache files for release generator configs.")
    parser.add_argument("--data-path", default=IMAGENET_PATH, help="ImageNet root containing train/ and val/.")
    parser.add_argument("--target-path", default=IMAGENET_CACHE_PATH, help="Output cache root for latent .pt files.")
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional JAX backend for cache building. Use gpu/cuda on A800, tpu on TPU, or leave unset for JAX default.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=128,
        help="Per-process cache batch size. Must divide the selected local device count.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory for the cache build.")
    parser.add_argument(
        "--save-workers",
        type=int,
        default=0,
        help="Optional process count for asynchronous latent file writes on each host.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    build_cache_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
