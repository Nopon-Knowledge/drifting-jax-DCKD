#!/usr/bin/env python3
"""Write an immutable, stdlib-only provenance manifest before a new run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SUFFIXES = {".py", ".sh"}
SOURCE_DIRS = ("dataset", "models", "scripts", "utils")
ROOT_SOURCE_FILES = ("drift_loss.py", "inference.py", "main.py", "memory_bank.py", "train.py")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(project_root: Path) -> list[Path]:
    files: set[Path] = set()
    for relative in ROOT_SOURCE_FILES:
        path = project_root / relative
        if path.is_file():
            files.add(path)
    for dirname in SOURCE_DIRS:
        root = project_root / dirname
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.suffix in SOURCE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                files.add(path)
    return sorted(files, key=lambda item: item.relative_to(project_root).as_posix())


def git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def pip_freeze() -> list[str]:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--protocol-path", default="")
    parser.add_argument("--train-seed", default="config-default")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    workdir = Path(args.workdir)
    if not workdir.is_absolute():
        workdir = project_root / workdir
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    manifest_path = workdir / "run_manifest.json"
    if manifest_path.exists() and not args.allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing provenance manifest: {manifest_path}"
        )

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")

    snapshot_path = workdir / "config_snapshot.yaml"
    if snapshot_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite config snapshot: {snapshot_path}")
    shutil.copy2(config_path, snapshot_path)

    protocol_path: Path | None = None
    if args.protocol_path:
        protocol_path = Path(args.protocol_path)
        if not protocol_path.is_absolute():
            protocol_path = project_root / protocol_path
        protocol_path = protocol_path.resolve()
        if not protocol_path.is_file():
            raise FileNotFoundError(f"Missing protocol file: {protocol_path}")
        shutil.copy2(protocol_path, workdir / "protocol_snapshot.yaml")

    entries = []
    aggregate = hashlib.sha256()
    for path in source_files(project_root):
        relative = path.relative_to(project_root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        entries.append({"path": relative, "size": size, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")

    freeze = pip_freeze()
    freeze_path = workdir / "environment.freeze.txt"
    freeze_path.write_text("\n".join(freeze) + ("\n" if freeze else ""), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_id": args.protocol_id,
        "protocol_path": (
            protocol_path.relative_to(project_root).as_posix()
            if protocol_path and protocol_path.is_relative_to(project_root)
            else str(protocol_path or "")
        ),
        "project_root": str(project_root),
        "workdir": str(workdir),
        "train_seed": args.train_seed,
        "config": {
            "source_path": (
                config_path.relative_to(project_root).as_posix()
                if config_path.is_relative_to(project_root)
                else str(config_path)
            ),
            "snapshot_path": snapshot_path.name,
            "sha256": sha256_file(snapshot_path),
        },
        "source": {
            "aggregate_sha256": aggregate.hexdigest(),
            "file_count": len(entries),
            "files": entries,
            "git_head": git_value(project_root, "rev-parse", "HEAD"),
            "git_status_porcelain": git_value(project_root, "status", "--porcelain"),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "pip_freeze_path": freeze_path.name,
            "pip_freeze_sha256": sha256_file(freeze_path),
            "selected_env": {
                key: os.environ.get(key)
                for key in (
                    "CUDA_VISIBLE_DEVICES",
                    "JAX_PLATFORMS",
                    "XLA_FLAGS",
                    "XLA_PYTHON_CLIENT_PREALLOCATE",
                )
            },
        },
    }
    if protocol_path is not None:
        manifest["protocol_sha256"] = sha256_file(protocol_path)

    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "source_aggregate_sha256": aggregate.hexdigest(),
                "config_sha256": manifest["config"]["sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
