from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_run_manifest(
    output_path: str | Path,
    *,
    command: list[str],
    config_path: str | Path,
    input_paths: Iterable[str | Path] = (),
    extra: dict | None = None,
) -> dict:
    inputs = []
    for raw_path in input_paths:
        path = Path(raw_path)
        inputs.append(
            {
                "path": str(path.resolve()),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.is_file() else None,
                "sha256": sha256_file(path) if path.is_file() else None,
            }
        )

    try:
        pip_freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except OSError:
        pip_freeze = []

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "inputs": inputs,
        "packages": pip_freeze,
        "extra": extra or {},
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
