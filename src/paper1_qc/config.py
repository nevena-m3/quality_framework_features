from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve all project paths without changing the working directory."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

    data_root = Path(cfg["paths"]["data_root"]).expanduser()
    output_root = Path(cfg["project"].get("output_root", "outputs"))
    if not output_root.is_absolute():
        output_root = config_path.parent.parent / output_root
    main_output_root = Path(cfg["project"].get("main_output_root", "MAIN outputs"))
    if not main_output_root.is_absolute():
        main_output_root = config_path.parent.parent / main_output_root

    cfg["_config_path"] = str(config_path)
    cfg["_project_root"] = str(config_path.parent.parent.resolve())
    cfg["_data_root"] = str(data_root)
    cfg["_output_root"] = str(output_root.resolve())
    cfg["_main_output_root"] = str(main_output_root.resolve())
    return cfg


def resolve_project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(cfg["_project_root"]) / path


def resolve_data_path(cfg: dict[str, Any], key: str) -> Path:
    value = Path(cfg["paths"][key])
    return value if value.is_absolute() else Path(cfg["_data_root"]) / value


def resolve_executable(value: str, default_name: str) -> str:
    candidate = default_name if value.lower() == "auto" else value
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Could not find {default_name!r}. Add it to PATH or set software.{default_name}."
        )
    return resolved


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
