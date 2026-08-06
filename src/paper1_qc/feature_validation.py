"""Reusable validation and publication-output helpers for feature families."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito palette: color-vision-deficiency friendly for categorical figures.
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#6B6B6B",
}


@dataclass(frozen=True)
class ValidationCheck:
    """One prespecified validation assertion."""

    layer: str
    check: str
    passed: bool
    observed: str
    required: str
    action_if_failed: str
    blocking: bool = True


def set_publication_style() -> None:
    """Apply the common visual contract used by all feature-family notebooks."""

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validation_frame(checks: list[ValidationCheck]) -> pd.DataFrame:
    """Return a stable, auditable gate table."""

    frame = pd.DataFrame([asdict(check) for check in checks])
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "layer",
                "check",
                "passed",
                "observed",
                "required",
                "action_if_failed",
                "blocking",
            ]
        )
    frame["passed"] = frame["passed"].astype(bool)
    frame["blocking"] = frame["blocking"].astype(bool)
    return frame


def gate_passed(frame: pd.DataFrame) -> bool:
    """True only when every blocking validation check passed."""

    if frame.empty:
        return False
    blocking = frame.loc[frame["blocking"].astype(bool)]
    return bool(len(blocking) and blocking["passed"].astype(bool).all())


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _is_missing_scalar(value) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    return False


def _json_storage_text(value) -> str | None:
    if _is_missing_scalar(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Path):
        value = str(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _storage_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize object columns so CSV and Parquet store the same table."""

    safe = frame.copy(deep=True).reset_index(drop=True)
    for column in safe.columns:
        series = safe[column]
        if not pd.api.types.is_object_dtype(series.dtype):
            continue
        nonmissing = [value for value in series.tolist() if not _is_missing_scalar(value)]
        if not nonmissing or all(isinstance(value, str) for value in nonmissing):
            safe[column] = series.astype("string")
        elif all(isinstance(value, (bool, np.bool_)) for value in nonmissing):
            safe[column] = series.astype("boolean")
        elif all(
            isinstance(value, (int, np.integer))
            and not isinstance(value, (bool, np.bool_))
            for value in nonmissing
        ):
            safe[column] = pd.to_numeric(series, errors="raise").astype("Int64")
        elif all(
            isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
            for value in nonmissing
        ):
            safe[column] = pd.to_numeric(series, errors="raise").astype("Float64")
        else:
            safe[column] = pd.Series(
                [_json_storage_text(value) for value in series.tolist()],
                index=series.index,
                dtype="string",
            )
    return safe


def save_table_bundle(frame: pd.DataFrame, directory: str | Path, stem: str) -> dict:
    """Save matched CSV/Parquet representations after a round-trip check."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    safe = _storage_safe_frame(frame)
    csv_path = directory / f"{stem}.csv"
    parquet_path = directory / f"{stem}.parquet"
    csv_tmp = directory / f".{stem}.csv.tmp"
    parquet_tmp = directory / f".{stem}.parquet.tmp"
    try:
        safe.to_csv(csv_tmp, index=False)
        safe.to_parquet(parquet_tmp, index=False)
        roundtrip = pd.read_parquet(parquet_tmp)
        if list(roundtrip.columns) != list(safe.columns) or len(roundtrip) != len(safe):
            raise RuntimeError(f"Parquet round-trip failed for {stem}")
        csv_tmp.replace(csv_path)
        parquet_tmp.replace(parquet_path)
    finally:
        csv_tmp.unlink(missing_ok=True)
        parquet_tmp.unlink(missing_ok=True)
    return {"csv": csv_path, "parquet": parquet_path}


def save_publication_figure(
    fig,
    directory: str | Path,
    stem: str,
    *,
    caption: str,
    alt_text: str,
    dpi: int = 600,
) -> dict:
    """Save editable/vector and high-resolution raster figure artifacts."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": directory / f"{stem}.png",
        "svg": directory / f"{stem}.svg",
        "pdf": directory / f"{stem}.pdf",
    }
    fig.savefig(paths["png"], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths["svg"], bbox_inches="tight", facecolor="white")
    fig.savefig(paths["pdf"], bbox_inches="tight", facecolor="white")
    sidecar = directory / f"{stem}.figure.json"
    write_json(
        {
            "stem": stem,
            "caption": caption,
            "alt_text": alt_text,
            "dpi": dpi,
            "files": {kind: path.name for kind, path in paths.items()},
        },
        sidecar,
    )
    return {**paths, "metadata": sidecar}
