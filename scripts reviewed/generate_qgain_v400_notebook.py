from __future__ import annotations

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "notebooks reviewed" / "01_QGAIN" / "02b_gain_dynamics_QGAIN_v4_0_0_REVIEWED_SOURCE.ipynb"
target = source
print(f"Authoritative reviewed notebook: {target}")
print("The notebook is source-controlled directly in Stage 1. Future revisions should update the notebook and this message together.")
