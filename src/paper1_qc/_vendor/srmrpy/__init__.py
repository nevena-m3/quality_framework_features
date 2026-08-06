"""Vendored SRMRpy reference implementation.

Upstream: https://github.com/jfsantos/SRMRpy
Commit: fee009779cef96bed34db3a7e31d10f3ad1ea133
License: MIT; see LICENSE.md in this directory.

Only import through :mod:`paper1_qc.qrev`, which pins the scientifically
validated normalized-fast parameterization.
"""

from .srmr import srmr

__all__ = ["srmr"]
