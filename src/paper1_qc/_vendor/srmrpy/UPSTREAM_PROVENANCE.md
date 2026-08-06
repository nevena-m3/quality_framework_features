# SRMRpy provenance

- Upstream repository: `https://github.com/jfsantos/SRMRpy`
- Upstream commit: `fee009779cef96bed34db3a7e31d10f3ad1ea133`
- Retrieved: 2026-07-29
- License: MIT (`LICENSE.md`)
- Vendored files: `hilbert.py`, `modulation_filters.py`, `segmentaxis.py`,
  and `srmr.py`

The only source modification is replacing absolute `srmrpy.*` imports with
package-relative imports. Numerical logic is unchanged. QREV v3 uses the
upstream regression-tested configuration `fast=True`, `norm=True`, and
`max_cf=30`.
