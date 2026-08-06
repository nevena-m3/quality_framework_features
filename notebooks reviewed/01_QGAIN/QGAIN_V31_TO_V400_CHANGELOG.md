# QGAIN v3.1 → v4.0 candidate change log

| Area | v3.1 | v4.0 reviewed candidate |
|---|---|---|
| Family label | gain and recorded-level dynamics | recorded level and level dynamics |
| Analysis features | 4 | 4; names preserved |
| Step detector | rejected audit still executed in module | removed from production module; legacy negative result retained in documentation |
| Signal-view label | incorrectly says DC preserved | exact mono/global-DC-removed/resampled view |
| Floor logic | sub-floor nonzero frames could be clamped but marked valid | all frames at/below floor are marked floor affected |
| Within IQR | could include frames from unusable short segments | only usable-segment frames are pooled |
| MAD scale | rounded 1.4826 implicit | exact factor parameterized; unscaled MAD exported |
| Weighting audit | limited | pooled-vs-segment-balanced companions |
| G5 | synthetic mechanism response only | explicit source non-identifiability and constant-RMS spectral controls |
| Drift audit | slope and CI | adds pair count and time-order permutation null |
| ML output | scientific wide table | scientific wide + standardized long + non-imputed ML interface |
| Deployment threshold | unspecified | explicitly not calibrated; standalone gate prohibited |
