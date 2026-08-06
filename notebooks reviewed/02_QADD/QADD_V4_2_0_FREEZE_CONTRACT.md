# QADD v4.2.0 Freeze Contract

## Acceptance token

`ACCEPT_QADD_V420`

## Immutable scientific decisions

- `qadd_pause_ac_level_dbfs_median` — RETAIN_PRIMARY_CONTEXTUAL
- `qadd_pause_level_iqr_db` — RETAIN_SECONDARY_CONDITIONAL
- `qadd_speech_pause_level_contrast_db` — RETAIN_SECONDARY_MIXED_NONINDEPENDENT
- `qadd_pause_spectral_flatness` — RETAIN_SECONDARY_NONORDINAL
- `qadd_mains_hum_comb_score_db` — RETAIN_TARGETED_CONDITIONAL

## Non-negotiable constraints

- No QADD scalar.
- No standalone recording accept/reject threshold.
- Missing and floor-censored values remain missing with status/support.
- Speech–pause contrast is not physical SNR and is not independent of pause level.
- Hum-comb prominence does not prove electrical mains interference.
- Panel I is explicitly N/A because no event detector is retained.
- Finalization must preserve all five numerical feature columns exactly.
- The freeze is immutable; any change requires a new semantic version.
