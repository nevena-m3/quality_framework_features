# QCHAN v4.0.0 cohort target-robustness hotfix R1

## Failure diagnosis

The G6 target/parameter-sensitivity cell failed with `KeyError: 'variant'`
because `parameter_sensitivity` was an empty DataFrame with no declared
columns. The empty table was a secondary symptom, not the primary failure.

The primary failure was the declared `octave_fraction_6` sensitivity variant.
The frozen QCHAN PSD grid is based on 16 kHz and `n_fft=2048`, giving 7.8125-Hz
frequency-bin spacing. One-sixth-octave bands beginning at 100 Hz are too narrow
to contain the minimum two frequency bins required by the LTAS integrator. The
variant therefore raised `ValueError: One-third-octave LTAS contains non-finite
bands` for every sampled recording. Because all parameter variants were inside
one recording-level `try` block, the error was captured but no parameter rows
were retained. The later gate-construction code then attempted to access a
nonexistent `variant` column and obscured the actual error.

## Scientific correction

The baseline feature definition remains one-third-octave LTAS. Smoothing
sensitivity is now characterized using technically resolvable one-octave and
one-half-octave alternatives, alongside the one-third-octave baseline. This
assesses robustness to progressively broader smoothing without pretending that
unresolvable one-sixth-octave information exists on the saved PSD grid.

No recording-level QCHAN feature value, reference, spectrum, support rule,
reference-vintage definition, or feature role was changed.

## Engineering correction

- target and parameter variants are isolated per variant;
- sensitivity DataFrames are created with explicit schemas even when empty;
- errors retain recording, stage, variant, exception type and message;
- G6 fails with the first real variant error instead of a secondary KeyError;
- completed extraction spectra and references remain restart-safe;
- two regression tests enforce the feasible smoothing grid and notebook schema.

## Rerun scope

The existing spectrum checkpoints, recording-level features, references,
reference ledger and reconstruction evidence are retained. Run the fresh local
notebook from the beginning with `REBUILD_CHECKPOINTS = False`; extraction will
reload completed checkpoints, then rerun G6 and all downstream evidence and
figures. Freezing remains prohibited.
