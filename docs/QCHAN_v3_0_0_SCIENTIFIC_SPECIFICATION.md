# QCHAN v3.0.0 scientific measurement specification

## Construct

QCHAN describes cohort-relative manifestations of acquisition-channel spectral
coloration and bandwidth limitation in frozen strict speech. It does not
identify a device, recover a microphone transfer function, or separate
acquisition effects from speech phenotype.

Every analysis value is tied to a frozen, task-matched,
leave-one-subject-out reference. References are constructed by taking a
binwise median across recordings within each reference subject and then a
binwise median across subjects. This prevents participants with repeated
recordings from receiving greater weight.

References require at least five other subjects and eight recordings. A
reference is unavailable if this requirement is not met. No global or
cross-task fallback is permitted.

## Analysis features

1. `qchan_ltas_distance_db` — primary, nonordinal. RMS difference in dB
   between target and reference gain-normalized one-third-octave LTAS.
2. `qchan_rolloff95_deficit_hz` — primary, oriented.
   `max(0, reference rolloff95 - target rolloff95)`.
3. `qchan_highband_ratio_deficit` — secondary, oriented.
   `max(0, reference - target)` for the 3–7.5-kHz to 0.1–7.5-kHz integrated
   power ratio.
4. `qchan_tilt_steepening_db_per_oct` — secondary and phenotype-sensitive.
   `max(0, reference slope - target slope)` for a Theil–Sen log-LTAS slope
   over 100–4000 Hz.

There is no scalar QCHAN score.

## Signal-processing contract

- Frozen strict-speech intervals only
- 200-ms guard at each segment boundary
- Minimum 3 seconds of guarded strict speech
- 16-kHz DC-removed analysis waveform
- Native sample rate, Nyquist, codec, and bandwidth-limitation flag retained
- 40-ms frames, 10-ms hop, Hann window, 2048-point FFT
- Linear-power averaging before logarithmic conversion
- Power normalization over 100–7500 Hz
- Fixed one-third-octave LTAS representation
- No pre-emphasis, equalization, peak normalization, or compression

Reference members must have native Nyquist support through 7.5 kHz. A target
with lower source bandwidth is still measured and explicitly flagged because
bandwidth limitation is part of the QCHAN construct. Upsampling is never
treated as restoring missing frequencies.

## Reference identity

The following are frozen as measurement identity:

- QCHAN measurement and parameter version
- Reference task stratum
- Excluded target subject
- Reference recording and subject membership
- Reference spectrum hash
- Cohort reference-vintage hash

QCHAN values are comparable only within the same frozen reference vintage.
Because the features are cohort-relative, a channel effect shared across the
cohort is not identifiable as a cohort-relative abnormality.

## Required validation

- Uniform-gain and polarity invariance
- Identity condition
- Low-pass and high-shelf dose response
- Two-sided spectral-coloration response
- Broad-notch response
- Additive-noise and reverberation discriminant characterization
- Common-mode reference limitation
- Native/source-bandwidth audit
- Logarithmic-floor sensitivity
- Lossless serialization stability
- Leave-one-reference-subject robustness
- Subject bootstrap
- Recording-weighted-reference comparison
- Boundary-guard, frame-length, and whole-segment deletion robustness
- Empirical availability and within-family redundancy
- Label-blind reviewer gallery

Clinical variables and human quality annotations are prohibited from feature
construction, reference membership, parameter selection, and validation
gates. Phenotype-confounding analyses belong downstream and cannot be used to
retune QCHAN.

## Retired legacy items

Raw centroid, flatness, and low/mid-band ratios are audit-only because they are
strongly affected by speech content and additive noise. Absolute tilt
deviation is removed because it overlaps LTAS distance. Recording-weighted
references and cross-task/global fallback references are prohibited.

