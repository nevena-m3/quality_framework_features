# QCHAN v4.0.0 reviewed scientific contract

## Measurement question

How far does the long-term spectrum of guarded strict speech deviate from a frozen, task-matched, subject-balanced leave-one-subject-out reference, and is the deviation compatible with reduced upper-frequency support or other spectral coloration?

## Required input

- deterministic mono 16-kHz analysis waveform;
- no amplitude normalization, spectral equalization, denoising or bandwidth restoration;
- frozen `strict_speech / primary` intervals;
- native sample rate, Nyquist, channel geometry and media provenance retained;
- minimum 3.0 s guarded strict speech and 100 valid frames.

## Reference contract

For each target recording, restrict reference candidates to the same task stratum and exclude every recording from the target subject. First take the median recording spectrum within each reference subject, then the median across subjects. Require at least five other subjects and eight recordings. Otherwise all reference-relative QCHAN features are unavailable. There is no global or cross-task fallback.

The reference member list, task, parameters, spectral floor, frequency grid and SHA-256 vintage are part of feature identity. Values from different reference vintages are not assumed interchangeable.

## Feature contract

### qchan_ltas_distance_db
RMS difference over 100–7500 Hz between gain-anchored log-LTAS and the target-specific reference. Unit: dB RMS. Higher means more deviation, not necessarily worse.

### qchan_rolloff95_deficit_hz
`max(0, reference rolloff95 - observed rolloff95)`. Unit: Hz. Signed difference is retained in audit fields.

### qchan_highband_ratio_deficit
`max(0, reference high-band ratio - observed high-band ratio)` for 3–7.5 kHz relative to 100–7.5 kHz. Unit: proportion. Signed difference is retained.

### qchan_tilt_steepening_db_per_oct
One-sided steepening of the 100–4000-Hz log-spectral slope relative to reference. Unit: dB/octave. Signed difference is retained.

## Allowed claims

- QCHAN summarizes cohort-relative speech-spectrum deviation and upper-band attenuation proxies.
- Controlled low-pass filtering, shelves, notches, source bandwidth and codecs can move QCHAN features.
- LTAS distance is nonordinal; rolloff/high-band/tilt features are one-sided by design.
- Support, native bandwidth, task stratum and reference vintage accompany every value.

## Forbidden claims

- device, microphone, browser, codec or platform identification;
- pure microphone transfer-function estimation;
- proof that an observed value is an acquisition artifact rather than phenotype or phonetic composition;
- a universal QCHAN scalar or generic recording rejection threshold;
- treating an unavailable feature as zero;
- treating a one-sided zero as no spectral deviation.

## Governance

Human-QC labels, diagnosis, ALSFRS and downstream prediction are excluded from analytical validation. Any change to formulas, parameters, reference membership rules, floor, feature roles, schema or claims requires a new measurement version.
