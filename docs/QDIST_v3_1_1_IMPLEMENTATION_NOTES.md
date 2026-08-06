# QDIST v3.1.1 — corrected candidate implementation

## Purpose

QDIST v3.1.1 measures **native-waveform evidence compatible with hard clipping or saturation**. It is not a comprehensive nonlinear-distortion estimator. Soft clipping, compression, limiting, AGC, total harmonic distortion, codec distortion, and perceptual distortion remain outside the claim.

The retained outputs are three reconstructable views of one accepted event system:

- `qdist_hard_clipped_frame_fraction`
- `qdist_hard_clip_event_rate_per_min`
- `qdist_hard_clipped_sample_fraction`

Near-full-scale occupancy and histogram-edge concentration remain diagnostics only.

## Corrections from v3.1.0

1. **Low-level false-positive path removed.** Candidate acceptance now requires both recording-relative amplitude plausibility and local-context prominence. Local prominence alone can no longer accept near-zero repeated sample codes.
2. **Candidate generation is prefiltered.** Only flat runs above the recording-relative amplitude floor are materialized. All-amplitude flat-run counts remain available as diagnostics, but the cohort no longer creates a million-row low-level candidate ledger.
3. **Recording amplitude floor recalibrated.** The default floor is 0.45 of the channel robust peak, not 0.85. This rejects near-zero quantization while retaining hard clipping occurring in a lower local gain state. The notebook evaluates 0.40 and 0.50 alternatives.
4. **Sparse-burden robustness corrected.** Relative changes remain reported, but the blocking sample-burden criterion uses absolute change in clipped milliseconds per analyzed minute. Relative percentages are unstable when the baseline is nearly zero.
5. **G8 decoupled from G6.** G8 now evaluates exact ledger reconstruction, related-view status, and merge-gap stability. An unrelated detector-threshold failure cannot automatically fail G8.
6. **Complete event review.** Every accepted real plateau is included in the blinded review package, together with near-threshold rejected candidates and valid-zero controls.
7. **Synthetic truth corrected.** Injection windows are selected independently of detector output to ensure every dose-grid cell contains nonzero known hard clipping. Sample precision, recall, F1, and burden error are now calculated against exact truth masks.
8. **Output isolation.** The measurement version and filenames are `qdist-v3.1.1` / `qdist_v311_*`, so stale v3.1.0 artifacts cannot mix with the corrected run.

## Runtime and storage redesign

- Native audio is decoded once per recording during main extraction.
- A single compressed checkpoint bundle is written per recording rather than several parquet files.
- Checkpoints are reused only when the implementation, media decoder, parameters, and frozen inputs have the same signature.
- Low-level flat runs are counted without creating candidate rows.
- Edge evidence uses merged local windows rather than a full-record Boolean mask for each level cluster.
- Empirical robustness runs on all positive recordings plus a fixed label-blind sample of 40 valid-zero recordings; each selected waveform is decoded once and reused across parameter variants.
- Gallery construction decodes each selected recording once and renders all of its review items.

These changes should make the corrected run substantially faster and reduce output size. Exact wall-clock improvement depends on media codec, disk speed, CPU, and the number of real high-amplitude candidates.

## Scientific acceptance logic

A plateau is accepted only when all required components pass:

- quantization-aware flat morphology;
- duration bounds;
- recording-relative magnitude floor;
- local-context prominence;
- bilateral active context;
- directionally compatible entry and exit transitions;
- sufficient polarity-specific level-cluster support;
- local edge-zone concentration relative to interior shells;
- negligible support beyond the proposed limiting edge;
- coarse-quantization guard;
- square-like/two-level ambiguity guard.

No weighted score permits one strong component to compensate for failure of a required component.

## Validation status at delivery

The corrected source, notebook, generator, media decoder, and focused tests were checked locally. The QDIST/media test set passes. Synthetic construct-recovery conditions contain nonzero known truth and meet the notebook's prespecified sample-level precision, recall, F1, monotonicity, and rank-order gates.

The family is still a **candidate** until the user runs the full frozen cohort and completes the versioned blinded event review. The notebook intentionally defaults to:

```python
PUBLISH_AND_FREEZE_QDIST_V311 = False
QDIST_REVIEW_DECISION = "PENDING"
```


## Final two-tier amplitude rule

Candidate generation uses a permissive recording-relative floor of 0.25. Final
acceptance requires either (a) a strong recording-edge ratio of at least 0.45,
or (b) a lower-level repeated-saturation path with at least four clustered
plateau candidates, 24 plateau samples, and 24 edge-zone samples. Both paths
also require local prominence and all morphology, context, transition,
terminality, quantization, and square-wave guards. This rule preserves
sensitivity to severe clipping in a lower gain state while rejecting isolated
quantized natural extrema.
