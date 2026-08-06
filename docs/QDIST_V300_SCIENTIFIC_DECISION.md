# QDIST v3.0.0 scientific decision record

## Final candidate construct

QDIST v3.0.0 measures native-waveform evidence compatible with **hard clipping
or saturation**. It does not claim to measure the complete nonlinear-distortion
term in the observation model. In particular, it does not provide a retained
measure of smooth soft clipping, limiting, dynamic-range compression, total
harmonic distortion, automatic gain control, or the physical acquisition stage
at which saturation occurred.

The authoritative input is the decoded native multichannel waveform. Detection
occurs before resampling, amplitude normalization, filtering, denoising,
interpolation, DC removal, channel averaging, or codec re-encoding. The
analyzed exposure is the continuous natural task span from the first frozen
speech onset to the final frozen speech offset, with internal pauses preserved.
This avoids diluting burden with arbitrary leading/trailing idle time and avoids
creating artificial joins by concatenating strict-speech intervals.

## Retained candidate features

All three candidate features are exact views of one accepted-event ledger:

1. `qdist_hard_clipped_frame_fraction` — fraction of complete 30-ms native
   task-span frames intersecting at least one accepted plateau on any channel.
   This is the primary time-prevalence view.
2. `qdist_hard_clip_event_rate_per_min` — merged accepted clipping episodes per
   minute of finite native task-span exposure. Exact event counts, exposure,
   Poisson intervals, and the 20-ms plateau-merging rule are retained. This is
   the primary event-frequency view and the highest-risk candidate.
3. `qdist_hard_clipped_sample_fraction` — accepted plateau channel-samples
   divided by all finite eligible channel-samples. This is a secondary burden
   view.

The features are related views of one detector, not three independent pieces of
evidence for separate nonlinear mechanisms. No QDIST scalar score is created.

## Detector contract

A plateau is accepted only when it jointly satisfies:

- candidate-level adaptive high-tail amplitude-distribution evidence evaluated in deterministic fixed 250-ms and 1,000-ms context windows;
- a short, near-flat time-domain plateau under a tight scale-relative tolerance;
- sufficient consecutive flat support;
- limited within-plateau range;
- sharp entry and exit morphology;
- local-extremum consistency within a frozen 5-ms neighborhood;
- minimum edge-zone occupancy and edge-excess evidence beyond adjacent histogram shells;
- the frozen maximum-duration rule.

Candidate-level rather than global-maximum anchoring permits detection below
final digital full scale and permits more than one clipping level within a
recording. Fixed context windows are placed on a deterministic 50%-overlap grid;
each occupied window is sorted once, and exact shell counts are recovered by
binary search. This makes runtime bounded without stochastic subsampling or
truncating accepted events. Near-full-scale occupancy and histogram spikes alone remain audit
components because high headroom use is not proof of nonlinear saturation.

Accepted plateaus are merged into episodes using the frozen 20-ms gap. The
candidate, accepted-plateau, and episode ledgers retain native sample indices,
time, channel, polarity, edge level, histogram evidence, plateau morphology,
rejection reasons, and the parameter hash. Recording-level outputs must be
exactly reconstructable from the ledgers.

## Rarity and valid zeros

Hard clipping is expected to be sparse. Rarity is not a reason to loosen the
detector, tune thresholds to cohort prevalence, or coerce missing values to
zero.

- A detector-available recording with no accepted event has a valid value of
  zero for all three features.
- Event-rate uncertainty is represented with the exact event count, exposure,
  support tier, and exact Poisson interval.
- The number of positive recordings determines downstream statistical
  suitability only: continuous positive-part/two-part modeling, binary and
  descriptive analysis, or descriptive-only reporting.
- Analytical correctness is judged independently of whether QDIST predicts
  diagnosis, ALS severity, or a human-QC label.

## Coarse quantization

Very coarse regular amplitude lattices can create repeated values and short
plateaus throughout an unclipped waveform. QDIST v3.0.0 therefore performs a
conservative native-channel quantization audit. When any channel occupies at
most 4096 sampled levels, at least 90% of adjacent level gaps are compatible
with a regular lattice, and at least 0.2% of adjacent sampled values repeat, the detector fails closed with
`unavailable_coarse_quantization`.

For such recordings:

- retained QDIST features remain missing, not zero;
- event and plateau counts remain missing;
- detector availability is false;
- channel-level quantization diagnostics remain auditable;
- no candidate enumeration is performed because candidate morphology is not
  interpretable under that input condition.

Digital zero remains a valid no-event waveform and is not misclassified as
coarse quantization.

The adjacent-repeat requirement prevents deterministic analytic waveforms with a
regular set of levels but no quantization plateaus from being incorrectly marked
unavailable. The 4096-level ceiling conservatively treats effectively 12-bit or
coarser native amplitude lattices as unsuitable for plateau-based clipping
measurement.

## Soft clipping decision

No soft-clipping feature is retained. In no-reference connected speech, smooth
waveform curvature cannot be uniquely separated from speech-source shape,
vocal effort, microphone response, compression, limiting, or codec processing.
Adding a soft-clipping feature would therefore overstate identifiability.

Instead, the notebook characterizes the hard-clipping detector under tanh
saturation doses and smooth dynamic compression. Moderate and severe smooth
saturation controls are expected not to trigger the hard-plateau detector. Any
response under extreme plateau-like saturation is reported as cross-sensitivity,
not promoted to a new feature. This explicitly documents the family scope:
QDIST under-covers soft limiting and dynamic-range compression by design.

## Analytical validation

The governed notebook implements the common G1–G10 architecture:

- exact native-input and preprocessing provenance;
- hand-computable numerical tests and exact ledger reconstruction;
- polarity, time-shift, non-saturating gain, and post-clipping attenuation
  behavior;
- hard-clipping dose grids over deterministic speech-like signals;
- sample-level and event-level precision, recall, F1, and count error;
- unclipped speech, tones, triangle/saw/square-like signals, impulses, music,
  percussive signals, DC offset, coarse quantization, smooth compression, and
  soft-saturation controls;
- native-versus-resampled and Opus/AAC codec characterization;
- threshold, context, frame, plateau, and episode-merging sensitivity;
- complete cohort extraction with restart-safe hash-bound checkpoints;
- empirical sparsity, participant clustering, repeated-recording behavior,
  exact reconstructability, and feature redundancy;
- a fixed blinded event/non-event gallery with synthetic positive and negative
  controls and real cases when available;
- independent pass/drop decisions for each retained candidate before freeze.

The synthetic validation package passed clean-signal specificity, monotonic
frame and sample burden, median sample precision above 0.99, median sample
recall above 0.70, median eligible-event precision of 0.90, and median
eligible-event recall of 1.00. Lossless floating-point WAV round-trip values
were identical. Resampling and lossy codec encoding removed plateau evidence in
the controlled example, confirming that native-waveform inspection is
mandatory and that pre-codec clipping may be invisible after codec smearing.

## Freeze decision rule

QDIST v3.0.0 remains a candidate until the real cohort run, event-merging
robustness, empirical sparsity audit, and blinded event review are complete.
Each candidate receives a separate final decision:

- `PASS_PRIMARY`, `PASS_PRIMARY_EVENT`, or `PASS_SECONDARY`;
- or `DROP`.

A failed feature cannot be rescued by another feature or by downstream clinical
associations. At least one primary feature must pass for the family to freeze.
