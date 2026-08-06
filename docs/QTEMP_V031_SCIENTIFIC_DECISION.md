# QTEMP scientific disposition for finalization

## Retained dropout detector

The bracketed dropout-like detector passed synthetic and participant-disjoint real-speech recovery for exact-zero and constant-low-information gaps, rejected attenuated speech, preserved native-waveform ordering, and yielded a small set of cohort positives suitable for blinded review.

## Retained decoded-repetition detector

The retained observable is not general “frozen audio.” It is near-exact consecutive decoded-waveform repetition with:
- non-silent support;
- waveform and spectral similarity;
- periodicity/low-entropy guards;
- accepted target duration of at least 40 ms.

The 40 ms lower bound follows the held-out real-speech validation: 20 ms recovery was insufficient, whereas 40–160 ms recovery met the retained-scope requirement. The cohort contained no accepted events even under the more permissive development scan, so applying the stricter final scope preserves the measured-zero result. These features remain valid prevalence/absence measurements but are excluded from continuous standardization, PCA, correlation, or family-score aggregation when their IQR is zero.

## Dropped splice detector

The splice-like detector is excluded from the final feature registry because:
- held-out source-replacement boundary recovery was inadequate;
- smooth deletion joins were generally not identifiable;
- cohort positivity was implausibly high;
- the event rate was strongly concentrated by native sample rate;
- many accepted events had one-sample support and could not be distinguished confidently from ordinary speech transients without a reference.

The dropped feature is not exported to the final analysis table or downstream manuscripts.

## Irreducible final step

Blinded adjudication is mandatory. Code can generate a blinded review package and calculate prespecified metrics, but it cannot replace the human perceptual/event verification required by G9.

## r2 implementation correction

No scientific feature decision changed. The r2 notebook fixes governance and
execution defects that incorrectly displayed G1 as failed. The splice-like
feature remains formally dropped; the four retained features and their bounded
claim scope are unchanged.
