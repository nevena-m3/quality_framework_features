# QTEMP v0.3 scientific contract

QTEMP measures observable decoded-stream continuity violations on native decoded channels before resampling, mono conversion, normalization, denoising, interpolation, filtering, or codec re-encoding.

## Claim boundaries

- Dropout-like features measure accepted bracketed near-zero or constant-low-information runs. They are not packet-loss fractions.
- Repetition features measure accepted near-exact consecutive decoded waveform repetition. They do not detect every freeze or packet-loss concealment strategy.
- The splice-like feature measures strong localized bilateral prediction/context mismatch after speech-boundary, QADD impulse, QGAIN persistent-level, QDIST clipping-edge, and within-QTEMP exclusions. Smooth or phase-compatible deletion joins may be unidentifiable in no-reference audio.

## Required evidence

Each retained feature requires numerical correctness, exact ledger reconstruction, synthetic recovery, realistic null controls, participant-disjoint real-speech injection, signal-chain characterization, support and availability auditing, parameter sensitivity, empirical rare-event characterization, and blinded adjudication. Rate and duration views from one event ledger are complementary summaries, not independent evidence.

## Freeze rule

v0.3 is development-only. It cannot write to the immutable family freeze or central manuscript feature table. After G1–G9 pass and every feature has an explicit retain/secondary/audit-only/drop decision, create a new immutable freeze version rather than converting v0.3 in place.
