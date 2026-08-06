# QTEMP revised — experimental waveform anomaly family

This folder is separate from `06_QTEMP`; the original implementation and its analytical disposition remain unchanged.

The revised family directly measures:

- robust sample-to-sample derivative outliers (glitch/pop candidates);
- short-time RMS collapses bracketed by active audio (dropout candidates).

Ordinary leading/trailing silence is excluded by an edge guard. A low-energy run is accepted only when active context exists on both sides, reducing—but not eliminating—confusion with speech pauses and stop closures. Features describe decoded-waveform anomalies and do not prove packet loss, buffering, or a particular acquisition failure.

Run `01_extract_and_review.ipynb` to extract recording features and event timestamps from the governed Bamboo freeze. Outputs are written to `MAIN outputs/03_EXPERIMENTAL/QTEMP_REVISED/` and are not merged into the validated family freezes.
