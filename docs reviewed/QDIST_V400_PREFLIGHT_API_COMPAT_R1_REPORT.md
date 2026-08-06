# QDIST v4.0.0 Preflight API-Compatibility Hotfix R1

## Failure diagnosed

The original reviewed tests treated three synthetic validation helpers as mandatory exports of the frozen `paper1_qc.qdist` production module: `synthetic_speech_like`, `hard_clip`, and `soft_clip_tanh`. The locally installed frozen detector does not expose those optional notebook/test helpers. This caused 18 `AttributeError` failures before detector validation could run. Two additional notebook tests used brittle literal-source checks for quote style and panel-stem placement.

## Correction

R1 keeps the frozen qdist-v3.1.1 detector unchanged and moves all synthetic fixture generation into the reviewed validation layer. The wrapper now requires only the production detector interface, adapts both supported `quantize_pcm` return signatures, accepts either `.summary` or `.recording` result mappings, and reads the frozen `frame_length_ms` parameter correctly. The time-translation fixture is frame-aligned and non-wrapping. Notebook governance is verified semantically with Python AST parsing, and Panels A-C are explicitly declared in the notebook.

## Scientific impact

No detector threshold, morphology rule, episode merge rule, feature definition, frozen qdist-v3.1.1 artifact, or cohort value is changed. This is a reviewed-layer API and governance correction only. Cohort extraction and freezing remain disabled.

## Quantization-control correction

The G5 clean-quantization guard is evaluated only on clean 8-, 10-, and 12-bit fixtures. Intentionally hard-clipped low-bit-depth fixtures are retained as construct characterizations and are not incorrectly required to be negative. This preserves the distinction between quantization-induced false plateaus and genuine imposed clipping on a coarse lattice.
