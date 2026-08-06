# QDIST v4.0.0 Stage 1 Preflight Implementation Report

The preflight patch adds a reviewed wrapper, 26 focused tests, a six-code-cell notebook, the 50-item checklist, A-J figure plan, provisional feature decisions, and guarded packaging/install scripts.

The wrapper imports the already-frozen paper1_qc.qdist implementation and refuses to run when its version, feature registry, or required API differs from qdist-v3.1.1. It does not copy or modify the detector.

Panels A-C are generated with complete PNG/SVG/PDF/source/caption/provenance bundles. Cohort extraction and freezing are disabled. Panel I is explicitly applicable but pending because QDIST has a retained event detector.
