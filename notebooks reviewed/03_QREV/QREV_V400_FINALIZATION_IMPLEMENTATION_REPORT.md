# QREV v4.0.0 — Finalization implementation report

The finalization package performs no audio decoding and does not recompute the four QREV analysis features. It verifies the completed 519-recording candidate, preserves the numerical values exactly, resolves G10 roles, replaces the invalid shorter-horizon sensitivity rows using saved boundary first-return/censor evidence, updates support terminology, regenerates four scientifically corrected figures, creates feature passports, completes the 50-item checklist, and prepares atomic measurement and figure freezes.

The finalization notebook requires the acceptance token `ACCEPT_QREV_V400`. The measurement freeze script refuses to overwrite an existing freeze and seals the executed notebook, artifact inventory and hashes. The figure-freeze script verifies all 22 applicable figures plus explicit Panel I N/A and seals the final workbook and provenance documents.
