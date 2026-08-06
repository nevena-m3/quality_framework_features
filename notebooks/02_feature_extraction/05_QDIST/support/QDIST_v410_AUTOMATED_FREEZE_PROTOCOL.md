# QDIST v4.1 automated reviewer-free freeze protocol

1. Verify all 923 candidate-cohort artifacts against the completed source manifest with zero exclusions.
2. Verify the reviewer-free computational-verification revision and its scientific scope.
3. Recreate all 144 exact hard-limit interventions from the frozen cohort inputs.
4. Derive reference sample masks, reference 20-ms-merged episodes, and reference 30-ms frame occupancy.
5. Re-run the unchanged QDIST v4.1 detector and quantify sample, event, and frame recovery.
6. Apply prespecified gates. A failed primary sample or occurrence gate blocks freezing. A failed event gate automatically demotes event rate to audit-only.
7. Generate the final registry, analysis table, model interface, feature passports, checklist, arbitration contract, manuscript wording, figures, and manifests.
8. Execute the finalization notebook headlessly and verify its completion marker.
9. Re-run governed tests immediately before sealing.
10. Copy selected candidate evidence and all finalization evidence into a non-overwriting staging directory, hash every retained artifact, and atomically rename the staging directory to `qdist-v4.1.0`.
11. Publish versioned canonical registry, analysis, model-interface, and passport artifacts without overwriting existing files.

No listening, subjective visual scoring, reviewer form, adjudication, or manual feature decision is used.

