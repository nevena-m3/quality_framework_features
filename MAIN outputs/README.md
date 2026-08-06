# MAIN outputs

This directory is reserved for the small set of investigator-approved, paper-facing
deliverables. Routine intermediate files remain under `outputs/`.

The first approved output is `00_DATA_FREEZE/`, created by:

```powershell
paper1-qc --config config/project.yaml freeze
```

That directory contains the frozen recording tables, inclusion/exclusion ledger,
diagnosis provenance, freeze summary, and a hash-bearing run manifest. Its generated
contents contain participant identifiers and are ignored by Git. Do not copy arbitrary
exploratory figures or temporary tables here.

The second approved output is the versioned `01_SEGMENTATION_FREEZE/`, created only
after the interactive all-recording Silero review. Its companion publication tree is
`outputs/01_segmentation_after_review/`: accepted and flagged recordings are retained
for downstream measurement; excluded recordings remain present there for audit but are
not analysis-eligible.
