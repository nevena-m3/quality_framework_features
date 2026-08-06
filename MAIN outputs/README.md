# Main Outputs

This directory contains the pipeline's authoritative and reviewed deliverables. Generated contents are excluded from Git; this index documents their expected organization.

```text
00_DATA_FREEZE/                 Immutable input-data freeze
01_SEGMENTATION_FREEZE/         Approved segmentation freeze
02_FEATURE_FAMILY_SNAPSHOTS/   Feature-family snapshots
02_FEATURE_FREEZE/              Approved feature freeze
02_FEATURE_TABLES/              Main feature tables
02_FEATURE_TABLES_EXPLORATORY/ Exploratory feature tables
02_FEATURE_LATEST/             Compact latest reviewed release (start here)
02_FEATURE_REVIEWED/            Reviewed feature workflow and deliverables
```

For normal use, open `02_FEATURE_LATEST/`. It contains one merged recording-level
table, one normalized registry, and exactly one numbered folder per feature family.
The larger `02_FEATURE_REVIEWED/` tree is the scientific evidence and provenance
area; it is not the main analysis input.

`02_FEATURE_REVIEWED/` is organized as follows:

```text
00_working_candidates/         Review-stage outputs, grouped by feature family
00_feature_registry/           Reviewed feature registry
01_analysis_features/          Reviewed analysis features
02_support_and_availability/   Support and availability artifacts
03_event_summaries/            Reviewed event summaries
04_model_ready_features/       Model-ready reviewed features
05_feature_passports/          Feature passports
06_family_freezes/             Approved family freezes
07_figure_packages/            Approved figure packages
08_validation_workbooks/       Validation workbooks
```

Only `00_working_candidates/` is mutable review-stage workspace. Treat the approved freeze and package directories as immutable published artifacts.
