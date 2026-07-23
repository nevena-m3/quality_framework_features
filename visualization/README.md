# Visualization and audit notebooks

This folder is the paper-facing audit layer. It deliberately contains substantial
in-notebook code for loading saved stage outputs, checking denominators, constructing
tables, plotting intermediate states, and exporting candidate figures. Reusable signal
processing and statistical algorithms remain in `src/paper1_qc` and are tested there.

Run the notebooks in order:

1. `00_preflight_and_run_order.ipynb`
2. `01_segmentation_visual_audit.ipynb`
3. `02_goal1_occurrence_and_acquisition_variability.ipynb`
4. `03_goal2_participant_persistence.ipynb`
5. `04_goal3_multidimensional_structure_and_robustness.ipynb`
6. `05_goal4_perceptual_family_alignment.ipynb`
7. `06_results_registry_and_manuscript_tables.ipynb`

Each notebook has a `RUN_... = False` gate. On the first local run, change that gate to
`True` to execute the required pipeline command(s). On later reporting runs, leave it
`False` to visualize the frozen saved outputs without recomputing the analysis.

All tables are saved as CSV and all figures as both PNG and SVG under
`outputs/visualization/<notebook-stage>/`. Notebook outputs are not the source of truth;
the saved tables, figures, run manifests, configuration hash, and input hashes are.

## Four study goals used here

The original manuscript stated three broad goals. The visualization layer separates the
first broad goal into two estimands so the repeated-recording design is not obscured:

1. occurrence, support, distributions, and acquisition variability;
2. participant persistence across repeated recordings (not test-retest reliability);
3. multidimensional structure and robustness, including exact-session Rest and
   WAV/WEBM/segmentation sensitivities;
4. perceptual family alignment using (a) the distributed detailed annotations, (b) the
   crossed four-RA reliability subset, and (c) the separate merged 2RA metadata labels.

This is an analysis/reporting split, not permission to change hypotheses after inspecting
results. Freeze the Statistical Analysis Plan and category/direction mappings first.

## Goal 4 gate

The schema expects the actual folder design:

```text
Bamboo_passage_HumanQC/
  Abbas/
  Liya/
  Samaana/
  Samara/
  Reliability/
    Abbas/
    Liya/
    Samaana/
    Samara/
```

The four top-level RA folders contain the distributed main annotations: approximately
170–173 different files per RA and one independent rating per recording. Their primary
family-alignment effect is estimated within rater and then combined, so rater thresholds
are not confused with Q alignment. Inter-rater agreement is not estimable from this
distributed portion.

The `Reliability` subfolders must contain the same approximately 70 recordings
independently rated by all four RAs. Only complete four-rater recording/family items enter
the primary Gwet AC1, Fleiss kappa, and consensus analysis. Incomplete items are reported
and excluded; a three-of-four consensus is saved only as sensitivity output.

The following are perceptual Q-family targets:

| Human annotation | Objective family |
|---|---|
| Environmental noise | additive interference |
| Volume unstable | gain dynamics |
| Reverberation/echo | reverberation tail |
| Platform effects | channel/device |
| Clipping | nonlinear distortion |
| Temporal discontinuities | temporal discontinuity |

`Any non-task related content` and `Competing speech` are retained as contextual audit
variables but are excluded from matched-family validation.

The merged 2RA metadata comparison is valid only for overlapping explicit families
(`Background Noise` and `Volume is Unstable`). `Poor Audio Quality` is multidimensional
and is not treated as a family label. Confirm in the RA codebook that `Yes` means artifact
present and `No` means artifact absent before setting
`broad_metadata.direction_confirmed: true`.
