# Input contract

No original signal, private clinical data, OOF predictions, weights, or patient identifier mappings are included.

## Binary metrics

Default input: `predictions/binary/background_reference/binary_fold{1..5}/test_predictions.csv`.
Required columns: `target`, `prediction`, `probability_1_1_SOP`. Optional columns: `patient_id`, `event_id`, `fold`. Labels are 0 (non-SOP) and 1 (SOP). Exactly five test folds are required.

## Patient-paired bootstrap

Default input: `predictions/interictal/patient_rates.csv`.
One row represents one patient, with columns:

- `patient_id`: a unique nonempty group identifier
- `seizure_adjacent`: the patient's positive decision-output proportion near seizures
- `interictal`: the patient's matched interictal positive decision-output proportion

Both proportions must be finite fractions between 0 and 1. Every patient must have both conditions; incomplete pairs are rejected.
Before preparing this table, positive outputs must be determined with the fixed thresholds and sustained-confirmation rule used for event timing. For each recording, divide positive decision outputs by evaluable outputs. Average matched interictal recordings within events, then average events within patients for both conditions. Do not pool all output counts across patients.

The script draws patient rows with replacement 10,000 times, using the same draws for both conditions. It computes equally weighted patient means and percentile 95% confidence intervals. The `--seed` argument controls bootstrap reproducibility and does not specify the training seed.

Outputs: `interictal_summary.csv`, `patient_pairs.csv`, and `method.json` under the output directory. The summary columns are `condition`, `mean`, `ci_lower`, `ci_upper`, and `patients`. This script starts from patient-level proportions; it does not generate predictions, select thresholds, or apply the sustained-confirmation rule to raw outputs.

## Frozen plot inputs

Place these files in `plot_inputs/`:

- binary_roc.csv
- binary_pr.csv
- binary_summary.json
- timing_bundle.json
- typing_per_class.csv
- typing_folds.csv
- typing_plot_constants.json
- typing_confusion_counts.csv
- interictal_summary.csv

In `timing_bundle.json`, `trajectory.sample_time_s` contains the time coordinates, and `models.sop_comparisons.transformer` contains the Transformer comparison. Field names must match `scripts/plots.py`.

A path-free `manifest.json` for the frozen plot inputs is included. For different inputs, regenerate it with shape `{"inputs": {"filename.csv": {"sha256": "hexadecimal SHA-256"}}}`. List all nine inputs with paths relative to the plot input directory. No original machine paths are needed.

Data sharing and final-paper alignment must be reviewed separately before adding real files to this preparation package. Condition labels describe the retained analysis scope and do not change the underlying predictions.
