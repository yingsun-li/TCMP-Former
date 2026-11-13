# TCMP-Former: evaluation and figure code

Preparation copy for partial code release. This package contains OOF metric calculation, patient-paired bootstrap analysis of interictal comparisons, and plotting from prepared summaries. It does not contain model implementation, training, weights, or prediction data. It does not reproduce model training.

## Setup

Use Python 3.10 or newer. From this package directory:

```bash
python -m pip install -r requirements.txt
python scripts/binary_metrics.py --help
python scripts/patient_bootstrap.py --help
python scripts/render_figures.py --help
```

## Run with your inputs

```bash
python scripts/binary_metrics.py
python scripts/patient_bootstrap.py
python scripts/render_figures.py
python scripts/render_figures.py --final
```

Defaults are resolved from the package directory, regardless of the current working directory. All tools accept explicit input and output arguments. Relative command-line paths resolve from the caller's working directory. Outputs stay under `outputs/` by default. Figure previews are written before final exports.

See `data/README.md` for input layout. No real data are bundled in this preparation copy. The interictal comparison uses 10,000 patient-paired bootstrap resamples and percentile 95% confidence intervals. Its input consists of paired patient-level positive decision-output proportions. Plot labels retain historical figure identifiers and should be checked against final manuscript numbering before upload.

Plotting reads frozen summaries and does not recompute model predictions or statistical intervals. Binary metrics use saved class predictions and SOP probabilities. The bootstrap resamples both conditions together and estimates equally weighted patient means. It does not calculate classification-metric intervals.

Binary metrics read saved predictions for the background reference condition. Input schema and condition names are defined in `data/README.md`.
