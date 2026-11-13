# Estimate patient-paired bootstrap intervals for positive decision-output proportions.
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

REPLICATES = 10000
CONDITIONS = ['seizure_adjacent', 'interictal']
LABELS = ['Seizure-adjacent', 'Matched interictal']


def load_patient_rates(path):
    frame = pd.read_csv(path, dtype={'patient_id': str})
    required = ['patient_id', *CONDITIONS]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f'Missing input columns: {sorted(missing)}')
    frame = frame[required].copy()
    if frame.isna().any().any() or len(frame) < 2:
        raise ValueError('At least two complete patient pairs are required')
    if frame['patient_id'].str.strip().eq('').any() or frame['patient_id'].duplicated().any():
        raise ValueError('Each patient must have one unique, nonempty identifier')
    values = frame[CONDITIONS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Patient proportions must be finite values between 0 and 1')
    return frame


def paired_bootstrap(frame, seed):
    values = frame[CONDITIONS].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(REPLICATES, len(values)))
    estimates = values[indices].mean(axis=1)
    lower, upper = np.quantile(estimates, [0.025, 0.975], axis=0)
    return pd.DataFrame({
        'condition': LABELS,
        'mean': values.mean(axis=0),
        'ci_lower': lower,
        'ci_upper': upper,
        'patients': len(values),
    })


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='Patient-paired bootstrap for interictal comparisons.')
    parser.add_argument('--input-csv', type=Path, default=root/'data'/'predictions'/'interictal'/'patient_rates.csv')
    parser.add_argument('--output-dir', type=Path, default=root/'outputs'/'bootstrap')
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    frame = load_patient_rates(args.input_csv)
    summary = paired_bootstrap(frame, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary.to_csv(args.output_dir/'interictal_summary.csv', index=False)
    frame.to_csv(args.output_dir/'patient_pairs.csv', index=False)
    method = {
        'cluster': 'patient',
        'resampling': 'patient pairs with replacement; both conditions use the same patient draw',
        'replicates': REPLICATES,
        'seed': args.seed,
        'estimator': 'equally weighted mean of patient-level proportions',
        'interval': 'percentile 2.5% to 97.5%',
    }
    (args.output_dir/'method.json').write_text(json.dumps(method, indent=2)+'\n')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
