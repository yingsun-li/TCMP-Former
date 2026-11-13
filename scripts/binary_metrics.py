# Compute binary classification metrics from saved predictions.
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)




def load_oof(prediction_dir: Path) -> pd.DataFrame:
    paths = sorted(prediction_dir.glob("binary_fold*/test_predictions.csv"))
    if len(paths) != 5:
        raise RuntimeError(f"Expected five test folds for {prediction_dir}, found {len(paths)}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        fold = int(path.parent.name.removeprefix("binary_fold"))
        if "fold" in frame.columns:
            saved_folds = set(frame["fold"].dropna().astype(int).unique())
            if saved_folds != {fold}:
                raise RuntimeError(
                    f"Fold column mismatch in {path}: expected {fold}, found {saved_folds}"
                )
        else:
            frame.insert(0, "fold", fold)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    required = {"target", "prediction", "probability_1_1_SOP"}
    missing = required.difference(result.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {prediction_dir}: {sorted(missing)}")
    return result


def metrics(name: str, frame: pd.DataFrame) -> dict[str, object]:
    y = frame["target"].astype(int).to_numpy()
    pred = frame["prediction"].astype(int).to_numpy()
    prob = frame["probability_1_1_SOP"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "oof_samples": len(frame),
        "patients": frame["patient_id"].nunique() if "patient_id" in frame else None,
        "events": frame["event_id"].nunique() if "event_id" in frame else None,
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "auroc": roc_auc_score(y, prob),
        "auprc": average_precision_score(y, prob),
        "sensitivity": recall_score(y, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "precision_sop": precision_score(y, pred, pos_label=1, zero_division=0),
        "f1_sop": f1_score(y, pred, pos_label=1, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--prediction-dir", type=Path, default=root/"data"/"predictions"/"binary"/"background_reference")
    parser.add_argument("--model-name", default="TCMP-Former")
    parser.add_argument("--output-dir", type=Path, default=root/"outputs"/"metrics")
    args = parser.parse_args()

    rows = []
    contracts = {}
    frame = load_oof(args.prediction_dir.resolve())
    rows.append(metrics(args.model_name, frame))
    contracts[args.model_name] = str(args.prediction_dir.resolve())

    args.output_dir.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "binary_basic_oof_metrics.csv", index=False)
    with (args.output_dir / "analysis_contract.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold_aggregation": "pooled five-fold test OOF predictions",
                "decision_rule": "saved model class prediction; no test threshold tuning",
                "probability_metrics": "AUROC and AUPRC use saved SOP probability",
                "excluded": [
                    "alarm confirmation",
                    "time-to-detection endpoints",
                    "patient-equal temporal metrics",
                    "pre-onset positive-sample fractions",
                ],
                "experiments": contracts,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
