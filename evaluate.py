"""Evaluate predictions against a separately supplied, approved label CSV."""

import argparse
from pathlib import Path

from paths import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=OUTPUT_DIR / "evaluation.csv")
    args = parser.parse_args()

    import pandas as pd
    from evaluation.binary_evaluation import binary_eval
    from llm_utils.constants import labels

    ground = pd.read_csv(args.ground_truth, dtype={"Linking_ID": str})
    predicted = pd.read_csv(args.predictions, dtype={"Linking_ID": str})
    for name, frame in (("ground truth", ground), ("predictions", predicted)):
        missing = set(["Linking_ID", *labels]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing {sorted(missing)}")
        if frame["Linking_ID"].isna().any():
            raise ValueError(f"{name} has missing Linking_ID values")
        if frame["Linking_ID"].duplicated().any():
            raise ValueError(f"{name} has duplicate Linking_ID values")
    paired = ground[["Linking_ID", *labels]].merge(
        predicted[["Linking_ID", *labels]], on="Linking_ID",
        how="inner", validate="one_to_one", suffixes=("_true", "_pred"))
    if len(paired) != len(ground) or len(paired) != len(predicted):
        raise ValueError("Ground truth and predictions must contain the same IDs")
    true = paired[[f"{label}_true" for label in labels]].set_axis(labels, axis=1)
    pred = paired[[f"{label}_pred" for label in labels]].set_axis(labels, axis=1)
    metrics = binary_eval(true, pred, col=labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))
    print(f"Saved aggregate metrics to {args.output}")


if __name__ == "__main__":
    main()
