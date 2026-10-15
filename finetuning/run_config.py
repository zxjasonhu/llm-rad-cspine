"""Input and training configuration shared by the LoRA entry points."""

import json
from pathlib import Path

import pandas as pd

from finetuning.sft_settings import DEFAULT_SEED


def load_csvs(paths, required_columns):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, dtype={"Linking_ID": str})
        missing = set(required_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        empty = frame[required_columns].columns[frame[required_columns].isna().any()].tolist()
        if empty:
            raise ValueError(f"{path} has missing values in: {empty}")
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("Input CSV files contain no rows")
    if data["Linking_ID"].isna().any() or data["Linking_ID"].duplicated().any():
        raise ValueError("Linking_ID must be present and unique across input CSV files")
    return data


PAPER_SUBSET_POSITIVES = {20: 3, 100: 11, 500: 55, 800: 88, 1064: 117}


def sample_training_data(data, size=None, seed=DEFAULT_SEED):
    if size is not None and size < 1:
        raise ValueError("--training-dataset-size must be positive")
    if size is None:
        return data.reset_index(drop=True)
    if size > len(data):
        raise ValueError(f"Requested {size} reports, but only {len(data)} are available")
    if size == len(data):
        return data.reset_index(drop=True)
    if size not in PAPER_SUBSET_POSITIVES:
        return data.sample(n=size, random_state=seed).reset_index(drop=True)
    if "Fracture_Case" not in data.columns:
        raise ValueError("Published subset sampling requires Fracture_Case")
    positive = data[data["Fracture_Case"] == 1]
    negative = data[data["Fracture_Case"] == 0]
    positive_count = PAPER_SUBSET_POSITIVES[size]
    negative_count = size - positive_count
    if len(positive) < positive_count or len(negative) < negative_count:
        raise ValueError(f"Cannot sample the published {size}-report class counts from this input")
    return pd.concat([
        positive.sample(n=positive_count, random_state=seed),
        negative.sample(n=negative_count, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_overrides(path):
    if path is None:
        return {}
    settings = json.loads(Path(path).read_text())
    if not isinstance(settings, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return settings
