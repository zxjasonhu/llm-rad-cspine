"""Generate manuscript tables from the released numeric inputs."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
MODEL_NAMES = {
    "Gemma-3": "gemma-3-4b-it",
    "Llama-3.2": "Llama_3_2-3B-Instruct",
    "Phi-3.5": "phi-3_5-mini-instruct",
    "1B": "gemma-3-1b-it",
    "4B": "gemma-3-4b-it",
    "12B": "gemma-3-12b-it",
    "Gemma-3-1B": "gemma-3-1b-it",
    "Gemma-3-4B": "gemma-3-4b-it",
    "Gemma-3-12B": "gemma-3-12b-it",
    "Gemma-3-27B": "gemma-3-27b-it",
    "Llama-3.2-3B": "Llama_3_2-3B-Instruct",
    "Phi-3.5-Mini": "phi-3_5-mini-instruct",
}
LEVELS = ["Occ_Condyle_Fracture", *[f"C{i}_Fracture" for i in range(1, 8)]]
METRICS = ("macro_f1", "macro_precision", "macro_recall",
           "report_sensitivity", "report_specificity")


def records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def table_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream))


def summary(values: list[float], digits: int = 3) -> str:
    if not values:
        raise ValueError("Missing released numeric values for a generated table cell")
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{mean(values):.{digits}f} ± {stdev(values):.{digits}f}"


def lora_method(label: str) -> str | None:
    lower = label.lower()
    if "lora" not in lower:
        return None
    if "standard" in lower or "(direct)" in lower:
        return "standard"
    if "regularized" in lower or "guided" in lower:
        return "impression_regularized"
    return None


def rows_for(data: list[dict[str, str]], **filters: str) -> list[dict[str, str]]:
    return [row for row in data if all(row[key] == value for key, value in filters.items())]


def full_data_metrics(rows: list[list[str]], table: str,
                      seeds: list[dict[str, str]], counts: list[dict[str, str]]) -> None:
    current_model = ""
    for row in rows[1:]:
        current_model = row[0] or current_model
        method = lora_method(row[1])
        if method is None:
            continue  # No released run-level observations for zero-shot rows.
        selected = rows_for(seeds, model=MODEL_NAMES[current_model],
                            method=method, training_n="1064")
        for column, metric in enumerate(METRICS, 2):
            row[column] = summary([float(entry[metric]) for entry in selected])
        if table in {"I", "J"}:
            for column, level in enumerate(LEVELS, 7):
                selected_counts = rows_for(counts, model=MODEL_NAMES[current_model],
                                           method=method, training_n="1064", category=level)
                row[column] = summary([float(entry["F1"]) for entry in selected_counts])


def temporal_table(rows: list[list[str]]) -> None:
    methods = {"Zero-shot (Direct)": "0shot", "Zero-shot (Impression-guided)": "cot",
               "LoRA (Standard)": "baseline", "LoRA (Impression-regularized)": "proposed"}
    current_model = ""
    for row in rows[1:]:
        current_model = row[0] or current_model
        source = DATA / "figure_data/temporal_windows" / (
            f"figure3_temporal_{MODEL_NAMES[current_model]}_{methods[row[1]]}.csv")
        if not source.exists():
            continue  # The published Gemma-27B aggregate has no released window series.
        values = [float(entry["macro_f1_mean"]) for entry in records(source)]
        row[2] = summary(values)
        row[3] = f"{stdev(values) / mean(values) * 100:.2f}"


def per_level_table(rows: list[list[str]], counts: list[dict[str, str]]) -> None:
    for row in rows[2:]:
        category = "Occ_Condyle_Fracture" if row[0] == "OCC" else f"{row[0]}_Fracture"
        for method, offset in (("standard", 4), ("impression_regularized", 6)):
            selected = rows_for(counts, model="gemma-3-4b-it", method=method,
                                training_n="1064", category=category)
            row[1] = str(int(selected[0]["TP"]) + int(selected[0]["FN"]))
            row[offset] = summary([float(entry["precision"]) for entry in selected])
            row[offset + 1] = summary([float(entry["recall"]) for entry in selected])


def error_table(rows: list[list[str]]) -> None:
    categories = records(DATA / "statistical_results/error_categories.csv")
    totals = defaultdict(int)
    for entry in categories:
        totals[entry["error_type"]] += int(entry["count"])
    for row in rows[1:]:
        entry = next(entry for entry in categories if
                     entry["error_type"] == row[0] and entry["reason_category"] == row[1])
        count = int(entry["count"])
        row[2] = str(count)
        row[3] = f"{count / totals[row[0]] * 100:.1f}%"


def trial_table(rows: list[list[str]], seeds: list[dict[str, str]]) -> None:
    trials = records(DATA / "model_metrics/two_pass_trial_metrics.csv")
    for row in rows[1:]:
        if row[0] == "LoRA (2 Pass)":
            selected = trials
        else:
            method = lora_method(row[0])
            if method is None:
                continue
            selected = rows_for(seeds, model="Llama_3_2-3B-Instruct",
                                method=method, training_n="1064")
        for column, metric in enumerate(METRICS, 1):
            key = metric
            row[column] = summary([float(entry[key]) for entry in selected])


def precision_recall_table(rows: list[list[str]], counts: list[dict[str, str]]) -> None:
    current_model = ""
    for row in rows[2:]:
        current_model = row[0] or current_model
        method = lora_method(row[1])
        if method is None:
            continue
        for level_index, level in enumerate(LEVELS):
            selected = rows_for(counts, model=MODEL_NAMES[current_model], method=method,
                                training_n="1064", category=level)
            for metric_index, metric in enumerate(("precision", "recall")):
                row[2 + 2 * level_index + metric_index] = summary(
                    [float(entry[metric]) for entry in selected])


def window_size_table(rows: list[list[str]]) -> None:
    source = records(DATA / "figure_data/window_size_sensitivity.csv")
    names = {"Gemma-3-1B": "Gemma-1B", "Gemma-3-4B": "Gemma-4B",
             "Gemma-3-12B": "Gemma-12B", "Gemma-3-27B": "Gemma-27B"}
    current_model = ""
    for row in rows[1:]:
        current_model = row[0] or current_model
        method = row[1].replace("Zero-Shot (Impression-guided)",
                                "Zero-Shot (Impression-Guided)")
        method = method.replace("LoRA (Standard)", "LoRA (Direct)")
        method = method.replace("LoRA (Impression-regularized)", "LoRA (Impression-Guided)")
        match = rows_for(source, Model=names.get(current_model, current_model), Method=method)
        if len(match) != 1:
            raise ValueError(f"Missing window-size summary for {current_model}, {method}")
        for column, days in enumerate((30, 60, 90, 120, 180), 2):
            row[column] = match[0][f"{days}-Day Temp F1"]


def bootstrap_table(rows: list[list[str]]) -> None:
    source = {(entry["Model"], entry["Training Size"]): entry for entry in
              records(DATA / "statistical_results/hierarchical_bootstrap.csv")
              if entry["Metric"] == "Macro F1"}
    names = {"gemma-3-1b": "gemma-3-1b-it", "gemma-3-4b": "gemma-3-4b-it",
             "gemma-3-12b": "gemma-3-12b-it", "Llama_3_2-3B": "Llama_3_2-3B-Instruct",
             "phi-3_5-mini": "phi-3_5-mini-instruct"}
    current_model = ""
    for row in rows[1:]:
        current_model = row[0] or current_model
        size = "1100" if row[1] == "1064" else row[1]
        entry = source[(names[current_model], size)]
        row[2:] = [entry["Impression-Guided LoRA"], entry["Direct LoRA"],
                   f"{entry['Δ (Obs)']} {entry['95% CI']}", entry["P-value"], entry["Sig"]]


def ablation_table(rows: list[list[str]], seeds: list[dict[str, str]]) -> None:
    verbalization = records(DATA / "model_metrics/label_verbalization_runs.csv")
    for row in rows[1:]:
        if row[0] == "Label Verbalization":
            selected = verbalization
        elif row[0] == "Standard LoRA (Baseline)":
            selected = rows_for(seeds, model="gemma-3-4b-it", method="standard", training_n="1064")
        elif row[0] == "Ours (score-based selection)":
            selected = rows_for(seeds, model="gemma-3-4b-it", method="impression_regularized",
                                training_n="1064")
        else:
            continue  # Random-sampling run metrics are available only as published aggregates.
        for column, metric in enumerate(METRICS, 1):
            key = metric.replace("report_", "patient_") if row[0] == "Label Verbalization" else metric
            row[column] = summary([float(entry[key]) for entry in selected])


def ablation_bootstrap_table(rows: list[list[str]]) -> None:
    import numpy as np

    source = {entry["Method"]: entry for entry in
              records(DATA / "statistical_results/ablation_bootstrap.csv")}
    names = {"Standard LoRA vs. Ours": ("Standard LoRA (Baseline)", "standard_lora_baseline"),
             "Random Sampling vs. Ours": ("Random Sampling", "random_sampling"),
             "Label Verbalization vs. Ours": ("Label Verbalization", "label_verbalization")}
    with np.load(DATA / "statistical_results/ablation_bootstrap_distributions.npz",
                 allow_pickle=False) as draws:
        for row in rows[1:]:
            method, distribution = names[row[0]]
            entry = source[method]
            values = draws[distribution]
            delta = -float(entry["Delta (Observed)"])
            lower, upper = -np.quantile(values, .975), -np.quantile(values, .025)
            p = 2 * min(np.mean(values >= 0), np.mean(values <= 0))
            row[1] = f"{delta:.3f} [{lower:.3f}, {upper:.3f}]"
            row[2] = f"<{2 / len(values):.3f}" if p == 0 else f"{p:.3f}"
            row[3] = entry["Sig"]


def error_profile_table(rows: list[list[str]], seeds: list[dict[str, str]],
                        counts: list[dict[str, str]]) -> None:
    mediphi = records(DATA / "model_metrics/mediphi_per_seed.csv")
    current_model = ""
    for row in rows[1:]:
        current_model = row[0] or current_model
        if current_model == "MediPhi-Instruct":
            method = row[1]
            if method == "Zero-shot":
                selected = rows_for(mediphi, method="MedPhi Baseline")
            elif method == "Standard LoRA":
                selected = rows_for(mediphi, method="Standard LoRA")
            else:
                selected = rows_for(mediphi, method="Impression-Reg LoRA",
                                    setting="Impression-Reg LoRA (lambda=0.1)")
            row[2] = summary([float(entry["macro_f1"]) for entry in selected])
            continue  # No released precision/recall or confusion counts for MediPhi.
        method = lora_method(row[1])
        if method is None:
            continue
        selected_seeds = rows_for(seeds, model="phi-3_5-mini-instruct", method=method,
                                  training_n="1064")
        for column, metric in enumerate(METRICS[:3], 2):
            row[column] = summary([float(entry[metric]) for entry in selected_seeds])
        selected = rows_for(counts, model="phi-3_5-mini-instruct", method=method,
                            training_n="1064")
        seed_ids = sorted({entry["seed"] for entry in selected})
        for column, field, report_only in ((5, "FN", False), (6, "FP", False),
                                           (8, "FN", True), (9, "FP", True)):
            values = [sum(int(entry[field]) for entry in selected if entry["seed"] == seed
                          and (entry["category"] == "Any_Fracture") == report_only)
                      for seed in seed_ids]
            row[column] = summary(values, digits=1)
        report_rows = rows_for(selected, category="Any_Fracture")
        row[7] = summary([2 * int(entry["TP"]) /
                          (2 * int(entry["TP"]) + int(entry["FP"]) + int(entry["FN"]))
                          for entry in report_rows])


def render_table(rows: list[list[str]], destination: Path) -> None:
    width = max(map(len, rows))
    padded = [row + [""] * (width - len(row)) for row in rows]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".csv").open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    with destination.open("w", encoding="utf-8") as stream:
        stream.write("| " + " | ".join(padded[0]) + " |\n")
        stream.write("| " + " | ".join(["---"] * width) + " |\n")
        for row in padded[1:]:
            stream.write("| " + " | ".join(cell.replace("|", "\\|").replace("\n", "<br>")
                                          for cell in row) + " |\n")


def main() -> None:
    seeds = records(DATA / "model_metrics/per_seed_metrics.csv")
    counts = records(DATA / "model_metrics/final_confusion_counts.csv")
    for source in sorted((DATA / "table_data").glob("*.csv")):
        table = source.stem.removeprefix("table_")
        rows = table_rows(source)
        if table in {"1", "2", "I", "J"}:
            full_data_metrics(rows, table, seeds, counts)
        elif table == "3":
            temporal_table(rows)
        elif table == "4":
            per_level_table(rows, counts)
        elif table == "5":
            error_table(rows)
        elif table == "H":
            trial_table(rows, seeds)
        elif table == "K":
            precision_recall_table(rows, counts)
        elif table == "L":
            window_size_table(rows)
        elif table == "M":
            bootstrap_table(rows)
        elif table == "N":
            ablation_table(rows, seeds)
        elif table == "N_panel_B":
            ablation_bootstrap_table(rows)
        elif table == "O":
            error_profile_table(rows, seeds, counts)
        render_table(rows, BASE / "outputs/tables" / f"{source.stem}.md")


if __name__ == "__main__":
    main()
