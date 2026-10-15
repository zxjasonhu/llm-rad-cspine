"""Draw Figs 1, 2, A, and B from released aggregate inputs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data/figure_data"
OUTPUT = BASE / "outputs/figures"
BLUE = "#2563a6"
ORANGE = "#d66c35"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def zero_shot(model: str, guided: bool = False) -> float:
    if model in {"llama_3b", "gemma_4b", "phi_3_5_mini"}:
        table = rows(BASE / "data/table_data/table_1.csv")
        names = {"llama_3b": "Llama-3.2", "gemma_4b": "Gemma-3",
                 "phi_3_5_mini": "Phi-3.5"}
        current = ""
        for entry in table:
            current = entry["Model Variant"] or current
            if current == names[model] and entry["Method"] == (
                "Zero-shot (Impression-guided)" if guided else "Zero-shot (Direct)"):
                return float(entry["Macro-F1"])
    if not guided and model in {"gemma_1b", "gemma_12b"}:
        current = ""
        for entry in rows(BASE / "data/table_data/table_2.csv"):
            current = entry["Model Size"] or current
            if current == {"gemma_1b": "1B", "gemma_12b": "12B"}[model] and entry["Method"] == "Zero-shot (Direct)":
                return float(entry["Macro-F1"])
    raise KeyError((model, guided))


def learning_curves(models: list[tuple[str, str, float, float | None]], number: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    model_ids = {"llama_3b": "Llama_3_2-3B-Instruct", "gemma_1b": "gemma-3-1b-it",
                 "gemma_4b": "gemma-3-4b-it", "gemma_12b": "gemma-3-12b-it",
                 "phi_3_5_mini": "phi-3_5-mini-instruct"}
    pvalues = {}
    for entry in rows(BASE / "data/statistical_results/hierarchical_bootstrap.csv"):
        if entry["Metric"] == "Macro F1":
            pvalues[(entry["Model"], int(entry["Training Size"]))] = entry["P-value"]
    for panel, (key, title, direct, guided) in enumerate(models):
        ax = axes[panel]
        data = rows(DATA / f"learning_curve_{key}.csv")
        x = np.array([1064 if int(row["sample_size"]) == 1100 else int(row["sample_size"]) for row in data])
        for method, color, label in (("baseline", BLUE, "Standard LoRA"),
                                     ("proposed", ORANGE, "Impression-regularized LoRA")):
            y = np.array([float(row[f"{method}_mean"]) for row in data])
            sd = np.array([float(row[f"{method}_std"]) for row in data])
            ax.errorbar(x, y, yerr=sd, marker="o", linewidth=2, capsize=3, color=color, label=label)
        ax.axhline(direct, color="#4f5964", linestyle="--", linewidth=1.2, label="Zero-shot direct")
        if guided is not None:
            ax.axhline(guided, color="#6a947d", linestyle="--", linewidth=1.2,
                       label="Zero-shot impression-guided")
        ax.set_title(f"({chr(97 + panel)}) {title}")
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("Training reports (N)")
        ax.grid(alpha=.25)
        ax.set_ylim(.1 if number == 2 else .6, 1.0)
        for position, row in zip(x, data):
            p = pvalues.get((model_ids[key], 1100 if position == 1064 else int(position)), "")
            stars = "***" if p.startswith("<") else ("**" if p and float(p) < .01 else
                    "*" if p and float(p) < .05 else "")
            if stars:
                ytop = max(float(row["baseline_mean"]) + float(row["baseline_std"]),
                           float(row["proposed_mean"]) + float(row["proposed_std"]))
                ax.text(position, min(ytop + .025, .98), stars, ha="center", fontsize=9)
    axes[0].set_ylabel("Macro-F1 (mean ± SD over five runs)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, .09, 1, 1))
    fig.savefig(OUTPUT / f"Fig{number}.png", dpi=200)
    plt.close(fig)


def temporal_curves() -> None:
    models = [
        ("gemma-3-1b-it", "Gemma-3-1B"),
        ("gemma-3-4b-it", "Gemma-3-4B"),
        ("gemma-3-12b-it", "Gemma-3-12B"),
        ("Llama_3_2-3B-Instruct", "Llama-3.2-3B"),
        ("phi-3_5-mini-instruct", "Phi-3.5-Mini"),
    ]
    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    styles = {"0shot": ("Zero-shot direct", "#777777"),
              "cot": ("Zero-shot impression-guided", "#6a947d"),
              "baseline": ("Standard LoRA", BLUE),
              "proposed": ("Impression-regularized LoRA", ORANGE)}
    for i, (key, display) in enumerate(models):
        for method, (label, color) in styles.items():
            source = DATA / "temporal_windows" / f"figure3_temporal_{key}_{method}.csv"
            if not source.exists():
                continue
            data = rows(source)
            x = [int(row["window_index"]) for row in data]
            for column, ax in (("macro_f1_mean", axes[i, 0]),
                               ("any_fracture_sensitivity_mean", axes[i, 1])):
                y = [float(row[column]) for row in data]
                ax.plot(x, y, color=color, linewidth=1.3, label=label)
                ax.set_ylim(0, 1.02)
                ax.grid(alpha=.15)
        axes[i, 0].set_ylabel(display + "\nMacro-F1")
        axes[i, 1].set_ylabel("Report sensitivity")
    axes[0, 0].set_title("(a) Temporal macro-F1")
    axes[0, 1].set_title("(b) Temporal report sensitivity")
    axes[-1, 0].set_xlabel("90-day window index (5-day step)")
    axes[-1, 1].set_xlabel("90-day window index (5-day step)")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(OUTPUT / "FigA.png", dpi=180)
    plt.close(fig)


def epoch_figure() -> None:
    summary = rows(DATA / "epoch_summary.csv")
    raw = rows(BASE / "data/model_metrics/epoch_per_seed.csv")
    selected = [row for row in summary if row["model"] == "gemma-3-4b-it"]
    points = [row for row in raw if row["model"] == "gemma-3-4b-it"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for row in points:
        ax.scatter(float(row["epoch"]), float(row["macro_f1"]), color="#9aa3ad", s=22, alpha=.8)
    ax.plot([int(row["epoch"]) for row in selected],
            [float(row["mean_macro_f1"]) for row in selected], marker="D", color=BLUE)
    ax.set(xlabel="Finetuning epochs", ylabel="Macro-F1", title="Gemma-3-4B epoch sensitivity")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUTPUT / "FigB.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    learning_curves([
        (key, title, zero_shot(key), zero_shot(key, guided=True))
        for key, title in (("llama_3b", "Llama-3.2-3B"),
                           ("gemma_4b", "Gemma-3-4B"),
                           ("phi_3_5_mini", "Phi-3.5-Mini"))
    ], 1)
    learning_curves([
        (key, title, zero_shot(key), None)
        for key, title in (("gemma_1b", "Gemma-3-1B"),
                           ("gemma_4b", "Gemma-3-4B"),
                           ("gemma_12b", "Gemma-3-12B"))
    ], 2)
    temporal_curves()
    epoch_figure()


if __name__ == "__main__":
    main()
