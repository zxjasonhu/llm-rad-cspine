"""Run the paper's JSON prompt with a base model or a trained LoRA adapter."""

import argparse
from pathlib import Path

from paths import PROMPTS_DIR


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="CSV with Linking_ID and Report_Text")
    parser.add_argument("--model", required=True,
                        help="Model key in finetuning/model_constants.py")
    parser.add_argument("--adapter", type=Path,
                        help="Optional LoRA checkpoint directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")

    import pandas as pd
    from llm_utils.constants import labels

    required = {"Linking_ID", "Report_Text"}
    reports = pd.read_csv(args.input, dtype={"Linking_ID": str})
    if not required.issubset(reports.columns):
        raise ValueError(f"Input needs columns {sorted(required)}")
    if reports[["Linking_ID", "Report_Text"]].isna().any().any():
        raise ValueError("Linking_ID and Report_Text must be present for every report")
    if reports["Linking_ID"].duplicated().any():
        raise ValueError("Linking_ID must be unique")
    if reports.empty:
        raise ValueError("Input CSV contains no reports")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm
    from transformers import pipeline

    from llm_datasets.report_dataset import ReportDataset
    from llm_utils import extract_json_labels
    from processor.core import get_model_settings, get_pipeline

    rows = []

    settings = get_model_settings(args.model)
    prompt = (PROMPTS_DIR / "prompt-0shot_final.txt").read_text()
    dataset = ReportDataset(reports, system_instruction=prompt)
    if args.adapter:
        import torch
        from finetuning.zoo import load_model_bundle
        bundle = load_model_bundle(args.model, adapter_path=str(args.adapter),
                                   merge_and_unload=True, torch_dtype=torch.bfloat16)
        pipe = pipeline("text-generation", model=bundle["model"],
                        tokenizer=bundle["tokenizer"], device_map="auto")
        max_new_tokens = settings["max_new_tokens"]
    else:
        state = get_pipeline(settings)
        pipe = state["pipeline"]
        max_new_tokens = state["max_new_tokens"]

    generated = (out[0]["generated_text"][-1]["content"] for out in
                 pipe(dataset, batch_size=args.batch_size,
                      max_new_tokens=max_new_tokens, do_sample=False))

    for index, response in tqdm(enumerate(generated), total=len(dataset)):
        row = {"Linking_ID": reports.iloc[index]["Linking_ID"],
               "Response": response}
        parsed = extract_json_labels(response) or {}
        row.update({label: parsed.get(label) for label in labels})
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
