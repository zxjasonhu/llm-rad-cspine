"""Evaluate a trained LoRA checkpoint on an explicitly supplied cohort."""

import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import pipeline

from finetuning.llm_dataset import get_eval_dataset
from finetuning.run_config import load_csvs
from finetuning.zoo import get_last_checkpoint, load_model_bundle
from llm_utils import extract_json_labels
from llm_utils.constants import labels
from evaluation import binary_eval


def get_model_pipe_for_eval(model_id, checkpoint_dir):
    checkpoint = get_last_checkpoint(checkpoint_dir)
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
    bundle = load_model_bundle(
        base_model_id=model_id,
        adapter_path=str(Path(checkpoint_dir) / checkpoint),
        merge_and_unload=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_pipe = pipeline(
        task="text-generation", model=bundle["model"], tokenizer=bundle["tokenizer"],
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    model_pipe.tokenizer.padding_side = "left"
    return model_pipe


def eval_ft_llm(model_id, settings, eval_csvs, batch_size=1, system_instruction=None):
    if batch_size < 1:
        raise ValueError("Evaluation batch size must be positive")
    test_df = load_csvs(eval_csvs, ["Linking_ID", "Report_Text", *labels])
    dataset = get_eval_dataset(
        df=test_df, system_instruction=system_instruction or settings["system_instruction"],
        user_instructions=None,
    )
    model_pipe = get_model_pipe_for_eval(model_id, settings["working_folder"])
    kwargs = {"batch_size": batch_size, "max_new_tokens": 512, "do_sample": False}
    if "llama" in model_id.lower():
        kwargs["pad_token_id"] = model_pipe.tokenizer.eos_token_id

    rows = []
    try:
        for index, output in tqdm(enumerate(model_pipe(dataset, **kwargs)), total=len(dataset)):
            response = output[0]["generated_text"][-1]["content"]
            extracted = extract_json_labels(response) or {}
            rows.append({
                "Linking_ID": test_df.iloc[index]["Linking_ID"],
                "Response": response,
                **{label: extracted.get(label) for label in labels},
            })
    finally:
        model_pipe.model.cpu()
        del model_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    predictions = pd.DataFrame(rows)
    predictions.to_csv(settings["output_file_path"], index=False)
    metrics = binary_eval(test_df, predictions, labels)
    metrics_path = Path(settings["working_folder"]) / "metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Predictions: {settings['output_file_path']}")
    print(f"Metrics: {metrics_path}")
