import pandas as pd
from datasets import Dataset
from trl.data_utils import maybe_apply_chat_template

from llm_datasets.report_dataset import ReportDataset
from llm_utils import preprocess_report


def format_train_dataset(example, instructions: str):
    completion = example.get("Output_report", example.get("Impression_Response"))
    return {
        "prompt": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": preprocess_report(example["Report_Text"])},
        ],
        "completion": [{"role": "assistant", "content": completion}],
    }


def get_dataset_from_pandas(df: pd.DataFrame, columns: list, instructions: str):
    dataset = Dataset.from_pandas(df[columns])
    return dataset.map(
        format_train_dataset,
        fn_kwargs={"instructions": instructions},
        remove_columns=dataset.column_names,
    )


def materialize_prompt_completion_chat_template(dataset: Dataset, tokenizer, dataset_name="train"):
    if len(dataset) == 0:
        return dataset
    first = dataset[0]
    if not isinstance(first.get("prompt"), list) or not isinstance(first.get("completion"), list):
        return dataset
    return dataset.map(
        maybe_apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        desc=f"Applying chat template to {dataset_name} dataset",
    )


def add_completion_type_flag(dataset: Dataset, is_json_completion: bool):
    if "is_json_completion" in dataset.column_names:
        dataset = dataset.remove_columns("is_json_completion")
    return dataset.add_column("is_json_completion", [is_json_completion] * len(dataset))


def get_eval_dataset(df: pd.DataFrame, system_instruction: str, user_instructions: str | None):
    return ReportDataset(
        df,
        system_instruction=system_instruction,
        user_prompts=[user_instructions] if user_instructions else None,
    )
