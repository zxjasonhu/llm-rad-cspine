from transformers import pipeline
import torch
from finetuning.model_constants import model_mapping


def get_model_settings(model):
    if model in model_mapping:
        return model_mapping[model]
    else:
        raise ValueError(
            f"Model {model} is not supported. Available models: {list(model_mapping.keys())}"
        )


def get_pipeline(settings):
    model_id = settings["model"]
    print(f"Loading pipeline for model: {model_id}")
    if "llama-3.2" in model_id.lower():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        pipe = pipeline(
            "text-generation",
            model=model_id,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        eos_token_id = pipe.model.config.eos_token_id
        pipe.tokenizer.pad_token_id = (
            eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        )
        pipe.model.generation_config.pad_token_id = pipe.tokenizer.pad_token_id

    else:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            do_sample=False,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    return {
        "pipeline": pipe,
        "model_id": model_id,
        "max_new_tokens": settings["max_new_tokens"],
    }
