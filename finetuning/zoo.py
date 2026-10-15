from finetuning.model_constants import model_mapping
from peft import LoraConfig, get_peft_model
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Gemma3ForCausalLM,
)

from peft import PeftModel

def get_last_checkpoint(working_folder):
    checkpoints = [
        path for path in Path(working_folder).iterdir()
        if path.is_dir() and path.name.startswith("checkpoint-")
        and path.name.removeprefix("checkpoint-").isdigit()
    ]
    return max(checkpoints, key=lambda path: int(path.name.split("-")[-1])).name if checkpoints else None


def _get_model_settings(base_model_id):
    if base_model_id in model_mapping:
        return model_mapping[base_model_id]

    raise ValueError(
        f"Model {base_model_id} is not supported. Available models: {list(model_mapping.keys())}"
    )


def _configure_tokenizer_for_training(base_model_id, tokenizer, model):
    if "llama_3" in base_model_id.lower():
        if tokenizer.pad_token_id is None:
            eos_token_id = model.config.eos_token_id
            if isinstance(eos_token_id, list):
                tokenizer.pad_token_id = eos_token_id[0]
            else:
                tokenizer.pad_token_id = eos_token_id
        print(f"Set pad_token_id for Llama training tokenizer: {tokenizer.pad_token_id}")
    elif "phi" in base_model_id.lower():
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        print(f"Set pad_token to unk_token for Phi training tokenizer: {tokenizer.pad_token_id}")


def load_model_bundle(
    base_model_id,
    adapter_path=None,
    is_trainable=False,
    merge_and_unload=False,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation=None,
    tokenizer_source=None,
):
    settings = _get_model_settings(base_model_id)
    model_id = settings["model"]

    tokenizer_source = tokenizer_source or adapter_path or model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)

    model_kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation

    if "gemma" in base_model_id.lower():
        print("Loading Gemma model for training...")
        model = Gemma3ForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        print("Gemma model loaded:", model.__class__.__name__)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

    if adapter_path is not None:
        print(f"Loading PEFT model from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=is_trainable)
        if merge_and_unload:
            model = model.merge_and_unload()

    _configure_tokenizer_for_training(base_model_id, tokenizer, model)

    return {"model": model, "tokenizer": tokenizer}


def lora_ft(model_id, lora_settings):
    config = LoraConfig(**lora_settings)

    _model_dict = get_model_ft(model_id)
    _model = _model_dict["model"]
    _model = get_peft_model(_model, config)
    _model.print_trainable_parameters()

    return {
        "model": _model,
        "tokenizer": _model_dict["tokenizer"],
    }


def get_model_ft(base_model_id):
    return load_model_bundle(base_model_id=base_model_id, torch_dtype=torch.float32)
