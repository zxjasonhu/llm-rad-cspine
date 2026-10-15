DEFAULT_SEED = 42

sft_config_template = {
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "max_length": 2560,
    "num_train_epochs": 2,
    "learning_rate": 1e-4,
    "max_grad_norm": 0.3,
    "report_to": "none",
}

PAPER_EPOCHS_BY_TRAINING_SIZE = {20: 12, 100: 6, 500: 3, 800: 2, 1064: 2}


def make_sft_config(mixed, output_dir, model_id, training_size, overrides=None, *, seed=DEFAULT_SEED):
    """Build SFTConfig values from the primary paper setting and caller overrides."""
    settings = sft_config_template.copy()
    settings["num_train_epochs"] = PAPER_EPOCHS_BY_TRAINING_SIZE.get(training_size, 2)
    model_name = model_id.lower()
    if "gemma-3-1b" in model_name:
        batch_size = 16
    elif "gemma-3-12b" in model_name:
        batch_size = 4
    elif any(name in model_name for name in ("gemma-3-4b", "llama_3_2-3b", "phi-3_5-mini")):
        batch_size = 8
    else:
        raise ValueError(f"No LoRA training configuration was reported for {model_id}")
    settings["per_device_train_batch_size"] = batch_size
    settings["gradient_accumulation_steps"] = (32 if mixed else 16) // batch_size
    settings["completion_only_loss"] = True
    if mixed:
        settings["packing"] = False
        settings["remove_unused_columns"] = False
    settings.update(overrides or {})
    settings["seed"] = seed
    settings["output_dir"] = str(output_dir)
    return settings
