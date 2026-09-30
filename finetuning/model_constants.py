MODEL_IDS = {
    "Llama_3_2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3_5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

model_mapping = {
    key: {
        "model": model_id,
        "max_new_tokens": 512,
    }
    for key, model_id in MODEL_IDS.items()
}
