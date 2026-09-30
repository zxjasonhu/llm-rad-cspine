"""Load paper prompts and create a run output directory."""

from pathlib import Path
from typing import List

from paths import OUTPUT_DIR, PROMPTS_DIR


def load_prompts(prompt_name: str | List[str]):
    if isinstance(prompt_name, str):
        prompt_name = [prompt_name]

    settings = {}
    for index, name in enumerate(prompt_name):
        prompt_path = PROMPTS_DIR / f"{name}.txt"
        prompt = prompt_path.read_text()
        key = "system_instruction" if index == 0 else f"user_prompt_{index}"
        settings[key] = prompt
    return settings


def setup_trail(model_id: str, prompt_name: str | List[str], suffix: str = None,
                output_dir: Path = OUTPUT_DIR):
    settings = load_prompts(prompt_name)
    if isinstance(prompt_name, str):
        prompt_name = [prompt_name]

    prompt_label = "cot" if len(prompt_name) > 1 else prompt_name[0]
    base_name = f"{model_id.replace(':', '_')}_{prompt_label}"
    if suffix:
        base_name += f"_{suffix}"
    working_folder = output_dir / base_name
    if working_folder.exists():
        raise FileExistsError(f"Training run already exists: {working_folder}")
    working_folder.mkdir(parents=True)
    settings["working_folder"] = str(working_folder)
    settings["output_file_path"] = str(working_folder / f"{base_name}_predictions.csv")
    return settings
