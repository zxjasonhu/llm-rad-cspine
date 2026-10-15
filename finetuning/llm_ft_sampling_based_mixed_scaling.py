"""Train impression-regularized LoRA using selected auxiliary impressions."""

import argparse
from pathlib import Path

from paths import OUTPUT_DIR, PROMPTS_DIR
from finetuning.run_config import load_csvs, load_overrides, sample_training_data
from finetuning.sft_settings import DEFAULT_SEED, make_sft_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Model key in model_constants.py")
    parser.add_argument("--train-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--beam-samples-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-csv", type=Path, nargs="+", help="Optional held-out CSV file(s)")
    parser.add_argument("--training-dataset-size", type=int,
                        help="Optional subset size; published sizes use S1 Table E class counts")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for sampling and training (default: {DEFAULT_SEED})")
    parser.add_argument("--sft-config", type=Path, help="JSON object overriding SFTConfig values")
    parser.add_argument("--impression-weight", type=float, default=0.1,
                        help="Auxiliary impression loss weight (paper default: 0.1)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    return parser.parse_args()


args = parse_args()

import torch
import gc
import logging

from trl import SFTConfig, SFTTrainer
from llm_utils import setup_trail
from finetuning.lora_settings import lora_constant
from finetuning.finetuning_utils import eval_ft_llm
from finetuning.llm_dataset import (
    add_completion_type_flag,
    get_dataset_from_pandas,
    materialize_prompt_completion_chat_template,
)
from finetuning.zoo import lora_ft
from datasets import concatenate_datasets
from finetuning.custom_data_collator import ImpressionAwareWeightedDataCollator
from finetuning.custom_loss_fn import impression_weighted_causal_lm_loss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MixedSamplingBasedTrainer:
    def __init__(
        self,
        model_id,
        settings,
        sft_config,
        lora_settings,
        training_df,
        impression_weight_max_value,
        seed=DEFAULT_SEED,
    ):
        self.model_id = model_id
        self.settings = settings
        self.sft_config = sft_config
        self.lora_settings = lora_settings
        self.training_df = training_df
        self.impression_weight_max_value = impression_weight_max_value
        self.seed = seed
        with open(PROMPTS_DIR / "prompt-0shot_final.txt", "r") as f:
            json_sys_prompt = f.read()

        self.json_dataset = get_dataset_from_pandas(
            df=self.training_df,
            columns=["Report_Text", "Output_report"],
            instructions=json_sys_prompt,
        )
        self.json_dataset = add_completion_type_flag(
            self.json_dataset,
            is_json_completion=True,
        )

    def load_impression_dataset_from_local(self, paths):
        beam = load_csvs(paths, ["Linking_ID", "Impression_Response"])
        selected = self.training_df[["Linking_ID", "Report_Text"]].merge(
            beam[["Linking_ID", "Impression_Response"]],
            on="Linking_ID", how="left", validate="one_to_one",
        )
        if selected["Impression_Response"].isna().any():
            raise ValueError("Beam samples are missing for some training reports")
        return selected

    def train_on_selected_samples(self, selected_samples_df):
        """Train the model on selected best samples using weighted loss"""

        if len(selected_samples_df) == 0:
            logger.warning("No valid samples to train on")
            return

        logger.info(
            f"Training on {len(selected_samples_df)} selected samples with weighted loss"
        )

        model_dict = None
        model = None
        tokenizer = None
        dataset = None
        trainer = None

        try:
            model_dict = lora_ft(self.model_id, self.lora_settings)

            model = model_dict["model"]
            tokenizer = model_dict["tokenizer"]

            dataset = get_dataset_from_pandas(
                df=selected_samples_df,
                columns=["Report_Text", "Impression_Response"],
                instructions=self.settings["system_instruction"],
            )
            dataset = add_completion_type_flag(
                dataset,
                is_json_completion=False,
            )

            dataset = concatenate_datasets([dataset, self.json_dataset]).shuffle(
                **({"seed": self.seed} if self.seed is not None else {})
            )
            dataset = materialize_prompt_completion_chat_template(
                dataset,
                tokenizer,
                dataset_name="mixed prompt-completion",
            )
            print("Final training dataset size after mixing:", len(dataset))

            impression_weight_max_value = self.impression_weight_max_value

            data_collator = ImpressionAwareWeightedDataCollator(
                pad_token_id=tokenizer.pad_token_id,
                completion_only_loss=True,
                padding_free=False,
                tokenizer=tokenizer,
                impression_weight_max_value=impression_weight_max_value,
            )

            impression_loss_fn = impression_weighted_causal_lm_loss
            if hasattr(model, "base_model"):
                print("original loss function:", model.base_model.model.loss_function)
                model.base_model.model._loss_function = impression_loss_fn
                model.base_model.model.loss_function = impression_loss_fn
                print(
                    "Registered weighted loss function with PEFT base model",
                    model.base_model.model.loss_function,
                )
            else:
                model._loss_function = impression_loss_fn
                model.loss_function = impression_loss_fn
                print(
                    "Registered weighted loss function with model", model.loss_function
                )
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                args=self.sft_config,
                train_dataset=dataset,
                data_collator=data_collator,
            )

            trainer.train()

            final_checkpoint = Path(self.settings["working_folder"]) / f"checkpoint-{trainer.state.global_step}"
            trainer.save_model(str(final_checkpoint))
            tokenizer.save_pretrained(str(final_checkpoint))
            logger.info(f"Final LoRA checkpoint: {final_checkpoint}")

            logger.info("Weighted training completed")

        except Exception as e:
            logger.error(f"Error during weighted training: {e}")
            raise
        finally:

            if trainer is not None:
                try:
                    if hasattr(trainer, "model") and trainer.model is not None:
                        trainer.model.cpu()
                    del trainer
                except Exception as e:
                    logger.warning(f"Error cleaning up trainer: {e}")

            if model is not None:
                try:
                    model.cpu()
                    del model
                except Exception as e:
                    logger.warning(f"Error cleaning up model: {e}")

            if tokenizer is not None:
                try:
                    del tokenizer
                except Exception as e:
                    logger.warning(f"Error cleaning up tokenizer: {e}")

            if model_dict is not None:
                try:
                    del model_dict
                except Exception as e:
                    logger.warning(f"Error cleaning up model_dict: {e}")

            if dataset is not None:
                try:
                    del dataset
                except Exception as e:
                    logger.warning(f"Error cleaning up dataset: {e}")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

def main():
    if not 0 <= args.impression_weight <= 1:
        raise ValueError("--impression-weight must be between 0 and 1")
    torch.manual_seed(args.seed)

    training_df = load_csvs(args.train_csv, ["Linking_ID", "Report_Text", "Output_report"])
    training_df = sample_training_data(training_df, args.training_dataset_size, args.seed)
    prompt_names = ["prompt-cot-sys", "prompt-cot-pass1", "prompt-cot-pass2"]
    run_name = f"impression_regularized_n{len(training_df)}"
    run_name += f"_seed{args.seed}"
    settings = setup_trail(args.model_id, prompt_names, run_name, args.output_dir)
    lora_settings = lora_constant.copy()
    sft_settings = make_sft_config(True, settings["working_folder"], args.model_id,
                                   len(training_df),
                                   load_overrides(args.sft_config), seed=args.seed)
    trainer = MixedSamplingBasedTrainer(
        model_id=args.model_id,
        settings=settings,
        sft_config=SFTConfig(**sft_settings),
        lora_settings=lora_settings,
        training_df=training_df,
        impression_weight_max_value=args.impression_weight,
        seed=args.seed,
    )
    selected_samples = trainer.load_impression_dataset_from_local(args.beam_samples_csv)
    trainer.train_on_selected_samples(selected_samples)

    if args.eval_csv:
        eval_ft_llm(args.model_id, settings, args.eval_csv, args.eval_batch_size,
                    (PROMPTS_DIR / "prompt-0shot_final.txt").read_text())


if __name__ == "__main__":
    main()
