import argparse
from pathlib import Path
from finetuning.run_config import load_csvs

NUM_BEAMS = 16
NUM_BEAM_GROUPS = 8
DIVERSITY_PENALTY = 10.0
MAX_NEW_TOKENS = 512

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id", type=str, required=True, help="Model ID for training"
    )
    parser.add_argument("--train-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Selected impression CSV path")
    parser.add_argument("--batch-size", type=int, default=1)
    parsed = parser.parse_args()
    if parsed.batch_size < 1:
        parser.error("--batch-size must be positive")
    return parsed

args = parse_args()

import pandas as pd
import torch
import gc
from tqdm import tqdm
import logging
import traceback

from llm_utils import load_prompts
from llm_utils.report_processing import white_space_fix
from finetuning.llm_dataset import ReportDataset
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BeamSampler:
    def __init__(self, model_id, settings, batch_size=1):
        self.model_id = model_id
        self.settings = settings
        self.batch_size = batch_size

    def get_model_pipe_for_sampling(self):
        """Get model pipeline for sampling intermediate impressions"""
        from finetuning.model_constants import model_mapping

        path_to_model = model_mapping[self.model_id]["model"]

        pipe = pipeline(
            "text-generation", model=path_to_model,
            torch_dtype=torch.bfloat16, device_map="auto",
        )

        if "llama_3_2" in self.model_id.lower():
            eos_token_id = pipe.model.config.eos_token_id
            if isinstance(eos_token_id, list):
                pipe.tokenizer.pad_token_id = eos_token_id[0]
            else:
                pipe.tokenizer.pad_token_id = eos_token_id
        elif "phi" in self.model_id.lower():
            pipe.tokenizer.pad_token = pipe.tokenizer.unk_token
            pipe.tokenizer.pad_token_id = pipe.tokenizer.unk_token_id
        pipe.tokenizer.padding_side = "left"

        return pipe

    def beam_kwargs(self, batch_size, model_pipe):
        kwargs = {
            "batch_size": batch_size,
            "max_new_tokens": MAX_NEW_TOKENS,
            "num_return_sequences": NUM_BEAMS,
            "num_beams": NUM_BEAMS,
            "num_beam_groups": NUM_BEAM_GROUPS,
            "diversity_penalty": DIVERSITY_PENALTY,
            "do_sample": False,
        }
        if "llama" in self.model_id.lower():
            kwargs["pad_token_id"] = model_pipe.tokenizer.eos_token_id
        return kwargs

    def cleanup_model_pipe(self, model_pipe):
        """Properly cleanup model pipeline to free GPU memory"""
        try:
            if hasattr(model_pipe, "model") and model_pipe.model is not None:
                model_pipe.model.cpu()

            del model_pipe

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")

    def sample_intermediate_impressions_batch(self, datapoints, model_pipe):
        """Sample multiple intermediate impressions for a batch of datapoints efficiently"""

        user_prompt_1 = self.settings["user_prompt_1"]

        batch_df = pd.DataFrame(datapoints)

        dataset = ReportDataset(
            dataframe=batch_df,
            system_instruction=self.settings["system_instruction"],
            user_prompts=[user_prompt_1],
        )

        pipe_kwargs = self.beam_kwargs(min(len(datapoints), self.batch_size), model_pipe)

        print("processing all results together")
        all_sampled_impressions = []
        for dp_outputs in tqdm(
            model_pipe(dataset, **pipe_kwargs),
            total=len(dataset),
            desc="Sampling Impressions",
        ):
            sampled_impressions = []
            for output in dp_outputs:
                response = output["generated_text"][-1]["content"]
                response = white_space_fix(response)
                sampled_impressions.append(response)
            all_sampled_impressions.append(sampled_impressions)

        return all_sampled_impressions

    def sample_intermediate_impressions(self, datapoint, model_pipe):
        """Sample multiple intermediate impressions for a single datapoint"""

        user_prompt_1 = self.settings["user_prompt_1"]

        single_df = pd.DataFrame([datapoint])

        dataset = ReportDataset(
            dataframe=single_df,
            system_instruction=self.settings["system_instruction"],
            user_prompts=[user_prompt_1],
        )

        pipe_kwargs = self.beam_kwargs(1, model_pipe)

        sampled_impressions = []
        for i, output in enumerate(
            tqdm(
                model_pipe(dataset[0], **pipe_kwargs),
                total=NUM_BEAMS,
                desc="Sampling Impressions",
            )
        ):
            response = output["generated_text"][-1]["content"]
            response = white_space_fix(response)
            sampled_impressions.append(response)

        return sampled_impressions

    def select_best_samples_batch(self, datapoints, model_pipe):
        """Sample and select best samples for a batch of datapoints efficiently using reference JSON negative log-likelihood"""

        all_sampled_impressions = self.sample_intermediate_impressions_batch(
            datapoints, model_pipe
        )

        best_samples = []

        for dp_idx, (datapoint, sampled_impressions) in tqdm(
            enumerate(zip(datapoints, all_sampled_impressions)),
            total=len(datapoints),
            desc="Selecting Best Samples",
        ):

            all_candidate_impressions = sampled_impressions.copy()

            ground_truths = [datapoint["Output_report"]] * len(
                all_candidate_impressions
            )
            candidate_losses = self.calculate_loss_scores_batch(
                all_candidate_impressions, ground_truths, model_pipe
            )

            best_loss = float("inf")
            best_impression_length = float("inf")
            best_sample = None

            for i, (impression, loss) in enumerate(
                zip(all_candidate_impressions, candidate_losses)
            ):
                impression_length = len(impression)
                is_better_sample = False

                if loss < best_loss:
                    is_better_sample = True
                elif loss == best_loss and impression_length < best_impression_length:
                    is_better_sample = True

                if is_better_sample:
                    best_loss = loss
                    best_impression_length = impression_length
                    best_sample = {
                        "Linking_ID": datapoint["Linking_ID"],
                        "Report_Text": datapoint["Report_Text"],
                        "Impression_Response": impression,
                        "Output_report": datapoint["Output_report"],
                        "candidate_nll": loss,
                    }

            if best_sample is not None:
                best_samples.append(best_sample)

        return best_samples

    def calculate_candidate_nll(self, impression, ground_truth, model_pipe):
        """Score the complete reference JSON continuation after a candidate impression."""
        conversation = [
            {"role": "system", "content": self.settings["system_instruction"]},
            {"role": "user", "content": self.settings["user_prompt_1"]},
            {"role": "assistant", "content": impression},
            {"role": "user", "content": self.settings["user_prompt_2"]},
            {"role": "assistant", "content": ground_truth},
        ]
        tokenizer = model_pipe.tokenizer
        full_text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        prompt_text = tokenizer.apply_chat_template(
            conversation[:-1], tokenize=False, add_generation_prompt=True
        )
        full_tokens = tokenizer.encode(full_text, return_tensors="pt")
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors="pt")
        prefix_length = prompt_tokens.shape[1]
        if prefix_length >= full_tokens.shape[1]:
            raise ValueError("Candidate has no JSON completion tokens")
        labels = full_tokens.clone()
        labels[:, :prefix_length] = -100
        device = next(model_pipe.model.parameters()).device
        with torch.no_grad():
            output = model_pipe.model(full_tokens.to(device), labels=labels.to(device))
        return output.loss.item()

    def calculate_loss_scores_batch(self, impressions, ground_truths, model_pipe):
        if len(impressions) != len(ground_truths):
            raise ValueError("Each impression needs a reference JSON output")
        return [
            self.calculate_candidate_nll(impression, truth, model_pipe)
            for impression, truth in zip(impressions, ground_truths)
        ]

    def select_best_sample(self, datapoint, model_pipe):
        """Sample multiple impressions and select the best one based on token-wise teacher forcing loss"""

        sampled_impressions = self.sample_intermediate_impressions(
            datapoint, model_pipe
        )

        all_candidate_impressions = sampled_impressions.copy()

        best_sample = None
        best_loss = float("inf")
        best_impression_length = float("inf")

        for i, impression in enumerate(all_candidate_impressions):
            try:
                loss = self.calculate_candidate_nll(
                    impression, datapoint["Output_report"], model_pipe
                )

                impression_length = len(impression)
                is_better_sample = False

                if loss < best_loss:
                    is_better_sample = True
                elif loss == best_loss and impression_length < best_impression_length:
                    is_better_sample = True

                if is_better_sample:
                    best_loss = loss
                    best_impression_length = impression_length
                    best_sample = {
                        "Linking_ID": datapoint["Linking_ID"],
                        "Report_Text": datapoint["Report_Text"],
                        "Impression_Response": impression,
                        "Output_report": datapoint["Output_report"],
                        "candidate_nll": loss,
                    }

            except Exception as e:
                logger.warning(f"Error processing sample: {e}")
                continue

        return best_sample

    def process_training_batch(self, train_batch):
        """Process a batch of training data with sampling using efficient batched processing"""

        model_pipe = self.get_model_pipe_for_sampling()

        logger.info(
            f"Processing batch of {len(train_batch)} samples using efficient batching"
        )

        try:
            datapoints = [row.to_dict() for _, row in train_batch.iterrows()]

            selected_samples = self.select_best_samples_batch(datapoints, model_pipe)

        except Exception as e:
            logger.warning(
                f"Batch processing failed: {e}. Falling back to sequential processing."
            )
            traceback.print_exc()

            selected_samples = []
            for idx, (_, datapoint) in enumerate(
                tqdm(
                    train_batch.iterrows(),
                    total=len(train_batch),
                    desc="Processing Samples",
                ),
            ):
                try:
                    best_sample = self.select_best_sample(datapoint, model_pipe)
                    if best_sample is not None:
                        selected_samples.append(best_sample)

                except Exception as e:
                    logger.error(f"Error processing datapoint {idx}: {e}")
                    continue

        self.cleanup_model_pipe(model_pipe)
        del model_pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(selected_samples) != len(train_batch):
            raise ValueError(
                f"Selected impressions for {len(selected_samples)} of "
                f"{len(train_batch)} training reports"
            )
        return pd.DataFrame(selected_samples)

def main():

    logger.info(f"Starting sampling-based finetuning for {args.model_id}")
    logger.info(f"Number of samples per datapoint: {NUM_BEAMS}")

    model_id = args.model_id
    prompt_name = ["prompt-cot-sys", "prompt-cot-pass1", "prompt-cot-pass2"]

    training_df = load_csvs(args.train_csv, ["Linking_ID", "Report_Text", "Output_report"])
    print(f"dataset length:", training_df.shape)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")

    settings = load_prompts(prompt_name)

    trainer = BeamSampler(
        model_id=model_id,
        settings=settings,
        batch_size=args.batch_size,
    )

    selected_samples = trainer.process_training_batch(training_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_samples.to_csv(args.output, index=False)

    logger.info("Sampling completed!")

if __name__ == "__main__":
    main()
