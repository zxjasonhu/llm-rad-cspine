"""Train the paper's direct report-to-JSON LoRA baseline."""

import argparse
from pathlib import Path

from paths import OUTPUT_DIR
from finetuning.run_config import load_csvs, load_overrides, sample_training_data
from finetuning.sft_settings import DEFAULT_SEED, make_sft_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_id", required=True, help="Model key in model_constants.py")
    parser.add_argument("--train-csv", type=Path, nargs="+", required=True,
                        help="Training CSV file(s), using the schema in examples/synthetic_cohort.csv")
    parser.add_argument("--eval-csv", type=Path, nargs="+", help="Optional held-out CSV file(s)")
    parser.add_argument("--training-dataset-size", type=int,
                        help="Optional subset size; published sizes use S1 Table E class counts")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for sampling and training (default: {DEFAULT_SEED})")
    parser.add_argument("--sft-config", type=Path, help="JSON object overriding SFTConfig values")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    import gc
    import torch
    from trl import SFTConfig, SFTTrainer

    from finetuning.finetuning_utils import eval_ft_llm
    from finetuning.llm_dataset import get_dataset_from_pandas
    from finetuning.lora_settings import lora_constant
    from finetuning.zoo import lora_ft
    from llm_utils import setup_trail

    torch.manual_seed(args.seed)

    training_df = load_csvs(args.train_csv, ["Linking_ID", "Report_Text", "Output_report"])
    training_df = sample_training_data(training_df, args.training_dataset_size, args.seed)
    run_name = f"lora_n{len(training_df)}"
    run_name += f"_seed{args.seed}"
    settings = setup_trail(args.model_id, "prompt-0shot_final", run_name, args.output_dir)
    lora_settings = lora_constant.copy()
    sft_settings = make_sft_config(False, settings["working_folder"], args.model_id,
                                   len(training_df),
                                   load_overrides(args.sft_config), seed=args.seed)
    model_bundle = lora_ft(args.model_id, lora_settings)
    dataset = get_dataset_from_pandas(
        training_df, ["Report_Text", "Output_report"], settings["system_instruction"]
    )
    trainer = SFTTrainer(
        model=model_bundle["model"], processing_class=model_bundle["tokenizer"],
        args=SFTConfig(**sft_settings), train_dataset=dataset,
    )
    trainer.train()
    checkpoint = Path(settings["working_folder"]) / f"checkpoint-{trainer.state.global_step}"
    trainer.save_model(str(checkpoint))
    model_bundle["tokenizer"].save_pretrained(str(checkpoint))
    print(f"Final LoRA checkpoint: {checkpoint}")

    if args.eval_csv:
        trainer.model.cpu()
        del trainer, dataset, model_bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        eval_ft_llm(args.model_id, settings, args.eval_csv, args.eval_batch_size)


if __name__ == "__main__":
    main()
