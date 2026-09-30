from dataclasses import dataclass
from typing import Any, Optional

import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling


@dataclass
class ImpressionAwareWeightedDataCollator(DataCollatorForLanguageModeling):
    pad_token_id: int = 0
    completion_only_loss: bool = True
    padding_free: bool = False
    return_position_ids: bool = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    tokenizer: Optional[Any] = None
    impression_weight_max_value: float = 0.1

    def torch_call(self, examples):
        output = super().torch_call(examples)
        if "input_ids" not in output or "labels" not in output:
            return output
        if any("is_json_completion" not in example for example in examples):
            raise ValueError("Each training example needs is_json_completion")

        weights = torch.tensor(
            [1.0 if example["is_json_completion"] else self.impression_weight_max_value
             for example in examples],
            dtype=torch.float,
        )
        output["impression_weight"] = weights
        output["batch_size"] = len(examples)
        return output
