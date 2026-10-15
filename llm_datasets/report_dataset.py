from torch.utils.data import Dataset

from llm_utils import preprocess_report

class ReportDataset(Dataset):
    def __init__(
        self,
        dataframe,
        system_instruction=None,
        user_prompts: list = None,
    ):
        self.dataframe = dataframe
        self.system_instruction = system_instruction
        self.user_prompts = user_prompts

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        _prompt = preprocess_report(row["Report_Text"])

        if self.user_prompts is not None:
            _prompt = self.user_prompts[0] + "\n" + _prompt

        messages = [
            {"role": "system", "content": self.system_instruction},
            {"role": "user", "content": _prompt},
        ]
        if "Impression_Response" in row and self.user_prompts is not None and len(self.user_prompts) > 1:
            messages.extend([
                {"role": "assistant", "content": row["Impression_Response"]},
                {"role": "user", "content": self.user_prompts[1]},
            ])

        return messages
