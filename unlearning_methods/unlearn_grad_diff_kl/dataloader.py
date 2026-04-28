"""Dataset and collator for grad_diff_KL unlearning."""

import datasets
import torch
from torch.utils.data import Dataset

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


class GradDiffKLDataset(Dataset):
    """Return forget/retain/normal triples for grad_diff_KL training."""

    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget10", language="en"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.language = language
        self.normal_data = datasets.load_dataset("truthful_qa", "generation", split="validation")

        if language == "en":
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            self.forget_data = datasets.load_from_disk(data_path.forget)["train"]
            self.retain_data = datasets.load_from_disk(data_path.retain)["train"]

    def __len__(self):
        return len(self.forget_data)

    def _convert(self, row, answer_key="answer"):
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            row["question"],
            row[answer_key],
            self.model_configs,
            self.language,
        )

    def __getitem__(self, idx):
        retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        normal_idx = (idx + torch.randint(0, len(self.normal_data), (1,)).item()) % len(self.normal_data)

        forget_sample = self._convert(self.forget_data[idx])
        retain_sample = self._convert(self.retain_data[retain_idx])
        normal_sample = self._convert(self.normal_data[normal_idx], answer_key="best_answer")
        return forget_sample, retain_sample, normal_sample


def grad_diff_kl_collator(samples):
    rets = []
    for part_idx in range(3):
        data = [sample[part_idx] for sample in samples]
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets
