"""Dataset and collator for RCP-SLoRA unlearning.

Returns paired (forget, retain) batches.
Mirrors unlearn_npo/dataloader.py structure but kept independent.
"""

import datasets
import torch
from torch.utils.data import Dataset

# PROJECT_ROOT를 sys.path에 추가한 뒤 import하므로 train.py가 먼저 경로를 설정함
from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


class RCPForgetDataset(Dataset):
    """Return paired (forget_sample, retain_sample) for RCP-SLoRA training.

    - 영어(en): locuslab/TOFU HuggingFace Hub에서 로드
    - 비영어: cfg.data_path.forget / cfg.data_path.retain 디스크 경로에서 로드
    """

    def __init__(self, data_path, tokenizer, model_family, max_length=512, split="forget01", language="en"):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.language = language

        if language == "en":
            self.forget_data = datasets.load_dataset(data_path, split)["train"]
            retain_split = "retain" + str(100 - int(split.replace("forget", ""))).zfill(2)
            self.retain_data = datasets.load_dataset(data_path, retain_split)["train"]
        else:
            # data_path는 Hydra structured config: data_path.forget / data_path.retain
            self.forget_data = datasets.load_from_disk(data_path.forget)["train"]
            self.retain_data = datasets.load_from_disk(data_path.retain)["train"]

    def __len__(self):
        return len(self.forget_data)

    def __getitem__(self, idx):
        # retain은 forget과 랜덤하게 페어링
        retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(self.retain_data)
        forget_row = self.forget_data[idx]
        retain_row = self.retain_data[retain_idx]

        forget_sample = convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            forget_row["question"],
            forget_row["answer"],
            self.model_configs,
            self.language,
        )
        retain_sample = convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            retain_row["question"],
            retain_row["answer"],
            self.model_configs,
            self.language,
        )
        return forget_sample, retain_sample


def rcp_collator(samples):
    """Collate (forget, retain) pairs into stacked tensors.

    Returns: [forget_batch, retain_batch]
      where each batch is (input_ids, labels, attention_mask).
    """
    rets = []
    for part_idx in range(2):  # 0=forget, 1=retain
        data = [sample[part_idx] for sample in samples]
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets
