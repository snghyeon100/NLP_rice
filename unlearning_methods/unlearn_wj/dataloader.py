"""Datasets for subspace-aware cross-lingual unlearning."""

from pathlib import Path

import datasets
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


def _as_plain(value):
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _get(mapping, key, default=None):
    if mapping is None:
        return default
    if isinstance(mapping, DictConfig):
        return mapping.get(key, default)
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _select_train_split(dataset):
    if isinstance(dataset, datasets.DatasetDict):
        if "train" in dataset:
            return dataset["train"]
        first_key = next(iter(dataset.keys()))
        return dataset[first_key]
    return dataset


def load_qa_dataset(entry, default_split=None):
    """Load a local HF dataset-from-disk or a hub dataset/config pair."""

    entry = _as_plain(entry)
    if isinstance(entry, dict):
        path = entry["path"]
        split = entry.get("split", default_split)
    else:
        path = entry
        split = default_split

    path = str(path)
    if Path(path).exists():
        return _select_train_split(datasets.load_from_disk(path))

    if split:
        return _select_train_split(datasets.load_dataset(path, split))
    return _select_train_split(datasets.load_dataset(path))


def _normalize_text(value):
    return " ".join(str(value).casefold().split())


def _row_signature(row, question_key, answer_key):
    return (_normalize_text(row.get(question_key, "")), _normalize_text(row.get(answer_key, "")))


def decontaminate_by_forget_set(utility_data, forget_data, question_key="question", answer_key="answer"):
    """Remove exact QA overlaps with the English forget set.

    This is an exact-string guard. Entity alias and translated paraphrase
    decontamination still needs an external fact/entity map.
    """

    forget_signatures = {
        _row_signature(row, question_key, answer_key)
        for row in forget_data
    }
    clean_indices = [
        idx
        for idx, row in enumerate(utility_data)
        if _row_signature(row, question_key, answer_key) not in forget_signatures
    ]
    return utility_data.select(clean_indices)


class SubspaceXlingualDataset(Dataset):
    """Return forget, retain, utility, and parallel batches.

    Target-language forget/retain evaluation datasets are intentionally not
    loaded here so zero-shot training cannot accidentally use them.
    """

    def __init__(
        self,
        cfg,
        tokenizer,
        model_family,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_length = int(cfg.max_length)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.question_key = cfg.get("question_key", "question")
        self.answer_key = cfg.get("answer_key", "answer")
        self.utility_language = cfg.get("utility_language", None)
        self.target_language = cfg.get("primary_target_language", "en")

        self.forget_data = load_qa_dataset(cfg.forget_data_path, cfg.get("forget_split", cfg.get("split", None)))
        self.retain_data = load_qa_dataset(cfg.retain_data_path, cfg.get("retain_split", None))
        utility_path = cfg.get("utility_data_path", None)
        self.utility_data = load_qa_dataset(utility_path) if utility_path else self.retain_data
        if cfg.get("decontaminate_utility", True):
            self.utility_data = decontaminate_by_forget_set(
                self.utility_data,
                self.forget_data,
                question_key=self.question_key,
                answer_key=self.answer_key,
            )

        parallel_cfg = cfg.get("parallel_utility_data_path", None)
        if parallel_cfg is None:
            self.parallel_en_data = self.retain_data
            self.parallel_tgt_data = self.utility_data
            self.parallel_en_language = "en"
            self.parallel_tgt_language = self.target_language
        else:
            self.parallel_en_data = load_qa_dataset(parallel_cfg.en, _get(parallel_cfg.en, "split"))
            self.parallel_tgt_data = load_qa_dataset(parallel_cfg.tgt, _get(parallel_cfg.tgt, "split"))
            self.parallel_en_language = _get(parallel_cfg.en, "language", "en")
            self.parallel_tgt_language = _get(parallel_cfg.tgt, "language", self.target_language)

    def __len__(self):
        return len(self.forget_data)

    def _sample_index(self, idx, dataset):
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
        offset = torch.randint(0, len(dataset), (1,)).item()
        return (idx + offset) % len(dataset)

    def _row_language(self, row, default_language):
        return row.get("language", default_language)

    def _convert(self, row, default_language):
        language = self._row_language(row, default_language)
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            row[self.question_key],
            row[self.answer_key],
            self.model_configs,
            language,
        )

    def __getitem__(self, idx):
        retain_idx = self._sample_index(idx, self.retain_data)
        utility_idx = self._sample_index(idx, self.utility_data)
        parallel_idx = self._sample_index(idx, self.parallel_en_data)
        parallel_tgt_idx = parallel_idx % len(self.parallel_tgt_data)

        forget_inputs = self._convert(self.forget_data[idx], "en")
        retain_inputs = self._convert(self.retain_data[retain_idx], "en")
        utility_inputs = self._convert(
            self.utility_data[utility_idx],
            self.utility_language or self._row_language(self.utility_data[utility_idx], "en"),
        )
        parallel_en_inputs = self._convert(self.parallel_en_data[parallel_idx], self.parallel_en_language)
        parallel_tgt_inputs = self._convert(self.parallel_tgt_data[parallel_tgt_idx], self.parallel_tgt_language)
        return forget_inputs, retain_inputs, utility_inputs, parallel_en_inputs, parallel_tgt_inputs


def subspace_xlingual_collator(samples):
    batches = []
    for part_idx in range(5):
        part = [sample[part_idx] for sample in samples]
        input_ids = torch.stack([sample[0] for sample in part])
        labels = torch.stack([sample[1] for sample in part])
        attention_mask = torch.stack([sample[2] for sample in part])
        batches.append((input_ids, labels, attention_mask))
    return tuple(batches)
