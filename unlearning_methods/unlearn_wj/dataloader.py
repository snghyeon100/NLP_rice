"""Data loading for WJ likelihood-based cross-lingual unlearning."""

import random
from pathlib import Path

import datasets
import torch
from torch.utils.data import Dataset

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


def _resolve_path(path, project_root):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((project_root / path).resolve())
    return path


def _load_dataset(path, split=None, project_root=None):
    """Load either a HuggingFace dataset split or a local load_from_disk dataset."""
    resolved = _resolve_path(path, project_root or Path.cwd())
    if resolved is not None and Path(resolved).exists():
        loaded = datasets.load_from_disk(resolved)
        return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded
    if split is None:
        loaded = datasets.load_dataset(path)
    else:
        loaded = datasets.load_dataset(path, split)
    return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded


def _retain_split_from_forget(forget_split):
    if not str(forget_split).startswith("forget"):
        raise ValueError(f"Cannot infer retain split from {forget_split!r}.")
    forget_pct = int(str(forget_split).replace("forget", ""))
    return "retain" + str(100 - forget_pct).zfill(2)


def _first_answer(value):
    if isinstance(value, list):
        if not value:
            raise ValueError("Empty answer list encountered.")
        return value[0]
    return value


class WJUnlearningDataset(Dataset):
    """Return source forget plus source/multilingual preservation examples.

    The dataset length follows the source forget set. Retain and utility examples
    are sampled randomly, so each epoch sees different preservation pairs.
    """

    def __init__(
        self,
        cfg,
        tokenizer,
        project_root,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.project_root = Path(project_root)
        self.max_length = int(cfg.max_length)
        self.source_language = str(cfg.source_language)
        self.question_key = str(cfg.question_key)
        self.answer_key = str(cfg.answer_key)
        self.model_configs = get_model_identifiers_from_yaml(cfg.model_family)

        self.forget_source = _load_dataset(
            cfg.source_data_path,
            split=cfg.forget_split,
            project_root=self.project_root,
        )
        source_retain_split = cfg.get("source_retain_split", None)
        if source_retain_split is None:
            source_retain_split = _retain_split_from_forget(cfg.forget_split)
        self.retain_source = _load_dataset(
            cfg.source_data_path,
            split=source_retain_split,
            project_root=self.project_root,
        )
        self.retain_multi = _load_dataset(cfg.retain_multi_path, project_root=self.project_root)
        utility_path = cfg.utility_multi_path if cfg.utility_multi_path is not None else cfg.retain_multi_path
        self.utility_multi = _load_dataset(utility_path, project_root=self.project_root)

        max_train_examples = cfg.get("max_train_examples", None)
        if max_train_examples is not None:
            max_train_examples = int(max_train_examples)
            self.forget_source = self.forget_source.select(range(min(max_train_examples, len(self.forget_source))))

    def __len__(self):
        return len(self.forget_source)

    def _convert(self, row, language):
        question = row[self.question_key]
        answer = _first_answer(row[self.answer_key])
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            question,
            answer,
            self.model_configs,
            language,
        )

    def _random_row(self, data):
        return data[random.randrange(len(data))]

    def _row_language(self, row, fallback):
        return row["language"] if "language" in row else fallback

    def __getitem__(self, idx):
        forget_row = self.forget_source[idx]
        retain_source_row = self._random_row(self.retain_source)
        retain_multi_row = self._random_row(self.retain_multi)
        utility_multi_row = self._random_row(self.utility_multi)

        return {
            "forget_source": self._convert(forget_row, self.source_language),
            "retain_source": self._convert(retain_source_row, self.source_language),
            "retain_multi": self._convert(
                retain_multi_row,
                self._row_language(retain_multi_row, self.source_language),
            ),
            "utility_multi": self._convert(
                utility_multi_row,
                self._row_language(utility_multi_row, self.source_language),
            ),
        }


class ParallelAnchorDataset(Dataset):
    """Aligned source-target samples for hidden-alignment layer localization."""

    def __init__(self, cfg, tokenizer, project_root):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.project_root = Path(project_root)
        self.max_length = int(cfg.max_length)
        self.source_language = str(cfg.source_language)
        self.question_key = str(cfg.question_key)
        self.answer_key = str(cfg.answer_key)
        self.model_configs = get_model_identifiers_from_yaml(cfg.model_family)

        data = _load_dataset(cfg.parallel_anchor_path, project_root=self.project_root)
        by_language = {}
        for row in data:
            lang = row["language"] if "language" in row else self.source_language
            by_language.setdefault(lang, []).append(row)

        if self.source_language not in by_language:
            raise ValueError(
                f"parallel_anchor_path has no source language {self.source_language!r}."
            )

        self.pairs = []
        source_rows = by_language[self.source_language]
        for target_language in list(cfg.target_languages):
            target_rows = by_language.get(str(target_language), [])
            pair_count = min(len(source_rows), len(target_rows))
            for i in range(pair_count):
                self.pairs.append((source_rows[i], target_rows[i], str(target_language)))

        if not self.pairs:
            raise ValueError("No source-target parallel anchor pairs were found.")

    def __len__(self):
        return len(self.pairs)

    def _convert(self, row, language):
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            row[self.question_key],
            _first_answer(row[self.answer_key]),
            self.model_configs,
            language,
        )

    def __getitem__(self, idx):
        source_row, target_row, target_language = self.pairs[idx]
        return {
            "source": self._convert(source_row, self.source_language),
            "target": self._convert(target_row, target_language),
            "target_language": target_language,
        }


def _stack_tuples(samples):
    input_ids = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]
    attention_mask = [sample[2] for sample in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)


def wj_collator(samples):
    keys = ["forget_source", "retain_source", "retain_multi", "utility_multi"]
    return {key: _stack_tuples([sample[key] for sample in samples]) for key in keys}


def parallel_anchor_collator(samples):
    return {
        "source": _stack_tuples([sample["source"] for sample in samples]),
        "target": _stack_tuples([sample["target"] for sample in samples]),
        "target_language": [sample["target_language"] for sample in samples],
    }
