"""Datasets for probe-based representation erasure."""

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
        return dataset[next(iter(dataset.keys()))]
    return dataset


def load_qa_dataset(entry, default_split=None):
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


def question_only_to_model_format(tokenizer, max_length, question, model_configs, language="en"):
    use_chat_template = str(model_configs.get("use_chat_template", "false")).lower() == "true"
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            raise ValueError("use_chat_template=true but tokenizer has no chat template.")
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
        )
    else:
        question_start = model_configs["question_start_tag"][language]
        question_end = model_configs["question_end_tag"]
        answer_tag = model_configs["answer_tag"][language]
        prompt_text = question_start + question + question_end + answer_tag
        encoded = tokenizer(
            prompt_text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    pad_length = max_length - len(encoded["input_ids"])
    input_ids = encoded["input_ids"] + [eos_token_id] * pad_length
    attention_mask = encoded["attention_mask"] + [0] * pad_length
    return torch.tensor(input_ids), torch.tensor(attention_mask)


def answer_to_model_format(tokenizer, max_length, answer):
    encoded = tokenizer(
        str(answer),
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if not input_ids:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = [eos_token_id]
        attention_mask = [1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    pad_length = max_length - len(input_ids)
    return (
        torch.tensor(input_ids + [pad_token_id] * pad_length),
        torch.tensor(attention_mask + [0] * pad_length),
    )


def _dedupe_texts(texts):
    seen = set()
    output = []
    for text in texts:
        text = str(text)
        key = _normalize_text(text)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _as_text_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


class RepErasureDataset(Dataset):
    """Return forget question-only inputs, answer candidates, and preserve batches."""

    def __init__(self, cfg, tokenizer, model_family):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.max_length = int(cfg.max_length)
        self.answer_max_length = int(cfg.get("answer_max_length", 128))
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.question_key = cfg.get("question_key", "question")
        self.answer_key = cfg.get("answer_key", "answer")
        self.paraphrased_answer_key = cfg.get("paraphrased_answer_key", "paraphrased_answer")
        self.perturbed_answer_key = cfg.get("perturbed_answer_key", "perturbed_answer")
        self.utility_language = cfg.get("utility_language", None)
        self.target_language = cfg.get("primary_target_language", "en")
        probe_cfg = cfg.get("probe", {})
        self.max_positive_targets = int(_get(probe_cfg, "max_positive_targets", 2))
        self.max_negative_targets = int(_get(probe_cfg, "max_negative_targets", 0))
        self.use_paraphrase_positive = bool(_get(probe_cfg, "use_paraphrase_positive", True))
        self.use_perturbed_negatives = bool(_get(probe_cfg, "use_perturbed_negatives", False))

        self.forget_data = load_qa_dataset(cfg.forget_data_path, cfg.get("forget_split", cfg.get("split", None)))
        self.forget_probe_data = load_qa_dataset(
            cfg.get("forget_probe_data_path", cfg.forget_data_path),
            cfg.get("forget_probe_split", cfg.get("forget_split", cfg.get("split", None))),
        )
        self.retain_data = load_qa_dataset(cfg.retain_data_path, cfg.get("retain_split_name", cfg.get("retain_split", None)))
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
        return len(self.forget_probe_data)

    def _sample_index(self, idx, dataset):
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")
        offset = torch.randint(0, len(dataset), (1,)).item()
        return (idx + offset) % len(dataset)

    def _row_language(self, row, default_language):
        return row.get("language", default_language)

    def _qa_convert(self, row, default_language):
        language = self._row_language(row, default_language)
        return convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            row[self.question_key],
            row[self.answer_key],
            self.model_configs,
            language,
        )

    def _question_convert(self, row, default_language):
        language = self._row_language(row, default_language)
        return question_only_to_model_format(
            self.tokenizer,
            self.max_length,
            row[self.question_key],
            self.model_configs,
            language,
        )

    def _candidate_texts(self, row):
        positives = [row[self.answer_key]]
        if self.use_paraphrase_positive and self.paraphrased_answer_key in row:
            positives.extend(_as_text_list(row.get(self.paraphrased_answer_key)))
        positives = _dedupe_texts(positives)[:self.max_positive_targets]
        while len(positives) < self.max_positive_targets:
            positives.append(positives[0])

        negatives = []
        original_negative_count = 0
        if self.use_perturbed_negatives and self.perturbed_answer_key in row:
            negatives = _dedupe_texts(_as_text_list(row.get(self.perturbed_answer_key)))[:self.max_negative_targets]
            original_negative_count = len(negatives)
        while len(negatives) < self.max_negative_targets:
            negatives.append(positives[0])

        candidates = positives + negatives
        positive_mask = [1] * self.max_positive_targets + [0] * self.max_negative_targets
        candidate_mask = [1] * self.max_positive_targets + [idx < original_negative_count for idx in range(self.max_negative_targets)]
        return candidates, positive_mask, candidate_mask

    def _candidate_convert(self, row):
        candidates, positive_mask, candidate_mask = self._candidate_texts(row)
        answer_ids = []
        answer_masks = []
        for text in candidates:
            input_ids, attention_mask = answer_to_model_format(self.tokenizer, self.answer_max_length, text)
            answer_ids.append(input_ids)
            answer_masks.append(attention_mask)
        return (
            torch.stack(answer_ids),
            torch.stack(answer_masks),
            torch.tensor(candidate_mask, dtype=torch.bool),
            torch.tensor(positive_mask, dtype=torch.bool),
        )

    def __getitem__(self, idx):
        retain_idx = self._sample_index(idx, self.retain_data)
        utility_idx = self._sample_index(idx, self.utility_data)
        parallel_idx = self._sample_index(idx, self.parallel_en_data)
        parallel_tgt_idx = parallel_idx % len(self.parallel_tgt_data)

        forget_row = self.forget_probe_data[idx]
        retain_row = self.retain_data[retain_idx]
        utility_row = self.utility_data[utility_idx]

        forget_question = self._question_convert(forget_row, "en")
        answer_candidates = self._candidate_convert(forget_row)
        retain_inputs = self._qa_convert(retain_row, "en")
        utility_inputs = self._qa_convert(
            utility_row,
            self.utility_language or self._row_language(utility_row, "en"),
        )
        parallel_en_inputs = self._question_convert(self.parallel_en_data[parallel_idx], self.parallel_en_language)
        parallel_tgt_inputs = self._question_convert(self.parallel_tgt_data[parallel_tgt_idx], self.parallel_tgt_language)
        return (
            forget_question,
            answer_candidates,
            retain_inputs,
            utility_inputs,
            parallel_en_inputs,
            parallel_tgt_inputs,
        )


def _stack_tuple(samples):
    return tuple(torch.stack([sample[idx] for sample in samples]) for idx in range(len(samples[0])))


def rep_erasure_collator(samples):
    forget_questions = _stack_tuple([sample[0] for sample in samples])
    answer_candidates = _stack_tuple([sample[1] for sample in samples])
    retain_inputs = _stack_tuple([sample[2] for sample in samples])
    utility_inputs = _stack_tuple([sample[3] for sample in samples])
    parallel_en_inputs = _stack_tuple([sample[4] for sample in samples])
    parallel_tgt_inputs = _stack_tuple([sample[5] for sample in samples])
    return (
        forget_questions,
        answer_candidates,
        retain_inputs,
        utility_inputs,
        parallel_en_inputs,
        parallel_tgt_inputs,
    )
