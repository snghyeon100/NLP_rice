import csv
import gc
import json
import math
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import datasets
import hydra
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_module import convert_raw_data_to_model_format
from utils import get_model_identifiers_from_yaml


def cfg_get(section, key, default=None):
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return section[key] if key in section and section[key] is not None else default


def path_or_model_id(value):
    if value is None:
        return None
    value = str(value)
    if value.startswith(("/", "./", "../", "~")):
        return to_absolute_path(os.path.expanduser(value))
    return value


def resolve_torch_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype_name in {"float16", "fp16"}:
        return torch.float16
    if dtype_name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_tokenizer(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_eval_tokenizer(cfg, model_configs):
    explicit_tokenizer = cfg_get(cfg, "tokenizer_path", None)
    prefer_checkpoint = bool(cfg_get(cfg, "prefer_checkpoint_tokenizer", True))

    candidates = []
    if explicit_tokenizer is not None:
        candidates.append(("tokenizer_path", explicit_tokenizer))
    elif prefer_checkpoint and cfg_get(cfg, "model_path", None) is not None:
        candidates.append(("model_path", cfg.model_path))
        candidates.append(("base_hf_key", model_configs["hf_key"]))
    else:
        candidates.append(("base_hf_key", model_configs["hf_key"]))

    errors = []
    for label, source in candidates:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                path_or_model_id(source),
                trust_remote_code=bool(cfg_get(cfg, "trust_remote_code", True)),
            )
            print(f"Loaded tokenizer from {label}: {source}")
            return ensure_tokenizer(tokenizer)
        except Exception as exc:
            errors.append(f"{label}={source}: {exc}")
            if explicit_tokenizer is not None:
                break

    raise RuntimeError("Could not load tokenizer. Tried:\n" + "\n".join(errors))


def normalize_eval_text(text, language, cfg):
    form = cfg_get(cfg, "unicode_normalization", None)
    if form is None or str(form).lower() in {"", "none", "false"}:
        return text

    languages = cfg_get(cfg, "normalize_languages", None)
    if isinstance(languages, str):
        languages = {languages}
    elif languages is not None:
        languages = set(languages)
    if languages is not None and language not in languages:
        return text

    return unicodedata.normalize(str(form), text)


def get_row_language(row, cfg):
    language_key = cfg_get(cfg, "language_key", "language")
    default_language = cfg_get(cfg, "language", "en")
    if language_key is not None and language_key in row:
        return row.get(language_key, default_language)
    return default_language


def truth_ratio_enabled(cfg):
    truth_cfg = cfg_get(cfg, "truth_ratio", None)
    return bool(cfg_get(truth_cfg, "enabled", False))


def normalize_for_match(text, language, cfg):
    match_cfg = cfg_get(cfg, "match_normalization", None)
    text = normalize_eval_text(str(text), language, cfg)

    if bool(cfg_get(match_cfg, "casefold", True)):
        text = text.casefold()
    if bool(cfg_get(match_cfg, "collapse_whitespace", True)):
        text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if bool(cfg_get(match_cfg, "strip_terminal_punctuation", True)):
        while text and unicodedata.category(text[0]).startswith("P"):
            text = text[1:].strip()
        while text and unicodedata.category(text[-1]).startswith("P"):
            text = text[:-1].strip()

    return text


def generation_match_metrics(generated, gold, language, cfg):
    normalized_generated = normalize_for_match(generated, language, cfg)
    normalized_gold = normalize_for_match(gold, language, cfg)
    contains = bool(normalized_gold) and normalized_gold in normalized_generated
    return {
        "generated_answer": generated,
        "normalized_generated_answer": normalized_generated,
        "normalized_gold_answer": normalized_gold,
        "normalized_exact_match": normalized_generated == normalized_gold,
        "normalized_contains": contains,
    }


class FullTofuDataset(Dataset):
    def __init__(self, cfg, tokenizer, model_configs):
        loaded = datasets.load_from_disk(path_or_model_id(cfg.data_path))
        self.data = loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.model_configs = model_configs
        self.max_length = cfg.max_length
        self.question_key = cfg.question_key
        self.answer_key = cfg.answer_key
        self.language_key = cfg_get(cfg, "language_key", "language")
        self.truth_cfg = cfg_get(cfg, "truth_ratio", None)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        language = get_row_language(row, self.cfg)
        question = normalize_eval_text(row[self.question_key], language, self.cfg)
        answer = normalize_eval_text(row[self.answer_key], language, self.cfg)
        input_ids, labels, attention_mask = convert_raw_data_to_model_format(
            self.tokenizer,
            self.max_length,
            question,
            answer,
            self.model_configs,
            language=language,
        )
        return {
            "index": idx,
            "language": language,
            "question": question,
            "answer": answer,
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        if truth_ratio_enabled(self.cfg):
            correct_answer_key = cfg_get(self.truth_cfg, "correct_answer_key", "paraphrased_answer")
            perturbed_answer_key = cfg_get(self.truth_cfg, "perturbed_answer_key", "perturbed_answer")
            if correct_answer_key not in row:
                raise KeyError(f"truth_ratio.correct_answer_key={correct_answer_key!r} not found in dataset row.")
            if perturbed_answer_key not in row:
                raise KeyError(f"truth_ratio.perturbed_answer_key={perturbed_answer_key!r} not found in dataset row.")

            correct_answer = normalize_eval_text(row[correct_answer_key], language, self.cfg)
            truth_input_ids, truth_labels, truth_attention_mask = convert_raw_data_to_model_format(
                self.tokenizer,
                self.max_length,
                question,
                correct_answer,
                self.model_configs,
                language=language,
            )

            perturbed_answers = row[perturbed_answer_key]
            if isinstance(perturbed_answers, str):
                perturbed_answers = [perturbed_answers]

            perturb_input_ids = []
            perturb_labels = []
            perturb_attention_mask = []
            for perturbed_answer in perturbed_answers:
                perturbed_answer = normalize_eval_text(perturbed_answer, language, self.cfg)
                converted = convert_raw_data_to_model_format(
                    self.tokenizer,
                    self.max_length,
                    question,
                    perturbed_answer,
                    self.model_configs,
                    language=language,
                )
                perturb_input_ids.append(converted[0])
                perturb_labels.append(converted[1])
                perturb_attention_mask.append(converted[2])

            sample.update(
                {
                    "truth_correct_answer": correct_answer,
                    "truth_input_ids": truth_input_ids,
                    "truth_labels": truth_labels,
                    "truth_attention_mask": truth_attention_mask,
                    "perturb_input_ids": torch.stack(perturb_input_ids),
                    "perturb_labels": torch.stack(perturb_labels),
                    "perturb_attention_mask": torch.stack(perturb_attention_mask),
                }
            )
        return sample


def collate_full_tofu(samples):
    batch = {
        "indices": [sample["index"] for sample in samples],
        "languages": [sample["language"] for sample in samples],
        "questions": [sample["question"] for sample in samples],
        "answers": [sample["answer"] for sample in samples],
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in samples]),
    }
    if "truth_input_ids" in samples[0]:
        perturb_input_ids = []
        perturb_labels = []
        perturb_attention_mask = []
        perturb_row_indices = []
        for row_idx, sample in enumerate(samples):
            num_perturbs = sample["perturb_input_ids"].shape[0]
            perturb_input_ids.append(sample["perturb_input_ids"])
            perturb_labels.append(sample["perturb_labels"])
            perturb_attention_mask.append(sample["perturb_attention_mask"])
            perturb_row_indices.extend([row_idx] * num_perturbs)

        batch.update(
            {
                "truth_correct_answers": [sample["truth_correct_answer"] for sample in samples],
                "truth_input_ids": torch.stack([sample["truth_input_ids"] for sample in samples]),
                "truth_labels": torch.stack([sample["truth_labels"] for sample in samples]),
                "truth_attention_mask": torch.stack([sample["truth_attention_mask"] for sample in samples]),
                "perturb_input_ids": torch.cat(perturb_input_ids, dim=0),
                "perturb_labels": torch.cat(perturb_labels, dim=0),
                "perturb_attention_mask": torch.cat(perturb_attention_mask, dim=0),
                "perturb_row_indices": torch.tensor(perturb_row_indices),
            }
        )
    return batch


def answer_token_stats(logits, labels):
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)

    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_loss = F.cross_entropy(
        shifted_logits.transpose(1, 2),
        safe_labels,
        reduction="none",
    ).float()
    token_loss = token_loss * mask
    token_count = mask.sum(dim=-1).clamp_min(1)
    loss_sum = token_loss.sum(dim=-1)
    avg_nll = loss_sum / token_count

    preds = shifted_logits.argmax(dim=-1)
    correct = (preds.eq(shifted_labels) & mask).sum(dim=-1)
    token_accuracy = correct.float() / token_count.float()

    return {
        "loss_sum": loss_sum.detach().cpu(),
        "avg_nll": avg_nll.detach().cpu(),
        "token_count": token_count.detach().cpu(),
        "token_correct": correct.detach().cpu(),
        "token_accuracy": token_accuracy.detach().cpu(),
    }


def grouped_mean(values, group_indices, num_groups):
    sums = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    counts = torch.zeros(num_groups, dtype=values.dtype, device=values.device)
    sums.index_add_(0, group_indices, values)
    counts.index_add_(0, group_indices, torch.ones_like(values))
    return sums / counts.clamp_min(1)


def build_prompt(question, language, tokenizer, model_configs):
    use_chat_template = str(model_configs.get("use_chat_template", "false")).lower() == "true"
    if use_chat_template:
        if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
            raise ValueError("use_chat_template=true but tokenizer has no chat template.")
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    question_start = model_configs["question_start_tag"][language]
    question_end = model_configs["question_end_tag"]
    answer_tag = model_configs["answer_tag"][language]
    return question_start + question + question_end + answer_tag


def load_model(model_path, cfg, model_configs):
    model_path = path_or_model_id(model_path)
    if model_path is None:
        raise ValueError("model_path must be provided.")

    kwargs = {
        "torch_dtype": resolve_torch_dtype(cfg.dtype),
        "trust_remote_code": bool(cfg.trust_remote_code),
        "low_cpu_mem_usage": bool(cfg.low_cpu_mem_usage),
    }
    attn_implementation = cfg.attn_implementation
    if attn_implementation == "auto":
        attn_implementation = "flash_attention_2" if model_configs.get("flash_attention2") == "true" else None
    if attn_implementation:
        kwargs["attn_implementation"] = str(attn_implementation)
    if cfg.device_map is not None:
        kwargs["device_map"] = cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    if cfg.device_map is None:
        model.to(cfg.device)
    return model


def evaluate_model(model, dataloader, cfg, model_label):
    results = {}
    device = next(model.parameters()).device
    truth_cfg = cfg_get(cfg, "truth_ratio", None)
    truth_mode = str(cfg_get(truth_cfg, "mode", "retain")).lower()
    progress = tqdm(dataloader, desc=f"Evaluating {model_label}")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            stats = answer_token_stats(outputs.logits, labels)

            truth_stats = None
            perturb_avg_nll = None
            raw_truth_ratio = None
            truth_ratio_score = None
            if "truth_input_ids" in batch:
                truth_outputs = model(
                    input_ids=batch["truth_input_ids"].to(device),
                    attention_mask=batch["truth_attention_mask"].to(device),
                )
                truth_stats = answer_token_stats(truth_outputs.logits, batch["truth_labels"].to(device))

                perturb_outputs = model(
                    input_ids=batch["perturb_input_ids"].to(device),
                    attention_mask=batch["perturb_attention_mask"].to(device),
                )
                perturb_stats = answer_token_stats(perturb_outputs.logits, batch["perturb_labels"].to(device))
                perturb_row_indices = batch["perturb_row_indices"].to(device)
                perturb_avg_nll = grouped_mean(
                    perturb_stats["avg_nll"].to(device),
                    perturb_row_indices,
                    input_ids.shape[0],
                ).detach().cpu()

                correct_avg_nll = truth_stats["avg_nll"]
                raw_truth_ratio = torch.exp(correct_avg_nll - perturb_avg_nll)
                if truth_mode == "forget":
                    truth_ratio_score = torch.minimum(raw_truth_ratio, 1 / raw_truth_ratio.clamp_min(1e-12))
                else:
                    truth_ratio_score = torch.clamp(1 - raw_truth_ratio, min=0)

        for row_idx, dataset_idx in enumerate(batch["indices"]):
            token_count = int(stats["token_count"][row_idx].item())
            token_correct = int(stats["token_correct"][row_idx].item())
            avg_nll = float(stats["avg_nll"][row_idx].item())
            row_result = {
                f"{model_label}_loss_sum": float(stats["loss_sum"][row_idx].item()),
                f"{model_label}_avg_nll": avg_nll,
                f"{model_label}_perplexity": safe_exp(avg_nll),
                f"{model_label}_prob_correct": safe_exp(-avg_nll),
                f"{model_label}_token_count": token_count,
                f"{model_label}_token_correct": token_correct,
                f"{model_label}_token_accuracy": float(stats["token_accuracy"][row_idx].item()),
            }
            if truth_stats is not None:
                truth_avg_nll = float(truth_stats["avg_nll"][row_idx].item())
                perturb_nll = float(perturb_avg_nll[row_idx].item())
                row_result.update(
                    {
                        f"{model_label}_truth_correct_avg_nll": truth_avg_nll,
                        f"{model_label}_truth_correct_prob": safe_exp(-truth_avg_nll),
                        f"{model_label}_perturb_avg_nll": perturb_nll,
                        f"{model_label}_perturb_prob": safe_exp(-perturb_nll),
                        f"{model_label}_truth_ratio_raw": float(raw_truth_ratio[row_idx].item()),
                        f"{model_label}_truth_ratio_score": float(truth_ratio_score[row_idx].item()),
                    }
                )
            results[dataset_idx] = row_result
    return results


def safe_exp(value):
    if value is None:
        return None
    if value > 80:
        return None
    if value < -80:
        return 0.0
    return float(math.exp(value))


def merge_records(dataset, ft_results, base_results=None, ft_generation_results=None, base_generation_results=None):
    records = []
    for idx in range(len(dataset)):
        row = dataset.data[idx]
        language = get_row_language(row, dataset.cfg)
        record = {
            "index": idx,
            "language": language,
            "question": normalize_eval_text(row[dataset.question_key], language, dataset.cfg),
            "gold_answer": normalize_eval_text(row[dataset.answer_key], language, dataset.cfg),
        }
        record.update(ft_results[idx])
        if ft_generation_results is not None and idx in ft_generation_results:
            record.update({f"ft_{key}": value for key, value in ft_generation_results[idx].items()})
        if base_results is not None:
            record.update(base_results[idx])
            if base_generation_results is not None and idx in base_generation_results:
                record.update({f"base_{key}": value for key, value in base_generation_results[idx].items()})
            record["delta_avg_nll"] = record["base_avg_nll"] - record["ft_avg_nll"]
            record["delta_token_accuracy"] = record["ft_token_accuracy"] - record["base_token_accuracy"]
            record["delta_prob_correct"] = record["ft_prob_correct"] - record["base_prob_correct"]
            if record["base_prob_correct"] not in {0, None}:
                record["prob_ratio_ft_to_base"] = record["ft_prob_correct"] / record["base_prob_correct"]
            if "ft_truth_ratio_score" in record and "base_truth_ratio_score" in record:
                record["delta_truth_ratio_score"] = record["ft_truth_ratio_score"] - record["base_truth_ratio_score"]
            if "ft_normalized_exact_match" in record and "base_normalized_exact_match" in record:
                record["delta_normalized_exact_match"] = (
                    float(record["ft_normalized_exact_match"]) - float(record["base_normalized_exact_match"])
                )
        records.append(record)
    return records


def mean_present(records, key):
    values = [float(row[key]) for row in records if key in row and row[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_records(records, prefix):
    loss_sum = sum(float(row[f"{prefix}_loss_sum"]) for row in records)
    token_count = sum(int(row[f"{prefix}_token_count"]) for row in records)
    token_correct = sum(int(row[f"{prefix}_token_correct"]) for row in records)
    avg_nll = loss_sum / max(1, token_count)
    result = {
        "num_examples": len(records),
        "num_answer_tokens": token_count,
        "avg_nll": avg_nll,
        "perplexity": safe_exp(avg_nll),
        "prob_correct": mean_present(records, f"{prefix}_prob_correct"),
        "token_accuracy": token_correct / max(1, token_count),
    }
    truth_ratio_score = mean_present(records, f"{prefix}_truth_ratio_score")
    if truth_ratio_score is not None:
        result.update(
            {
                "truth_correct_prob": mean_present(records, f"{prefix}_truth_correct_prob"),
                "perturb_prob": mean_present(records, f"{prefix}_perturb_prob"),
                "truth_ratio_raw": mean_present(records, f"{prefix}_truth_ratio_raw"),
                "truth_ratio_score": truth_ratio_score,
            }
        )
    exact_match = mean_present(records, f"{prefix}_normalized_exact_match")
    if exact_match is not None:
        result.update(
            {
                "normalized_exact_match": exact_match,
                "normalized_contains": mean_present(records, f"{prefix}_normalized_contains"),
            }
        )
    return result


def aggregate_by_language(records, compare_base):
    groups = defaultdict(list)
    for record in records:
        groups[record["language"]].append(record)

    rows = []
    for language in sorted(groups):
        lang_records = groups[language]
        ft = aggregate_records(lang_records, "ft")
        row = {
            "language": language,
            "num_examples": ft["num_examples"],
            "num_answer_tokens": ft["num_answer_tokens"],
            "ft_avg_nll": ft["avg_nll"],
            "ft_perplexity": ft["perplexity"],
            "ft_prob_correct": ft["prob_correct"],
            "ft_token_accuracy": ft["token_accuracy"],
        }
        for metric in [
            "truth_correct_prob",
            "perturb_prob",
            "truth_ratio_raw",
            "truth_ratio_score",
            "normalized_exact_match",
            "normalized_contains",
        ]:
            if metric in ft:
                row[f"ft_{metric}"] = ft[metric]
        if compare_base:
            base = aggregate_records(lang_records, "base")
            row.update(
                {
                    "base_avg_nll": base["avg_nll"],
                    "base_perplexity": base["perplexity"],
                    "base_prob_correct": base["prob_correct"],
                    "base_token_accuracy": base["token_accuracy"],
                    "delta_avg_nll": base["avg_nll"] - ft["avg_nll"],
                    "delta_prob_correct": ft["prob_correct"] - base["prob_correct"],
                    "delta_token_accuracy": ft["token_accuracy"] - base["token_accuracy"],
                }
            )
            if base["prob_correct"] not in {0, None}:
                row["prob_ratio_ft_to_base"] = ft["prob_correct"] / base["prob_correct"]
            for metric in [
                "truth_correct_prob",
                "perturb_prob",
                "truth_ratio_raw",
                "truth_ratio_score",
                "normalized_exact_match",
                "normalized_contains",
            ]:
                if metric in base:
                    row[f"base_{metric}"] = base[metric]
                if metric in ft and metric in base:
                    row[f"delta_{metric}"] = ft[metric] - base[metric]
        rows.append(row)
    return rows


def select_case_studies(records, cfg):
    case_cfg = cfg.case_study
    rng = random.Random(int(case_cfg.seed))
    by_language = defaultdict(list)
    for record in records:
        by_language[record["language"]].append(record)

    selected = {}

    def add(record, tag):
        idx = record["index"]
        if idx not in selected:
            selected[idx] = dict(record)
            selected[idx]["case_tags"] = []
        if tag not in selected[idx]["case_tags"]:
            selected[idx]["case_tags"].append(tag)

    for language in sorted(by_language):
        rows = by_language[language]
        best = sorted(rows, key=lambda row: row["ft_avg_nll"])
        worst = sorted(rows, key=lambda row: row["ft_avg_nll"], reverse=True)
        for row in best[: int(case_cfg.num_best_per_language)]:
            add(row, "best_ft")
        for row in worst[: int(case_cfg.num_worst_per_language)]:
            add(row, "worst_ft")
        if cfg.compare_base:
            improved = sorted(rows, key=lambda row: row.get("delta_avg_nll", float("-inf")), reverse=True)
            for row in improved[: int(case_cfg.num_improved_per_language)]:
                add(row, "largest_base_to_ft_improvement")
        random_count = int(case_cfg.num_random_per_language)
        if random_count > 0:
            for row in rng.sample(rows, min(random_count, len(rows))):
                add(row, "random_seeded")

    selected_records = sorted(selected.values(), key=lambda row: (row["language"], row["index"]))
    max_total = cfg_get(case_cfg, "max_total", None)
    if max_total is not None:
        selected_records = selected_records[: int(max_total)]
    return selected_records


def generate_outputs(model, tokenizer, model_configs, selected_records, cfg, output_key):
    if not selected_records:
        return
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"
    prompts = [
        build_prompt(record["question"], record["language"], tokenizer, model_configs)
        for record in selected_records
    ]
    batch_size = int(cfg.generation.batch_size)
    progress = tqdm(range(0, len(prompts), batch_size), desc=f"Generating {output_key}")

    generation_kwargs = {
        "max_new_tokens": int(cfg.generation.max_new_tokens),
        "do_sample": bool(cfg.generation.do_sample),
        "num_beams": int(cfg.generation.num_beams),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if cfg.generation.temperature is not None:
        generation_kwargs["temperature"] = float(cfg.generation.temperature)
    if cfg.generation.top_p is not None:
        generation_kwargs["top_p"] = float(cfg.generation.top_p)

    for start in progress:
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg.max_length),
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_kwargs,
            )
        decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        for offset, text in enumerate(decoded):
            selected_records[start + offset][output_key] = text.strip()


def generation_eval_enabled(cfg):
    gen_eval_cfg = cfg_get(cfg, "generation_eval", None)
    return bool(cfg_get(gen_eval_cfg, "enabled", False))


def generation_kwargs_from_cfg(cfg, section_name):
    gen_cfg = cfg_get(cfg, section_name, None)
    fallback_cfg = cfg_get(cfg, "generation", None)
    max_new_tokens = cfg_get(gen_cfg, "max_new_tokens", cfg_get(fallback_cfg, "max_new_tokens", 128))
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(cfg_get(gen_cfg, "do_sample", cfg_get(fallback_cfg, "do_sample", False))),
        "num_beams": int(cfg_get(gen_cfg, "num_beams", cfg_get(fallback_cfg, "num_beams", 1))),
    }
    temperature = cfg_get(gen_cfg, "temperature", cfg_get(fallback_cfg, "temperature", None))
    top_p = cfg_get(gen_cfg, "top_p", cfg_get(fallback_cfg, "top_p", None))
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if top_p is not None:
        kwargs["top_p"] = float(top_p)
    return kwargs


def generate_dataset_metrics(model, tokenizer, model_configs, dataset, cfg, model_label):
    results = {}
    device = next(model.parameters()).device
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    gen_eval_cfg = cfg_get(cfg, "generation_eval", None)
    batch_size = int(cfg_get(gen_eval_cfg, "batch_size", cfg_get(cfg_get(cfg, "generation", None), "batch_size", 4)))
    generation_kwargs = generation_kwargs_from_cfg(cfg, "generation_eval")
    generation_kwargs.update({"pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id})

    prompts = []
    rows = []
    for idx in range(len(dataset)):
        row = dataset.data[idx]
        language = get_row_language(row, dataset.cfg)
        question = normalize_eval_text(row[dataset.question_key], language, dataset.cfg)
        gold = normalize_eval_text(row[dataset.answer_key], language, dataset.cfg)
        prompts.append(build_prompt(question, language, tokenizer, model_configs))
        rows.append((idx, language, gold))

    progress = tqdm(range(0, len(prompts), batch_size), desc=f"Generating full {model_label}")
    for start in progress:
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(cfg.max_length),
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_kwargs,
            )
        decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        for offset, text in enumerate(decoded):
            dataset_idx, language, gold = rows[start + offset]
            results[dataset_idx] = generation_match_metrics(text.strip(), gold, language, cfg)

    tokenizer.padding_side = previous_padding_side
    return results


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)


def write_examples_jsonl(path, records, selected_records):
    selected_by_idx = {record["index"]: record for record in selected_records}
    with open(path, "w") as f:
        for record in records:
            row = dict(record)
            selected = selected_by_idx.get(record["index"])
            if selected is not None:
                row["case_tags"] = selected.get("case_tags", [])
                for key in ["ft_generated_answer", "base_generated_answer"]:
                    if key in selected:
                        row[key] = selected[key]
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_per_language_csv(path, rows, compare_base):
    preferred_fields = [
        "language",
        "num_examples",
        "num_answer_tokens",
        "ft_avg_nll",
        "ft_perplexity",
        "ft_prob_correct",
        "ft_token_accuracy",
        "ft_truth_correct_prob",
        "ft_perturb_prob",
        "ft_truth_ratio_raw",
        "ft_truth_ratio_score",
        "ft_normalized_exact_match",
        "ft_normalized_contains",
    ]
    if compare_base:
        preferred_fields.extend(
            [
                "base_avg_nll",
                "base_perplexity",
                "base_prob_correct",
                "base_token_accuracy",
                "base_truth_correct_prob",
                "base_perturb_prob",
                "base_truth_ratio_raw",
                "base_truth_ratio_score",
                "base_normalized_exact_match",
                "base_normalized_contains",
                "delta_avg_nll",
                "delta_prob_correct",
                "delta_token_accuracy",
                "delta_truth_ratio_score",
                "delta_normalized_exact_match",
                "prob_ratio_ft_to_base",
            ]
        )
    present = set().union(*(row.keys() for row in rows)) if rows else set(preferred_fields)
    fields = [field for field in preferred_fields if field in present]
    fields.extend(sorted(present - set(fields)))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_metric(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_case_study_md(path, summary, per_language_rows, selected_records, compare_base):
    with open(path, "w") as f:
        f.write("# Fine-tuned Full TOFU Evaluation\n\n")
        f.write("This report evaluates the fine-tuned model on the full multilingual TOFU fiction QA dataset used for fine-tuning.\n\n")
        f.write("## Overall Metrics\n\n")
        ft = summary["models"]["finetuned"]
        f.write(f"- Fine-tuned avg NLL: {format_metric(ft['avg_nll'])}\n")
        f.write(f"- Fine-tuned perplexity: {format_metric(ft['perplexity'])}\n")
        f.write(f"- Fine-tuned correct-answer probability: {format_metric(ft['prob_correct'])}\n")
        f.write(f"- Fine-tuned token accuracy: {format_metric(ft['token_accuracy'])}\n")
        if "truth_ratio_score" in ft:
            f.write(f"- Fine-tuned truth ratio score: {format_metric(ft['truth_ratio_score'])}\n")
        if "normalized_exact_match" in ft:
            f.write(f"- Fine-tuned normalized exact match: {format_metric(ft['normalized_exact_match'])}\n")
            f.write(f"- Fine-tuned normalized contains: {format_metric(ft['normalized_contains'])}\n")
        if compare_base:
            base = summary["models"]["base"]
            improvement = summary["improvement"]
            f.write(f"- Base avg NLL: {format_metric(base['avg_nll'])}\n")
            f.write(f"- Base perplexity: {format_metric(base['perplexity'])}\n")
            f.write(f"- Base correct-answer probability: {format_metric(base['prob_correct'])}\n")
            f.write(f"- Base token accuracy: {format_metric(base['token_accuracy'])}\n")
            f.write(f"- Delta avg NLL (base - fine-tuned): {format_metric(improvement['delta_avg_nll'])}\n")
            f.write(f"- Delta correct-answer probability (fine-tuned - base): {format_metric(improvement['delta_prob_correct'])}\n")
            f.write(f"- Delta token accuracy (fine-tuned - base): {format_metric(improvement['delta_token_accuracy'])}\n")

        f.write("\n## Per-language Metrics\n\n")
        has_truth = any("ft_truth_ratio_score" in row for row in per_language_rows)
        has_generation = any("ft_normalized_exact_match" in row for row in per_language_rows)
        header_values = ["Language", "#Examples", "FT NLL", "FT PPL", "FT Prob", "FT Acc"]
        aligns = ["---", "---:", "---:", "---:", "---:", "---:"]
        if has_truth:
            header_values.extend(["FT Truth"])
            aligns.extend(["---:"])
        if has_generation:
            header_values.extend(["FT EM", "FT Contains"])
            aligns.extend(["---:", "---:"])
        if compare_base:
            header_values.extend(["Base NLL", "Base PPL", "Base Prob", "Base Acc", "Delta NLL", "Delta Prob"])
            aligns.extend(["---:", "---:", "---:", "---:", "---:", "---:"])
            if has_truth:
                header_values.extend(["Base Truth", "Delta Truth"])
                aligns.extend(["---:", "---:"])
            if has_generation:
                header_values.extend(["Base EM", "Base Contains"])
                aligns.extend(["---:", "---:"])
        f.write("| " + " | ".join(header_values) + " |\n")
        f.write("|" + "|".join(aligns) + "|\n")
        for row in per_language_rows:
            values = [
                row["language"],
                row["num_examples"],
                format_metric(row["ft_avg_nll"]),
                format_metric(row["ft_perplexity"]),
                format_metric(row["ft_prob_correct"]),
                format_metric(row["ft_token_accuracy"]),
            ]
            if has_truth:
                values.append(format_metric(row.get("ft_truth_ratio_score")))
            if has_generation:
                values.extend(
                    [
                        format_metric(row.get("ft_normalized_exact_match")),
                        format_metric(row.get("ft_normalized_contains")),
                    ]
                )
            if compare_base:
                values.extend(
                    [
                        format_metric(row["base_avg_nll"]),
                        format_metric(row["base_perplexity"]),
                        format_metric(row["base_prob_correct"]),
                        format_metric(row["base_token_accuracy"]),
                        format_metric(row["delta_avg_nll"]),
                        format_metric(row["delta_prob_correct"]),
                    ]
                )
                if has_truth:
                    values.extend(
                        [
                            format_metric(row.get("base_truth_ratio_score")),
                            format_metric(row.get("delta_truth_ratio_score")),
                        ]
                    )
                if has_generation:
                    values.extend(
                        [
                            format_metric(row.get("base_normalized_exact_match")),
                            format_metric(row.get("base_normalized_contains")),
                        ]
                    )
            f.write("| " + " | ".join(map(str, values)) + " |\n")

        f.write("\n## Case Studies\n\n")
        for record in selected_records:
            title = f"{record['language']} / index {record['index']} / {', '.join(record.get('case_tags', []))}"
            f.write(f"### {title}\n\n")
            f.write("Question:\n\n")
            f.write(f"{record['question']}\n\n")
            f.write("Gold answer:\n\n")
            f.write(f"{record['gold_answer']}\n\n")
            if compare_base:
                f.write("Base output:\n\n")
                f.write(f"{record.get('base_generated_answer', '')}\n\n")
            f.write("Fine-tuned output:\n\n")
            f.write(f"{record.get('ft_generated_answer', '')}\n\n")
            f.write("Metrics:\n\n")
            f.write(f"- FT avg NLL: {format_metric(record.get('ft_avg_nll'))}\n")
            f.write(f"- FT correct-answer probability: {format_metric(record.get('ft_prob_correct'))}\n")
            f.write(f"- FT token accuracy: {format_metric(record.get('ft_token_accuracy'))}\n")
            if "ft_truth_ratio_score" in record:
                f.write(f"- FT truth ratio score: {format_metric(record.get('ft_truth_ratio_score'))}\n")
            if "ft_normalized_exact_match" in record:
                f.write(f"- FT normalized exact match: {format_metric(record.get('ft_normalized_exact_match'))}\n")
                f.write(f"- FT normalized contains: {format_metric(record.get('ft_normalized_contains'))}\n")
            if compare_base:
                f.write(f"- Base avg NLL: {format_metric(record.get('base_avg_nll'))}\n")
                f.write(f"- Base correct-answer probability: {format_metric(record.get('base_prob_correct'))}\n")
                f.write(f"- Base token accuracy: {format_metric(record.get('base_token_accuracy'))}\n")
                f.write(f"- Delta avg NLL: {format_metric(record.get('delta_avg_nll'))}\n")
                f.write(f"- Delta correct-answer probability: {format_metric(record.get('delta_prob_correct'))}\n")
                if "base_truth_ratio_score" in record:
                    f.write(f"- Base truth ratio score: {format_metric(record.get('base_truth_ratio_score'))}\n")
                if "base_normalized_exact_match" in record:
                    f.write(f"- Base normalized exact match: {format_metric(record.get('base_normalized_exact_match'))}\n")
                    f.write(f"- Base normalized contains: {format_metric(record.get('base_normalized_contains'))}\n")
            f.write("\n")


@hydra.main(version_base=None, config_path="config", config_name="eval_finetune_full")
def main(cfg):
    os.environ["WANDB_DISABLED"] = "true"
    if cfg.model_path is None:
        raise ValueError("Set model_path=/path/to/finetuned/model.")

    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.overwrite and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Set overwrite=true to reuse it.")
    OmegaConf.save(config=cfg, f=str(output_dir / "config_resolved.yaml"))

    model_configs = get_model_identifiers_from_yaml(cfg.model_family)
    tokenizer = load_eval_tokenizer(cfg, model_configs)

    dataset = FullTofuDataset(cfg, tokenizer, model_configs)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        collate_fn=collate_full_tofu,
    )

    ft_model = load_model(cfg.model_path, cfg, model_configs)
    ft_results = evaluate_model(ft_model, dataloader, cfg, "ft")
    ft_generation_results = None
    if generation_eval_enabled(cfg):
        ft_generation_results = generate_dataset_metrics(ft_model, tokenizer, model_configs, dataset, cfg, "ft")
    selected_records = []

    base_results = None
    base_generation_results = None
    if cfg.compare_base:
        unload_model(ft_model)

        base_model_path = cfg.base_model_path or model_configs["hf_key"]
        base_model = load_model(base_model_path, cfg, model_configs)
        base_results = evaluate_model(base_model, dataloader, cfg, "base")
        if generation_eval_enabled(cfg):
            base_generation_results = generate_dataset_metrics(base_model, tokenizer, model_configs, dataset, cfg, "base")

        records = merge_records(dataset, ft_results, base_results, ft_generation_results, base_generation_results)
        selected_records = select_case_studies(records, cfg) if cfg.case_study.enabled else []
        if cfg.generation.enabled and selected_records and not generation_eval_enabled(cfg):
            # Regenerate FT outputs if base comparison changed the selected set.
            unload_model(base_model)
            ft_model = load_model(cfg.model_path, cfg, model_configs)
            generate_outputs(ft_model, tokenizer, model_configs, selected_records, cfg, "ft_generated_answer")
            unload_model(ft_model)
            base_model = load_model(base_model_path, cfg, model_configs)
            generate_outputs(base_model, tokenizer, model_configs, selected_records, cfg, "base_generated_answer")
        unload_model(base_model)
    else:
        records = merge_records(dataset, ft_results, None, ft_generation_results, None)
        selected_records = select_case_studies(records, cfg) if cfg.case_study.enabled else []
        if cfg.generation.enabled and selected_records and not generation_eval_enabled(cfg):
            generate_outputs(ft_model, tokenizer, model_configs, selected_records, cfg, "ft_generated_answer")
        unload_model(ft_model)

    per_language_rows = aggregate_by_language(records, bool(cfg.compare_base))
    summary = {
        "model_family": cfg.model_family,
        "model_path": str(cfg.model_path),
        "base_model_path": str(cfg.base_model_path) if cfg.compare_base else None,
        "data_path": str(cfg.data_path),
        "models": {
            "finetuned": aggregate_records(records, "ft"),
        },
        "per_language": per_language_rows,
        "num_case_studies": len(selected_records),
    }
    if cfg.compare_base:
        summary["models"]["base"] = aggregate_records(records, "base")
        summary["improvement"] = {
            "delta_avg_nll": summary["models"]["base"]["avg_nll"] - summary["models"]["finetuned"]["avg_nll"],
            "delta_prob_correct": summary["models"]["finetuned"]["prob_correct"] - summary["models"]["base"]["prob_correct"],
            "delta_token_accuracy": summary["models"]["finetuned"]["token_accuracy"] - summary["models"]["base"]["token_accuracy"],
        }
        if "truth_ratio_score" in summary["models"]["finetuned"] and "truth_ratio_score" in summary["models"]["base"]:
            summary["improvement"]["delta_truth_ratio_score"] = (
                summary["models"]["finetuned"]["truth_ratio_score"] - summary["models"]["base"]["truth_ratio_score"]
            )
        if "normalized_exact_match" in summary["models"]["finetuned"] and "normalized_exact_match" in summary["models"]["base"]:
            summary["improvement"]["delta_normalized_exact_match"] = (
                summary["models"]["finetuned"]["normalized_exact_match"]
                - summary["models"]["base"]["normalized_exact_match"]
            )

    write_json(output_dir / "summary.json", summary)
    write_per_language_csv(output_dir / "per_language.csv", per_language_rows, bool(cfg.compare_base))
    write_examples_jsonl(output_dir / "examples.jsonl", records, selected_records)
    write_json(output_dir / "case_study.json", selected_records)
    if cfg.case_study.enabled:
        write_case_study_md(output_dir / "case_study.md", summary, per_language_rows, selected_records, bool(cfg.compare_base))

    print(f"Saved summary to {output_dir / 'summary.json'}")
    print(f"Saved per-language metrics to {output_dir / 'per_language.csv'}")
    print(f"Saved per-example records to {output_dir / 'examples.jsonl'}")
    if cfg.case_study.enabled:
        print(f"Saved case study report to {output_dir / 'case_study.md'}")


if __name__ == "__main__":
    main()
