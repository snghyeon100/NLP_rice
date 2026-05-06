from tqdm import tqdm
from data_module import (
    TextDatasetQAStat,
    custom_data_collator_with_indices,
    normalize_eval_text,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os, hydra
import evaluate
import json
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from rouge_score import rouge_scorer
from utils import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality
import torch.nn as nn
import csv 
import numpy as np 

from evaluate import load


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_TASKS = [
    "eval_log",
    "eval_real_author_wo_options",
    "eval_real_world_wo_options",
    "eval_log_forget",
]
DEFAULT_QUESTION_KEYS = ["question", "question", "question", "question"]
DEFAULT_ANSWER_KEYS = ["answer", "answer", "answer", "answer"]
DEFAULT_BASE_ANSWER_KEYS = ["paraphrased_answer", "answer", "answer", "paraphrased_answer"]
DEFAULT_PERTURBED_ANSWER_KEYS = ["perturbed_answer", "perturbed_answer", "perturbed_answer", "perturbed_answer"]
SUMMARY_COLUMNS = [
    "language",
    "Model Utility",
    "Prob. Retain",
    "Prob. Forget",
    "Truth Ratio Forget",
    "Prob. Real Authors",
    "1 - Truth Ratio Real Authors",
    "Prob. Real World",
    "1 - Truth Ratio Real World",
    "1 - Truth Ratio Retain",
]
SUMMARY_CSV_COLUMNS = [
    "language",
    "Model Utility",
    "Prob. Retain",
    "Truth Ratio Retain",
    "Prob. Forget",
    "Truth Ratio Forget",
]
SUMMARY_TASKS = {
    "eval_real_author_wo_options.json": "Real Authors",
    "eval_real_world_wo_options.json": "Real World",
    "eval_log.json": "Retain",
    "eval_log_forget.json": "Forget",
}


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, DictConfig):
        return cfg.get(key, default)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def list_cfg(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def resolve_project_path(path):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


def infer_languages(cfg):
    languages = list_cfg(cfg_get(cfg, "languages", None))
    if languages:
        return languages
    return [cfg_get(cfg, "language", "en")]


def _forward_logits(model, input_ids, attention_mask):
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits


def _batch_loss_from_logits(logits, labels):
    shifted_labels = labels[..., 1:].contiguous()
    shifted_logits = logits[..., :-1, :]
    valid_mask = shifted_labels != -100
    batch_size = shifted_labels.shape[0]
    loss = torch.zeros(batch_size, device=logits.device, dtype=torch.float32)
    if not valid_mask.any():
        return loss

    token_logits = shifted_logits[valid_mask].float()
    token_labels = shifted_labels[valid_mask]
    token_losses = torch.nn.functional.cross_entropy(token_logits, token_labels, reduction="none")
    row_ids = torch.arange(batch_size, device=logits.device).unsqueeze(1).expand_as(shifted_labels)[valid_mask]
    loss.index_add_(0, row_ids, token_losses)
    return loss


def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model, cfg=None):
    eval_logs = {}
    keep_detailed_logs = bool(cfg_get(cfg, "save_raw_logs", False)) or bool(cfg_get(cfg, "save_legacy_aggregate_stat", False))
    perturb_chunk_size = int(cfg_get(cfg, "perturb_eval_chunk_size", 1))
    perturb_chunk_size = max(1, perturb_chunk_size)
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices = batch
        input_ids = input_ids.to(model.device)
        labels = labels.to(model.device)
        attention_mask = attention_mask.to(model.device)
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        flat_perturb_input_ids = perturb_input_ids.reshape(bsz * seq_len, -1)
        flat_perturb_labels = perturb_labels.reshape(bsz * seq_len, -1)
        flat_perturb_attention_mask = perturb_attention_mask.reshape(bsz * seq_len, -1)

        with torch.inference_mode():
            logits = _forward_logits(model, input_ids, attention_mask)
            gt_loss = _batch_loss_from_logits(logits, labels).cpu()
            del logits

        perturb_losses = []
        for start in range(0, flat_perturb_input_ids.shape[0], perturb_chunk_size):
            stop = min(start + perturb_chunk_size, flat_perturb_input_ids.shape[0])
            chunk_input_ids = flat_perturb_input_ids[start:stop].to(model.device)
            chunk_labels = flat_perturb_labels[start:stop].to(model.device)
            chunk_attention_mask = flat_perturb_attention_mask[start:stop].to(model.device)
            with torch.inference_mode():
                logits = _forward_logits(model, chunk_input_ids, chunk_attention_mask)
                chunk_loss = _batch_loss_from_logits(logits, chunk_labels).cpu()
                del logits
            perturb_losses.append(chunk_loss)
            del chunk_input_ids, chunk_labels, chunk_attention_mask

        perturb_loss = torch.cat(perturb_losses, dim=0).view(bsz, seq_len)
        num_token_gt = (labels.cpu() != -100).sum(-1).clamp_min(1)
        num_token_perturb = (flat_perturb_labels != -100).view(bsz, seq_len, -1).sum(-1).clamp_min(1)

        perturb_loss_per_token_tensor = perturb_loss / num_token_perturb
        gt_loss_per_token_tensor = gt_loss / num_token_gt
        truth_ratio = None
        if keep_detailed_logs:
            truth_ratio = torch.exp(gt_loss_per_token_tensor.unsqueeze(-1) - perturb_loss_per_token_tensor).mean(-1)

        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token_tensor.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token_tensor.cpu().numpy().tolist()))

        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if keep_detailed_logs and 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if keep_detailed_logs and 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if keep_detailed_logs and 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if keep_detailed_logs and 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if keep_detailed_logs and 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
        if keep_detailed_logs:
            truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
            gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
            perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
            num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
            num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))
            eval_logs['truth_ratio'].update(truth_ratio)
            eval_logs['paraphrased_loss'].update(gt_loss)
            eval_logs['perturb_loss'].update(perturb_loss)
            eval_logs['num_token_paraphrased'].update(num_token_gt)
            eval_logs['num_token_perturb'].update(num_token_perturb)

    return eval_logs

def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key, language):
    max_length = cfg_get(cfg, "input_max_length", cfg.generation.max_length)
    unicode_normalization = cfg_get(cfg, "unicode_normalization", None)
    normalize_languages = cfg_get(cfg, "normalize_languages", None)

    torch_format_dataset = TextDatasetQAStat( 
        folder, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=max_length,
        split=split, 
        question_key=question_key, 
        answer_key=answer_key,
        language=language,
        unicode_normalization=unicode_normalization,
        normalize_languages=normalize_languages,
    ) 
    base_torch_format_dataset = TextDatasetQAStat(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=max_length,
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key,
        language=language,
        unicode_normalization=unicode_normalization,
        normalize_languages=normalize_languages,
    )

    perturb_torch_format_dataset = TextDatasetQAStat(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=max_length,
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key,
        language=language,
        unicode_normalization=unicode_normalization,
        normalize_languages=normalize_languages,
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(cfg.ds_size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))


    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    perturb_batch_size = max(1, cfg.batch_size // 4)
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset, batch_size=perturb_batch_size, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=perturb_batch_size, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader

def eval_chrf(gen_outputs, ground_truths):
    chrf = load("chrf") 
    results = chrf.compute(predictions=gen_outputs, references=ground_truths)
    eval_result = {
        'chrf': results,
    }
    return eval_result


def _as_text_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_text_value(value, language, cfg):
    return normalize_eval_text(
        value,
        language,
        cfg_get(cfg, "unicode_normalization", None),
        cfg_get(cfg, "normalize_languages", None),
    )


def _normalize_text_list(value, language, cfg):
    return [_normalize_text_value(item, language, cfg) for item in _as_text_list(value)]


def _dataloader_row_lookup(dataloader, indices=None):
    dataset = getattr(dataloader, "dataset", None)
    data = getattr(dataset, "data", None)
    if data is None:
        return {}

    wanted = None if indices is None else {int(index) for index in indices}
    lookup = {}
    for row in data:
        if "index" in row:
            row_index = int(row["index"])
            if wanted is None or row_index in wanted:
                lookup[row_index] = row
                if wanted is not None and len(lookup) == len(wanted):
                    break
    return lookup


def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False, language='en'):
    eval_logs = {}
    keep_detailed_logs = bool(cfg_get(cfg, "save_raw_logs", False)) or bool(cfg_get(cfg, "save_legacy_aggregate_stat", False))
    compute_generation_metrics = bool(cfg_get(cfg, "compute_generation_metrics", True)) or bool(cfg_get(cfg, "save_legacy_aggregate_stat", False))
    save_generated_text = bool(cfg_get(cfg, "save_generated_text", False))
    collect_case_studies = bool(cfg_get(cfg, "save_case_studies", False)) and eval_task == "eval_log_forget"
    need_generation = compute_generation_metrics or save_generated_text

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        index_values = indices.cpu().numpy().tolist()
        if compute_generation_metrics:
            all_indices.extend(index_values)
        labels_cpu = labels
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.inference_mode():
            logits = _forward_logits(model, batch["input_ids"], batch["attention_mask"])
            gt_loss = _batch_loss_from_logits(logits, batch['labels']).cpu()
            del logits
            input_string = [None] * len(indices)
            gen_output = [None] * len(indices)
            gt = [None] * len(indices)
            normalized_gen_output = gen_output
            normalized_gt = gt
            if need_generation:
                input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer, language=language)
                normalized_gen_output = [
                    _normalize_text_value(text, language, cfg)
                    for text in gen_output
                ]
                normalized_gt = [
                    _normalize_text_value(text, language, cfg)
                    for text in gt
                ]
                if compute_generation_metrics:
                    gen_outputs.extend(normalized_gen_output)
                    ground_truths.extend(normalized_gt)
                    input_strings.extend(input_string)

        num_token_gt = (labels_cpu != -100).sum(-1).clamp_min(1)
        gt_loss_per_token = gt_loss / num_token_gt



        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if keep_detailed_logs and 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if keep_detailed_logs and 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if save_generated_text and 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs['avg_gt_loss'].update(dict(zip(index_values, gt_loss_per_token.cpu().numpy().tolist())))
        if keep_detailed_logs:
            eval_logs['gt_loss'].update(dict(zip(index_values, gt_loss.cpu().numpy().tolist())))
            eval_logs['num_token_gt'].update(dict(zip(index_values, num_token_gt.cpu().numpy().tolist())))
        if save_generated_text:
            eval_logs['generated_text'].update(dict(zip(index_values, zip(input_string, gen_output, gt))))

    if compute_generation_metrics and gen_outputs:
        eval_logs.update(eval_chrf(gen_outputs, ground_truths))
        eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
        eval_logs.update(eval_bleu(gen_outputs, ground_truths))
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model, cfg))
    
    if normalize_gt and keep_detailed_logs:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    if collect_case_studies:
        eval_logs["case_study_candidates"] = build_case_study_candidates(eval_logs)

    return eval_logs


def language_data_path(cfg, language):
    data_path_by_language = cfg_get(cfg, "data_path_by_language", None)
    if data_path_by_language is not None and language in data_path_by_language:
        data_path = data_path_by_language[language]
        if isinstance(data_path, str):
            return [data_path] * 4
        return list(data_path)

    if cfg_get(cfg, "data_path", None) is not None and cfg_get(cfg, "languages", None) is None:
        return list(cfg.data_path)

    if language == "en":
        return [cfg_get(cfg, "english_data_path", "locuslab/TOFU")] * 4

    data_root = cfg_get(cfg, "data_root", "./dataset")
    split = cfg_get(cfg, "split", "forget01_perturbed")
    return [
        f"{data_root}/retain_perturbed_{language}",
        f"{data_root}/real_authors_perturbed_{language}",
        f"{data_root}/world_facts_perturbed_{language}",
        f"{data_root}/{split}_{language}",
    ]


def language_eval_cfg(cfg, language):
    split = cfg_get(cfg, "split", "forget01_perturbed")
    language_cfg = {
        "model_family": cfg.model_family,
        "language": language,
        "data_path": language_data_path(cfg, language),
        "split": split,
        "split_list": list_cfg(cfg_get(cfg, "split_list", None)) or [
            "retain_perturbed",
            "real_authors_perturbed",
            "world_facts_perturbed",
            split,
        ],
        "eval_task": list_cfg(cfg_get(cfg, "eval_task", None)) or DEFAULT_EVAL_TASKS,
        "question_key": list_cfg(cfg_get(cfg, "question_key", None)) or DEFAULT_QUESTION_KEYS,
        "answer_key": list_cfg(cfg_get(cfg, "answer_key", None)) or DEFAULT_ANSWER_KEYS,
        "base_answer_key": list_cfg(cfg_get(cfg, "base_answer_key", None)) or DEFAULT_BASE_ANSWER_KEYS,
        "perturbed_answer_key": list_cfg(cfg_get(cfg, "perturbed_answer_key", None)) or DEFAULT_PERTURBED_ANSWER_KEYS,
        "generation": OmegaConf.to_container(cfg.generation, resolve=True),
        "input_max_length": cfg_get(cfg, "input_max_length", cfg.generation.max_length),
        "unicode_normalization": cfg_get(cfg, "unicode_normalization", None),
        "normalize_languages": cfg_get(cfg, "normalize_languages", None),
        "save_generated_text": cfg_get(cfg, "save_generated_text", False),
        "ds_size": cfg_get(cfg, "ds_size", None),
        "overwrite": cfg_get(cfg, "overwrite", True),
        "use_pretrained": cfg_get(cfg, "use_pretrained", False),
        "batch_size": int(cfg_get(cfg, "batch_size", 4)),
        "perturb_eval_chunk_size": int(cfg_get(cfg, "perturb_eval_chunk_size", 1)),
        "retain_result": retain_result_for_language(cfg, language),
        "save_raw_logs": bool(cfg_get(cfg, "save_raw_logs", False)),
        "save_case_studies": bool(cfg_get(cfg, "save_case_studies", False)),
        "case_study_k": int(cfg_get(cfg, "case_study_k", 3)),
        "case_study_seed": int(cfg_get(cfg, "case_study_seed", 42)),
        "compute_generation_metrics": bool(cfg_get(cfg, "compute_generation_metrics", False)),
        "save_legacy_aggregate_stat": bool(cfg_get(cfg, "save_legacy_aggregate_stat", False)),
    }
    return OmegaConf.create(language_cfg)


def retain_result_for_language(cfg, language):
    retain_result_by_language = cfg_get(cfg, "retain_result_by_language", None)
    if retain_result_by_language is not None and language in retain_result_by_language:
        return retain_result_by_language[language]

    retain_result_template = cfg_get(cfg, "retain_result_template", None)
    if retain_result_template:
        return str(retain_result_template).format(language=language)

    return cfg_get(cfg, "retain_result", None)


def _json_float(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (int, float)):
        value = float(value)
        if np.isfinite(value):
            return value
    return None


def _exp_neg(value):
    return np.exp(-1 * np.array(value, dtype=np.float64))


def _common_keys(first, second):
    return [key for key in first.keys() if key in second]


def _metric_value(metric_dict, key):
    if metric_dict is None:
        return None
    if key in metric_dict:
        return metric_dict[key]
    str_key = str(key)
    if str_key in metric_dict:
        return metric_dict[str_key]
    return None


def _json_float_list(values):
    result = []
    for value in np.atleast_1d(np.array(values, dtype=np.float64)):
        result.append(_json_float(value))
    return result


def build_case_study_candidates(eval_logs):
    candidates = []
    avg_gt_loss = eval_logs.get("avg_gt_loss", {})
    avg_paraphrased_loss = eval_logs.get("avg_paraphrased_loss", {})
    avg_perturb_loss = eval_logs.get("average_perturb_loss", {})

    for index in avg_gt_loss:
        gt_loss = _metric_value(avg_gt_loss, index)
        paraphrase_loss = _metric_value(avg_paraphrased_loss, index)
        perturb_losses = _metric_value(avg_perturb_loss, index)
        if gt_loss is None or paraphrase_loss is None or perturb_losses is None:
            continue

        gt_loss = float(gt_loss)
        paraphrase_loss = float(paraphrase_loss)
        perturb_losses = np.atleast_1d(np.array(perturb_losses, dtype=np.float64))
        perturb_probs = np.exp(-perturb_losses)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            truth_ratio_values = np.exp(np.clip(paraphrase_loss - perturb_losses, -745, 700))

        record = {
            "index": int(index),
            "avg_gt_loss": _json_float(gt_loss),
            "gt_prob": _json_float(np.exp(-gt_loss)),
            "paraphrase_avg_nll": _json_float(paraphrase_loss),
            "paraphrase_prob": _json_float(np.exp(-paraphrase_loss)),
            "perturb_avg_nll": _json_float_list(perturb_losses),
            "perturb_prob": _json_float(np.mean(perturb_probs)),
            "perturb_probs": _json_float_list(perturb_probs),
            "truth_ratio": _json_float(np.mean(truth_ratio_values)),
            "truth_ratio_values": _json_float_list(truth_ratio_values),
        }
        candidates.append(record)

    return candidates


def _finite_metric(record, key, default):
    value = record.get(key)
    if value is None:
        return default
    value = float(value)
    if np.isfinite(value):
        return value
    return default


def select_case_studies(candidates, k, seed):
    if not candidates:
        return {
            "worst_forget": [],
            "best_forget": [],
            "highest_gt_prob": [],
            "random_seeded": [],
        }

    k = max(1, int(k))
    worst_forget = sorted(
        candidates,
        key=lambda row: (
            _finite_metric(row, "truth_ratio", np.inf),
            -_finite_metric(row, "gt_prob", -np.inf),
        ),
    )[:k]

    best_forget = sorted(
        candidates,
        key=lambda row: abs(np.log(max(_finite_metric(row, "truth_ratio", 0.0), 1e-12))),
    )[:k]

    highest_gt_prob = sorted(
        candidates,
        key=lambda row: _finite_metric(row, "gt_prob", -np.inf),
        reverse=True,
    )[:k]

    rng = np.random.default_rng(int(seed))
    random_indices = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
    random_seeded = [candidates[int(idx)] for idx in random_indices]

    return {
        "worst_forget": worst_forget,
        "best_forget": best_forget,
        "highest_gt_prob": highest_gt_prob,
        "random_seeded": random_seeded,
    }


def _case_study_prompt(question, cfg, tokenizer, language):
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    use_chat_template = str(model_cfg.get('use_chat_template', 'false')).lower() == 'true'
    if use_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )

    question_start = model_cfg['question_start_tag'][language if language in model_cfg['question_start_tag'] else 'en']
    question_end = model_cfg['question_end_tag']
    answer_tag = model_cfg['answer_tag'][language if language in model_cfg['answer_tag'] else 'en']
    return question_start + question + question_end + answer_tag


def _add_case_study_generations(selected, cfg, model, tokenizer, language):
    if model is None or tokenizer is None:
        return selected

    records_by_index = {}
    for records in selected.values():
        for record in records:
            if record.get("question") is not None and record.get("index") not in records_by_index:
                records_by_index[record.get("index")] = record

    if not records_by_index:
        return selected

    original_padding_side = tokenizer.padding_side
    prompts = [
        _case_study_prompt(record["question"], cfg, tokenizer, language)
        for record in records_by_index.values()
    ]
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    inputs = tokenizer(
        prompts,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    generation_kwargs = {
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if cfg.generation.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = cfg.generation.max_new_tokens
    elif cfg.generation.max_length is not None:
        generation_kwargs["max_length"] = cfg.generation.max_length

    with torch.inference_mode():
        outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs)

    decoded = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    for record, output in zip(records_by_index.values(), decoded):
        record["generated_output"] = _normalize_text_value(output, language, cfg)

    tokenizer.padding_side = original_padding_side
    return selected


def _attach_case_study_context(selected, dataloader, language, cfg):
    selected_indices = {
        int(record["index"])
        for records in selected.values()
        for record in records
        if record.get("index") is not None
    }
    rows = _dataloader_row_lookup(dataloader, selected_indices)

    for records in selected.values():
        for record in records:
            row = rows.get(int(record["index"]), {})
            question_values = _normalize_text_list(row.get("question"), language, cfg)
            gold_answers = _normalize_text_list(row.get("answer"), language, cfg)
            paraphrased_answers = _normalize_text_list(row.get("paraphrased_answer"), language, cfg)
            perturbed_answers = _normalize_text_list(row.get("perturbed_answer"), language, cfg)

            record["language"] = language
            record["question"] = question_values[0] if question_values else None
            record["gold_answer"] = gold_answers[0] if gold_answers else None
            record["paraphrased_answer"] = paraphrased_answers[0] if paraphrased_answers else None
            record["perturbed_answers"] = perturbed_answers
            record.setdefault("generated_output", None)

    return selected


def write_case_study_files(save_dir, language, candidates, cfg, dataloader=None, model=None, tokenizer=None):
    if not candidates:
        return

    selected = select_case_studies(
        candidates,
        cfg_get(cfg, "case_study_k", 3),
        cfg_get(cfg, "case_study_seed", 42),
    )
    if dataloader is not None:
        selected = _attach_case_study_context(selected, dataloader, language, cfg)
    selected = _add_case_study_generations(selected, cfg, model, tokenizer, language)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "language": language,
        "case_study_k": int(cfg_get(cfg, "case_study_k", 3)),
        "case_study_seed": int(cfg_get(cfg, "case_study_seed", 42)),
        "groups": selected,
    }

    with open(save_dir / "case_studies_forget.json", "w") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    with open(save_dir / "case_studies_forget.md", "w") as f:
        f.write(f"# Forget Set Case Studies ({language})\n\n")
        for group_name, records in selected.items():
            f.write(f"## {group_name}\n\n")
            for record in records:
                f.write(f"### index {record.get('index')}\n\n")
                f.write(f"- Truth ratio: {record.get('truth_ratio')}\n")
                f.write(f"- Gold probability: {record.get('gt_prob')}\n")
                f.write(f"- Paraphrase probability: {record.get('paraphrase_prob')}\n")
                f.write(f"- Mean perturb probability: {record.get('perturb_prob')}\n\n")
                f.write("Question:\n\n")
                f.write(f"{record.get('question')}\n\n")
                f.write("Gold answer:\n\n")
                f.write(f"{record.get('gold_answer')}\n\n")
                f.write("Generated output:\n\n")
                f.write(f"{record.get('generated_output')}\n\n")
                f.write("Paraphrased answer:\n\n")
                f.write(f"{record.get('paraphrased_answer')}\n\n")
                f.write("Perturbed answers:\n\n")
                for answer in record.get("perturbed_answers", []):
                    f.write(f"- {answer}\n")
                f.write("\n")


def _probability_values(eval_logs, task_name):
    task_logs = eval_logs.get(task_name)
    if task_logs is None or "avg_gt_loss" not in task_logs:
        return None

    avg_gt_loss = task_logs["avg_gt_loss"]
    if "eval_log" in task_name:
        return _exp_neg(list(avg_gt_loss.values()))

    avg_perturb_loss = task_logs.get("average_perturb_loss")
    if avg_perturb_loss is None:
        return None

    keys = _common_keys(avg_gt_loss, avg_perturb_loss)
    if not keys:
        return None

    probs = []
    for key in keys:
        true_prob = float(_exp_neg([avg_gt_loss[key]])[0])
        false_probs = _exp_neg(np.atleast_1d(avg_perturb_loss[key]))
        all_prob = true_prob + float(np.sum(false_probs))
        if all_prob > 0 and np.isfinite(all_prob):
            probs.append(true_prob / all_prob)

    return np.array(probs, dtype=np.float64)


def _summary_probability(eval_logs, task_name):
    probs = _probability_values(eval_logs, task_name)
    if probs is None or probs.size == 0:
        return None
    return _json_float(np.mean(probs))


def _truth_ratio_values(eval_logs, task_name):
    task_logs = eval_logs.get(task_name)
    if task_logs is None:
        return None

    avg_paraphrased_loss = task_logs.get("avg_paraphrased_loss")
    avg_perturb_loss = task_logs.get("average_perturb_loss")
    if avg_paraphrased_loss is None or avg_perturb_loss is None:
        return None

    keys = _common_keys(avg_paraphrased_loss, avg_perturb_loss)
    if not keys:
        return None

    truth_ratios = []
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for key in keys:
            paraphrase_loss = float(avg_paraphrased_loss[key])
            perturb_losses = np.array(avg_perturb_loss[key], dtype=np.float64)
            perturb_losses = np.atleast_1d(perturb_losses)
            mean_perturb_loss = float(np.mean(perturb_losses))
            ratio = np.exp(np.clip(paraphrase_loss - mean_perturb_loss, -745, 700))
            truth_ratios.append(ratio)

    return np.array(truth_ratios, dtype=np.float64)


def _summary_truth_ratio(eval_logs, task_name):
    truth_ratios = _truth_ratio_values(eval_logs, task_name)
    if truth_ratios is None or truth_ratios.size == 0:
        return None
    if "forget" in task_name:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            truth_ratios = np.minimum(truth_ratios, 1 / truth_ratios)
    return _json_float(np.mean(truth_ratios))


def _summary_one_minus_truth_ratio(eval_logs, task_name):
    truth_ratios = _truth_ratio_values(eval_logs, task_name)
    if truth_ratios is None or truth_ratios.size == 0:
        return None
    return _json_float(np.mean(np.maximum(0, 1 - truth_ratios)))


def _harmonic_mean(values):
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return None
    if any(value == 0 for value in valid):
        return 0.0
    return _json_float(len(valid) / sum(1 / value for value in valid))


def _load_retain_result(path):
    if path is None:
        return None
    with open(resolve_project_path(path), "r") as f:
        return json.load(f)


def build_summary_accumulator(eval_logs):
    accumulator = {}
    for task_name in SUMMARY_TASKS:
        task_logs = eval_logs.get(task_name)
        if task_logs is None:
            continue

        task_acc = accumulator.setdefault(task_name, {
            "prob_sum": 0.0,
            "prob_count": 0,
            "truth_ratio_sum": 0.0,
            "truth_ratio_count": 0,
            "one_minus_truth_ratio_sum": 0.0,
            "one_minus_truth_ratio_count": 0,
        })

        probs = _probability_values(eval_logs, task_name)
        if probs is not None and probs.size > 0:
            task_acc["prob_sum"] += float(np.sum(probs))
            task_acc["prob_count"] += int(probs.size)

        truth_ratios = _truth_ratio_values(eval_logs, task_name)
        if truth_ratios is not None and truth_ratios.size > 0:
            if "forget" in task_name:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    summary_truth_ratios = np.minimum(truth_ratios, 1 / truth_ratios)
            else:
                summary_truth_ratios = truth_ratios
            finite_truth_ratios = summary_truth_ratios[np.isfinite(summary_truth_ratios)]
            if finite_truth_ratios.size > 0:
                task_acc["truth_ratio_sum"] += float(np.sum(finite_truth_ratios))
                task_acc["truth_ratio_count"] += int(finite_truth_ratios.size)
            one_minus = np.maximum(0, 1 - truth_ratios)
            task_acc["one_minus_truth_ratio_sum"] += float(np.sum(one_minus))
            task_acc["one_minus_truth_ratio_count"] += int(one_minus.size)

    return accumulator


def merge_summary_accumulators(target, source):
    for task_name, source_task in source.items():
        target_task = target.setdefault(task_name, {
            "prob_sum": 0.0,
            "prob_count": 0,
            "truth_ratio_sum": 0.0,
            "truth_ratio_count": 0,
            "one_minus_truth_ratio_sum": 0.0,
            "one_minus_truth_ratio_count": 0,
        })
        for key, value in source_task.items():
            target_task[key] += value
    return target


def _acc_mean(accumulator, task_name, sum_key, count_key):
    task_acc = accumulator.get(task_name)
    if task_acc is None:
        return None
    count = task_acc.get(count_key, 0)
    if count == 0:
        return None
    return _json_float(task_acc.get(sum_key, 0.0) / count)


def build_summary_row_from_accumulator(language, accumulator):
    row = {"language": language}
    row["Prob. Real Authors"] = _acc_mean(accumulator, "eval_real_author_wo_options.json", "prob_sum", "prob_count")
    row["1 - Truth Ratio Real Authors"] = _acc_mean(accumulator, "eval_real_author_wo_options.json", "one_minus_truth_ratio_sum", "one_minus_truth_ratio_count")
    row["Prob. Real World"] = _acc_mean(accumulator, "eval_real_world_wo_options.json", "prob_sum", "prob_count")
    row["1 - Truth Ratio Real World"] = _acc_mean(accumulator, "eval_real_world_wo_options.json", "one_minus_truth_ratio_sum", "one_minus_truth_ratio_count")
    row["Prob. Retain"] = _acc_mean(accumulator, "eval_log.json", "prob_sum", "prob_count")
    row["1 - Truth Ratio Retain"] = _acc_mean(accumulator, "eval_log.json", "one_minus_truth_ratio_sum", "one_minus_truth_ratio_count")
    row["Prob. Forget"] = _acc_mean(accumulator, "eval_log_forget.json", "prob_sum", "prob_count")
    row["Truth Ratio Forget"] = _acc_mean(accumulator, "eval_log_forget.json", "truth_ratio_sum", "truth_ratio_count")

    utility_values = [
        row["Prob. Real Authors"],
        row["1 - Truth Ratio Real Authors"],
        row["Prob. Real World"],
        row["1 - Truth Ratio Real World"],
        row["Prob. Retain"],
        row["1 - Truth Ratio Retain"],
    ]
    row["Model Utility"] = _harmonic_mean(utility_values)

    return {column: row.get(column) for column in SUMMARY_COLUMNS}


def build_summary_row(language, eval_logs, retain_result=None):
    return build_summary_row_from_accumulator(language, build_summary_accumulator(eval_logs))


def combine_language_logs(logs_by_language):
    combined = {}
    for language, language_logs in logs_by_language.items():
        for task_name, task_logs in language_logs.items():
            combined_task = combined.setdefault(task_name, {})
            for metric_name, metric_values in task_logs.items():
                if not isinstance(metric_values, dict):
                    continue
                combined_metric = combined_task.setdefault(metric_name, {})
                for sample_idx, value in metric_values.items():
                    combined_metric[f"{language}:{sample_idx}"] = value
    return combined


def write_summary_files(root_save_dir, rows):
    root_save_dir = Path(root_save_dir)
    root_save_dir.mkdir(parents=True, exist_ok=True)

    json_rows = [
        {key: _json_float(value) if key != "language" else value for key, value in row.items()}
        for row in rows
    ]
    with open(root_save_dir / "eval_summary.json", "w") as f:
        json.dump(json_rows, f, indent=4, ensure_ascii=False)

    csv_rows = [
        {
            key: row.get("1 - Truth Ratio Retain") if key == "Truth Ratio Retain" else row.get(key)
            for key in SUMMARY_CSV_COLUMNS
        }
        for row in json_rows
    ]
    with open(root_save_dir / "eval_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({
                key: "" if value is None else value
                for key, value in row.items()
            })


def evaluate_one_language(model, tokenizer, cfg, language, save_dir):
    eval_cfg = language_eval_cfg(cfg, language)
    assert len(eval_cfg.data_path) == len(eval_cfg.split_list) == len(eval_cfg.eval_task) == len(eval_cfg.question_key) == len(eval_cfg.answer_key) == len(eval_cfg.base_answer_key) == len(eval_cfg.perturbed_answer_key), "data_path, split, eval_task, and answer key lists must have the same length"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_raw_logs = bool(cfg_get(eval_cfg, "save_raw_logs", False))
    save_legacy_stat = bool(cfg_get(eval_cfg, "save_legacy_aggregate_stat", False))

    aggregated_eval_logs = {} if (save_raw_logs or save_legacy_stat) else None
    summary_accumulator = {}
    for folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key in zip(
        eval_cfg.data_path,
        eval_cfg.split_list,
        eval_cfg.question_key,
        eval_cfg.answer_key,
        eval_cfg.eval_task,
        eval_cfg.base_answer_key,
        eval_cfg.perturbed_answer_key,
    ):
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if eval_task == "eval_log_forget":
            split = eval_cfg.split
        print(f"[{language}] Working on eval task {eval_task} with split {split}")
        save_filename = os.path.join(save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if save_raw_logs and os.path.exists(save_filename) and not eval_cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            with open(save_filename, "r") as f:
                eval_logs = json.load(f)
            merge_summary_accumulators(summary_accumulator, build_summary_accumulator({f"{eval_task}.json": eval_logs}))
            if aggregated_eval_logs is not None:
                aggregated_eval_logs[f"{eval_task}.json"] = eval_logs
            continue

        eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
            eval_cfg,
            eval_task,
            tokenizer,
            resolve_project_path(folder),
            split,
            question_key,
            answer_key,
            base_answer_key,
            perturbed_answer_key,
            language=language,
        )

        normalize_gt = "eval_log" not in eval_task
        eval_logs = get_all_evals(
            eval_cfg,
            model,
            tokenizer,
            eval_task,
            eval_dataloader,
            base_eval_dataloader,
            perturb_dataloader,
            normalize_gt=normalize_gt,
            language=language,
        )

        if save_raw_logs:
            with open(save_filename, "w") as f:
                json.dump(eval_logs, f, indent=4, ensure_ascii=False)

        if bool(cfg_get(eval_cfg, "save_case_studies", False)) and eval_task == "eval_log_forget":
            write_case_study_files(
                save_dir,
                language,
                eval_logs.get("case_study_candidates", []),
                eval_cfg,
                dataloader=eval_dataloader,
                model=model,
                tokenizer=tokenizer,
            )
            if not save_raw_logs:
                eval_logs.pop("case_study_candidates", None)

        merge_summary_accumulators(summary_accumulator, build_summary_accumulator({f"{eval_task}.json": eval_logs}))
        if aggregated_eval_logs is not None:
            aggregated_eval_logs[f"{eval_task}.json"] = eval_logs
        del eval_logs

    aggregated_eval_log_filename = os.path.join(save_dir, "eval_log_aggregated.json")
    if save_raw_logs and aggregated_eval_logs is not None:
        with open(aggregated_eval_log_filename, "w") as f:
            json.dump(aggregated_eval_logs, f, indent=4, ensure_ascii=False)

    if save_legacy_stat and eval_cfg.retain_result is not None and aggregated_eval_logs is not None:
        retain_result = json.load(open(resolve_project_path(eval_cfg.retain_result), "r"))
        aggregate_stat = {**get_model_utility(aggregated_eval_logs), **get_forget_quality(aggregated_eval_logs, retain_result)}
        with open(os.path.join(save_dir, "aggregate_stat.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_stat.keys()))
            writer.writeheader()
            writer.writerow(aggregate_stat)

    return summary_accumulator, aggregated_eval_logs


def evaluate_languages(model, tokenizer, cfg):
    languages = infer_languages(cfg)
    multilingual = cfg_get(cfg, "languages", None) is not None
    save_raw_logs = bool(cfg_get(cfg, "save_raw_logs", False))
    all_logs = {} if save_raw_logs else None
    summary_rows = []
    total_accumulator = {}
    root_save_dir = Path(resolve_project_path(cfg.save_dir))
    root_save_dir.mkdir(parents=True, exist_ok=True)
    with open(root_save_dir / "resolved_eval_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    for language in languages:
        save_dir = root_save_dir
        if multilingual:
            save_dir = save_dir / language
        language_accumulator, language_logs = evaluate_one_language(model, tokenizer, cfg, language, save_dir)
        merge_summary_accumulators(total_accumulator, language_accumulator)

        summary_rows.append(build_summary_row_from_accumulator(language, language_accumulator))
        if save_raw_logs and language_logs is not None:
            all_logs[language] = language_logs
        del language_logs

    if multilingual and save_raw_logs:
        with open(root_save_dir / "multilingual_aggregated.json", "w") as f:
            json.dump(all_logs, f, indent=4, ensure_ascii=False)

    if len(languages) > 1:
        summary_rows.append(build_summary_row_from_accumulator("total", total_accumulator))

    write_summary_files(root_save_dir, summary_rows)

    return all_logs if save_raw_logs else summary_rows


def load_eval_model(cfg, model_cfg, model_id):
    device_map = None
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    config = AutoConfig.from_pretrained(model_id)
    model = None
    for attempt in range(3):
        try:
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=device_map,
                )
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                model = AutoModelForCausalLM.from_pretrained(
                    resolve_project_path(cfg.model_path),
                    config=config,
                    attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map=device_map,
                )
        except Exception as e:
            print(e)
            continue
        else:
            break
    else:
        raise RuntimeError("Could not load model.")

    model = model.eval()
    if device_map is None:
        model.to("cuda")
    return model


def load_eval_tokenizer(cfg, model_id):
    tokenizer_path = cfg_get(cfg, "tokenizer_path", None)
    if tokenizer_path is not None:
        source = resolve_project_path(tokenizer_path)
    elif cfg_get(cfg, "prefer_checkpoint_tokenizer", False) and not cfg_get(cfg, "use_pretrained", False):
        model_path = resolve_project_path(cfg.model_path)
        source = model_path if Path(model_path, "tokenizer_config.json").exists() else model_id
    else:
        source = model_id
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def reinitialize_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg):
    os.chdir(PROJECT_ROOT)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = load_eval_tokenizer(cfg, model_id)
    model = load_eval_model(cfg, model_cfg, model_id)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    evaluate_languages(model, tokenizer, cfg)
                    

def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}



def run_generation(cfg, batch, model, tokenizer, language='en'):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    generation_inputs = input_strings

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    use_chat_template = str(model_cfg.get('use_chat_template', 'false')).lower() == 'true'

    if use_chat_template:
        # chat template 형식: "...user\n질문\nassistant\n정답..."
        # rsplit으로 마지막 "assistant\n" 기준으로 나눔
        ground_truth = []
        display_inputs = []
        generation_inputs = []
        for s in input_strings:
            prompt_text, answer_text = s.rsplit("assistant\n", 1)
            question_text = prompt_text.rsplit("user\n", 1)[-1].strip()
            ground_truth.append(answer_text)
            display_inputs.append(prompt_text)
            generation_inputs.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": question_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        input_strings = display_inputs
    else:
        split_symbols = {
            "en": "Answer: ",
            "fr": "Reponse: ",
            "ar": "الإجابة: ",
            "fa": "پاسخ: ",
            "hi": "उत्तर: ",
            "iw": "תשובה: ",        # Hebrew
            "id": "Jawaban: ",       # Indonesian
            "ja": "回答: ",          # Japanese
            "ko": "답변: ",          # Korean
            "ru": "Ответ: ",         # Russian
        }
        split_symbol = split_symbols.get(language, "Answer: ")
        ground_truth = [s.split(split_symbol)[1] for s in input_strings]
        input_strings = [s.split(split_symbol)[0] for s in input_strings]
        generation_inputs = input_strings

    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id


    inputs = left_pad_tokenizer(
        generation_inputs,
        add_special_tokens=not use_chat_template,
        return_tensors='pt',
        padding=True,
    ).to(model.device)
    #now generate
    generation_kwargs = {
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": left_pad_tokenizer.eos_token_id,
    }
    if cfg.generation.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = cfg.generation.max_new_tokens
    elif cfg.generation.max_length is not None:
        generation_kwargs["max_length"] = cfg.generation.max_length
    out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, **generation_kwargs)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    return input_strings, strs, ground_truth



def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores['rouge1'].recall
        rougeL_recall[idx] = rouge_scores['rougeL'].recall


    return {'rouge1_recall': rouge1_recall, 'rougeL_recall': rougeL_recall}



if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    main()
