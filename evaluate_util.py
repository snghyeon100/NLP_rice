from tqdm import tqdm
from data_module import (
    TextDatasetQAStat,
    custom_data_collator,
    get_batch_loss,
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
    "Prob. Real Authors",
    "Truth Ratio Real Authors",
    "Prob. Real World",
    "Truth Ratio Real World",
    "Prob. Retain",
    "Truth Ratio Retain",
    "Prob. Forget",
    "Truth Ratio Forget",
    "Model Utility",
    "Forget Quality",
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



def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        input_ids, labels, attention_mask, indices = batch
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1
        perturb_batch = {"input_ids": perturb_input_ids.view(bsz*seq_len, -1), "labels": perturb_labels.view(bsz*seq_len, -1), "attention_mask": perturb_attention_mask.view(bsz*seq_len, -1)}


        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)
        for k, v in perturb_batch.items():
            perturb_batch[k] = v.to(model.device)


        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)
        gt_loss = gt_loss.to(torch.float32)
        perturb_loss = perturb_loss.to(torch.float32)
        num_token_gt = (batch['labels']!=-100).sum(-1)
        num_token_perturb = (perturb_batch['labels']!=-100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        
        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss/num_token_perturb
        gt_loss_per_token = gt_loss/num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))


        # zip index and each stat into a dict
        perturb_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), perturb_loss_per_token.cpu().numpy().tolist()))
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))


        # merge dicts

        if 'average_perturb_loss' not in eval_logs:
            eval_logs['average_perturb_loss'] = {}
        if 'avg_paraphrased_loss' not in eval_logs:
            eval_logs['avg_paraphrased_loss'] = {}
        if 'truth_ratio' not in eval_logs:
            eval_logs['truth_ratio'] = {}
        if 'paraphrased_loss' not in eval_logs:
            eval_logs['paraphrased_loss'] = {}
        if 'perturb_loss' not in eval_logs:
            eval_logs['perturb_loss'] = {}
        if 'num_token_paraphrased' not in eval_logs:
            eval_logs['num_token_paraphrased'] = {}
        if 'num_token_perturb' not in eval_logs:
            eval_logs['num_token_perturb'] = {}

        eval_logs['average_perturb_loss'].update(perturb_loss_per_token)
        eval_logs['avg_paraphrased_loss'].update(gt_loss_per_token)
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

def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False, language='en'):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices = batch
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer, language=language)
            gen_outputs.extend([
                normalize_eval_text(
                    text,
                    language,
                    cfg_get(cfg, "unicode_normalization", None),
                    cfg_get(cfg, "normalize_languages", None),
                )
                for text in gen_output
            ])
            ground_truths.extend([
                normalize_eval_text(
                    text,
                    language,
                    cfg_get(cfg, "unicode_normalization", None),
                    cfg_get(cfg, "normalize_languages", None),
                )
                for text in gt
            ])
            input_strings.extend(input_string)
            
        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        gt_loss = gt_loss.to(torch.float32)
        num_token_gt = (batch['labels']!=-100).sum(-1)
        gt_loss_per_token = gt_loss/num_token_gt



        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output,gt))))

    eval_logs.update(eval_chrf(gen_outputs, ground_truths))
    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    eval_logs.update(eval_bleu(gen_outputs, ground_truths))
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))
    
    if normalize_gt:
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
        "save_generated_text": cfg_get(cfg, "save_generated_text", True),
        "ds_size": cfg_get(cfg, "ds_size", None),
        "overwrite": cfg_get(cfg, "overwrite", True),
        "use_pretrained": cfg_get(cfg, "use_pretrained", False),
        "batch_size": int(cfg_get(cfg, "batch_size", 4)),
        "retain_result": retain_result_for_language(cfg, language),
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


def _summary_probability(eval_logs, task_name):
    task_logs = eval_logs.get(task_name)
    if task_logs is None or "avg_gt_loss" not in task_logs:
        return None

    avg_gt_loss = task_logs["avg_gt_loss"]
    if "eval_log" in task_name:
        probs = _exp_neg(list(avg_gt_loss.values()))
        return _json_float(np.mean(probs))

    avg_perturb_loss = task_logs.get("average_perturb_loss")
    if avg_perturb_loss is None:
        return None

    keys = _common_keys(avg_gt_loss, avg_perturb_loss)
    if not keys:
        return None

    true_probs = _exp_neg([avg_gt_loss[key] for key in keys])
    perturb_prob_sums = np.array([
        np.sum(_exp_neg(avg_perturb_loss[key]))
        for key in keys
    ], dtype=np.float64)
    all_probs = true_probs + perturb_prob_sums
    normalized_true_probs = np.divide(
        true_probs,
        all_probs,
        out=np.zeros_like(true_probs, dtype=np.float64),
        where=all_probs != 0,
    )
    return _json_float(np.mean(normalized_true_probs))


def _summary_truth_ratio(eval_logs, task_name):
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

    paraphrase_losses = np.array([avg_paraphrased_loss[key] for key in keys], dtype=np.float64)
    perturb_losses = np.array([
        np.mean(np.array(avg_perturb_loss[key], dtype=np.float64))
        for key in keys
    ], dtype=np.float64)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        curr_stat = np.exp(np.clip(perturb_losses - paraphrase_losses, -745, 700))
        inverse_stat = np.divide(
            1.0,
            curr_stat,
            out=np.full_like(curr_stat, np.inf, dtype=np.float64),
            where=curr_stat != 0,
        )
        if "forget" in task_name:
            truth_ratio = np.mean(np.minimum(curr_stat, inverse_stat))
        else:
            truth_ratio = np.mean(np.maximum(0, 1 - inverse_stat))
    return _json_float(truth_ratio)


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


def build_summary_row(language, eval_logs, retain_result=None):
    row = {"language": language}
    for task_name, task_label in SUMMARY_TASKS.items():
        row[f"Prob. {task_label}"] = _summary_probability(eval_logs, task_name)
        row[f"Truth Ratio {task_label}"] = _summary_truth_ratio(eval_logs, task_name)

    utility_values = [
        row["Prob. Real Authors"],
        row["Truth Ratio Real Authors"],
        row["Prob. Real World"],
        row["Truth Ratio Real World"],
        row["Prob. Retain"],
        row["Truth Ratio Retain"],
    ]
    row["Model Utility"] = _harmonic_mean(utility_values)

    row["Forget Quality"] = None
    if retain_result is not None:
        row["Forget Quality"] = _json_float(get_forget_quality(eval_logs, retain_result)["Forget Quality"])

    return {column: row.get(column) for column in SUMMARY_COLUMNS}


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
        json.dump(json_rows, f, indent=4)

    with open(root_save_dir / "eval_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in json_rows:
            writer.writerow({
                key: "" if value is None else value
                for key, value in row.items()
            })


def evaluate_one_language(model, tokenizer, cfg, language, save_dir):
    eval_cfg = language_eval_cfg(cfg, language)
    assert len(eval_cfg.data_path) == len(eval_cfg.split_list) == len(eval_cfg.eval_task) == len(eval_cfg.question_key) == len(eval_cfg.answer_key) == len(eval_cfg.base_answer_key) == len(eval_cfg.perturbed_answer_key), "data_path, split, eval_task, and answer key lists must have the same length"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    aggregated_eval_logs = {}
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

        if os.path.exists(save_filename) and not eval_cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            with open(save_filename, "r") as f:
                aggregated_eval_logs[f"{eval_task}.json"] = json.load(f)
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

        with open(save_filename, "w") as f:
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

    aggregated_eval_log_filename = os.path.join(save_dir, "eval_log_aggregated.json")
    with open(aggregated_eval_log_filename, "w") as f:
        json.dump(aggregated_eval_logs, f, indent=4)

    if eval_cfg.retain_result is not None:
        retain_result = json.load(open(resolve_project_path(eval_cfg.retain_result), "r"))
        aggregate_stat = {**get_model_utility(aggregated_eval_logs), **get_forget_quality(aggregated_eval_logs, retain_result)}
        with open(os.path.join(save_dir, "aggregate_stat.csv"), "w") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate_stat.keys()))
            writer.writeheader()
            writer.writerow(aggregate_stat)

    return aggregated_eval_logs


def evaluate_languages(model, tokenizer, cfg):
    languages = infer_languages(cfg)
    multilingual = cfg_get(cfg, "languages", None) is not None
    all_logs = {}
    summary_rows = []
    retain_logs_by_language = {}
    root_save_dir = Path(resolve_project_path(cfg.save_dir))

    for language in languages:
        save_dir = root_save_dir
        if multilingual:
            save_dir = save_dir / language
        all_logs[language] = evaluate_one_language(model, tokenizer, cfg, language, save_dir)

        retain_result_path = retain_result_for_language(cfg, language)
        retain_result = _load_retain_result(retain_result_path) if retain_result_path is not None else None
        if retain_result is not None:
            retain_logs_by_language[language] = retain_result
        summary_rows.append(build_summary_row(language, all_logs[language], retain_result))

    if multilingual:
        root_save_dir.mkdir(parents=True, exist_ok=True)
        with open(root_save_dir / "multilingual_aggregated.json", "w") as f:
            json.dump(all_logs, f, indent=4)

    if len(languages) > 1:
        combined_retain_result = None
        if len(retain_logs_by_language) == len(languages):
            combined_retain_result = combine_language_logs(retain_logs_by_language)
        summary_rows.append(build_summary_row("total", combine_language_logs(all_logs), combined_retain_result))

    write_summary_files(root_save_dir, summary_rows)

    return all_logs


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
