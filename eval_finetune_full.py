import csv
import gc
import json
import math
import os
import random
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


class FullTofuDataset(Dataset):
    def __init__(self, cfg, tokenizer, model_configs):
        loaded = datasets.load_from_disk(path_or_model_id(cfg.data_path))
        self.data = loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded
        self.tokenizer = tokenizer
        self.model_configs = model_configs
        self.max_length = cfg.max_length
        self.question_key = cfg.question_key
        self.answer_key = cfg.answer_key
        self.language_key = cfg.language_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        question = row[self.question_key]
        answer = row[self.answer_key]
        language = row.get(self.language_key, "en")
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


def collate_full_tofu(samples):
    return {
        "indices": [sample["index"] for sample in samples],
        "languages": [sample["language"] for sample in samples],
        "questions": [sample["question"] for sample in samples],
        "answers": [sample["answer"] for sample in samples],
        "input_ids": torch.stack([sample["input_ids"] for sample in samples]),
        "labels": torch.stack([sample["labels"] for sample in samples]),
        "attention_mask": torch.stack([sample["attention_mask"] for sample in samples]),
    }


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
    progress = tqdm(dataloader, desc=f"Evaluating {model_label}")
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            stats = answer_token_stats(outputs.logits, labels)

        for row_idx, dataset_idx in enumerate(batch["indices"]):
            token_count = int(stats["token_count"][row_idx].item())
            token_correct = int(stats["token_correct"][row_idx].item())
            avg_nll = float(stats["avg_nll"][row_idx].item())
            results[dataset_idx] = {
                f"{model_label}_loss_sum": float(stats["loss_sum"][row_idx].item()),
                f"{model_label}_avg_nll": avg_nll,
                f"{model_label}_perplexity": safe_exp(avg_nll),
                f"{model_label}_token_count": token_count,
                f"{model_label}_token_correct": token_correct,
                f"{model_label}_token_accuracy": float(stats["token_accuracy"][row_idx].item()),
            }
    return results


def safe_exp(value):
    if value is None:
        return None
    if value > 80:
        return None
    return float(math.exp(value))


def merge_records(dataset, ft_results, base_results=None):
    records = []
    for idx in range(len(dataset)):
        row = dataset.data[idx]
        record = {
            "index": idx,
            "language": row.get(dataset.language_key, "en"),
            "question": row[dataset.question_key],
            "gold_answer": row[dataset.answer_key],
        }
        record.update(ft_results[idx])
        if base_results is not None:
            record.update(base_results[idx])
            record["delta_avg_nll"] = record["base_avg_nll"] - record["ft_avg_nll"]
            record["delta_token_accuracy"] = record["ft_token_accuracy"] - record["base_token_accuracy"]
        records.append(record)
    return records


def aggregate_records(records, prefix):
    loss_sum = sum(float(row[f"{prefix}_loss_sum"]) for row in records)
    token_count = sum(int(row[f"{prefix}_token_count"]) for row in records)
    token_correct = sum(int(row[f"{prefix}_token_correct"]) for row in records)
    avg_nll = loss_sum / max(1, token_count)
    return {
        "num_examples": len(records),
        "num_answer_tokens": token_count,
        "avg_nll": avg_nll,
        "perplexity": safe_exp(avg_nll),
        "token_accuracy": token_correct / max(1, token_count),
    }


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
            "ft_token_accuracy": ft["token_accuracy"],
        }
        if compare_base:
            base = aggregate_records(lang_records, "base")
            row.update(
                {
                    "base_avg_nll": base["avg_nll"],
                    "base_perplexity": base["perplexity"],
                    "base_token_accuracy": base["token_accuracy"],
                    "delta_avg_nll": base["avg_nll"] - ft["avg_nll"],
                    "delta_token_accuracy": ft["token_accuracy"] - base["token_accuracy"],
                }
            )
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
    fields = [
        "language",
        "num_examples",
        "num_answer_tokens",
        "ft_avg_nll",
        "ft_perplexity",
        "ft_token_accuracy",
    ]
    if compare_base:
        fields.extend(
            [
                "base_avg_nll",
                "base_perplexity",
                "base_token_accuracy",
                "delta_avg_nll",
                "delta_token_accuracy",
            ]
        )
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
        f.write(f"- Fine-tuned token accuracy: {format_metric(ft['token_accuracy'])}\n")
        if compare_base:
            base = summary["models"]["base"]
            improvement = summary["improvement"]
            f.write(f"- Base avg NLL: {format_metric(base['avg_nll'])}\n")
            f.write(f"- Base perplexity: {format_metric(base['perplexity'])}\n")
            f.write(f"- Base token accuracy: {format_metric(base['token_accuracy'])}\n")
            f.write(f"- Delta avg NLL (base - fine-tuned): {format_metric(improvement['delta_avg_nll'])}\n")
            f.write(f"- Delta token accuracy (fine-tuned - base): {format_metric(improvement['delta_token_accuracy'])}\n")

        f.write("\n## Per-language Metrics\n\n")
        header = "| Language | #Examples | FT NLL | FT PPL | FT Acc |"
        sep = "|---|---:|---:|---:|---:|"
        if compare_base:
            header = "| Language | #Examples | FT NLL | FT PPL | FT Acc | Base NLL | Base PPL | Base Acc | Delta NLL |"
            sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        f.write(header + "\n")
        f.write(sep + "\n")
        for row in per_language_rows:
            values = [
                row["language"],
                row["num_examples"],
                format_metric(row["ft_avg_nll"]),
                format_metric(row["ft_perplexity"]),
                format_metric(row["ft_token_accuracy"]),
            ]
            if compare_base:
                values.extend(
                    [
                        format_metric(row["base_avg_nll"]),
                        format_metric(row["base_perplexity"]),
                        format_metric(row["base_token_accuracy"]),
                        format_metric(row["delta_avg_nll"]),
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
            f.write(f"- FT token accuracy: {format_metric(record.get('ft_token_accuracy'))}\n")
            if compare_base:
                f.write(f"- Base avg NLL: {format_metric(record.get('base_avg_nll'))}\n")
                f.write(f"- Base token accuracy: {format_metric(record.get('base_token_accuracy'))}\n")
                f.write(f"- Delta avg NLL: {format_metric(record.get('delta_avg_nll'))}\n")
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
    tokenizer_source = cfg.tokenizer_path or model_configs["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(
        path_or_model_id(tokenizer_source),
        trust_remote_code=bool(cfg.trust_remote_code),
    )
    tokenizer = ensure_tokenizer(tokenizer)

    dataset = FullTofuDataset(cfg, tokenizer, model_configs)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        collate_fn=collate_full_tofu,
    )

    ft_model = load_model(cfg.model_path, cfg, model_configs)
    ft_results = evaluate_model(ft_model, dataloader, cfg, "ft")
    selected_records = []

    base_results = None
    if cfg.compare_base:
        unload_model(ft_model)

        base_model_path = cfg.base_model_path or model_configs["hf_key"]
        base_model = load_model(base_model_path, cfg, model_configs)
        base_results = evaluate_model(base_model, dataloader, cfg, "base")

        records = merge_records(dataset, ft_results, base_results)
        selected_records = select_case_studies(records, cfg) if cfg.case_study.enabled else []
        if cfg.generation.enabled and selected_records:
            # Regenerate FT outputs if base comparison changed the selected set.
            unload_model(base_model)
            ft_model = load_model(cfg.model_path, cfg, model_configs)
            generate_outputs(ft_model, tokenizer, model_configs, selected_records, cfg, "ft_generated_answer")
            unload_model(ft_model)
            base_model = load_model(base_model_path, cfg, model_configs)
            generate_outputs(base_model, tokenizer, model_configs, selected_records, cfg, "base_generated_answer")
        unload_model(base_model)
    else:
        records = merge_records(dataset, ft_results, None)
        selected_records = select_case_studies(records, cfg) if cfg.case_study.enabled else []
        if cfg.generation.enabled and selected_records:
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
            "delta_token_accuracy": summary["models"]["finetuned"]["token_accuracy"] - summary["models"]["base"]["token_accuracy"],
        }

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
