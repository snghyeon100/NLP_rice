"""Subspace-aware zero-shot cross-lingual unlearning runner.

Run from the NLP_rice repo root with:
    python unlearning_methods/unlearn_wj/train.py
"""

import csv
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import transformers
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed

from evaluate_util import get_all_evals, get_dataloader
from unlearning_methods.unlearn_wj.dataloader import SubspaceXlingualDataset, subspace_xlingual_collator
from unlearning_methods.unlearn_wj.localization import load_selected_layers, run_localization
from unlearning_methods.unlearn_wj.loss import compute_subspace_xlingual_loss
from unlearning_methods.unlearn_wj.subspace import (
    apply_selected_lora,
    estimate_preserve_subspace,
    load_preserve_subspace,
    register_projection_hooks,
    trainable_parameter_summary,
)
from utils import get_forget_quality, get_model_identifiers_from_yaml, get_model_utility


def resolve_project_path(path):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


class SubspaceXlingualTrainer(Trainer):
    def __init__(self, *args, reference_model=None, alpha=1.0, beta=1.0, gamma=1.0, loss_log_path=None, **kwargs):
        self.reference_model = reference_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_logs = {}
        self.loss_log_path = Path(loss_log_path) if loss_log_path is not None else None
        super().__init__(*args, **kwargs)
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, logs = compute_subspace_xlingual_loss(
            model,
            self.reference_model,
            inputs,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        self.loss_logs = {key: float(value.cpu()) for key, value in logs.items()}
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self.loss_logs:
            logs = {**logs, **self.loss_logs}
        self._append_loss_log(logs)
        try:
            return super().log(logs, start_time=start_time)
        except TypeError:
            return super().log(logs)

    def _append_loss_log(self, logs):
        if self.loss_log_path is None:
            return

        self.loss_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "step": int(self.state.global_step),
            "epoch": float(logs["epoch"]) if "epoch" in logs else None,
        }
        for key, value in logs.items():
            if key == "epoch":
                continue
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float, str, bool)) or value is None:
                record[key] = value
            else:
                record[key] = str(value)

        with open(self.loss_log_path, "a") as f:
            f.write(json.dumps(record) + "\n")


def build_training_args(cfg, max_steps, steps_per_epoch, batch_size):
    num_cuda_devices = max(1, torch.cuda.device_count())
    save_steps = cfg.get("save_steps", None) or steps_per_epoch
    return transformers.TrainingArguments(
        per_device_train_batch_size=max(1, batch_size // num_cuda_devices),
        per_device_eval_batch_size=max(1, batch_size // num_cuda_devices),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=torch.cuda.is_available(),
        bf16_full_eval=torch.cuda.is_available(),
        logging_steps=max(1, int(cfg.get("logging_steps", 10))),
        logging_dir=str(Path(cfg.save_dir) / "logs"),
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        eval_steps=steps_per_epoch,
        eval_strategy="steps" if cfg.eval_while_train else "no",
        seed=cfg.seed,
        remove_unused_columns=False,
    )


def _load_model(path, model_cfg, model_id):
    return AutoModelForCausalLM.from_pretrained(
        path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )


def _tokenizer_source(model_path, model_id):
    model_path = Path(model_path)
    tokenizer_files = (
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "chat_template.jinja",
    )
    if model_path.exists() and any((model_path / filename).exists() for filename in tokenizer_files):
        return str(model_path)
    return model_id


def _device_for_main_model():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _device_for_reference_model():
    if torch.cuda.device_count() > 1:
        return "cuda:1"
    return _device_for_main_model()


def _make_loader(dataset, cfg, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        collate_fn=subspace_xlingual_collator,
    )


def run_eval_jobs(model, tokenizer, cfg, force=False):
    if not force and not cfg.get("run_eval_after_train", False):
        return

    aggregated_by_language = {}
    for job in cfg.eval.jobs:
        language = job.language
        save_dir = Path(resolve_project_path(job.save_dir))
        save_dir.mkdir(parents=True, exist_ok=True)
        job_cfg = OmegaConf.create(OmegaConf.to_container(job, resolve=True))
        job_cfg.model_family = cfg.model_family

        language_logs = {}
        for folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key in zip(
            job_cfg.data_path,
            job_cfg.split_list,
            job_cfg.question_key,
            job_cfg.answer_key,
            job_cfg.eval_task,
            job_cfg.base_answer_key,
            job_cfg.perturbed_answer_key,
        ):
            if eval_task == "eval_log_forget":
                split = job_cfg.split
            save_filename = save_dir / f"{eval_task}.json"
            if save_filename.exists() and not job_cfg.overwrite:
                continue
            eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                job_cfg,
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
            eval_logs = get_all_evals(
                job_cfg,
                model,
                tokenizer,
                eval_task,
                eval_dataloader,
                base_eval_dataloader,
                perturb_dataloader,
                normalize_gt="eval_log" not in eval_task,
                language=language,
            )
            with open(save_filename, "w") as f:
                json.dump(eval_logs, f, indent=4)
            language_logs[f"{eval_task}.json"] = eval_logs

        with open(save_dir / "eval_log_aggregated.json", "w") as f:
            json.dump(language_logs, f, indent=4)
        if job.get("retain_result", None) is not None and "eval_log_forget.json" in language_logs:
            model_utility = get_model_utility(language_logs)
            retain_result = json.load(open(resolve_project_path(job.retain_result), "r"))
            forget_quality = get_forget_quality(language_logs, retain_result)
            aggregate_stat = {**model_utility, **forget_quality}
            with open(save_dir / "aggregate_stat.csv", "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(aggregate_stat.keys()))
                writer.writeheader()
                writer.writerow(aggregate_stat)
        aggregated_by_language[language] = language_logs

    merged_path = Path(cfg.save_dir) / "eval_results" / "multilingual_aggregated.json"
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with open(merged_path, "w") as f:
        json.dump(aggregated_by_language, f, indent=4)


def prepare_model_for_save(model, cfg):
    if cfg.get("save_merged_model", True) and hasattr(model, "merge_and_unload"):
        print("Merging LoRA adapter into base model before saving.")
        return model.merge_and_unload()
    return model


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.chdir(PROJECT_ROOT)
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.reference_model_path = resolve_project_path(cfg.get("reference_model_path", cfg.model_path))
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.save_dir) / "cfg.yaml")

    tokenizer = AutoTokenizer.from_pretrained(
        _tokenizer_source(cfg.model_path, model_id),
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    dataset = SubspaceXlingualDataset(cfg, tokenizer=tokenizer, model_family=cfg.model_family)
    localization_loader = _make_loader(dataset, cfg, shuffle=False)

    batch_size = int(cfg.batch_size)
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = max(1, len(dataset) // (batch_size * cfg.gradient_accumulation_steps * num_devices))
    max_steps = max(1, int(cfg.num_epochs * len(dataset)) // (batch_size * cfg.gradient_accumulation_steps * num_devices))

    model = _load_model(cfg.model_path, model_cfg, model_id).to(_device_for_main_model())
    model.generation_config.do_sample = True
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if cfg.get("selected_layers_path", None):
        selected_layers = load_selected_layers(resolve_project_path(cfg.selected_layers_path))
    elif cfg.get("run_localization", True):
        selected_layers, _ = run_localization(model, localization_loader, cfg, cfg.save_dir)
    else:
        selected_layers = []

    model = apply_selected_lora(model, cfg, selected_layers)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    summary = trainable_parameter_summary(model)
    with open(Path(cfg.save_dir) / "trainable_parameters.json", "w") as f:
        json.dump({key: value for key, value in summary.items() if key != "trainable_names"}, f, indent=2)
    with open(Path(cfg.save_dir) / "trainable_parameter_names.txt", "w") as f:
        f.write("\n".join(summary["trainable_names"]))

    reference_model = None
    if float(cfg.get("gamma", 0.0)) > 0:
        reference_model = _load_model(cfg.reference_model_path, model_cfg, model_id).to(_device_for_reference_model())
        reference_model.eval()

    projection_handles = []
    if cfg.get("use_projection", True):
        basis_path = cfg.get("preserve_basis_path", None)
        if basis_path:
            preserve_basis = load_preserve_subspace(resolve_project_path(basis_path))
        else:
            preserve_basis = estimate_preserve_subspace(model, localization_loader, cfg, cfg.save_dir)
        projection_handles = register_projection_hooks(model, preserve_basis)

    training_args = build_training_args(cfg, max_steps, steps_per_epoch, batch_size)
    trainer = SubspaceXlingualTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=subspace_xlingual_collator,
        reference_model=reference_model,
        alpha=float(cfg.alpha),
        beta=float(cfg.beta),
        gamma=float(cfg.gamma),
        loss_log_path=Path(cfg.save_dir) / "training_loss_history.jsonl",
    )

    if cfg.eval_only:
        run_eval_jobs(model, tokenizer, cfg, force=True)
    else:
        trainer.train()

    for handle in projection_handles:
        handle.remove()

    if cfg.save_model and (not cfg.eval_only):
        model = prepare_model_for_save(model, cfg)
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        run_eval_jobs(model, tokenizer, cfg)

    for file in Path(cfg.save_dir).glob("checkpoint-*"):
        for global_step_dir in file.glob("global_step*"):
            shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
