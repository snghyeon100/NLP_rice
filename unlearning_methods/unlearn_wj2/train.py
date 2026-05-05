"""Probe-based representation erasure runner.

Run from the NLP_rice repo root with:
    python unlearning_methods/unlearn_wj2/train.py
"""

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

from evaluate_util import evaluate_languages
from unlearning_methods.unlearn_wj2.dataloader import RepErasureDataset, rep_erasure_collator
from unlearning_methods.unlearn_wj2.localization import load_selected_layers, run_localization
from unlearning_methods.unlearn_wj2.loss import compute_rep_erasure_loss
from unlearning_methods.unlearn_wj2.probes import (
    LayerProbeBank,
    evaluate_answer_probes,
    load_probes,
    resolve_candidate_layers,
    train_answer_probes,
)
from unlearning_methods.unlearn_wj2.subspace import apply_selected_lora, trainable_parameter_summary
from utils import get_model_identifiers_from_yaml


def resolve_project_path(path):
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


def _get(cfg, path, default=None):
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        if hasattr(current, "get"):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
    return current


class RepErasureTrainer(Trainer):
    def __init__(
        self,
        *args,
        reference_model=None,
        probes=None,
        answer_bank=None,
        cfg=None,
        selected_layers=None,
        loss_log_path=None,
        **kwargs,
    ):
        self.reference_model = reference_model
        self.probes = probes
        self.answer_bank = answer_bank
        self.cfg = cfg
        self.selected_layers = [int(layer_id) for layer_id in selected_layers]
        self.loss_log_path = Path(loss_log_path) if loss_log_path is not None else None
        self.loss_log_sums = {}
        self.loss_log_count = 0
        super().__init__(*args, **kwargs)
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        if self.probes is not None:
            self.probes.eval()
            for param in self.probes.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs, logs = compute_rep_erasure_loss(
            model,
            self.reference_model,
            self.probes,
            inputs,
            self.cfg,
            self.selected_layers,
            self.answer_bank,
        )
        loss_logs = {key: float(value.detach().cpu()) for key, value in logs.items()}
        for key, value in loss_logs.items():
            self.loss_log_sums[key] = self.loss_log_sums.get(key, 0.0) + value
        self.loss_log_count += 1
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self.loss_log_count:
            averaged = {key: value / self.loss_log_count for key, value in self.loss_log_sums.items()}
            logs = {**logs, **averaged}
        self._append_loss_log(logs)
        self.loss_log_sums = {}
        self.loss_log_count = 0
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
    erase_cfg = cfg.erase
    num_cuda_devices = max(1, torch.cuda.device_count())
    save_steps = erase_cfg.get("save_steps", None) or steps_per_epoch
    return transformers.TrainingArguments(
        per_device_train_batch_size=max(1, batch_size // num_cuda_devices),
        per_device_eval_batch_size=max(1, batch_size // num_cuda_devices),
        gradient_accumulation_steps=erase_cfg.gradient_accumulation_steps,
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=erase_cfg.lr,
        bf16=torch.cuda.is_available(),
        bf16_full_eval=torch.cuda.is_available(),
        logging_steps=max(1, int(erase_cfg.get("logging_steps", 10))),
        logging_dir=str(Path(cfg.save_dir) / "logs"),
        output_dir=cfg.save_dir,
        optim="paged_adamw_32bit",
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=save_steps,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=erase_cfg.weight_decay,
        eval_strategy="no",
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


def _make_loader(dataset, cfg, shuffle=True, batch_size=None):
    return DataLoader(
        dataset,
        batch_size=int(batch_size or cfg.erase.batch_size),
        shuffle=shuffle,
        collate_fn=rep_erasure_collator,
    )


def _needs_reference_model(cfg):
    erase_cfg = cfg.get("erase", {})
    weights = [
        erase_cfg.get("retain_kl_weight", 0.0),
        erase_cfg.get("utility_kl_weight", 0.0),
        erase_cfg.get("hidden_preserve_weight", 0.0),
        erase_cfg.get("forget_hidden_norm_weight", 0.0),
    ]
    return any(float(weight) > 0 for weight in weights)


def run_multilingual_eval(model, tokenizer, cfg, force=False):
    if not force and not cfg.get("run_eval_after_train", False):
        return

    eval_cfg = cfg.get("eval_after_train", None)
    eval_cfg = OmegaConf.create(OmegaConf.to_container(eval_cfg, resolve=True) if eval_cfg is not None else {})
    eval_cfg.model_family = cfg.model_family
    eval_cfg.model_path = cfg.save_dir
    eval_cfg.tokenizer_path = cfg.save_dir
    eval_cfg.prefer_checkpoint_tokenizer = True
    eval_cfg.use_pretrained = False
    eval_cfg.reinitialize_weights = False
    if "languages" not in eval_cfg:
        eval_cfg.languages = ["en", "ar", "fa", "fr", "hi", "id", "iw", "ja", "ko", "ru"]
    if "language" not in eval_cfg:
        eval_cfg.language = "en"
    if "save_dir" not in eval_cfg:
        eval_cfg.save_dir = str(Path(cfg.save_dir) / "eval_results" / "multilingual")
    if "generation" not in eval_cfg:
        eval_cfg.generation = {"max_length": 200, "max_new_tokens": 128}
    if "batch_size" not in eval_cfg:
        eval_cfg.batch_size = 4
    if "overwrite" not in eval_cfg:
        eval_cfg.overwrite = True

    print(f"Running multilingual evaluation after training. Results will be saved to {eval_cfg.save_dir}")
    evaluate_languages(model, tokenizer, eval_cfg)


def prepare_model_for_save(model, cfg):
    if cfg.get("save_merged_model", True) and hasattr(model, "merge_and_unload"):
        print("Merging LoRA adapter into base model before saving.")
        return model.merge_and_unload()
    return model


def save_probe_audit(path, stage, audit):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"stage": stage, "audit": audit}) + "\n")


def save_answer_bank_summary(path, answer_bank):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    source_counts = {}
    for source_type in answer_bank.source_types:
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
    payload = {
        "num_entries": int(answer_bank.input_ids.shape[0]),
        "num_samples": int(answer_bank.owner_indices.unique().numel()),
        "num_positive_entries": int(answer_bank.positive_mask.sum().item()),
        "num_negative_entries": int((~answer_bank.positive_mask).sum().item()),
        "source_counts": source_counts,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Answer bank: {payload}")


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.chdir(PROJECT_ROOT)
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.reference_model_path = resolve_project_path(cfg.get("reference_model_path", cfg.model_path))
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, Path(cfg.save_dir) / "cfg.yaml")

    tokenizer_source = _tokenizer_source(cfg.model_path, model_id)
    print(f"Loading tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = RepErasureDataset(cfg, tokenizer=tokenizer, model_family=cfg.model_family)
    save_answer_bank_summary(Path(cfg.save_dir) / "answer_bank_summary.json", dataset.answer_bank)
    probe_loader = _make_loader(dataset, cfg, shuffle=True, batch_size=cfg.erase.batch_size)
    localization_loader = _make_loader(dataset, cfg, shuffle=False, batch_size=cfg.erase.batch_size)

    batch_size = int(cfg.erase.batch_size)
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = max(1, len(dataset) // (batch_size * cfg.erase.gradient_accumulation_steps * num_devices))
    max_steps = max(1, int(cfg.erase.num_epochs * len(dataset)) // (batch_size * cfg.erase.gradient_accumulation_steps * num_devices))

    model = _load_model(cfg.model_path, model_cfg, model_id).to(_device_for_main_model())
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    candidate_layers = resolve_candidate_layers(model, cfg.probe.get("candidate_layers", "auto"))
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "n_embd", 0)))
    if hidden_size <= 0:
        raise ValueError("Could not infer model hidden size.")

    if cfg.probe.get("load_path", None):
        probes = load_probes(
            resolve_project_path(cfg.probe.load_path),
            hidden_size=hidden_size,
            rank=int(cfg.probe.get("hidden_rank", 64)),
            map_location=_device_for_main_model(),
        )
    else:
        probes = LayerProbeBank(candidate_layers, hidden_size, rank=int(cfg.probe.get("hidden_rank", 64)))

    if cfg.probe.get("train", True) and not cfg.probe.get("load_path", None):
        probes = train_answer_probes(model, probes, probe_loader, dataset.answer_bank, cfg, Path(cfg.save_dir) / "probes")
    else:
        probes.to(_device_for_main_model())

    pre_audit = evaluate_answer_probes(
        model,
        probes,
        localization_loader,
        dataset.answer_bank,
        cfg,
        max_batches=cfg.localization.get("num_batches", 8),
    )
    save_probe_audit(Path(cfg.save_dir) / "probe_audit.jsonl", "pre_erasure", pre_audit)

    if cfg.localization.get("selected_layers_path", None):
        selected_layers = load_selected_layers(resolve_project_path(cfg.localization.selected_layers_path))
    elif cfg.localization.get("run", True):
        selected_layers, _ = run_localization(model, probes, localization_loader, dataset.answer_bank, cfg, cfg.save_dir)
    else:
        selected_layers = candidate_layers[: int(cfg.localization.get("select_top_k_layers", len(candidate_layers)))]
    print(f"Selected layers for erasure: {selected_layers}")

    if cfg.eval_only:
        run_multilingual_eval(model, tokenizer, cfg, force=True)
        return

    model = apply_selected_lora(model, cfg, selected_layers)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    summary = trainable_parameter_summary(model)
    with open(Path(cfg.save_dir) / "trainable_parameters.json", "w") as f:
        json.dump({key: value for key, value in summary.items() if key != "trainable_names"}, f, indent=2)
    with open(Path(cfg.save_dir) / "trainable_parameter_names.txt", "w") as f:
        f.write("\n".join(summary["trainable_names"]))

    reference_model = None
    if _needs_reference_model(cfg):
        reference_model = _load_model(cfg.reference_model_path, model_cfg, model_id).to(_device_for_reference_model())
        reference_model.eval()

    training_args = build_training_args(cfg, max_steps, steps_per_epoch, batch_size)
    trainer = RepErasureTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=dataset,
        args=training_args,
        data_collator=rep_erasure_collator,
        reference_model=reference_model,
        probes=probes,
        answer_bank=dataset.answer_bank,
        cfg=cfg,
        selected_layers=selected_layers,
        loss_log_path=Path(cfg.save_dir) / "training_loss_history.jsonl",
    )
    trainer.train()

    post_audit = evaluate_answer_probes(
        model,
        probes,
        localization_loader,
        dataset.answer_bank,
        cfg,
        max_batches=cfg.localization.get("num_batches", 8),
    )
    save_probe_audit(Path(cfg.save_dir) / "probe_audit.jsonl", "post_erasure", post_audit)

    if cfg.save_model:
        model = prepare_model_for_save(model, cfg)
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)
        run_multilingual_eval(model, tokenizer, cfg)

    for file in Path(cfg.save_dir).glob("checkpoint-*"):
        for global_step_dir in file.glob("global_step*"):
            shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
