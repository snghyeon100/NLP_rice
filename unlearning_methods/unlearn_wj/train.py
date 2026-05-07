"""End-to-end WJ zero-shot cross-lingual likelihood unlearning runner.

Run from the repo root:
    CUDA_VISIBLE_DEVICES=0,1 python unlearning_methods/unlearn_wj/train.py
"""

import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from unlearning_methods.unlearn_wj.dataloader import WJUnlearningDataset, wj_collator
from unlearning_methods.unlearn_wj.localization import collect_lora_target_modules, select_layers
from unlearning_methods.unlearn_wj.loss import compute_wj_loss
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


def pick_device(index):
    if not torch.cuda.is_available():
        return torch.device("cpu")
    index = int(index)
    if index >= torch.cuda.device_count():
        raise ValueError(f"Requested cuda:{index}, but only {torch.cuda.device_count()} visible CUDA devices exist.")
    return torch.device(f"cuda:{index}")


def pick_dtype(cfg):
    if not torch.cuda.is_available():
        return torch.float32
    if bool(cfg.bf16):
        return torch.bfloat16
    if bool(cfg.fp16):
        return torch.float16
    return torch.float32


def ensure_save_dir(cfg):
    save_dir = Path(cfg.save_dir)
    if save_dir.exists() and any(save_dir.iterdir()) and not bool(cfg.overwrite_dir):
        raise FileExistsError(
            f"save_dir already exists and is not empty: {save_dir}\n"
            "Set overwrite_dir=true or choose a new save_dir."
        )
    if save_dir.exists() and bool(cfg.overwrite_dir):
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.lora_save_dir).mkdir(parents=True, exist_ok=True)
    if bool(cfg.save_merged_model):
        Path(cfg.merged_save_dir).mkdir(parents=True, exist_ok=True)


def count_trainable_parameters(model):
    trainable = 0
    total = 0
    tensors = []
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            tensors.append({"name": name, "numel": param.numel(), "shape": list(param.shape)})
    return {
        "trainable": trainable,
        "total": total,
        "trainable_fraction": trainable / max(1, total),
        "trainable_tensors": tensors,
    }


def move_batch_to_device(batch, device):
    moved = {}
    for key, tensors in batch.items():
        moved[key] = tuple(tensor.to(device) for tensor in tensors)
    return moved


def build_model(model_path, model_cfg, dtype, device):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)


def attach_lora(model, cfg, selected_layers):
    if not bool(cfg.use_lora):
        for param in model.parameters():
            param.requires_grad = True
        return model, None

    from peft import LoraConfig, TaskType, get_peft_model

    target_leaves = cfg.get("lora_target_modules", None)
    if target_leaves is None:
        target_leaves = sorted(
            {
                name.split(".")[-1]
                for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and name.split(".")[-1] != "lm_head"
            }
        )
    target_modules = collect_lora_target_modules(model, target_leaves, selected_layers)
    print(f"LoRA selected layers: {selected_layers}")
    print(f"LoRA target module count: {len(target_modules)}")

    lora_config = LoraConfig(
        r=int(cfg.lora_r),
        lora_alpha=int(cfg.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, target_modules


def save_training_artifacts(model, tokenizer, cfg, final=False):
    if bool(cfg.use_lora):
        model.save_pretrained(cfg.lora_save_dir)
        tokenizer.save_pretrained(cfg.lora_save_dir)
        if final and bool(cfg.save_merged_model):
            merged = model.merge_and_unload()
            merged.save_pretrained(cfg.merged_save_dir, safe_serialization=True)
            tokenizer.save_pretrained(cfg.merged_save_dir)
    else:
        model.save_pretrained(cfg.save_dir, safe_serialization=True)
        tokenizer.save_pretrained(cfg.save_dir)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ["WANDB_DISABLED"] = "true"
    os.chdir(PROJECT_ROOT)
    set_seed(int(cfg.seed))

    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    cfg.lora_save_dir = resolve_project_path(cfg.lora_save_dir)
    cfg.merged_save_dir = resolve_project_path(cfg.merged_save_dir)
    cfg.retain_multi_path = resolve_project_path(cfg.retain_multi_path)
    cfg.utility_multi_path = resolve_project_path(cfg.utility_multi_path)
    cfg.parallel_anchor_path = resolve_project_path(cfg.parallel_anchor_path)

    ensure_save_dir(cfg)
    OmegaConf.save(cfg, Path(cfg.save_dir) / "config_resolved.yaml")

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    tokenizer_source = cfg.model_path if Path(cfg.model_path).exists() else model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_device = pick_device(cfg.gpu_train)
    ref_device = pick_device(cfg.gpu_ref)
    dtype = pick_dtype(cfg)

    print("=" * 70)
    print("WJ zero-shot cross-lingual likelihood unlearning")
    print(f"model_path:       {cfg.model_path}")
    print(f"save_dir:         {cfg.save_dir}")
    print(f"source_language:  {cfg.source_language}")
    print(f"target_languages: {list(cfg.target_languages)}")
    print(f"forget_objective: {cfg.forget_objective}")
    print(f"train_device:     {train_device}")
    print(f"ref_device:       {ref_device}")
    print(f"dtype:            {dtype}")
    print("=" * 70)

    dataset = WJUnlearningDataset(cfg, tokenizer=tokenizer, project_root=PROJECT_ROOT)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        collate_fn=wj_collator,
        drop_last=False,
    )
    steps_per_epoch = max(1, len(dataloader))
    if cfg.max_steps is None:
        max_steps = max(1, steps_per_epoch * int(cfg.num_epochs))
    else:
        max_steps = int(cfg.max_steps)
    optimizer_steps = max(1, math.ceil(max_steps / int(cfg.gradient_accumulation_steps)))
    warmup_steps = int(float(cfg.warmup_ratio) * optimizer_steps)

    print(f"dataset size:     {len(dataset)}")
    print(f"steps_per_epoch:  {steps_per_epoch}")
    print(f"max_steps:        {max_steps}")
    print(f"optimizer_steps:  {optimizer_steps}")
    print(f"warmup_steps:     {warmup_steps}")

    model = build_model(cfg.model_path, model_cfg, dtype, train_device)
    ref_model = build_model(cfg.model_path, model_cfg, dtype, ref_device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    selected_layers, _ = select_layers(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        project_root=PROJECT_ROOT,
        save_dir=Path(cfg.save_dir) / "localization",
    )
    model, target_modules = attach_lora(model, cfg, selected_layers)
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False

    trainable_info = count_trainable_parameters(model)
    trainable_info["selected_layers"] = selected_layers
    trainable_info["target_modules"] = target_modules
    with open(Path(cfg.save_dir) / "trainable_parameters.json", "w") as f:
        json.dump(trainable_info, f, indent=2)

    if bool(cfg.eval_only):
        print("eval_only=true: model loaded and artifacts written; no training run.")
        return

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        eps=float(cfg.adam_epsilon),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_steps,
    )

    log_path = Path(cfg.save_dir) / "training_loss_history.jsonl"
    model.train()
    global_step = 0
    optimizer_step = 0
    running_start = time.time()

    while global_step < max_steps:
        for batch in dataloader:
            if global_step >= max_steps:
                break
            global_step += 1
            batch = move_batch_to_device(batch, train_device)

            loss, _, logs = compute_wj_loss(model, ref_model, batch, cfg)
            if not torch.isfinite(loss):
                message = f"Non-finite loss at step {global_step}: {logs}"
                if bool(cfg.abort_on_nonfinite):
                    raise FloatingPointError(message)
                print(f"[warning] {message}; skipping update")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaled_loss = loss / int(cfg.gradient_accumulation_steps)
            scaled_loss.backward()

            should_step = global_step % int(cfg.gradient_accumulation_steps) == 0 or global_step == max_steps
            if should_step:
                trainable_params = [param for param in model.parameters() if param.requires_grad]
                nonfinite_grad = False
                for param in trainable_params:
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        nonfinite_grad = True
                        break
                if nonfinite_grad:
                    message = f"Non-finite gradient at step {global_step}: {logs}"
                    if bool(cfg.abort_on_nonfinite):
                        raise FloatingPointError(message)
                    print(f"[warning] {message}; skipping update")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                if float(cfg.max_grad_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.max_grad_norm))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

            logs.update(
                {
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "lr": scheduler.get_last_lr()[0],
                    "elapsed_sec": time.time() - running_start,
                    "batch_forget_source_size": int(batch["forget_source"][0].shape[0]),
                    "batch_retain_source_size": int(batch["retain_source"][0].shape[0]),
                    "batch_retain_multi_size": int(batch["retain_multi"][0].shape[0]),
                    "batch_utility_multi_size": int(batch["utility_multi"][0].shape[0]),
                }
            )
            with open(log_path, "a") as f:
                f.write(json.dumps(logs, ensure_ascii=False) + "\n")

            if global_step == 1 or global_step % int(cfg.log_steps) == 0:
                print(
                    f"step {global_step}/{max_steps} "
                    f"loss={logs['loss_total']:.4f} "
                    f"forget={logs['loss_forget']:.4f} "
                    f"lr={logs['lr']:.3e}"
                )

        if bool(cfg.save_every_epoch):
            save_training_artifacts(model, tokenizer, cfg, final=False)

    save_training_artifacts(model, tokenizer, cfg, final=True)
    print(f"Training complete. Artifacts saved to {cfg.save_dir}")


if __name__ == "__main__":
    main()
