"""LoRA selection and preserve-subspace gradient projection."""

import re
from pathlib import Path

import torch

from unlearning_methods.unlearn_wj.localization import extract_layer_id
from unlearning_methods.unlearn_wj.loss import forward_outputs


def apply_selected_lora(model, cfg, selected_layers):
    if not cfg.get("use_lora", True):
        if not cfg.get("full_finetune", False):
            freeze_unselected_layers(model, selected_layers)
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=int(cfg.get("lora_alpha", 16)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=list(cfg.get("lora_target_modules", [])),
    )
    model = get_peft_model(model, lora_config)
    freeze_unselected_lora(model, selected_layers)
    return model


def freeze_unselected_layers(model, selected_layers):
    selected = set(int(layer_id) for layer_id in selected_layers)
    for name, param in model.named_parameters():
        layer_id = extract_layer_id(name)
        param.requires_grad = layer_id in selected


def freeze_unselected_lora(model, selected_layers):
    selected = set(int(layer_id) for layer_id in selected_layers)
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            continue
        layer_id = extract_layer_id(name)
        param.requires_grad = layer_id in selected


def trainable_parameter_summary(model):
    trainable = 0
    total = 0
    trainable_names = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            trainable_names.append(name)
    return {
        "trainable": trainable,
        "total": total,
        "ratio": trainable / max(total, 1),
        "trainable_names": trainable_names,
    }


def _safe_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _preserve_loss(model, inputs):
    _, retain_inputs, utility_inputs, _, _ = inputs
    retain_outputs = forward_outputs(model, retain_inputs)
    utility_outputs = forward_outputs(model, utility_inputs)
    return retain_outputs.loss + utility_outputs.loss


def estimate_preserve_subspace(model, dataloader, cfg, save_dir):
    max_batches = int(cfg.get("max_preserve_batches", 8))
    rank = int(cfg.get("preserve_rank", 4))
    gradients = {}
    used_batches = 0

    model.train()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        model.zero_grad(set_to_none=True)
        loss = _preserve_loss(model, batch)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients.setdefault(name, []).append(param.grad.detach().flatten().float().cpu())
        used_batches += 1

    if used_batches == 0:
        raise ValueError("Preserve subspace dataloader produced no batches.")

    basis_by_name = {}
    param_shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}
    output_dir = Path(save_dir) / "subspace"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, grad_list in gradients.items():
        matrix = torch.stack(grad_list, dim=1)
        local_rank = min(rank, matrix.shape[1], matrix.shape[0])
        if local_rank <= 0:
            continue
        basis = torch.linalg.svd(matrix, full_matrices=False).U[:, :local_rank].contiguous()
        basis_by_name[name] = basis
        torch.save({"name": name, "basis": basis, "shape": param_shapes[name]}, output_dir / f"{_safe_name(name)}.pt")

    torch.save({"basis": basis_by_name, "num_batches": used_batches, "rank": rank}, output_dir / "preserve_basis.pt")
    model.zero_grad(set_to_none=True)
    return basis_by_name


def load_preserve_subspace(path):
    payload = torch.load(path, map_location="cpu")
    return payload["basis"] if isinstance(payload, dict) and "basis" in payload else payload


def register_projection_hooks(model, basis_by_name):
    handles = []
    basis_cache = {}

    for name, param in model.named_parameters():
        basis = basis_by_name.get(name)
        if basis is None or not param.requires_grad:
            continue

        def _hook(grad, param_name=name, cpu_basis=basis):
            cached = basis_cache.get(param_name)
            if cached is None or cached.device != grad.device:
                cached = cpu_basis.to(device=grad.device, dtype=torch.float32)
                basis_cache[param_name] = cached
            flat = grad.detach().flatten().float()
            projected = flat - cached @ (cached.transpose(0, 1) @ flat)
            return projected.view_as(grad).to(dtype=grad.dtype)

        handles.append(param.register_hook(_hook))

    return handles
