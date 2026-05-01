"""LoRA layer selection helpers for representation erasure."""

from unlearning_methods.unlearn_wj2.localization import extract_layer_id


def apply_selected_lora(model, cfg, selected_layers):
    if not cfg.get("use_lora", True):
        if cfg.get("full_finetune", False):
            for param in model.parameters():
                param.requires_grad = True
        else:
            freeze_unselected_layers(model, selected_layers)
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=int(cfg.get("lora_rank", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
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
