"""Layer localization for cross-lingual subspace-aware unlearning."""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from unlearning_methods.unlearn_wj.loss import batch_to_device, model_device


LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\."),
    re.compile(r"(?:^|\.)transformer\.h\.(\d+)\."),
    re.compile(r"(?:^|\.)h\.(\d+)\."),
)


def extract_layer_id(name):
    for pattern in LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def _matches_target_module(name, target_modules):
    if not target_modules:
        return True
    return any(f".{module}." in name or name.endswith(f".{module}.weight") or name.endswith(f".{module}.bias") for module in target_modules)


def trainable_layer_ids(model):
    layer_ids = set()
    for name, _ in model.named_parameters():
        layer_id = extract_layer_id(name)
        if layer_id is not None:
            layer_ids.add(layer_id)
    return sorted(layer_ids)


def _zero_scores(layer_ids):
    return {str(layer_id): 0.0 for layer_id in layer_ids}


def _zscore(values, eps=1e-12):
    keys = list(values.keys())
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float32)
    if tensor.numel() <= 1:
        return {key: 0.0 for key in keys}
    std = tensor.std(unbiased=False).clamp_min(float(eps))
    normalized = (tensor - tensor.mean()) / std
    return {key: float(value) for key, value in zip(keys, normalized)}


def _score_layer_gradients(model, loss, layer_ids, target_modules=None, eps=1e-12):
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sq = defaultdict(float)
    param_sq = defaultdict(float)
    for name, param in model.named_parameters():
        if param.grad is None or not _matches_target_module(name, target_modules):
            continue
        layer_id = extract_layer_id(name)
        if layer_id is None:
            continue
        grad_sq[layer_id] += float(param.grad.detach().float().pow(2).sum().cpu())
        param_sq[layer_id] += float(param.detach().float().pow(2).sum().cpu())

    scores = {}
    for layer_id in layer_ids:
        grad_norm = grad_sq[layer_id] ** 0.5
        param_norm = param_sq[layer_id] ** 0.5
        scores[str(layer_id)] = grad_norm / (param_norm + float(eps))
    return scores


def _add_scores(total, scores):
    for key, value in scores.items():
        total[key] = total.get(key, 0.0) + float(value)


def _forward_loss(model, inputs):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    return model(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss


def _pool_hidden(hidden, attention_mask, pooling="mean"):
    mask = attention_mask.to(hidden.device).float()
    if pooling == "last":
        lengths = mask.sum(dim=1).long().clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
    masked_hidden = hidden * mask.unsqueeze(-1)
    return masked_hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


@torch.no_grad()
def _hidden_share_scores(model, en_inputs, tgt_inputs, layer_ids, pooling="mean"):
    device = model_device(model)
    en_input_ids, _, en_attention_mask = batch_to_device(en_inputs, device)
    tgt_input_ids, _, tgt_attention_mask = batch_to_device(tgt_inputs, device)
    en_outputs = model(
        input_ids=en_input_ids,
        attention_mask=en_attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    tgt_outputs = model(
        input_ids=tgt_input_ids,
        attention_mask=tgt_attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )

    scores = {}
    for layer_id in layer_ids:
        hidden_idx = layer_id + 1
        if hidden_idx >= len(en_outputs.hidden_states) or hidden_idx >= len(tgt_outputs.hidden_states):
            continue
        en_pool = _pool_hidden(en_outputs.hidden_states[hidden_idx], en_attention_mask, pooling=pooling)
        tgt_pool = _pool_hidden(tgt_outputs.hidden_states[hidden_idx], tgt_attention_mask, pooling=pooling)
        scores[str(layer_id)] = float(F.cosine_similarity(en_pool.float(), tgt_pool.float(), dim=-1).mean().cpu())
    return scores


def _layer_grad_vectors(model, loss, layer_ids, target_modules=None):
    model.zero_grad(set_to_none=True)
    loss.backward()
    vectors = defaultdict(list)
    for name, param in model.named_parameters():
        if param.grad is None or not _matches_target_module(name, target_modules):
            continue
        layer_id = extract_layer_id(name)
        if layer_id in layer_ids:
            vectors[layer_id].append(param.grad.detach().flatten().float().cpu())
    return {layer_id: torch.cat(parts) for layer_id, parts in vectors.items() if parts}


def _grad_share_scores(model, en_inputs, tgt_inputs, layer_ids, target_modules=None, eps=1e-12):
    en_vectors = _layer_grad_vectors(model, _forward_loss(model, en_inputs), layer_ids, target_modules=target_modules)
    tgt_vectors = _layer_grad_vectors(model, _forward_loss(model, tgt_inputs), layer_ids, target_modules=target_modules)
    scores = {}
    for layer_id in layer_ids:
        if layer_id not in en_vectors or layer_id not in tgt_vectors:
            continue
        en_vec = en_vectors[layer_id]
        tgt_vec = tgt_vectors[layer_id]
        denom = en_vec.norm() * tgt_vec.norm()
        scores[str(layer_id)] = float((en_vec @ tgt_vec / denom.clamp_min(float(eps))).item())
    return scores


def _final_scores(forget_scores, share_scores, util_scores, cfg):
    eps = float(cfg.get("score_eps", 1e-12))
    z_forget = _zscore(forget_scores, eps=eps)
    z_share = _zscore(share_scores, eps=eps)
    z_util = _zscore(util_scores, eps=eps)
    eta = float(cfg.get("share_eta", 1.0))
    util_lambda = float(cfg.get("util_lambda", 1.0))
    mode = cfg.get("score_mode", "additive")

    final = {}
    for key in forget_scores:
        if mode == "multiplicative":
            score = z_forget[key] * z_share.get(key, 0.0) - util_lambda * z_util.get(key, 0.0)
        else:
            score = z_forget[key] + eta * z_share.get(key, 0.0) - util_lambda * z_util.get(key, 0.0)
        final[key] = float(score)
    return final, z_forget, z_share, z_util


def _select_layers(final_scores, cfg):
    ranked = sorted(((int(key), value) for key, value in final_scores.items()), key=lambda item: item[1], reverse=True)
    threshold = cfg.get("select_score_threshold", None)
    if threshold is not None:
        selected = [layer_id for layer_id, score in ranked if score >= float(threshold)]
    else:
        top_k = int(cfg.get("select_top_k_layers", len(ranked)))
        selected = [layer_id for layer_id, _ in ranked[:top_k]]
    return selected


def run_localization(model, dataloader, cfg, save_dir):
    model.train()
    target_modules = list(cfg.get("lora_target_modules", []))
    layer_ids = trainable_layer_ids(model)
    if not layer_ids:
        raise ValueError("No transformer layer ids were found in model parameter names.")

    max_batches = int(cfg.get("localization_num_batches", 8))
    forget_total = _zero_scores(layer_ids)
    util_total = _zero_scores(layer_ids)
    share_total = _zero_scores(layer_ids)
    used_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        forget_inputs, retain_inputs, utility_inputs, parallel_en_inputs, parallel_tgt_inputs = batch

        forget_scores = _score_layer_gradients(
            model,
            _forward_loss(model, forget_inputs),
            layer_ids,
            target_modules=target_modules,
            eps=cfg.get("score_eps", 1e-12),
        )
        util_loss = _forward_loss(model, retain_inputs) + _forward_loss(model, utility_inputs)
        util_scores = _score_layer_gradients(
            model,
            util_loss,
            layer_ids,
            target_modules=target_modules,
            eps=cfg.get("score_eps", 1e-12),
        )

        share_type = cfg.get("share_score_type", "hidden")
        if share_type in ("hidden", "mixed"):
            share_scores = _hidden_share_scores(
                model,
                parallel_en_inputs,
                parallel_tgt_inputs,
                layer_ids,
                pooling=cfg.get("pooling", "mean"),
            )
        else:
            share_scores = _zero_scores(layer_ids)
        if share_type in ("grad", "mixed"):
            grad_scores = _grad_share_scores(
                model,
                parallel_en_inputs,
                parallel_tgt_inputs,
                layer_ids,
                target_modules=target_modules,
                eps=cfg.get("score_eps", 1e-12),
            )
            if share_type == "mixed":
                for key, value in grad_scores.items():
                    share_scores[key] = 0.5 * share_scores.get(key, 0.0) + 0.5 * value
            else:
                share_scores = grad_scores

        _add_scores(forget_total, forget_scores)
        _add_scores(util_total, util_scores)
        _add_scores(share_total, share_scores)
        used_batches += 1

    if used_batches == 0:
        raise ValueError("Localization dataloader produced no batches.")

    forget_avg = {key: value / used_batches for key, value in forget_total.items()}
    util_avg = {key: value / used_batches for key, value in util_total.items()}
    share_avg = {key: value / used_batches for key, value in share_total.items()}
    final, z_forget, z_share, z_util = _final_scores(forget_avg, share_avg, util_avg, cfg)
    selected_layers = _select_layers(final, cfg)

    output_dir = Path(save_dir) / "localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "forget": forget_avg,
        "share": share_avg,
        "utility": util_avg,
        "z_forget": z_forget,
        "z_share": z_share,
        "z_utility": z_util,
        "final": final,
        "selected_layers": selected_layers,
        "num_batches": used_batches,
    }
    with open(output_dir / "layer_scores.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(output_dir / "selected_layers.json", "w") as f:
        json.dump(selected_layers, f, indent=2)
    model.zero_grad(set_to_none=True)
    return selected_layers, payload


def load_selected_layers(path):
    with open(path, "r") as f:
        selected = json.load(f)
    return [int(layer_id) for layer_id in selected]

