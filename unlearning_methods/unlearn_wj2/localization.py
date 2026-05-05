"""Probe-decodability localization for representation erasure."""

import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from unlearning_methods.unlearn_wj2.probes import (
    answer_bank_embeddings,
    batch_to_device,
    contrastive_probe_loss_bank,
    model_device,
    question_hidden_states,
)


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


def load_selected_layers(path):
    with open(path, "r") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [int(item) for item in payload]
    return [int(item) for item in payload["selected_layers"]]


def trainable_layer_ids(model):
    layer_ids = set()
    for name, _ in model.named_parameters():
        layer_id = extract_layer_id(name)
        if layer_id is not None:
            layer_ids.add(layer_id)
    return sorted(layer_ids)


def _matches_target_module(name, target_modules):
    if not target_modules:
        return True
    return any(
        f".{module}." in name or name.endswith(f".{module}.weight") or name.endswith(f".{module}.bias")
        for module in target_modules
    )


def _zero_scores(layer_ids):
    return {str(layer_id): 0.0 for layer_id in layer_ids}


def _add_scores(total, scores):
    for key, value in scores.items():
        total[key] = total.get(key, 0.0) + float(value)


def _zscore(values, eps=1e-12):
    keys = list(values.keys())
    tensor = torch.tensor([float(values[key]) for key in keys], dtype=torch.float32)
    if tensor.numel() <= 1:
        return {key: 0.0 for key in keys}
    std = tensor.std(unbiased=False).clamp_min(float(eps))
    normalized = (tensor - tensor.mean()) / std
    return {key: float(value) for key, value in zip(keys, normalized)}


def _forward_loss(model, inputs):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    return model(input_ids=input_ids, labels=labels, attention_mask=attention_mask).loss


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


@torch.no_grad()
def _probe_decodability_scores(model, probes, forget_question, sample_indices, bank_cache, layer_ids, cfg):
    bank_embeddings, candidate_mask, owner_indices, bank_positive_mask = bank_cache
    hidden_by_layer = question_hidden_states(
        model,
        forget_question,
        layer_ids,
        pooling=cfg.get("probe", {}).get("pooling", "last"),
    )
    temperature = float(cfg.get("probe", {}).get("temperature", 0.07))
    scores = {}
    for layer_id, hidden in hidden_by_layer.items():
        if str(layer_id) not in probes.probes:
            continue
        projected = probes(layer_id, hidden)
        loss, metrics = contrastive_probe_loss_bank(
            projected,
            bank_embeddings,
            candidate_mask,
            owner_indices,
            bank_positive_mask,
            sample_indices,
            temperature=temperature,
        )
        # Higher score means the layer encodes the target answer more recoverably.
        scores[str(layer_id)] = float(metrics["probe_positive_prob"].cpu()) - float(loss.cpu())
    return scores


@torch.no_grad()
def _share_scores(model, en_question, tgt_question, layer_ids, pooling="mean"):
    en_hidden = question_hidden_states(model, en_question, layer_ids, pooling=pooling)
    tgt_hidden = question_hidden_states(model, tgt_question, layer_ids, pooling=pooling)
    scores = {}
    for layer_id in layer_ids:
        if layer_id not in en_hidden or layer_id not in tgt_hidden:
            continue
        scores[str(layer_id)] = float(F.cosine_similarity(en_hidden[layer_id].float(), tgt_hidden[layer_id].float(), dim=-1).mean().cpu())
    return scores


def _final_scores(decodability_scores, share_scores, util_scores, cfg):
    loc_cfg = cfg.get("localization", {})
    eps = float(loc_cfg.get("score_eps", 1e-12))
    z_dec = _zscore(decodability_scores, eps=eps)
    z_share = _zscore(share_scores, eps=eps)
    z_util = _zscore(util_scores, eps=eps)
    dec_w = float(loc_cfg.get("decodability_weight", 1.0))
    share_w = float(loc_cfg.get("share_weight", 1.0))
    util_w = float(loc_cfg.get("utility_weight", 1.0))
    final = {}
    for key in decodability_scores:
        final[key] = float(dec_w * z_dec[key] + share_w * z_share.get(key, 0.0) - util_w * z_util.get(key, 0.0))
    return final, z_dec, z_share, z_util


def _select_layers(final_scores, cfg):
    loc_cfg = cfg.get("localization", {})
    ranked = sorted(((int(key), value) for key, value in final_scores.items()), key=lambda item: item[1], reverse=True)
    threshold = loc_cfg.get("select_score_threshold", None)
    if threshold is not None:
        return [layer_id for layer_id, score in ranked if score >= float(threshold)]
    top_k = int(loc_cfg.get("select_top_k_layers", len(ranked)))
    return [layer_id for layer_id, _ in ranked[:top_k]]


def _set_model_requires_grad(model, requires_grad):
    previous = {}
    for name, param in model.named_parameters():
        previous[name] = param.requires_grad
        param.requires_grad = requires_grad
    return previous


def _restore_requires_grad(model, previous):
    for name, param in model.named_parameters():
        if name in previous:
            param.requires_grad = previous[name]


def run_localization(model, probes, dataloader, answer_bank, cfg, save_dir):
    loc_cfg = cfg.get("localization", {})
    layer_ids = [int(layer_id) for layer_id in probes.layer_ids]
    if not layer_ids:
        layer_ids = trainable_layer_ids(model)
    if not layer_ids:
        raise ValueError("No transformer layer ids were found.")

    max_batches = int(loc_cfg.get("num_batches", 8))
    target_modules = list(cfg.get("lora_target_modules", []))
    dec_total = _zero_scores(layer_ids)
    share_total = _zero_scores(layer_ids)
    util_total = _zero_scores(layer_ids)
    used_batches = 0

    previous_requires_grad = _set_model_requires_grad(model, True)
    model.train()
    probes.eval()
    bank_cache = answer_bank_embeddings(
        model,
        answer_bank,
        chunk_size=int(cfg.get("probe", {}).get("bank_embedding_batch_size", 256)),
    )
    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            forget_question, sample_indices, retain_inputs, utility_inputs, parallel_en_inputs, parallel_tgt_inputs = batch

            dec_scores = _probe_decodability_scores(
                model,
                probes,
                forget_question,
                sample_indices,
                bank_cache,
                layer_ids,
                cfg,
            )
            util_loss = _forward_loss(model, retain_inputs) + _forward_loss(model, utility_inputs)
            util_scores = _score_layer_gradients(
                model,
                util_loss,
                layer_ids,
                target_modules=target_modules,
                eps=loc_cfg.get("score_eps", 1e-12),
            )
            share_type = loc_cfg.get("share_score_type", "hidden")
            if share_type == "hidden":
                share_scores = _share_scores(
                    model,
                    parallel_en_inputs,
                    parallel_tgt_inputs,
                    layer_ids,
                    pooling=loc_cfg.get("pooling", "mean"),
                )
            else:
                share_scores = _zero_scores(layer_ids)

            _add_scores(dec_total, dec_scores)
            _add_scores(share_total, share_scores)
            _add_scores(util_total, util_scores)
            used_batches += 1
    finally:
        _restore_requires_grad(model, previous_requires_grad)
        model.zero_grad(set_to_none=True)

    if used_batches == 0:
        raise ValueError("Localization dataloader produced no batches.")

    dec_avg = {key: value / used_batches for key, value in dec_total.items()}
    share_avg = {key: value / used_batches for key, value in share_total.items()}
    util_avg = {key: value / used_batches for key, value in util_total.items()}
    final, z_dec, z_share, z_util = _final_scores(dec_avg, share_avg, util_avg, cfg)
    selected_layers = _select_layers(final, cfg)

    output_dir = Path(save_dir) / "localization"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_layers": selected_layers,
        "scores": final,
        "decodability_scores": dec_avg,
        "share_scores": share_avg,
        "utility_scores": util_avg,
        "z_decodability": z_dec,
        "z_share": z_share,
        "z_utility": z_util,
        "num_batches": used_batches,
    }
    with open(output_dir / "selected_layers.json", "w") as f:
        json.dump(payload, f, indent=2)
    return selected_layers, payload
