"""Losses for probe-based representation erasure."""

import torch
import torch.nn.functional as F

from unlearning_methods.unlearn_wj2.probes import (
    _pool_hidden,
    answer_embeddings,
    batch_to_device,
    model_device,
    question_hidden_states,
    uniform_probe_loss,
)


def forward_outputs(model, inputs, output_hidden_states=False):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    return model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        output_hidden_states=output_hidden_states,
        use_cache=False,
    )


def reference_kl(model, reference_model, inputs):
    if reference_model is None:
        return torch.zeros((), device=model_device(model))

    input_ids, _, attention_mask = batch_to_device(inputs, model_device(model))
    current_outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    with torch.no_grad():
        ref_input_ids, _, ref_attention_mask = batch_to_device(inputs, model_device(reference_model))
        ref_outputs = reference_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask, use_cache=False)

    current_log_probs = F.log_softmax(current_outputs.logits.float(), dim=-1)
    ref_log_probs = F.log_softmax(ref_outputs.logits.float().to(current_log_probs.device), dim=-1)
    token_kl = F.kl_div(current_log_probs, ref_log_probs, reduction="none", log_target=True).sum(dim=-1)
    mask = attention_mask.to(token_kl.device).float()
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)


def hidden_preservation_loss(model, reference_model, inputs, layer_ids, pooling="mean"):
    if reference_model is None or not layer_ids:
        return torch.zeros((), device=model_device(model))

    input_ids, _, attention_mask = batch_to_device(inputs, model_device(model))
    current_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    with torch.no_grad():
        ref_input_ids, _, ref_attention_mask = batch_to_device(inputs, model_device(reference_model))
        ref_outputs = reference_model(
            input_ids=ref_input_ids,
            attention_mask=ref_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    losses = []
    for layer_id in layer_ids:
        hidden_index = int(layer_id) + 1
        if hidden_index >= len(current_outputs.hidden_states) or hidden_index >= len(ref_outputs.hidden_states):
            continue
        current_pool = _pool_hidden(current_outputs.hidden_states[hidden_index], attention_mask, pooling=pooling)
        ref_pool = _pool_hidden(
            ref_outputs.hidden_states[hidden_index].to(current_pool.device),
            ref_attention_mask.to(current_pool.device),
            pooling=pooling,
        )
        losses.append(F.mse_loss(current_pool.float(), ref_pool.float()))
    if not losses:
        return torch.zeros((), device=model_device(model))
    return torch.stack(losses).mean()


def forget_hidden_norm_loss(model, reference_model, forget_question, layer_ids, pooling="last"):
    if reference_model is None or not layer_ids:
        return torch.zeros((), device=model_device(model))

    current_hidden = question_hidden_states(model, forget_question, layer_ids, pooling=pooling)
    with torch.no_grad():
        ref_question = batch_to_device(forget_question, model_device(reference_model))
        ref_hidden = question_hidden_states(reference_model, ref_question, layer_ids, pooling=pooling)

    losses = []
    for layer_id in layer_ids:
        if layer_id not in current_hidden or layer_id not in ref_hidden:
            continue
        current_norm = current_hidden[layer_id].float().norm(dim=-1)
        ref_norm = ref_hidden[layer_id].to(current_norm.device).float().norm(dim=-1)
        losses.append(F.mse_loss(current_norm, ref_norm))
    if not losses:
        return torch.zeros((), device=model_device(model))
    return torch.stack(losses).mean()


def compute_probe_erasure_loss(model, probes, forget_question, answer_candidates, layer_ids, temperature, pooling):
    candidates, candidate_mask, _ = answer_embeddings(model, answer_candidates)
    hidden_by_layer = question_hidden_states(model, forget_question, layer_ids, pooling=pooling)
    losses = []
    entropies = []
    for layer_id, hidden in hidden_by_layer.items():
        if str(layer_id) not in probes.probes:
            continue
        projected = probes(layer_id, hidden)
        loss, metrics = uniform_probe_loss(
            projected,
            candidates,
            candidate_mask,
            temperature=temperature,
        )
        losses.append(loss)
        entropies.append(metrics["probe_entropy"])
    if not losses:
        raise ValueError("No probe erasure losses were produced. Check selected layers and probe layers.")
    return torch.stack(losses).mean(), torch.stack(entropies).mean()


def compute_rep_erasure_loss(model, reference_model, probes, inputs, cfg, selected_layers):
    forget_question, answer_candidates, retain_inputs, utility_inputs, _, _ = inputs
    erase_cfg = cfg.get("erase", {})
    probe_cfg = cfg.get("probe", {})
    selected_layers = [int(layer_id) for layer_id in selected_layers]

    erase_loss, probe_entropy = compute_probe_erasure_loss(
        model,
        probes,
        forget_question,
        answer_candidates,
        selected_layers,
        temperature=float(probe_cfg.get("temperature", 0.07)),
        pooling=probe_cfg.get("pooling", "last"),
    )
    retain_outputs = forward_outputs(model, retain_inputs)
    utility_outputs = forward_outputs(model, utility_inputs)
    retain_kl = reference_kl(model, reference_model, retain_inputs)
    utility_kl = reference_kl(model, reference_model, utility_inputs)
    retain_hidden = hidden_preservation_loss(
        model,
        reference_model,
        retain_inputs,
        selected_layers,
        pooling=erase_cfg.get("hidden_pooling", "mean"),
    )
    utility_hidden = hidden_preservation_loss(
        model,
        reference_model,
        utility_inputs,
        selected_layers,
        pooling=erase_cfg.get("hidden_pooling", "mean"),
    )
    hidden_preserve = 0.5 * (retain_hidden + utility_hidden)
    norm_loss = forget_hidden_norm_loss(
        model,
        reference_model,
        forget_question,
        selected_layers,
        pooling=probe_cfg.get("pooling", "last"),
    )

    loss = (
        float(erase_cfg.get("erase_weight", 1.0)) * erase_loss
        + float(erase_cfg.get("retain_ce_weight", 0.5)) * retain_outputs.loss
        + float(erase_cfg.get("utility_ce_weight", 0.5)) * utility_outputs.loss
        + float(erase_cfg.get("retain_kl_weight", 0.1)) * retain_kl
        + float(erase_cfg.get("utility_kl_weight", 0.1)) * utility_kl
        + float(erase_cfg.get("hidden_preserve_weight", 0.1)) * hidden_preserve
        + float(erase_cfg.get("forget_hidden_norm_weight", 0.01)) * norm_loss
    )
    logs = {
        "objective_loss": loss.detach(),
        "erase_loss": erase_loss.detach(),
        "probe_entropy": probe_entropy.detach(),
        "retain_loss": retain_outputs.loss.detach(),
        "utility_loss": utility_outputs.loss.detach(),
        "retain_kl": retain_kl.detach(),
        "utility_kl": utility_kl.detach(),
        "hidden_preserve_loss": hidden_preserve.detach(),
        "forget_hidden_norm_loss": norm_loss.detach(),
    }
    return loss, retain_outputs, logs
