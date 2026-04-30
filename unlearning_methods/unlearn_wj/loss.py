"""Loss for subspace-aware zero-shot cross-lingual unlearning."""

import torch
import torch.nn.functional as F


def model_device(model):
    return next(model.parameters()).device


def batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def forward_outputs(model, inputs):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    return model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def reference_kl(model, reference_model, inputs):
    if reference_model is None:
        return torch.zeros((), device=model_device(model))

    input_ids, _, attention_mask = batch_to_device(inputs, model_device(model))
    current_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    with torch.no_grad():
        ref_input_ids, _, ref_attention_mask = batch_to_device(inputs, model_device(reference_model))
        ref_outputs = reference_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask)

    current_log_probs = F.log_softmax(current_outputs.logits.float(), dim=-1)
    ref_log_probs = F.log_softmax(ref_outputs.logits.float().to(current_log_probs.device), dim=-1)
    token_kl = F.kl_div(current_log_probs, ref_log_probs, reduction="none", log_target=True).sum(dim=-1)
    mask = attention_mask.to(token_kl.device).float()
    return (token_kl * mask).sum() / mask.sum().clamp_min(1.0)


def compute_subspace_xlingual_loss(
    model,
    reference_model,
    inputs,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
):
    forget_inputs, retain_inputs, utility_inputs, _, _ = inputs

    forget_outputs = forward_outputs(model, forget_inputs)
    retain_outputs = forward_outputs(model, retain_inputs)
    utility_outputs = forward_outputs(model, utility_inputs)
    kl_loss = reference_kl(model, reference_model, utility_inputs)

    loss = -forget_outputs.loss + alpha * retain_outputs.loss + beta * utility_outputs.loss + gamma * kl_loss
    logs = {
        "loss": loss.detach(),
        "forget_loss": forget_outputs.loss.detach(),
        "retain_loss": retain_outputs.loss.detach(),
        "utility_loss": utility_outputs.loss.detach(),
        "kl_loss": kl_loss.detach(),
    }
    return loss, forget_outputs, logs

