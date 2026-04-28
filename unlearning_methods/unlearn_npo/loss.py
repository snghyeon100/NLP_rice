"""NPO loss for forget/retain paired batches."""

import torch
import torch.nn.functional as F
from torch import nn


def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def compute_batch_nll(model, inputs):
    device = _model_device(model)
    input_ids, labels, attention_mask = _batch_to_device(inputs, device)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if ref_model is None:
        raise ValueError("NPO requires an oracle/reference model.")
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    device = _model_device(model)
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss.to(device))

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss.to(device))

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


def compute_npo_loss(model, oracle_model, inputs, beta=1.0, gamma=1.0, alpha=1.0):
    forget_inputs, retain_inputs = inputs
    forget_loss, forget_outputs = compute_dpo_loss(
        model=model,
        ref_model=oracle_model,
        win_inputs=None,
        lose_inputs=forget_inputs,
        beta=beta,
    )
    retain_input_ids, retain_labels, retain_attention_mask = _batch_to_device(retain_inputs, _model_device(model))
    retain_outputs = model(retain_input_ids, labels=retain_labels, attention_mask=retain_attention_mask)
    return gamma * forget_loss + alpha * retain_outputs.loss, forget_outputs
