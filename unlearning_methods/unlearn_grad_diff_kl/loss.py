"""Gradient-difference plus KL loss."""

import torch
import torch.nn.functional as F
from torch import nn


def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _forward(model, inputs):
    input_ids, labels, attention_mask = _batch_to_device(inputs, _model_device(model))
    return model(input_ids, labels=labels, attention_mask=attention_mask)


def _reference_kl(model, oracle_model, inputs):
    if oracle_model is None:
        raise ValueError("grad_diff_KL requires an oracle/reference model.")

    current_outputs = _forward(model, inputs)
    current_probs = F.log_softmax(current_outputs.logits, dim=-1)
    current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

    with torch.no_grad():
        oracle_outputs = _forward(oracle_model, inputs)
    oracle_probs = F.log_softmax(oracle_outputs.logits, dim=-1)
    oracle_probs = oracle_probs.view(-1, oracle_outputs.logits.shape[-1])

    normal_loss = nn.functional.kl_div(
        current_probs,
        oracle_probs.to(_model_device(model)),
        reduction="batchmean",
        log_target=True,
    )
    return normal_loss


def compute_grad_diff_kl_loss(model, oracle_model, inputs):
    forget_inputs, retain_inputs, normal_inputs = inputs

    forget_outputs = _forward(model, forget_inputs)
    retain_outputs = _forward(model, retain_inputs)
    normal_loss = _reference_kl(model, oracle_model, normal_inputs)

    loss = -forget_outputs.loss + retain_outputs.loss + normal_loss
    return loss, forget_outputs
