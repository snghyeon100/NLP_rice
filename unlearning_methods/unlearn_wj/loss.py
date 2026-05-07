"""Likelihood-based WJ unlearning losses."""

import torch
import torch.nn.functional as F
from torch import nn


def model_device(model):
    return next(model.parameters()).device


def batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def forward_outputs(model, inputs):
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    return model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def sequence_nll(model, inputs):
    """Return per-example answer-token NLL sums and token counts."""
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    token_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")(
        logits.transpose(-1, -2),
        shifted_labels,
    )
    mask = shifted_labels.ne(-100)
    nll_sum = (token_loss * mask).sum(dim=-1)
    token_count = mask.sum(dim=-1).clamp_min(1)
    return nll_sum, token_count, outputs


def npo_forget_loss(model, ref_model, inputs, beta):
    """Forget-only NPO objective.

    Minimizing this loss increases the model NLL on the forget answer relative
    to the frozen reference model.
    """
    model_nll, token_count, outputs = sequence_nll(model, inputs)
    with torch.no_grad():
        ref_nll, _, _ = sequence_nll(ref_model, inputs)
    log_ratio = -(model_nll - ref_nll.to(model_device(model)))
    loss = -2.0 / beta * F.logsigmoid(-beta * log_ratio).mean()
    return loss, outputs, {
        "forget_model_nll": model_nll.detach().mean(),
        "forget_ref_nll": ref_nll.detach().to(model_device(model)).mean(),
        "forget_model_avg_nll": (model_nll / token_count).detach().mean(),
    }


def negative_ce_forget_loss(model, inputs):
    outputs = forward_outputs(model, inputs)
    return -outputs.loss, outputs, {"forget_ce": outputs.loss.detach()}


def masked_kl(model, ref_model, inputs):
    """KL(ref || model) averaged over answer tokens only."""
    input_ids, labels, attention_mask = batch_to_device(inputs, model_device(model))
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    shifted_labels = labels[..., 1:]
    mask = shifted_labels.ne(-100)
    if mask.sum().item() == 0:
        return outputs.logits.sum() * 0.0

    current_logits = outputs.logits[..., :-1, :][mask].contiguous()
    current_log_probs = F.log_softmax(current_logits, dim=-1)

    with torch.no_grad():
        ref_input_ids, _, ref_attention_mask = batch_to_device(inputs, model_device(ref_model))
        ref_outputs = ref_model(input_ids=ref_input_ids, attention_mask=ref_attention_mask)
        ref_mask = mask.to(model_device(ref_model))
        ref_logits = ref_outputs.logits[..., :-1, :][ref_mask].contiguous()
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        ref_log_probs = ref_log_probs.to(model_device(model))

    per_token_kl = F.kl_div(
        current_log_probs,
        ref_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    return per_token_kl.mean()


def _weighted_add(total, weight, value):
    if weight == 0:
        return total
    return total + float(weight) * value


def compute_wj_loss(model, ref_model, batch, cfg):
    weights = cfg.loss_weights
    objective = str(cfg.forget_objective).lower()
    logs = {}

    if objective == "npo":
        forget_loss, outputs, forget_logs = npo_forget_loss(
            model,
            ref_model,
            batch["forget_source"],
            beta=float(cfg.npo_beta),
        )
        logs.update(forget_logs)
    elif objective in {"neg_ce", "grad_diff"}:
        forget_loss, outputs, forget_logs = negative_ce_forget_loss(model, batch["forget_source"])
        logs.update(forget_logs)
    else:
        raise ValueError(f"Unsupported forget_objective: {cfg.forget_objective}")

    total = float(weights.forget) * forget_loss
    logs["loss_forget"] = forget_loss.detach()

    if float(weights.retain_source_ce) != 0:
        retain_source_outputs = forward_outputs(model, batch["retain_source"])
        total = _weighted_add(total, weights.retain_source_ce, retain_source_outputs.loss)
        logs["loss_retain_source_ce"] = retain_source_outputs.loss.detach()

    if float(weights.retain_multi_ce) != 0:
        retain_multi_outputs = forward_outputs(model, batch["retain_multi"])
        total = _weighted_add(total, weights.retain_multi_ce, retain_multi_outputs.loss)
        logs["loss_retain_multi_ce"] = retain_multi_outputs.loss.detach()

    if float(weights.utility_multi_ce) != 0:
        utility_multi_outputs = forward_outputs(model, batch["utility_multi"])
        total = _weighted_add(total, weights.utility_multi_ce, utility_multi_outputs.loss)
        logs["loss_utility_multi_ce"] = utility_multi_outputs.loss.detach()

    if float(weights.retain_source_kl) != 0:
        retain_source_kl = masked_kl(model, ref_model, batch["retain_source"])
        total = _weighted_add(total, weights.retain_source_kl, retain_source_kl)
        logs["loss_retain_source_kl"] = retain_source_kl.detach()

    if float(weights.retain_multi_kl) != 0:
        retain_multi_kl = masked_kl(model, ref_model, batch["retain_multi"])
        total = _weighted_add(total, weights.retain_multi_kl, retain_multi_kl)
        logs["loss_retain_multi_kl"] = retain_multi_kl.detach()

    if float(weights.utility_multi_kl) != 0:
        utility_multi_kl = masked_kl(model, ref_model, batch["utility_multi"])
        total = _weighted_add(total, weights.utility_multi_kl, utility_multi_kl)
        logs["loss_utility_multi_kl"] = utility_multi_kl.detach()

    logs["loss_total"] = total.detach()
    return total, outputs, {key: float(value.detach().float().cpu()) for key, value in logs.items()}
