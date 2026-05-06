"""Gradient-difference loss (no KL term)."""


def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _forward(model, inputs):
    input_ids, labels, attention_mask = _batch_to_device(inputs, _model_device(model))
    return model(input_ids, labels=labels, attention_mask=attention_mask)


def compute_grad_diff_loss(model, inputs):
    """grad_diff: -forget_loss + retain_loss (oracle model 불필요)."""
    forget_inputs, retain_inputs = inputs

    forget_outputs = _forward(model, forget_inputs)
    retain_outputs = _forward(model, retain_inputs)

    loss = -forget_outputs.loss + retain_outputs.loss
    return loss, forget_outputs
