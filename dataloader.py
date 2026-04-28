"""Custom Hugging Face Trainer for TOFU QA fine-tuning."""

import torch
from transformers import Trainer


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids, labels, attention_mask = inputs
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            if prediction_loss_only:
                return (loss, None, None)
            logits = outputs.logits
        return (loss, logits, labels)
