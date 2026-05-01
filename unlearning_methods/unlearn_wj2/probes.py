"""Answer-recovery probes for representation erasure."""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def model_device(model):
    return next(model.parameters()).device


def batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _pool_hidden(hidden, attention_mask, pooling="last"):
    mask = attention_mask.to(hidden.device).float()
    if pooling == "last":
        lengths = mask.sum(dim=1).long().clamp_min(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), lengths]
    masked_hidden = hidden * mask.unsqueeze(-1)
    return masked_hidden.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)


class LowRankAnswerProbe(nn.Module):
    def __init__(self, hidden_size, rank=64):
        super().__init__()
        rank = int(rank)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden):
        hidden = hidden.to(self.down.weight.dtype)
        return self.norm(self.up(torch.tanh(self.down(hidden))))


class LayerProbeBank(nn.Module):
    def __init__(self, layer_ids, hidden_size, rank=64):
        super().__init__()
        self.layer_ids = [int(layer_id) for layer_id in layer_ids]
        self.probes = nn.ModuleDict({
            str(layer_id): LowRankAnswerProbe(hidden_size, rank=rank)
            for layer_id in self.layer_ids
        })

    def forward(self, layer_id, hidden):
        return self.probes[str(int(layer_id))](hidden)


def resolve_candidate_layers(model, candidate_layers):
    if candidate_layers is None or str(candidate_layers).lower() == "auto":
        num_layers = getattr(model.config, "num_hidden_layers", None)
        if num_layers is None:
            num_layers = getattr(model.config, "n_layer", None)
        if num_layers is None:
            raise ValueError("Could not infer num_hidden_layers for probe candidate_layers=auto.")
        return list(range(int(num_layers)))
    return [int(layer_id) for layer_id in candidate_layers]


@torch.no_grad()
def answer_embeddings(model, answer_candidates):
    answer_input_ids, answer_attention_mask, candidate_mask, positive_mask = answer_candidates
    device = model_device(model)
    answer_input_ids = answer_input_ids.to(device)
    answer_attention_mask = answer_attention_mask.to(device)
    candidate_mask = candidate_mask.to(device)
    positive_mask = positive_mask.to(device)

    batch_size, num_candidates, seq_len = answer_input_ids.shape
    flat_ids = answer_input_ids.view(batch_size * num_candidates, seq_len)
    flat_mask = answer_attention_mask.view(batch_size * num_candidates, seq_len).float()
    embeddings = model.get_input_embeddings()(flat_ids)
    pooled = (embeddings * flat_mask.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / flat_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = pooled.view(batch_size, num_candidates, -1)
    return pooled.detach(), candidate_mask, positive_mask


def question_hidden_states(model, question_inputs, layer_ids, pooling="last"):
    input_ids, attention_mask = batch_to_device(question_inputs, model_device(model))
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_by_layer = {}
    for layer_id in layer_ids:
        hidden_index = int(layer_id) + 1
        if hidden_index >= len(outputs.hidden_states):
            continue
        pooled = _pool_hidden(outputs.hidden_states[hidden_index], attention_mask, pooling=pooling)
        hidden_by_layer[int(layer_id)] = pooled
    return hidden_by_layer


def _score_matrix(projected_hidden, candidate_embeddings, temperature):
    projected = F.normalize(projected_hidden.float(), dim=-1)
    candidates = F.normalize(candidate_embeddings.float(), dim=-1)
    flat_candidates = candidates.view(-1, candidates.shape[-1])
    return projected @ flat_candidates.transpose(0, 1) / float(temperature)


def contrastive_probe_loss(projected_hidden, candidate_embeddings, candidate_mask, positive_mask, temperature=0.07):
    batch_size, num_candidates, _ = candidate_embeddings.shape
    scores = _score_matrix(projected_hidden, candidate_embeddings, temperature)
    flat_valid = candidate_mask.reshape(-1).bool()
    positive_matrix = torch.zeros(
        batch_size,
        batch_size * num_candidates,
        device=scores.device,
        dtype=torch.bool,
    )
    for row_idx in range(batch_size):
        start = row_idx * num_candidates
        stop = start + num_candidates
        positive_matrix[row_idx, start:stop] = positive_mask[row_idx].bool()

    scores = scores.masked_fill(~flat_valid.unsqueeze(0), torch.finfo(scores.dtype).min)
    positive_scores = scores.masked_fill(~positive_matrix, torch.finfo(scores.dtype).min)
    log_positive = torch.logsumexp(positive_scores, dim=-1)
    log_denominator = torch.logsumexp(scores, dim=-1)
    loss = -(log_positive - log_denominator).mean()

    with torch.no_grad():
        preds = scores.argmax(dim=-1)
        accuracy = positive_matrix.gather(1, preds.unsqueeze(1)).float().mean()
        positive_prob = torch.exp(log_positive - log_denominator).mean()
    return loss, {"probe_acc": accuracy, "probe_positive_prob": positive_prob}


def uniform_probe_loss(projected_hidden, candidate_embeddings, candidate_mask, temperature=0.07):
    scores = _score_matrix(projected_hidden, candidate_embeddings, temperature)
    valid_scores = scores[:, candidate_mask.reshape(-1).bool()]
    log_probs = F.log_softmax(valid_scores, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
    loss = -log_probs.mean()
    return loss, {"probe_entropy": entropy}


def probe_losses_for_layers(probes, hidden_by_layer, candidate_embeddings, candidate_mask, positive_mask, temperature):
    losses = {}
    metrics = {}
    for layer_id, hidden in hidden_by_layer.items():
        if str(layer_id) not in probes.probes:
            continue
        projected = probes(layer_id, hidden)
        loss, layer_metrics = contrastive_probe_loss(
            projected,
            candidate_embeddings,
            candidate_mask,
            positive_mask,
            temperature=temperature,
        )
        losses[int(layer_id)] = loss
        metrics[int(layer_id)] = layer_metrics
    return losses, metrics


def save_probes(probes, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": probes.state_dict(),
            "layer_ids": probes.layer_ids,
        },
        path,
    )


def load_probes(path, hidden_size, rank=64, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    probes = LayerProbeBank(payload["layer_ids"], hidden_size, rank=rank)
    probes.load_state_dict(payload["state_dict"])
    return probes


def train_answer_probes(model, probes, dataloader, cfg, save_dir):
    probe_cfg = cfg.get("probe", {})
    temperature = float(probe_cfg.get("temperature", 0.07))
    pooling = probe_cfg.get("pooling", "last")
    num_epochs = int(probe_cfg.get("num_epochs", 20))
    max_batches = probe_cfg.get("max_batches", None)
    max_batches = int(max_batches) if max_batches is not None else None
    log_every = max(1, int(probe_cfg.get("log_every", 10)))

    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    probes.train()
    probes.to(model_device(model))

    optimizer = torch.optim.AdamW(
        probes.parameters(),
        lr=float(probe_cfg.get("lr", 1e-3)),
        weight_decay=float(probe_cfg.get("weight_decay", 0.01)),
    )

    logs = []
    global_step = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        running_prob = 0.0
        running_count = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"probe epoch {epoch + 1}/{num_epochs}")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            forget_question, answer_candidates, *_ = batch
            with torch.no_grad():
                candidates, candidate_mask, positive_mask = answer_embeddings(model, answer_candidates)
                hidden_by_layer = question_hidden_states(model, forget_question, probes.layer_ids, pooling=pooling)
                hidden_by_layer = {layer_id: hidden.detach() for layer_id, hidden in hidden_by_layer.items()}

            losses, metrics = probe_losses_for_layers(
                probes,
                hidden_by_layer,
                candidates,
                candidate_mask,
                positive_mask,
                temperature=temperature,
            )
            if not losses:
                raise ValueError("No probe losses were produced. Check candidate layer ids.")
            loss = torch.stack(list(losses.values())).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            layer_acc = torch.stack([item["probe_acc"] for item in metrics.values()]).mean()
            layer_prob = torch.stack([item["probe_positive_prob"] for item in metrics.values()]).mean()
            running_loss += float(loss.detach().cpu())
            running_acc += float(layer_acc.detach().cpu())
            running_prob += float(layer_prob.detach().cpu())
            running_count += 1
            global_step += 1

            if global_step % log_every == 0:
                record = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "probe_loss": running_loss / running_count,
                    "probe_acc": running_acc / running_count,
                    "probe_positive_prob": running_prob / running_count,
                }
                print(record)
                logs.append(record)
                running_loss = running_acc = running_prob = 0.0
                running_count = 0

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "probe_training_log.jsonl", "w") as f:
        for record in logs:
            f.write(json.dumps(record) + "\n")
    save_path = probe_cfg.get("save_path", None) or str(save_dir / "answer_probes.pt")
    save_probes(probes, save_path)
    return probes


@torch.no_grad()
def evaluate_answer_probes(model, probes, dataloader, cfg, max_batches=None):
    probes.eval()
    probes.to(model_device(model))
    pooling = cfg.get("probe", {}).get("pooling", "last")
    temperature = float(cfg.get("probe", {}).get("temperature", 0.07))
    max_batches = int(max_batches) if max_batches is not None else None
    totals = {
        int(layer_id): {"loss": 0.0, "acc": 0.0, "positive_prob": 0.0, "count": 0}
        for layer_id in probes.layer_ids
    }

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        forget_question, answer_candidates, *_ = batch
        candidates, candidate_mask, positive_mask = answer_embeddings(model, answer_candidates)
        hidden_by_layer = question_hidden_states(model, forget_question, probes.layer_ids, pooling=pooling)
        losses, metrics = probe_losses_for_layers(
            probes,
            hidden_by_layer,
            candidates,
            candidate_mask,
            positive_mask,
            temperature=temperature,
        )
        for layer_id, loss in losses.items():
            totals[layer_id]["loss"] += float(loss.cpu())
            totals[layer_id]["acc"] += float(metrics[layer_id]["probe_acc"].cpu())
            totals[layer_id]["positive_prob"] += float(metrics[layer_id]["probe_positive_prob"].cpu())
            totals[layer_id]["count"] += 1

    output = {}
    for layer_id, values in totals.items():
        count = max(1, values["count"])
        output[str(layer_id)] = {
            "probe_loss": values["loss"] / count,
            "probe_acc": values["acc"] / count,
            "probe_positive_prob": values["positive_prob"] / count,
        }
    return output
