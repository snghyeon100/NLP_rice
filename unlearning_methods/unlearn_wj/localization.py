"""Layer selection utilities for WJ unlearning."""

import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unlearning_methods.unlearn_wj.dataloader import ParallelAnchorDataset, parallel_anchor_collator


LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def get_num_layers(model):
    if getattr(model.config, "num_hidden_layers", None) is not None:
        return int(model.config.num_hidden_layers)
    for attr in ("model", "language_model", "transformer"):
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "layers"):
            return len(module.layers)
    raise ValueError("Could not infer number of transformer layers.")


def extract_layer_idx(module_name):
    match = LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def layer_window(num_layers, min_layer=None, max_layer=None):
    start = 0 if min_layer is None else max(0, int(min_layer))
    end = num_layers - 1 if max_layer is None else min(num_layers - 1, int(max_layer))
    if start > end:
        raise ValueError(f"Invalid layer window: min_layer={min_layer}, max_layer={max_layer}.")
    return list(range(start, end + 1))


def select_middle_layers(num_layers, top_k, min_layer=None, max_layer=None):
    candidates = layer_window(num_layers, min_layer, max_layer)
    if top_k >= len(candidates):
        return candidates
    center = (candidates[0] + candidates[-1]) / 2.0
    selected = sorted(candidates, key=lambda idx: (abs(idx - center), idx))[: int(top_k)]
    return sorted(selected)


def layers_with_target_modules(model, target_leaves, min_layer=None, max_layer=None):
    target_leaves = set(str(name) for name in target_leaves)
    allowed = set(layer_window(get_num_layers(model), min_layer=min_layer, max_layer=max_layer))
    layers = set()
    counts = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf not in target_leaves or leaf == "lm_head":
            continue
        layer_idx = extract_layer_idx(name)
        if layer_idx is None or layer_idx not in allowed:
            continue
        layers.add(layer_idx)
        counts[layer_idx] = counts.get(layer_idx, 0) + 1
    return sorted(layers), counts


def select_middle_target_layers(model, target_leaves, top_k, min_layer=None, max_layer=None):
    candidates, counts = layers_with_target_modules(
        model,
        target_leaves,
        min_layer=min_layer,
        max_layer=max_layer,
    )
    if not candidates:
        raise ValueError(f"No layers contain target modules: {list(target_leaves)}")
    if top_k >= len(candidates):
        return candidates, counts
    center = (candidates[0] + candidates[-1]) / 2.0
    selected = sorted(candidates, key=lambda idx: (abs(idx - center), idx))[: int(top_k)]
    return sorted(selected), counts


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


def _mean_pool(hidden_state, attention_mask):
    mask = attention_mask.to(hidden_state.dtype).unsqueeze(-1)
    return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


@torch.no_grad()
def compute_hidden_alignment_scores(model, cfg, tokenizer, project_root):
    device = next(model.parameters()).device
    dataset = ParallelAnchorDataset(cfg, tokenizer=tokenizer, project_root=project_root)
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.layer_selection.hidden_alignment_batch_size),
        shuffle=True,
        collate_fn=parallel_anchor_collator,
    )

    num_layers = get_num_layers(model)
    score_sum = torch.zeros(num_layers, dtype=torch.float64)
    score_count = torch.zeros(num_layers, dtype=torch.float64)

    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= int(cfg.layer_selection.hidden_alignment_batches):
            break

        source_input_ids, _, source_attention_mask = _batch_to_device(batch["source"], device)
        target_input_ids, _, target_attention_mask = _batch_to_device(batch["target"], device)

        source_outputs = model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        target_outputs = model(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        for layer_idx in range(num_layers):
            source_pool = _mean_pool(source_outputs.hidden_states[layer_idx + 1], source_attention_mask)
            target_pool = _mean_pool(target_outputs.hidden_states[layer_idx + 1], target_attention_mask)
            cosine = F.cosine_similarity(source_pool.float(), target_pool.float(), dim=-1).mean().item()
            score_sum[layer_idx] += cosine
            score_count[layer_idx] += 1

    if score_count.sum().item() == 0:
        raise ValueError("No hidden-alignment batches were processed.")

    scores = {}
    for layer_idx in range(num_layers):
        count = max(1.0, score_count[layer_idx].item())
        scores[layer_idx] = {"hidden_alignment": score_sum[layer_idx].item() / count}
    return scores


def select_layers(model, cfg, tokenizer, project_root, save_dir):
    """Select update layers and persist score artifacts."""
    num_layers = get_num_layers(model)
    strategy = str(cfg.layer_selection.strategy).lower()
    top_k = int(cfg.layer_selection.top_k)
    min_layer = cfg.layer_selection.get("min_layer", None)
    max_layer = cfg.layer_selection.get("max_layer", None)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if strategy == "manual":
        selected = list(cfg.layer_selection.selected_layers or [])
        if not selected:
            raise ValueError("layer_selection.strategy=manual requires selected_layers.")
        scores = {int(layer): {"manual": 1.0 if int(layer) in selected else 0.0} for layer in range(num_layers)}
    elif strategy == "middle":
        selected = select_middle_layers(num_layers, top_k, min_layer=min_layer, max_layer=max_layer)
        scores = {
            layer: {
                "middle_distance": abs(layer - ((selected[0] + selected[-1]) / 2.0)) if selected else 0.0,
                "selected": layer in selected,
            }
            for layer in range(num_layers)
        }
    elif strategy == "middle_targets":
        target_leaves = cfg.get("lora_target_modules", None)
        if target_leaves is None:
            target_leaves = sorted(
                {
                    name.split(".")[-1]
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear) and name.split(".")[-1] != "lm_head"
                }
            )
        selected, target_counts = select_middle_target_layers(
            model,
            target_leaves,
            top_k,
            min_layer=min_layer,
            max_layer=max_layer,
        )
        scores = {
            layer: {
                "target_module_count": target_counts.get(layer, 0),
                "selected": layer in selected,
            }
            for layer in range(num_layers)
        }
    elif strategy == "hidden_alignment":
        scores = compute_hidden_alignment_scores(model, cfg, tokenizer, project_root)
        candidates = layer_window(num_layers, min_layer=min_layer, max_layer=max_layer)
        selected = sorted(
            candidates,
            key=lambda layer: scores[layer]["hidden_alignment"],
            reverse=True,
        )[:top_k]
        selected = sorted(selected)
        for layer, layer_scores in scores.items():
            layer_scores["selected"] = layer in selected
    else:
        raise ValueError(f"Unsupported layer_selection.strategy: {cfg.layer_selection.strategy}")

    selected = [int(layer) for layer in selected]
    with open(save_dir / "selected_layers.json", "w") as f:
        json.dump({"strategy": strategy, "selected_layers": selected}, f, indent=2)
    with open(save_dir / "layer_scores.json", "w") as f:
        json.dump({str(k): v for k, v in scores.items()}, f, indent=2)
    return selected, scores


def collect_lora_target_modules(model, target_leaves, selected_layers):
    """Return full module names for LoRA attachment inside selected layers."""
    selected_layers = set(int(layer) for layer in selected_layers) if selected_layers is not None else None
    target_leaves = set(str(name) for name in target_leaves)
    target_modules = []

    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in target_leaves or leaf == "lm_head":
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        layer_idx = extract_layer_idx(name)
        if selected_layers is not None and layer_idx not in selected_layers:
            continue
        target_modules.append(name)

    if not target_modules:
        raise ValueError(
            f"No LoRA target modules found for leaves={sorted(target_leaves)} "
            f"and selected_layers={sorted(selected_layers) if selected_layers else None}."
        )
    return target_modules
