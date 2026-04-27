import json
import os
import random
from pathlib import Path

import datasets
import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from data_module import convert_prompt_to_model_format
from mcsu_utils import (
    compute_language_subspace,
    compute_shared_subspace,
    default_layer_ids,
    last_nonpad_pool,
    pairwise_subspace_overlap,
    parse_torch_dtype,
    resolve_layer_ids,
)
from utils import get_model_identifiers_from_yaml


def _cfg_get(cfg, key, default=None):
    return cfg[key] if key in cfg and cfg[key] is not None else default


def _format_template(template, language, split=None):
    return str(template).format(language=language, lang=language, split=split)


def _local_path(path):
    if path is None:
        return None
    path = str(path)
    candidate = Path(path).expanduser()
    if candidate.exists():
        return str(candidate)
    abs_candidate = Path(to_absolute_path(path)).expanduser()
    if abs_candidate.exists():
        return str(abs_candidate)
    return None


def _load_from_disk_train(path, split_name=None):
    loaded = datasets.load_from_disk(str(path))
    if isinstance(loaded, datasets.DatasetDict):
        if split_name is not None and split_name in loaded:
            return loaded[split_name]
        if "train" in loaded:
            return loaded["train"]
        first_split = next(iter(loaded.keys()))
        return loaded[first_split]
    return loaded


def _load_hf_subset(data_path, subset):
    try:
        loaded = datasets.load_dataset(str(data_path), str(subset))
    except TypeError:
        loaded = datasets.load_dataset(str(data_path), name=str(subset))
    return loaded["train"] if isinstance(loaded, datasets.DatasetDict) else loaded


def _load_dataset_any(data_path, subset=None):
    local = _local_path(data_path)
    if local is not None:
        return _load_from_disk_train(local, split_name=subset)
    return _load_hf_subset(data_path, subset)


def retain_split_from_forget(split):
    try:
        forget_pct = int(str(split).replace("forget", "").split("_")[0])
        return "retain" + str(100 - forget_pct).zfill(2)
    except Exception:
        return "retain99"


def _filter_language(dataset, language, data_name):
    if hasattr(dataset, "column_names") and "language" in dataset.column_names:
        dataset = dataset.filter(lambda row: row["language"] == language)
        if len(dataset) == 0:
            raise ValueError(f"{data_name} has a language column but no examples for language '{language}'.")
    return dataset


def _candidate_translated_path(cfg, template, language):
    root = _cfg_get(cfg, "translated_data_root")
    if root is None or template is None:
        return None
    return os.path.join(str(root), _format_template(template, language, _cfg_get(cfg, "split")))


def _first_existing_path(paths):
    for path in paths:
        local = _local_path(path)
        if local is not None:
            return local
    return None


def load_forget_retain_data(cfg, language):
    data_path = _cfg_get(cfg, "data_path")
    split = _cfg_get(cfg, "split", "forget01")
    retain_split = retain_split_from_forget(split)

    if isinstance(data_path, (dict, DictConfig)):
        forget_data = _load_dataset_any(data_path["forget"], split)
        retain_data = _load_dataset_any(data_path["retain"], retain_split)
        return _filter_language(forget_data, language, "forget_data"), _filter_language(
            retain_data, language, "retain_data"
        )

    forget_template = _cfg_get(cfg, "forget_template")
    retain_template = _cfg_get(cfg, "retain_template")
    translated_forget = _candidate_translated_path(cfg, forget_template, language)
    translated_retain = _candidate_translated_path(cfg, retain_template, language)

    fallback_forget = [
        translated_forget,
        os.path.join(str(_cfg_get(cfg, "translated_data_root", "")), f"{split}_{language}"),
        os.path.join(str(_cfg_get(cfg, "translated_data_root", "")), f"{split}_perturbed_{language}"),
    ]
    fallback_retain = [
        translated_retain,
        os.path.join(str(_cfg_get(cfg, "translated_data_root", "")), f"{retain_split}_{language}"),
        os.path.join(str(_cfg_get(cfg, "translated_data_root", "")), f"retain99_{language}"),
    ]

    forget_path = _first_existing_path([path for path in fallback_forget if path])
    retain_path = _first_existing_path([path for path in fallback_retain if path])

    if forget_path is not None:
        forget_data = _load_from_disk_train(forget_path)
    elif _local_path(data_path) is not None:
        forget_data = _filter_language(_load_dataset_any(data_path, split), language, "forget_data")
    elif language == "en":
        forget_data = _load_dataset_any(data_path, split)
    else:
        raise FileNotFoundError(
            f"Could not find forget data for language '{language}'. Tried translated_data_root/template "
            f"paths under '{_cfg_get(cfg, 'translated_data_root')}' and data_path='{data_path}'."
        )

    if retain_path is not None:
        retain_data = _load_from_disk_train(retain_path)
    elif language == "en" and _local_path(data_path) is None:
        retain_data = _load_dataset_any(data_path, retain_split)
    elif _local_path(data_path) is not None:
        retain_data = _filter_language(_load_dataset_any(data_path, retain_split), language, "retain_data")
    else:
        raise FileNotFoundError(
            f"Could not find retain/control data for language '{language}'. Tried translated retain template "
            f"under '{_cfg_get(cfg, 'translated_data_root')}'."
        )

    return _filter_language(forget_data, language, "forget_data"), _filter_language(
        retain_data, language, "retain_data"
    )


def get_question(row, data_name):
    for key in ("question", "prompt", "input", "text"):
        if key in row and row[key] is not None:
            return row[key]
    raise KeyError(
        f"Could not find a question field in {data_name}. Expected one of "
        f"question/prompt/input/text; available fields: {sorted(row.keys())}"
    )


def get_control_question(forget_row, retain_data, idx, language, control_idx=None):
    for key in ("control_prompt", "control_question", "negative_question"):
        if key in forget_row and forget_row[key]:
            return forget_row[key]
    if len(retain_data) == 0:
        raise ValueError(f"No retain/control examples are available for language '{language}'.")
    control_idx = idx % len(retain_data) if control_idx is None else control_idx
    return get_question(retain_data[control_idx], "retain_data")


def build_prompt_pairs(forget_data, retain_data, language, max_examples=None, random_control=False, seed=42):
    num_examples = len(forget_data)
    if max_examples is not None:
        num_examples = min(num_examples, int(max_examples))

    rng = random.Random(seed)
    pairs = []
    for idx in range(num_examples):
        forget_row = forget_data[idx]
        row_language = forget_row.get("language", language)
        if row_language != language:
            continue
        forget_question = get_question(forget_row, "forget_data")
        control_idx = rng.randrange(len(retain_data)) if random_control and len(retain_data) > 0 else None
        control_question = get_control_question(forget_row, retain_data, idx, language, control_idx=control_idx)
        pairs.append((forget_question, control_question))
    if not pairs:
        raise ValueError(f"No MCSU prompt pairs were built for language '{language}'.")
    return pairs


def _stack_prompt_batch(tokenizer, questions, max_length, model_configs, language, device):
    encoded = [
        convert_prompt_to_model_format(tokenizer, max_length, question, model_configs, language)
        for question in questions
    ]
    input_ids = torch.stack([item[0] for item in encoded]).to(device)
    attention_mask = torch.stack([item[1] for item in encoded]).to(device)
    return input_ids, attention_mask


@torch.no_grad()
def compute_language_differences(model, tokenizer, model_configs, pairs, language, layer_ids, cfg, device):
    z_by_layer = {layer: [] for layer in layer_ids}
    batch_size = int(_cfg_get(cfg, "batch_size", 4))
    prompt_max_length = int(_cfg_get(cfg, "prompt_max_length", 256))

    for start in tqdm(range(0, len(pairs), batch_size), desc=f"MCSU z[{language}]"):
        batch_pairs = pairs[start : start + batch_size]
        forget_questions = [pair[0] for pair in batch_pairs]
        control_questions = [pair[1] for pair in batch_pairs]
        forget_input_ids, forget_attention_mask = _stack_prompt_batch(
            tokenizer, forget_questions, prompt_max_length, model_configs, language, device
        )
        control_input_ids, control_attention_mask = _stack_prompt_batch(
            tokenizer, control_questions, prompt_max_length, model_configs, language, device
        )

        outputs_f = model(
            input_ids=forget_input_ids,
            attention_mask=forget_attention_mask,
            output_hidden_states=True,
        )
        outputs_c = model(
            input_ids=control_input_ids,
            attention_mask=control_attention_mask,
            output_hidden_states=True,
        )

        for layer in layer_ids:
            h_f = last_nonpad_pool(outputs_f.hidden_states[layer + 1], forget_attention_mask)
            h_c = last_nonpad_pool(outputs_c.hidden_states[layer + 1], control_attention_mask)
            z_by_layer[layer].append((h_f - h_c).detach().float().cpu())

    return {layer: torch.cat(z_list, dim=0) for layer, z_list in z_by_layer.items()}


def load_model_and_tokenizer(cfg):
    model_configs = get_model_identifiers_from_yaml(cfg.model_family)
    model_source = _cfg_get(cfg, "model_path", model_configs["hf_key"])
    tokenizer_source = model_source

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        if tokenizer_source == model_configs["hf_key"]:
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_configs["hf_key"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = parse_torch_dtype(_cfg_get(cfg, "torch_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        attn_implementation="flash_attention_2" if model_configs["flash_attention2"] == "true" else None,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    return model, tokenizer, model_configs


def build_diagnostics(examples_per_language, singular_values, shared_eigvals, B_by_lang, train_languages, k_lang):
    diagnostics = {
        "num_examples_per_language": examples_per_language,
        "top_singular_values": {},
        "top_shared_eigenvalues": {},
        "pairwise_cross_lingual_subspace_overlap": {},
    }

    for layer, layer_singular_values in singular_values.items():
        layer_key = str(layer)
        diagnostics["top_singular_values"][layer_key] = {}
        for language, values in layer_singular_values.items():
            diagnostics["top_singular_values"][layer_key][language] = values[:10].cpu().float().tolist()
        diagnostics["top_shared_eigenvalues"][layer_key] = shared_eigvals[layer][:10].cpu().float().tolist()

        overlaps = {}
        for i, lang1 in enumerate(train_languages):
            if lang1 not in B_by_lang[layer]:
                continue
            for lang2 in train_languages[i + 1 :]:
                if lang2 not in B_by_lang[layer]:
                    continue
                overlaps[f"{lang1}__{lang2}"] = pairwise_subspace_overlap(
                    B_by_lang[layer][lang1],
                    B_by_lang[layer][lang2],
                    normalizer=k_lang,
                )
        diagnostics["pairwise_cross_lingual_subspace_overlap"][layer_key] = overlaps
    return diagnostics


@hydra.main(version_base=None, config_path="config", config_name="mcsu_subspace")
def main(cfg):
    set_seed(int(_cfg_get(cfg, "seed", 42)))
    output_dir = Path(to_absolute_path(str(cfg.output_dir)))
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, model_configs = load_model_and_tokenizer(cfg)
    device_cfg = str(_cfg_get(cfg, "device", "auto"))
    if device_cfg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_cfg)
    model.to(device)
    model.eval()

    layer_ids = resolve_layer_ids(_cfg_get(cfg, "layer_ids"), model.config)
    if not layer_ids:
        layer_ids = default_layer_ids(model.config)

    languages = [str(language) for language in cfg.languages]
    train_languages = [str(language) for language in cfg.train_languages]
    missing_train_languages = [language for language in train_languages if language not in languages]
    if missing_train_languages:
        raise ValueError(
            f"train_languages must be included in languages. Missing: {missing_train_languages}"
        )

    z_by_language_layer = {}
    examples_per_language = {}
    for language in languages:
        forget_data, retain_data = load_forget_retain_data(cfg, language)
        retain_data = _filter_language(retain_data, language, "retain_data")
        pairs = build_prompt_pairs(
            forget_data,
            retain_data,
            language,
            max_examples=_cfg_get(cfg, "max_examples_per_language"),
            random_control=bool(_cfg_get(cfg, "random_control", False)),
            seed=int(_cfg_get(cfg, "seed", 42)),
        )
        examples_per_language[language] = len(pairs)
        z_by_language_layer[language] = compute_language_differences(
            model, tokenizer, model_configs, pairs, language, layer_ids, cfg, device
        )

    B_by_lang = {layer: {} for layer in layer_ids}
    singular_values = {layer: {} for layer in layer_ids}
    for layer in layer_ids:
        for language in languages:
            z = z_by_language_layer[language][layer]
            B, S = compute_language_subspace(
                z,
                k=int(cfg.k_lang),
                row_normalize_z=bool(cfg.row_normalize_z),
                mean_center_z=bool(cfg.mean_center_z),
            )
            gram = B.T @ B
            err = (gram - torch.eye(gram.shape[0])).abs().max().item()
            if err > 1e-3:
                print(f"[warn] basis orthogonality error for layer={layer}, language={language}: {err:.4e}")
            B_by_lang[layer][language] = B.cpu().float()
            singular_values[layer][language] = S.cpu().float()

    U_shared = {}
    shared_eigvals = {}
    for layer in layer_ids:
        train_basis = {language: B_by_lang[layer][language] for language in train_languages}
        U, eigvals = compute_shared_subspace(train_basis, k_shared=int(cfg.k_shared))
        U_shared[layer] = U.cpu().float()
        shared_eigvals[layer] = eigvals.cpu().float()

    subspace_obj = {
        "model_family": cfg.model_family,
        "model_path": cfg.model_path,
        "layer_ids": layer_ids,
        "languages": languages,
        "train_languages": train_languages,
        "k_lang": int(cfg.k_lang),
        "k_shared": int(cfg.k_shared),
        "U_shared": U_shared,
        "B_by_lang": B_by_lang,
        "singular_values": singular_values,
        "shared_eigvals": shared_eigvals,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(subspace_obj, output_dir / "subspaces.pt")

    diagnostics = build_diagnostics(
        examples_per_language,
        singular_values,
        shared_eigvals,
        B_by_lang,
        train_languages,
        int(cfg.k_lang),
    )
    with open(output_dir / "subspace_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"Saved MCSU subspaces to {output_dir / 'subspaces.pt'}")
    print(f"Saved MCSU diagnostics to {output_dir / 'subspace_diagnostics.json'}")


if __name__ == "__main__":
    main()
