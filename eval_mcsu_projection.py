import argparse
import importlib.util
import json
import os
from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf

from mcsu_subspace import (
    build_prompt_pairs,
    compute_language_differences,
    load_forget_retain_data,
    load_model_and_tokenizer,
)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _setup_wandb(args):
    if not _parse_bool(args.wandb_enabled):
        os.environ["WANDB_DISABLED"] = "true"
        return None
    if importlib.util.find_spec("wandb") is None:
        raise ImportError("--wandb_enabled true requires the wandb package. Install it with `pip install wandb`.")

    import wandb

    os.environ.pop("WANDB_DISABLED", None)
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()] if args.wandb_tags else []
    run_name = args.wandb_name
    if run_name is None:
        run_name = f"projection_{Path(str(args.model_path).rstrip('/')).name or args.model_family}"
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        tags=tags,
        mode=args.wandb_mode,
        config=vars(args),
    )
    return wandb


def _load_subspace(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _parse_languages(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    value = value.strip()
    if value.startswith("["):
        return [str(item) for item in OmegaConf.create(value)]
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_layer_ids(value, fallback):
    if value is None:
        return [int(layer) for layer in fallback]
    if value.strip().lower() in {"none", "null", ""}:
        return [int(layer) for layer in fallback]
    if value.strip().startswith("["):
        parsed = OmegaConf.create(value)
        return [int(layer) for layer in parsed]
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def projection_stats(z, U, eps):
    U = U.to(dtype=z.dtype)
    proj = z @ U
    energy = (proj ** 2).sum(dim=-1) / ((z ** 2).sum(dim=-1) + eps)
    return {
        "mean_projection_energy": energy.mean().item(),
        "std_projection_energy": energy.std(unbiased=False).item() if energy.numel() > 1 else 0.0,
        "num_examples": int(energy.numel()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MCSU projection energy diagnostics.")
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--subspace_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--languages", default=None, help="Comma list or OmegaConf list. Defaults to subspace languages.")
    parser.add_argument("--split", default="forget01")
    parser.add_argument("--data_path", default="locuslab/TOFU")
    parser.add_argument("--translated_data_root", default="./trans_tofu")
    parser.add_argument("--forget_template", default="forget01_perturbed_{language}")
    parser.add_argument("--retain_template", default="retain99_{language}")
    parser.add_argument("--layer_ids", default=None, help="Comma list or OmegaConf list. Defaults to subspace layers.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--prompt_max_length", type=int, default=256)
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_examples_per_language", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wandb_enabled", default="false")
    parser.add_argument("--wandb_entity", default="changwoolabs")
    parser.add_argument("--wandb_project", default="multilingual-amnesia")
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_tags", default="")
    parser.add_argument("--wandb_mode", default="online")
    parser.add_argument("--wandb_log_artifact", default="true")
    return parser.parse_args()


def main():
    args = parse_args()
    wandb_module = _setup_wandb(args)
    subspace_obj = _load_subspace(to_absolute_path(args.subspace_path))
    languages = _parse_languages(args.languages) or [str(language) for language in subspace_obj["languages"]]
    layer_ids = _parse_layer_ids(args.layer_ids, subspace_obj["layer_ids"])
    raw_u_shared = {int(layer): basis for layer, basis in subspace_obj["U_shared"].items()}

    cfg = OmegaConf.create(vars(args))
    model, tokenizer, model_configs = load_model_and_tokenizer(cfg)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model.to(device)
    model.eval()

    results = {
        "model_family": args.model_family,
        "model_path": args.model_path,
        "subspace_path": args.subspace_path,
        "languages": languages,
        "layer_ids": layer_ids,
        "projection_energy": {},
    }

    with torch.no_grad():
        for language in languages:
            forget_data, retain_data = load_forget_retain_data(cfg, language)
            pairs = build_prompt_pairs(
                forget_data,
                retain_data,
                language,
                max_examples=args.max_examples_per_language,
                random_control=False,
                seed=42,
            )
            z_by_layer = compute_language_differences(
                model,
                tokenizer,
                model_configs,
                pairs,
                language,
                layer_ids,
                cfg,
                device,
            )
            results["projection_energy"][language] = {}
            for layer in layer_ids:
                if layer not in raw_u_shared:
                    raise KeyError(
                        f"Subspace file has no U_shared for layer {layer}. "
                        f"Available layers: {sorted(raw_u_shared.keys())}"
                    )
                stats = projection_stats(z_by_layer[layer], raw_u_shared[layer], args.eps)
                results["projection_energy"][language][str(layer)] = stats
                if wandb_module is not None:
                    wandb_module.log({
                        f"projection/{language}/layer_{layer}/mean_projection_energy": stats["mean_projection_energy"],
                        f"projection/{language}/layer_{layer}/std_projection_energy": stats["std_projection_energy"],
                        f"projection/{language}/layer_{layer}/num_examples": stats["num_examples"],
                    })

    output_path = Path(to_absolute_path(args.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved MCSU projection diagnostics to {output_path}")
    if wandb_module is not None:
        if _parse_bool(args.wandb_log_artifact):
            artifact_name = f"{wandb_module.run.name}-projection-diagnostics".replace("/", "-")
            artifact = wandb_module.Artifact(artifact_name, type="mcsu_projection")
            artifact.add_file(str(output_path))
            wandb_module.log_artifact(artifact)
        wandb_module.finish()


if __name__ == "__main__":
    main()
