"""Merge a WJ LoRA adapter into its base model.

Example:
    python unlearning_methods/unlearn_wj/merge_lora.py \
      --base_model ./finetuned/finetuned_100 \
      --adapter ./finetuned/wj_npo_2e-5_forget01_5_en/lora_adapter \
      --output ./finetuned/wj_npo_2e-5_forget01_5_en/merged
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_project_path(path):
    path = str(path)
    if path.startswith("/"):
        return path
    return str((PROJECT_ROOT / path).resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    args = parser.parse_args()

    dtype = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    base_model = resolve_project_path(args.base_model)
    adapter = resolve_project_path(args.adapter)
    output = resolve_project_path(args.output)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload()
    Path(output).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    tokenizer.save_pretrained(output)
    print(f"Merged model saved to: {output}")


if __name__ == "__main__":
    main()
