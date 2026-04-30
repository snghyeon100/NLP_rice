"""LoRA adapter를 base model에 merge하여 일반 모델로 저장.

merge_and_unload()를 통해 LoRA 가중치를 base model에 흡수시킵니다.
  W_eff = W_base + (alpha/r) * B @ A

merge된 모델은 PEFT 없이 AutoModelForCausalLM.from_pretrained()으로 로드 가능하므로
기존 evaluate_util.py와 바로 호환됩니다.

사용법:
    python unlearning_methods/unlearn_sh/merge_lora.py \\
        --base_model_path ./finetuned \\
        --adapter_path    ./finetuned/rcp_slora_.../lora_adapter \\
        --output_path     ./outputs/merged_ko \\
        --model_family    qwen3_5_2b

평가 (merge 후):
    python evaluate_util.py \\
        model_path=./outputs/merged_ko \\
        'languages=[ko]' \\
        ...

eval_while_train 마지막 step 대비:
    - eval_while_train: eval_steps 배수 시점에만 평가 (마지막 step과 타이밍 차이 가능)
    - merge 후 평가:    실제 학습이 끝난 최종 가중치로 평가 → 더 정확
    - 수치:            float 오차 수준으로 동일
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import get_model_identifiers_from_yaml


def merge_lora(base_model_path: str, adapter_path: str, output_path: str, model_family: str | None = None):
    """LoRA adapter를 base model에 merge한 뒤 output_path에 저장.

    Args:
        base_model_path: fine-tuned base 모델 경로
        adapter_path:    학습된 LoRA adapter 경로 (adapter_model.safetensors 포함 폴더)
        output_path:     merge된 모델 저장 경로
        model_family:    model_config.yaml 키 (flash_attention 설정용, 없으면 None)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # flash_attention 설정 확인
    use_flash_attn = False
    if model_family is not None:
        try:
            model_cfg = get_model_identifiers_from_yaml(model_family)
            use_flash_attn = model_cfg.get("flash_attention2", "false") == "true"
        except Exception:
            pass

    # ── Base model 로드 ───────────────────────────────────
    print(f"Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        attn_implementation="flash_attention_2" if use_flash_attn else None,
        torch_dtype=torch.float16,   # merge 연산은 fp16으로 (Titan XP 호환)
        trust_remote_code=True,
    )

    # ── LoRA adapter 로드 ─────────────────────────────────
    print(f"Loading LoRA adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    # ── Merge: W_eff = W_base + (alpha/r) * B @ A ────────
    print("Merging LoRA weights into base model...")
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    # ── 저장 ──────────────────────────────────────────────
    print(f"Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)

    # tokenizer도 함께 저장 (adapter_path에 저장되어 있는 tokenizer 파일 우선)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    print(f"\n{'='*50}")
    print(f"Merge complete.")
    print(f"  Base model:   {base_model_path}")
    print(f"  LoRA adapter: {adapter_path}")
    print(f"  Merged model: {output_path}")
    print(f"{'='*50}")
    print(f"\n[다음 단계 — evaluate_util.py로 평가]")
    print(f"  python evaluate_util.py model_path={output_path} 'languages=[<lang>]'\n")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA adapter를 base model에 merge하여 evaluate_util.py 호환 모델로 저장"
    )
    parser.add_argument(
        "--base_model_path", required=True,
        help="fine-tuned base 모델 경로 (예: ./finetuned)",
    )
    parser.add_argument(
        "--adapter_path", required=True,
        help="학습된 LoRA adapter 경로 (lora_save_dir, 예: ./finetuned/rcp_slora_.../lora_adapter)",
    )
    parser.add_argument(
        "--output_path", required=True,
        help="merge된 모델 저장 경로 (예: ./outputs/merged_ko)",
    )
    parser.add_argument(
        "--model_family", default=None,
        help="model_config.yaml 키 (예: qwen3_5_2b). flash_attention 설정 자동 적용. 없으면 생략 가능.",
    )
    args = parser.parse_args()

    merge_lora(
        base_model_path=args.base_model_path,
        adapter_path=args.adapter_path,
        output_path=args.output_path,
        model_family=args.model_family,
    )


if __name__ == "__main__":
    main()
