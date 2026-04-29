"""End-to-end RCP-SLoRA unlearning runner.

Stage 1: Retain-Conflict Projected NPO with PEFT LoRA.

실행:
    python unlearning_methods/unlearn_sh/train.py
    python unlearning_methods/unlearn_sh/train.py language=ko \
        data_path.forget=./dataset/forget01_ko \
        data_path.retain=./dataset/retain99_ko

제약:
    - gradient_accumulation_steps == 1 만 지원 (1보다 크면 ValueError)
    - DeepSpeed 사용 안 함
    - gpu_train, gpu_oracle config로 GPU 번호 지정
"""

import csv
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, set_seed

from evaluate_util import get_all_evals, get_dataloader
from unlearning_methods.unlearn_sh.dataloader import RCPForgetDataset, rcp_collator
from unlearning_methods.unlearn_sh.loss import compute_rcp_gradients
from utils import get_forget_quality, get_model_identifiers_from_yaml, get_model_utility, merge_dicts


# ──────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────

def resolve_project_path(path):
    """상대경로를 PROJECT_ROOT 기준 절대경로로 변환."""
    if path is None:
        return None
    path = str(path)
    if path.startswith("/"):
        return path
    if path.startswith(("./", "../")):
        return str((PROJECT_ROOT / path).resolve())
    return path


def find_all_linear_names(model):
    """모델 내 모든 Linear 레이어의 leaf 이름을 반환 (LoRA target 자동 감지용).

    lm_head는 제외:
      - 많은 causal LM (Qwen 등)에서 embed_tokens와 weight를 공유(tied weight)하므로
        LoRA를 붙이면 weight tying 전제가 깨짐
      - PEFT 표준 관행: attention/MLP linear만 target으로 사용
    """
    linear_cls = [torch.nn.Linear]
    try:
        import bitsandbytes as bnb
        linear_cls += [bnb.nn.Linear4bit, bnb.nn.Linear8bitLt]
    except ImportError:
        pass
    linear_cls = tuple(linear_cls)

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, linear_cls):
            names.add(name.split(".")[-1])

    # lm_head: vocab projection, weight tying 가능성 있으므로 LoRA 대상에서 제외
    names.discard("lm_head")
    return list(names)


# ──────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────

class RCPTrainer(Trainer):
    """HuggingFace Trainer를 상속하여 training_step을 오버라이드.

    compute_loss() 대신 training_step()을 오버라이드하는 이유:
      - g_f와 g_r을 별도로 계산한 뒤 projection된 gradient를 param.grad에 수동 주입
      - Trainer가 loss.backward()를 재호출하지 않도록 training_step 전체를 제어
    """

    def __init__(self, *args, oracle_model=None, eval_cfg=None, tokenizer=None, cfg=None, **kwargs):
        self.oracle_model = oracle_model
        self.eval_cfg = eval_cfg
        self.tokenizer = tokenizer
        self.cfg = cfg
        super().__init__(*args, **kwargs)
        if self.oracle_model is None:
            raise ValueError("RCPTrainer requires an oracle/reference model.")
        self.oracle_model.eval()
        for p in self.oracle_model.parameters():
            p.requires_grad = False

    def _wrap_model(self, model, training=True, dataloader=None):
        # DeepSpeed 미사용 → 래핑 없이 그대로 반환
        return model

    # ── 핵심: training_step 오버라이드 ──────────────────────
    def training_step(self, model, inputs, num_items_in_batch=None):
        """RCP gradient projection을 수행하고 param.grad를 수동으로 주입.

        Trainer의 _inner_training_loop는 이 메서드의 반환값을 loss 추적에만 사용.
        backward()는 이 메서드 내부에서 이미 gradient를 param.grad에 써뒀으므로
        Trainer가 다시 backward()를 호출하지 않아도 됨.
        (training_step 내부에서 backward를 호출하지 않고 autograd.grad로 직접 처리)
        """
        model.train()

        # ── Gradient projection ──────────────────────────
        projected_grads, log_dict = compute_rcp_gradients(
            model=model,
            oracle_model=self.oracle_model,
            inputs=inputs,
            cfg=self.cfg,
        )

        # ── Projected gradient를 param.grad에 주입 ────────
        params = [p for p in model.parameters() if p.requires_grad]
        for p, gp in zip(params, projected_grads):
            p.grad = gp.detach()

        # ── 로깅 ──────────────────────────────────────────
        self.log(log_dict)

        # ── Trainer loss 추적용 scalar 반환 ───────────────
        # detach()된 tensor이므로 Trainer 내부에서 .backward()가 호출되더라도 no-op
        return torch.tensor(
            log_dict["forget_loss"],
            device=next(model.parameters()).device,
        )

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        return (outputs.loss, outputs.logits, labels)

    # ── 평가 ────────────────────────────────────────────────
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self._wrap_model(self.model, training=False)
        model.eval()

        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg
        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        aggregated_eval_logs = {}

        with torch.no_grad():
            for folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key in zip(
                eval_cfg.data_path,
                eval_cfg.split_list,
                eval_cfg.question_key,
                eval_cfg.answer_key,
                eval_cfg.eval_task,
                eval_cfg.base_answer_key,
                eval_cfg.perturbed_answer_key,
            ):
                if eval_task == "eval_log_forget":
                    split = eval_cfg.split

                print(f"Working on eval task {eval_task} with split {split}")
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")

                if os.path.exists(save_filename) and not eval_cfg.overwrite:
                    print(f"Skipping {eval_task} because {save_filename} already exists")
                    continue

                eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(
                    eval_cfg,
                    eval_task,
                    self.tokenizer,
                    folder,
                    split,
                    question_key,
                    answer_key,
                    base_answer_key,
                    perturbed_answer_key,
                    language=self.cfg.language,  # 언어 전달
                )

                eval_logs = get_all_evals(
                    eval_cfg,
                    model,
                    self.tokenizer,
                    eval_task,
                    eval_dataloader,
                    base_eval_dataloader,
                    perturb_dataloader,
                    normalize_gt=False,
                    language=self.cfg.language,
                )

                with open(save_filename, "w") as f:
                    json.dump(eval_logs, f, indent=4)
                aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

        aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")
        with open(aggregated_eval_log_filename, "w") as f:
            json.dump(aggregated_eval_logs, f, indent=4)

        if eval_cfg.retain_result is not None:
            model_utility = get_model_utility(aggregated_eval_logs)
            retain_result = json.load(open(eval_cfg.retain_result, "r"))
            forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
            aggregate_stat = {**model_utility, **forget_quality}
            with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), "w") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=list(aggregate_stat.keys()))
                writer.writeheader()
                writer.writerow(aggregate_stat)

        return aggregated_eval_logs


# ──────────────────────────────────────────────────────────
# TrainingArguments 빌더
# ──────────────────────────────────────────────────────────

def build_training_args(cfg, max_steps, steps_per_epoch, batch_size):
    return transformers.TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,   # RCP-SLoRA는 항상 1
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=cfg.bf16,                   # Titan XP(Pascal): false / Ampere+: true
        fp16=cfg.fp16,                   # Titan XP: true  / Ampere+: false
        bf16_full_eval=cfg.bf16,
        fp16_full_eval=cfg.fp16,
        logging_steps=max(1, max_steps // 20),
        logging_dir=str(Path(cfg.save_dir) / "logs"),
        output_dir=cfg.save_dir,
        optim=cfg.optim,                 # adamw_torch (기본) or paged_adamw_32bit
        save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters=False,
        weight_decay=cfg.weight_decay,
        eval_steps=steps_per_epoch,
        eval_strategy="steps" if cfg.eval_while_train else "no",
        seed=cfg.seed,
    )



# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    # ── gradient_accumulation_steps 검증 ──────────────────
    if cfg.gradient_accumulation_steps != 1:
        raise ValueError(
            f"RCP-SLoRA는 gradient_accumulation_steps == 1만 지원합니다. "
            f"현재 값: {cfg.gradient_accumulation_steps}\n"
            "이유: training_step 내부에서 param.grad를 수동으로 쓰기 때문에 "
            "gradient accumulation과 충돌합니다."
        )

    os.environ["WANDB_DISABLED"] = "true"
    set_seed(cfg.seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]
    cfg.model_path = resolve_project_path(cfg.model_path)
    cfg.save_dir = resolve_project_path(cfg.save_dir)
    cfg.lora_save_dir = resolve_project_path(cfg.lora_save_dir)

    print(f"{'='*50}")
    print(f"Saving to:        {cfg.save_dir}")
    print(f"LoRA adapter to:  {cfg.lora_save_dir}")
    print(f"Language:         {cfg.language}")
    print(f"GPU train:        cuda:{cfg.gpu_train}")
    print(f"GPU oracle:       cuda:{cfg.gpu_oracle}")
    print(f"{'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset / steps 계산 ──────────────────────────────
    dataset = RCPForgetDataset(
        cfg.data_path,
        tokenizer=tokenizer,
        model_family=cfg.model_family,
        max_length=500,
        split=cfg.split,
        language=cfg.language,
    )
    batch_size = cfg.batch_size
    steps_per_epoch = max(1, len(dataset) // batch_size)
    max_steps = max(1, int(cfg.num_epochs * len(dataset)) // batch_size)
    print(f"Dataset size:  {len(dataset)}")
    print(f"Steps/epoch:   {steps_per_epoch}")
    print(f"Max steps:     {max_steps}")

    training_args = build_training_args(cfg, max_steps, steps_per_epoch, batch_size)

    # ── 학습 모델 로드 ────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(f"cuda:{cfg.gpu_train}")

    # ── PEFT LoRA 부착 ────────────────────────────────────
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        target_modules = cfg.lora_target_modules
        if target_modules is None:
            target_modules = find_all_linear_names(model)
            print(f"LoRA target modules (auto): {target_modules}")
        else:
            print(f"LoRA target modules (config): {target_modules}")

        if cfg.lora_target_layers is not None:
            filtered = []
            for name, _ in model.named_modules():
                if not any(name.endswith(t) for t in target_modules):
                    continue
                if "layers." not in name:
                    continue
                layer_idx = int(name.split("layers.")[1].split(".")[0])
                if layer_idx in cfg.lora_target_layers:
                    filtered.append(name)
            target_modules = filtered
            print(f"LoRA target layers filter applied: {cfg.lora_target_layers}")
            print(f"Filtered target modules count: {len(target_modules)}")

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            target_modules=target_modules,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("LoRA 미사용: 전체 파라미터 학습")

    model.generation_config.do_sample = True
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    # ── Oracle 모델 로드 (frozen, LoRA 없음) ──────────────
    oracle_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(f"cuda:{cfg.gpu_oracle}")
    oracle_model.eval()

    # ── Trainer 구성 ─────────────────────────────────────
    trainer = RCPTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
        compute_metrics=None,
        args=training_args,
        data_collator=rcp_collator,
        oracle_model=oracle_model,
        eval_cfg=cfg.eval,
        cfg=cfg,
    )
    model.config.use_cache = False

    # ── 학습 / 평가 ────────────────────────────────────────
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    # ── LoRA adapter 저장 ─────────────────────────────────
    if cfg.save_model and (not cfg.eval_only):
        Path(cfg.lora_save_dir).mkdir(parents=True, exist_ok=True)
        if cfg.use_lora:
            # PEFT adapter만 저장 (adapter_model.safetensors + adapter_config.json)
            model.save_pretrained(cfg.lora_save_dir)
            tokenizer.save_pretrained(cfg.lora_save_dir)
            print(f"LoRA adapter saved to: {cfg.lora_save_dir}")
        else:
            model.save_pretrained(cfg.save_dir)
            tokenizer.save_pretrained(cfg.save_dir)

    # ── checkpoint 정리 ────────────────────────────────────
    for file in Path(cfg.save_dir).glob("checkpoint-*"):
        for global_step_dir in file.glob("global_step*"):
            shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
