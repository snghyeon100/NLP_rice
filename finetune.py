from data_module import TextDatasetQA, custom_data_collator
from dataloader import CustomTrainer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from transformers.trainer_utils import get_last_checkpoint
from hydra.utils import to_absolute_path

import hydra 
import transformers
import os
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_model_identifiers_from_yaml


def normalize_cuda_visible_devices():
    # Accept legacy/typo env var if provided and map it to CUDA_VISIBLE_DEVICES.
    legacy_key = "CUDA_VISIBE_DEVICE"
    standard_key = "CUDA_VISIBLE_DEVICES"
    legacy_value = os.environ.get(legacy_key)
    standard_value = os.environ.get(standard_key)

    if standard_value is None and legacy_value is not None:
        os.environ[standard_key] = legacy_value
        print(f"[env] {legacy_key} detected, using {standard_key}={legacy_value}")
    elif standard_value is not None:
        print(f"[env] {standard_key}={standard_value}")
        if legacy_value is not None and legacy_value != standard_value:
            print(f"[env] Warning: both {legacy_key} and {standard_key} are set. Using {standard_key}.")


@hydra.main(version_base=None, config_path="config", config_name="finetune")
def main(cfg):
    normalize_cuda_visible_devices()

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            print(f"[env] LOCAL_RANK={local_rank}, cuda_device={torch.cuda.current_device()}")
    set_seed(cfg.seed)
    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file

    if os.environ.get('LOCAL_RANK') is None or local_rank == 0:
        with open(f'{cfg.save_dir}/cfg.yaml', 'w') as f:
            OmegaConf.save(cfg, f)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500
    torch_format_dataset = TextDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length)

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")

    
    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps*num_devices)
    # max_steps=5
    print(f"max_steps: {max_steps}")
    save_dir = to_absolute_path(cfg.save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, max_steps//cfg.num_epochs),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//40),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=save_dir,
            optim="paged_adamw_32bit",
            save_steps=300,
            save_strategy="steps",
            save_total_limit=5,
            save_only_model=False,
            ddp_find_unused_parameters= False,
            eval_strategy="steps",
            eval_steps=300,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            prediction_loss_only=True,
            deepspeed=to_absolute_path("config/ds_config.json"),
            weight_decay = cfg.weight_decay,
            seed = cfg.seed,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
                                                 torch_dtype=torch.bfloat16, trust_remote_code = True,)
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    trainer = CustomTrainer(
        model=model,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=custom_data_collator,
    )
    model.config.use_cache = False

    last_ckpt = None
    if os.path.isdir(save_dir):
        last_ckpt = get_last_checkpoint(save_dir)   
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
        trainer.train(resume_from_checkpoint=last_ckpt)
    else:
        trainer.train()


    trainer.save_model(cfg.save_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(cfg.save_dir)

if __name__ == "__main__":
    main()
