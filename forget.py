from data_module import TextForgetDatasetQA, TextForgetDatasetQAMCSU
from dataloader import (
    CustomTrainerForgetting,
    custom_data_collator_forget,
    custom_data_collator_forget_kl,
    custom_data_collator_forget_mcsu,
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed

import hydra 
import transformers
import os
import importlib.util
from pathlib import Path
from utils import get_model_identifiers_from_yaml


def _cfg_get(section, key, default=None):
    if section is None:
        return default
    return section[key] if key in section and section[key] is not None else default


def setup_wandb(cfg):
    wandb_cfg = cfg.get("wandb", {})
    enabled = bool(_cfg_get(wandb_cfg, "enabled", False))
    if not enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return None, None

    if importlib.util.find_spec("wandb") is None:
        raise ImportError("wandb.enabled=true requires the wandb package. Install it with `pip install wandb`.")

    os.environ.pop("WANDB_DISABLED", None)
    os.environ["WANDB_ENTITY"] = str(_cfg_get(wandb_cfg, "entity", "changwoolabs"))
    os.environ["WANDB_PROJECT"] = str(_cfg_get(wandb_cfg, "project", "multilingual-amnesia"))
    os.environ["WANDB_MODE"] = str(_cfg_get(wandb_cfg, "mode", "online"))

    group = _cfg_get(wandb_cfg, "group", None)
    if group:
        os.environ["WANDB_RUN_GROUP"] = str(group)

    tags = _cfg_get(wandb_cfg, "tags", [])
    if tags:
        os.environ["WANDB_TAGS"] = ",".join([str(tag) for tag in tags])

    run_name = _cfg_get(wandb_cfg, "name", None)
    if run_name is None:
        mcsu_suffix = "mcsu" if bool(cfg.get("mcsu", {}).get("enabled", False)) else "baseline"
        run_name = f"{cfg.model_family}_{cfg.forget_loss}_{cfg.split}_{cfg.language}_{mcsu_suffix}"

    return "wandb", str(run_name)


def get_train_and_oracle_devices():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Set CUDA_VISIBLE_DEVICES to a valid GPU before training.")
    train_device = torch.device("cuda:0")
    oracle_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    return train_device, oracle_device


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

@hydra.main(version_base=None, config_path="config", config_name="forget")
def main(cfg):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    num_devices = int(os.environ.get('WORLD_SIZE', 1))
    print(f"num_devices: {num_devices}")


    set_seed(cfg.seed)

    report_to, wandb_run_name = setup_wandb(cfg)
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    if cfg.model_path is None:
        cfg.model_path = model_cfg["ft_model_path"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    train_device, oracle_device = get_train_and_oracle_devices()
    print(f"train_device: {train_device}, oracle_device: {oracle_device}")

    tokenizer_source = cfg.model_path if cfg.model_path is not None else model_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        if tokenizer_source == model_id:
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    max_length = 500

    mcsu_cfg = cfg.get("mcsu", {})
    mcsu_enabled = bool(mcsu_cfg.get("enabled", False))
    if mcsu_enabled:
        torch_format_dataset = TextForgetDatasetQAMCSU(
            cfg.data_path,
            tokenizer=tokenizer,
            model_family=cfg.model_family,
            max_length=max_length,
            split=cfg.split,
            loss_type=cfg.forget_loss,
            language=cfg.language,
            mcsu_prompt_max_length=mcsu_cfg.get("prompt_max_length", 256),
            mcsu_control_source=mcsu_cfg.get("control_source", "retain"),
        )
    else:
        torch_format_dataset = TextForgetDatasetQA(cfg.data_path, tokenizer=tokenizer, model_family = cfg.model_family, max_length=max_length, split=cfg.split, loss_type=cfg.forget_loss, language=cfg.language)
    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset)//(batch_size*gradient_accumulation_steps*num_devices)

    max_steps = int(cfg.num_epochs*len(torch_format_dataset))//(batch_size*gradient_accumulation_steps)
    print(f"max_steps: {max_steps}")
    print("batch_size:", batch_size)
    per_device_batch_size = max(1, batch_size // max(1, torch.cuda.device_count()))
    training_args = transformers.TrainingArguments(
            per_device_train_batch_size=per_device_batch_size,
            per_device_eval_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=max(1, steps_per_epoch),
            max_steps=max_steps,
            learning_rate=cfg.lr,
            bf16=True,
            bf16_full_eval=True,
            logging_steps=max(1,max_steps//20),
            logging_dir=f'{cfg.save_dir}/logs',
            output_dir=cfg.save_dir,
            optim="paged_adamw_32bit",
            save_strategy="steps" if cfg.save_model and (not cfg.eval_only) else "no",
            save_steps=steps_per_epoch,
            save_only_model=True,
            ddp_find_unused_parameters= False,
            # deepspeed='config/ds_config.json',
            weight_decay = cfg.weight_decay,
            eval_steps = steps_per_epoch,
            eval_strategy = "steps" if cfg.eval_while_train else "no",
            seed=cfg.seed,
            report_to=report_to,
            run_name=wandb_run_name,

        )
    # Load the main model on GPU 0 (cuda:0)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path, 
        attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(train_device)
    
    # Load the oracle model on GPU 1 (cuda:1), only if needed
    oracle_model = None
    if cfg.forget_loss == "KL" or cfg.forget_loss == "grad_diff_KL" or cfg.forget_loss == "npo":
        oracle_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path, 
            attn_implementation="flash_attention_2" if model_cfg["flash_attention2"] == "true" else None,
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        ).to(oracle_device)
        oracle_model.eval()

    model.generation_config.do_sample = True
    
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()


    data_collator = custom_data_collator_forget
    if mcsu_enabled:
        data_collator = custom_data_collator_forget_mcsu
    elif cfg.forget_loss == "grad_diff_KL":
        data_collator = custom_data_collator_forget_kl
       
    trainer = CustomTrainerForgetting(
        model=model,
        tokenizer=tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset = torch_format_dataset,
        compute_metrics=None,                # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=data_collator,
        oracle_model = oracle_model,
        beta = cfg.beta,
        gamma= cfg.gamma,
        forget_loss = cfg.forget_loss,
        eval_cfg = cfg.eval,
        mcsu_enabled=mcsu_enabled,
        mcsu_subspace_path=mcsu_cfg.get("subspace_path", None),
        mcsu_gamma=mcsu_cfg.get("gamma", 1.0),
        mcsu_layer_ids=mcsu_cfg.get("layer_ids", None),
        mcsu_eps=mcsu_cfg.get("eps", 1e-8),
    )
    model.config.use_cache = False  
    # trainer.train()
    if cfg.eval_only:
        trainer.evaluate()
    else:
        trainer.train()

    #save the tokenizer
    if cfg.save_model and (not cfg.eval_only):
        model.save_pretrained(cfg.save_dir)
        tokenizer.save_pretrained(cfg.save_dir)

    #delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                #delete the directory
                import shutil
                shutil.rmtree(global_step_dir)



if __name__ == "__main__":
    main()
