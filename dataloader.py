import torch
from torch import nn
from transformers import Trainer
import torch.nn.functional as F
import copy, os
import deepspeed
from evaluate_util import get_dataloader, get_all_evals
import copy
import json 
from pathlib import Path
from data_module import get_batch_loss 
from mcsu_utils import last_nonpad_pool
from utils import merge_dicts, interleave_eval_result_dict, get_forget_quality, get_model_utility
import numpy as np
from scipy.stats import ks_2samp, hmean
import csv 
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available

def printll(name, inp):
    print(name, [round(x, 4) for x in inp])

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
            if prediction_loss_only:
                return (loss, None, None)
            logits = outputs.logits
        return (loss, logits, labels)
    


class CustomTrainerForgetting(Trainer):
    def __init__(self, gamma=1.0, alpha=1.0, *args, **kwargs):
        self.loss_type = kwargs.pop('forget_loss')
        self.oracle_model = kwargs.pop('oracle_model')
        self.eval_cfg = kwargs.pop('eval_cfg')
        self.tokenizer = kwargs.pop('tokenizer', None)
        self.mcsu_enabled = kwargs.pop('mcsu_enabled', False)
        self.mcsu_subspace_path = kwargs.pop('mcsu_subspace_path', None)
        self.mcsu_gamma = kwargs.pop('mcsu_gamma', 1.0)
        self.mcsu_layer_ids = kwargs.pop('mcsu_layer_ids', None)
        self.mcsu_eps = kwargs.pop('mcsu_eps', 1e-8)
        self.U_shared = {}
        self.alpha = alpha
        self.gamma = gamma
        self.beta = kwargs.pop('beta')
        self.train_device = None
        self.oracle_device = None
        if self.mcsu_enabled:
            self._load_mcsu_subspace()
        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)

        if self.loss_type == "grad_diff_KL" or self.loss_type == "npo":
            self.oracle_model.eval()
        if self.model is not None:
            self.train_device = next(self.model.parameters()).device
        if self.oracle_model is not None:
            self.oracle_device = next(self.oracle_model.parameters()).device
        #     self.oracle_model = self._create_oracle_model(oracle_model)

    def _load_mcsu_subspace(self):
        if not self.mcsu_subspace_path:
            raise ValueError("mcsu_enabled=True requires mcsu_subspace_path.")
        if not os.path.exists(self.mcsu_subspace_path):
            raise FileNotFoundError(f"MCSU subspace file not found: {self.mcsu_subspace_path}")

        try:
            subspace_obj = torch.load(self.mcsu_subspace_path, map_location="cpu", weights_only=False)
        except TypeError:
            subspace_obj = torch.load(self.mcsu_subspace_path, map_location="cpu")

        raw_u_shared = subspace_obj.get("U_shared")
        if raw_u_shared is None:
            raise KeyError(f"MCSU subspace file {self.mcsu_subspace_path} does not contain 'U_shared'.")

        available_u = {int(layer): basis for layer, basis in raw_u_shared.items()}
        if self.mcsu_layer_ids is None:
            self.mcsu_layer_ids = [int(layer) for layer in subspace_obj.get("layer_ids", available_u.keys())]
        elif isinstance(self.mcsu_layer_ids, str):
            if self.mcsu_layer_ids.lower() in {"none", "null", ""}:
                self.mcsu_layer_ids = [int(layer) for layer in subspace_obj.get("layer_ids", available_u.keys())]
            else:
                self.mcsu_layer_ids = [int(layer.strip()) for layer in self.mcsu_layer_ids.split(",") if layer.strip()]
        else:
            self.mcsu_layer_ids = [int(layer) for layer in self.mcsu_layer_ids]

        for layer in self.mcsu_layer_ids:
            if layer not in available_u:
                raise KeyError(
                    f"MCSU subspace for layer {layer} is missing. "
                    f"Available layers: {sorted(available_u.keys())}"
                )
            self.U_shared[layer] = available_u[layer].detach().float().cpu()
            self.U_shared[layer].requires_grad_(False)

    def _wrap_model(self, model, training=True, dataloader=None):
        # The forgetting pipeline manually places the train model on cuda:0 and the
        # oracle/reference model on cuda:1. Letting Trainer apply DataParallel would
        # replicate the train model onto cuda:1 as well, which collides with the
        # oracle model and causes OOM.
        return model
        

    def _create_oracle_model(self, reference_model):
        with torch.no_grad():
            model_copy = reference_model.__class__(reference_model.config)  
            model_copy.to("cuda:1")  
            model_copy.load_state_dict(reference_model.state_dict(), strict=True) 

            model_copy.eval() 
            for param in model_copy.parameters():
                param.requires_grad = False  

        return model_copy
        
    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        #set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    

    def compute_mcsu_loss(self, model, mcsu_forget_inputs, mcsu_control_inputs):
        device = next(model.parameters()).device
        forget_input_ids, forget_attention_mask = mcsu_forget_inputs
        control_input_ids, control_attention_mask = mcsu_control_inputs
        forget_input_ids = forget_input_ids.to(device)
        forget_attention_mask = forget_attention_mask.to(device)
        control_input_ids = control_input_ids.to(device)
        control_attention_mask = control_attention_mask.to(device)

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
        if outputs_f.hidden_states is None or outputs_c.hidden_states is None:
            raise ValueError("Model did not return hidden states. MCSU requires output_hidden_states=True.")

        layer_losses = []
        for layer in self.mcsu_layer_ids:
            hidden_idx = layer + 1
            if hidden_idx >= len(outputs_f.hidden_states):
                raise ValueError(
                    f"MCSU layer {layer} is out of range for model hidden_states length "
                    f"{len(outputs_f.hidden_states)}."
                )
            h_f = last_nonpad_pool(outputs_f.hidden_states[hidden_idx], forget_attention_mask)
            h_c = last_nonpad_pool(outputs_c.hidden_states[hidden_idx], control_attention_mask)
            z = h_f - h_c
            U = self.U_shared[layer].to(device=z.device, dtype=z.dtype)
            if U.shape[0] != z.shape[-1]:
                raise ValueError(
                    f"MCSU subspace hidden dim {U.shape[0]} does not match model hidden dim {z.shape[-1]} "
                    f"for layer {layer}."
                )
            proj = z @ U
            num = (proj ** 2).sum(dim=-1)
            den = (z ** 2).sum(dim=-1) + self.mcsu_eps
            layer_losses.append((num / den).mean())

        return torch.stack(layer_losses).mean()


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = None
        mcsu_forget_inputs, mcsu_control_inputs = None, None
        if self.mcsu_enabled:
            if len(inputs) < 4:
                raise ValueError(
                    "MCSU batches must contain existing loss inputs plus "
                    "mcsu_forget_prompt_inputs and mcsu_control_prompt_inputs."
                )
            mcsu_forget_inputs = inputs[-2]
            mcsu_control_inputs = inputs[-1]
            inputs = inputs[:-2]

        if self.loss_type == "grad_ascent":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            loss = forget_loss

        elif self.loss_type == "grad_diff":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
            
            
        elif self.loss_type == "grad_diff_KL":
            forget_inputs, retain_inputs, normal_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1

            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss + retain_loss
            
            # Add KL term
            normal_input_ids, normal_labels, normal_attention_mask = normal_inputs
            current_outputs = model(normal_input_ids,labels=normal_labels, attention_mask=normal_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            
            with torch.no_grad():
                normal_outputs = self.oracle_model(
                    normal_input_ids.to(self.oracle_device),
                    labels=normal_labels.to(self.oracle_device),
                    attention_mask=normal_attention_mask.to(self.oracle_device),
                )
            normal_probs = F.log_softmax(normal_outputs.logits, dim=-1)
            normal_probs = normal_probs.view(-1, normal_outputs.logits.shape[-1])

            #minimum KL divergence
            normal_loss = nn.functional.kl_div(current_probs, normal_probs.to(current_probs.device), reduction='batchmean', log_target=True)
            loss += normal_loss
        
        elif self.loss_type == "npo":
            forget_inputs, retain_inputs = inputs

            forget_loss, outputs = compute_dpo_loss(
                model=model,
                ref_model=self.oracle_model,
                win_inputs=None,
                lose_inputs=forget_inputs,
                beta=self.beta,
            )
            retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

            loss = self.gamma * forget_loss + self.alpha * retain_loss
            
        elif self.loss_type == "KL":
            forget_inputs, retain_inputs = inputs
            input_ids, labels, attention_mask = forget_inputs
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_loss = outputs.loss
            forget_loss = forget_loss * -1
            
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            current_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])
            
            with torch.no_grad():
                retain_outputs = self.oracle_model(
                    retain_input_ids.to(self.oracle_device),
                    labels=retain_labels.to(self.oracle_device),
                    attention_mask=retain_attention_mask.to(self.oracle_device),
                )
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = nn.functional.kl_div(current_probs, retain_probs.to(current_probs.device), reduction='batchmean', log_target=True)
            loss = forget_loss + retain_loss

        elif self.loss_type == "idk":
            idk_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
            
            #concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((idk_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((idk_labels, retain_labels), dim=0)
            attention_mask = torch.cat((idk_attention_mask, retain_attention_mask), dim=0)
            
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        
        elif self.loss_type == "dpo":
            idk_inputs, forget_inputs, retain_inputs = inputs
            idk_input_ids, idk_labels, idk_attention_mask = idk_inputs
            forget_input_ids, forget_labels, forget_attention_mask = forget_inputs
            idk_outputs = model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids,labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(forget_input_ids,labels=forget_labels, attention_mask=forget_attention_mask)
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, labels)
            
            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, labels)


            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle

            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

            outputs = forget_outputs
        else:
            raise ValueError(f"Unsupported forget_loss: {self.loss_type}")

        if self.mcsu_enabled and float(self.mcsu_gamma) != 0.0:
            loss_mcsu = self.compute_mcsu_loss(model, mcsu_forget_inputs, mcsu_control_inputs)
            loss = loss + self.mcsu_gamma * loss_mcsu
            try:
                self.log({"loss_mcsu": loss_mcsu.detach().float().item()})
            except Exception:
                pass
        
        return (loss, outputs) if return_outputs else loss
        
    
    def compute_retain_loss(self, model, retain_inputs):
        retain_input_ids, retain_labels, retain_attention_mask = retain_inputs
        retain_outputs = model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
        retain_loss = 0.0
        retain_loss += retain_outputs.loss
        return retain_loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        args = self.args
        model = self._wrap_model(self.model, training=False, dataloader=None)
        print(self.is_in_train, args.device, model.dtype, self.args.dataloader_num_workers, self.eval_cfg.split_list, self.eval_cfg.split)
        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        model.eval()
        curr_step = self.state.global_step
        eval_cfg = self.eval_cfg

        curr_save_dir = os.path.join(eval_cfg.save_dir, f"checkpoint-{curr_step}")
        Path(curr_save_dir).mkdir(parents=True, exist_ok=True)
        forget_rate = eval_cfg.split.split('_')[0]
        with torch.no_grad():
            for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(eval_cfg.data_path, eval_cfg.split_list, eval_cfg.question_key, eval_cfg.answer_key, eval_cfg.eval_task, eval_cfg.base_answer_key, eval_cfg.perturbed_answer_key)):
                world_size = self.accelerator.num_processes

                # For some reason, Hydra is not interprating the split correctly
                if eval_task == 'eval_log_forget':
                    split = eval_cfg.split
                print(f'Working on eval task {eval_task} with split {split}')
                save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                save_filename = save_filename if world_size == 1 else os.path.join(curr_save_dir, f"{eval_task}_{self.accelerator.local_process_index}.json")
                # print(save_filename)
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
                    language=eval_cfg.language,
                )
                eval_dataloader = self.accelerator.prepare(eval_dataloader)
                base_eval_dataloader = self.accelerator.prepare(base_eval_dataloader)
                perturb_dataloader = self.accelerator.prepare(perturb_dataloader)
                normalize_gt = False 


                eval_logs = get_all_evals(eval_cfg, model, self.tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

                with open(save_filename, "w") as f:
                    # pretty write json to f
                    json.dump(eval_logs, f, indent=4)
            

            self.accelerator.wait_for_everyone()
            aggregated_eval_logs = {}
            for eval_task in eval_cfg.eval_task:
                #read the saved file as json and merge them using merge_dicts
                if world_size > 1:
                    if self.accelerator.is_local_main_process:
                        eval_logs = json.load(open(os.path.join(curr_save_dir, f"{eval_task}_0.json")))
                        for i in range(1, world_size):
                            filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                            eval_logs = merge_dicts(eval_logs, json.load(open(filename)))
                        
                        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

                        new_save_filename = os.path.join(curr_save_dir, f"{eval_task}.json")
                        with open(new_save_filename, "w") as f:
                            # pretty write json to f
                            json.dump(eval_logs, f, indent=4)

                            #delete old files use shutil

                            for i in range(world_size):
                                filename = os.path.join(curr_save_dir, f"{eval_task}_{i}.json")
                                os.remove(filename)
                                
            if self.accelerator.is_local_main_process:
                aggregated_eval_log_filename = os.path.join(curr_save_dir, "eval_log_aggregated.json")

                with open(aggregated_eval_log_filename, 'w') as f:
                    json.dump(aggregated_eval_logs, f, indent=4)

                if eval_cfg.retain_result is not None:
                    model_utility = get_model_utility(aggregated_eval_logs)
                    retain_result = json.load(open(eval_cfg.retain_result, 'r'))
                    forget_quality = get_forget_quality(aggregated_eval_logs, retain_result)
                    aggregate_stat = {**model_utility, **forget_quality}

                    # save aggregate_stat as csv
                    with open(os.path.join(curr_save_dir, "aggregate_stat.csv"), 'w') as csvfile:
                        field_names = list(aggregate_stat.keys())
                        writer = csv.DictWriter(csvfile, fieldnames=field_names)
                        writer.writeheader()
                        writer.writerow(aggregate_stat)


def compute_batch_nll(model, inputs):
    device = next(model.parameters()).device
    input_ids = inputs[0].to(device)
    labels = inputs[1].to(device)
    attention_mask = inputs[2].to(device)

    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits

    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    if win_inputs is None and lose_inputs is None:
        raise ValueError("Both win_inputs and lose_inputs can't be None")

    device = next(model.parameters()).device
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
            win_ref_loss = win_ref_loss.to(device)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
            lose_ref_loss = lose_ref_loss.to(device)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)



def custom_data_collator_forget(samples):
    forget_samples, retain_samples = [sample[0] for sample in samples], [sample[1] for sample in samples]
    rets = []
    for data_type in ["forget", "retain"]:
        data = forget_samples if data_type == "forget" else retain_samples
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def custom_data_collator_forget_kl(samples):
    forget_samples, retain_samples, normal_samples = [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]
    rets = []
    for data_type in ["forget", "retain", "normal"]:
        if data_type == "forget":
            data = forget_samples  
        elif data_type == "normal":
            data = normal_samples
        else:
            data = retain_samples
            
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))
    return rets


def custom_data_collator_forget_mcsu(samples):
    if len(samples[0]) < 4:
        raise ValueError(
            "MCSU samples must contain existing forget/retain inputs plus "
            "mcsu_forget_prompt_inputs and mcsu_control_prompt_inputs."
        )

    rets = []
    num_existing_inputs = len(samples[0]) - 2
    for input_pos in range(num_existing_inputs):
        data = [sample[input_pos] for sample in samples]
        input_ids = [s[0] for s in data]
        labels = [s[1] for s in data]
        attention_mask = [s[2] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)))

    for input_pos in [num_existing_inputs, num_existing_inputs + 1]:
        data = [sample[input_pos] for sample in samples]
        input_ids = [s[0] for s in data]
        attention_mask = [s[1] for s in data]
        rets.append((torch.stack(input_ids), torch.stack(attention_mask)))

    return rets

def compute_metrics(pred):
    logits, labels = torch.from_numpy(pred.predictions), torch.from_numpy(pred.label_ids)
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss  = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}

def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss
