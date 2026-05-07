# unlearn_wj

`unlearn_wj`는 TOFU finetuned model에서 source-language forget answer likelihood를 직접 낮추고, multilingual retain/utility CE와 KL로 보존하는 zero-shot cross-lingual unlearning 실험 코드다.

기존 `wj2`처럼 probe만 지우는 방식이 아니라, 평가 지표인 `Prob. Forget`과 `Truth Ratio Forget`이 실제로 움직이도록 token likelihood objective를 학습한다.

## 핵심 구성

- `train.py`: end-to-end 학습 실행
- `config.yaml`: 데이터, loss weight, LoRA, layer selection 설정
- `dataloader.py`: source forget, source retain, multilingual retain/utility batch 구성
- `loss.py`: NPO / negative CE forget loss, retain/utility CE, masked KL
- `localization.py`: manual, middle, hidden-alignment layer selection
- `merge_lora.py`: 저장된 LoRA adapter를 base model에 merge

## 기본 실행

```bash
CUDA_VISIBLE_DEVICES=4,5 python unlearning_methods/unlearn_wj/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./finetuned/wj_npo_forget01_en
```

`CUDA_VISIBLE_DEVICES=4,5`처럼 지정한 경우 config의 `gpu_train: 0`, `gpu_ref: 1`은 visible device 기준이다.

## 빠른 smoke run

```bash
CUDA_VISIBLE_DEVICES=4,5 python unlearning_methods/unlearn_wj/train.py \
  model_path=./finetuned/finetuned_100 \
  save_dir=./finetuned/wj_smoke \
  max_steps=3 \
  batch_size=1 \
  layer_selection.top_k=2 \
  log_steps=1 \
  overwrite_dir=true
```

## Layer Selection

기본값은 `middle`이다. 모델의 중간 layer 중 `top_k`개에만 LoRA를 붙인다.

```yaml
layer_selection:
  strategy: middle
  top_k: 12
  min_layer: 4
  max_layer: null
```

직접 지정하려면:

```bash
python unlearning_methods/unlearn_wj/train.py \
  layer_selection.strategy=manual \
  layer_selection.selected_layers=[8,9,10,11,12,13]
```

source-target hidden alignment로 고르려면:

```bash
python unlearning_methods/unlearn_wj/train.py \
  layer_selection.strategy=hidden_alignment \
  layer_selection.hidden_alignment_batches=8
```

선택 결과는 `<save_dir>/localization/selected_layers.json`과 `layer_scores.json`에 저장된다.

## Loss

기본 objective:

```text
L =
    w_forget * L_forget_source
  + w_retain_source * CE_retain_source
  + w_retain_multi * CE_retain_multi
  + w_utility_multi * CE_utility_multi
  + w_kl * KL(ref || model)
```

`forget_objective`는 아래 중 선택한다.

- `npo`: 기본값. reference model 대비 forget answer likelihood를 낮춘다.
- `neg_ce`: forget answer CE에 음수를 붙인다.
- `grad_diff`: 현재 구현에서는 `neg_ce`와 동일한 direct negative CE ablation이다.

## 산출물

학습 후 `<save_dir>` 아래에 다음 파일이 생긴다.

- `config_resolved.yaml`
- `localization/selected_layers.json`
- `localization/layer_scores.json`
- `training_loss_history.jsonl`
- `trainable_parameters.json`
- `lora_adapter/adapter_model.safetensors`

## 평가

LoRA adapter만 저장되므로 `evaluate_util.py`로 평가하려면 먼저 merge한다.

```bash
python unlearning_methods/unlearn_wj/merge_lora.py \
  --base_model ./finetuned/finetuned_100 \
  --adapter ./finetuned/wj_npo_forget01_en/lora_adapter \
  --output ./finetuned/wj_npo_forget01_en/merged
```

그 다음 전체 언어 평가:

```bash
CUDA_VISIBLE_DEVICES=4 python evaluate_util.py \
  model_path=./finetuned/wj_npo_forget01_en/merged \
  languages=[en,ar,fa,fr,hi,id,iw,ja,ko,ru] \
  batch_size=1 \
  perturb_eval_chunk_size=1 \
  save_dir=./finetuned/wj_npo_forget01_en/eval_full \
  save_raw_logs=false \
  save_case_studies=true
```

우선 봐야 하는 값은 `Prob. Forget` 감소, `Truth Ratio Forget` 증가, `Prob. Retain` 유지, `Model Utility` 유지다.
