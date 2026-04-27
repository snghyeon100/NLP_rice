# Multilingual Concept Subspace Unlearning

MCSU is an optional additive unlearning loss for this repository. It estimates a shared multilingual hidden-state subspace for the forget concept, then penalizes the current model when the forget-control prompt difference still lies in that subspace.

It differs from the naive multilingual baselines because the existing losses optimize output behavior on forget/retain data, while MCSU explicitly suppresses a hidden-state direction shared across languages.

## Build A Subspace

```bash
python mcsu_subspace.py \
  --config-name mcsu_subspace \
  model_family=qwen3.5-2B \
  model_path=Qwen/Qwen3.5-2B \
  output_dir=./runs/qwen_mcsu_subspace \
  train_languages='[en,ko,ja,fr]' \
  k_lang=16 \
  k_shared=8
```

For local translated datasets, set `translated_data_root`, `forget_template`, and `retain_template`. The script saves:

- `subspaces.pt`: `U_shared`, per-language bases, singular values, eigenvalues, and resolved config
- `subspace_diagnostics.json`: example counts, top singular/eigenvalues, and pairwise cross-lingual subspace overlap

## Train Baselines

MCSU is disabled by default in `config/forget.yaml`.

```bash
python forget.py \
  model_family=qwen3.5-2B \
  model_path=Qwen/Qwen3.5-2B \
  forget_loss=npo \
  mcsu.enabled=false \
  save_dir=./runs/qwen_naive_npo
```

## Train With MCSU

```bash
python forget.py \
  model_family=qwen3.5-2B \
  model_path=Qwen/Qwen3.5-2B \
  forget_loss=npo \
  mcsu.enabled=true \
  mcsu.subspace_path=./runs/qwen_mcsu_subspace/subspaces.pt \
  mcsu.gamma=1.0 \
  save_dir=./runs/qwen_mcsu_npo_gamma1
```

`mcsu.gamma=0` skips the MCSU hidden-state forward pass and is equivalent to the naive baseline.

## W&B Logging

Training uses the Hugging Face Trainer W&B integration. The default W&B entity is `changwoolabs`; enable logging with:

```bash
python forget.py \
  model_family=qwen3.5-2B \
  model_path=Qwen/Qwen3.5-2B \
  forget_loss=npo \
  mcsu.enabled=true \
  mcsu.subspace_path=./runs/qwen_mcsu_subspace/subspaces.pt \
  wandb.enabled=true \
  wandb.project=multilingual-amnesia \
  wandb.name=qwen_mcsu_npo_gamma1 \
  save_dir=./runs/qwen_mcsu_npo_gamma1
```

Useful overrides:

```bash
wandb.entity=changwoolabs
wandb.group=qwen-mcsu
wandb.tags='[qwen,npo,mcsu]'
wandb.mode=online
```

When MCSU is enabled, `loss_mcsu` is logged along with the normal Trainer metrics. Disable W&B with `wandb.enabled=false`.

Evaluation logging is separate because `evaluate_util.py` is a standalone script. Enable it with the same W&B fields:

```bash
python evaluate_util.py \
  model_family=qwen3.5-2B \
  model_path=./runs/qwen_mcsu_npo_gamma1 \
  save_dir=./runs/qwen_mcsu_npo_gamma1/eval_results \
  batch_size=1 \
  wandb.enabled=true \
  wandb.entity=changwoolabs \
  wandb.project=multilingual-amnesia \
  wandb.name=qwen_mcsu_npo_gamma1_eval \
  wandb.group=qwen-mcsu
```

The evaluation script logs scalar summaries such as mean losses, truth ratios, ROUGE/BLEU/chrF fields, and uploads the JSON outputs as a W&B artifact.

For a compact comparison view after running both MCSU and baseline, use:

```bash
python mcsu_report.py \
  --mcsu_eval ./runs/qwen_mcsu_npo_gamma1/eval_results/eval_log_aggregated.json \
  --baseline_eval ./runs/qwen_baseline_npo/eval_results/eval_log_aggregated.json \
  --mcsu_projection ./runs/qwen_mcsu_npo_gamma1/projection_diag.json \
  --baseline_projection ./runs/qwen_baseline_npo/projection_diag.json \
  --output_dir ./runs/qwen_mcsu_report \
  --wandb_enabled true \
  --wandb_entity changwoolabs \
  --wandb_project multilingual-amnesia \
  --wandb_name qwen_mcsu_baseline_summary \
  --wandb_group qwen-mcsu
```

This creates `summary.json`, `eval_summary.csv`, and `projection_summary.csv`, and logs only compact aggregate metrics plus W&B tables/artifacts.

## Projection Diagnostics

```bash
python eval_mcsu_projection.py \
  --model_family qwen3.5-2B \
  --model_path ./runs/qwen_mcsu_npo_gamma1 \
  --subspace_path ./runs/qwen_mcsu_subspace/subspaces.pt \
  --output_path ./runs/qwen_mcsu_npo_gamma1/projection_diag.json \
  --wandb_enabled true \
  --wandb_entity changwoolabs \
  --wandb_project multilingual-amnesia \
  --wandb_name qwen_mcsu_npo_gamma1_projection
```

The diagnostic reports mean/std projection energy by language and layer. An MCSU model should reduce this energy more than the naive multilingual unlearning baseline.

## Evaluation

The original `evaluate_util.py` metrics are still available for forget quality, retain utility, probability, truth ratio, and generation metrics. W&B is disabled by default for evaluation; set `wandb.enabled=true` to log scalar summaries and JSON artifacts. MCSU also adds the optional projection diagnostic.

## Memory Notes

Both subspace construction and the MCSU loss call the model with `output_hidden_states=True`, which is memory-heavy. Start with small values such as:

```yaml
batch_size: 1
prompt_max_length: 128
layer_ids: [8, 16]
```

Then increase batch size, prompt length, and layer count once memory use is stable.
