# NLP_rice Unlearning 실험 구조

이 레포는 TOFU QA 데이터로 fine-tuning된 모델에 여러 unlearning 방법론을 적용하고, 각 방법론을 독립적으로 실험하기 위한 코드베이스입니다.

핵심 구조는 다음과 같습니다.

```text
NLP_rice/
  config/                         # fine-tuning/evaluation용 Hydra config
  unlearning_methods/
    unlearn_npo/                  # NPO 계열 unlearning 파이프라인
      config.yaml
      dataloader.py
      loss.py
      train.py
    unlearn_grad_diff_kl/         # Grad-Diff + KL 계열 unlearning 파이프라인
      config.yaml
      dataloader.py
      loss.py
      train.py
  data_module.py                  # TOFU QA 포맷/평가용 공통 데이터 유틸
  dataloader.py                   # fine-tuning용 Trainer 유틸
  finetune.py                     # TOFU fine-tuning 실행 스크립트
  evaluate_finetune.py            # fine-tuned/base 모델 평가 스크립트
  evaluate_util.py                # unlearning 평가 스크립트
  aggregate_eval_stat.py          # 평가 결과 집계 유틸
  utils.py                        # 모델 저장/로그 등 공통 유틸
```

## 기본 원칙

각 unlearning 방법론은 `unlearning_methods/unlearn_<method_name>/` 아래에서 end-to-end로 실행되도록 구성합니다.

즉, 새로운 방법론을 추가할 때는 공통 unlearning 실행 스크립트에 끼워 넣는 방식이 아니라, 해당 방법론 디렉토리 안에 데이터 로딩, loss, 학습 실행 코드를 모두 둡니다. 방법론끼리 중복되는 코드가 일부 생기더라도, 빠르게 실험하고 수정할 수 있도록 독립성을 우선합니다.

각 방법론 디렉토리는 아래 파일 구성을 따릅니다.

```text
unlearning_methods/unlearn_<method_name>/
  config.yaml     # 해당 방법론의 Hydra config
  dataloader.py   # 해당 방법론에서 사용할 Dataset/Collator
  loss.py         # 해당 방법론의 objective/loss
  train.py        # 모델 로드, dataloader 생성, 학습, 평가, 저장까지 수행
```

## 현재 구현된 방법론

### `unlearn_npo`

NPO 계열 objective를 사용하는 unlearning 파이프라인입니다.

- forget set과 retain set을 함께 사용합니다.
- fine-tuned 모델을 unlearning 대상 모델로 불러옵니다.
- oracle/reference 모델을 함께 불러와 forget loss 계산에 사용합니다.
- `unlearning_methods/unlearn_npo/train.py` 하나로 모델 로드, dataloader 생성, 학습, 평가, 저장을 수행합니다.

실행:

```bash
cd /Users/woojin/Desktop/26-1/COSE461/project/NLP_rice
python unlearning_methods/unlearn_npo/train.py
```

Hydra override 예시:

```bash
python unlearning_methods/unlearn_npo/train.py model_path=./finetuned language=en split=forget01
```

주요 설정 파일:

```text
unlearning_methods/unlearn_npo/config.yaml
```

### `unlearn_grad_diff_kl`

Grad-Diff objective와 KL term을 함께 사용하는 unlearning 파이프라인입니다.

- forget set, retain set, normal data를 함께 사용합니다.
- forget loss는 낮추고 retain loss는 유지하도록 구성합니다.
- normal data에 대해서는 oracle/reference 모델과의 KL divergence를 사용합니다.
- `unlearning_methods/unlearn_grad_diff_kl/train.py` 하나로 모델 로드, dataloader 생성, 학습, 평가, 저장을 수행합니다.

실행:

```bash
cd /Users/woojin/Desktop/26-1/COSE461/project/NLP_rice
python unlearning_methods/unlearn_grad_diff_kl/train.py
```

Hydra override 예시:

```bash
python unlearning_methods/unlearn_grad_diff_kl/train.py model_path=./finetuned language=en split=forget01
```

주요 설정 파일:

```text
unlearning_methods/unlearn_grad_diff_kl/config.yaml
```

## Config 사용 방식

각 unlearning 방법론은 자기 디렉토리의 `config.yaml`을 기본 설정으로 사용합니다.

중요한 공통 필드는 다음과 같습니다.

```yaml
model_path: ./finetuned
save_dir: ${model_path}/${method_name}_${lr}_${split}_${num_epochs}_${language}
split: forget01
language: en
batch_size: 1
gradient_accumulation_steps: 4
num_epochs: 5
lr: 2e-5
```

`model_path: ./finetuned`는 `NLP_rice` 루트 기준 상대 경로입니다. 따라서 `train.py`를 어느 위치에서 실행하더라도 기본적으로 `NLP_rice/finetuned`를 바라보도록 처리합니다.

`save_dir`도 `model_path` 기준으로 만들어집니다. 기본 설정을 그대로 쓰면 unlearning 결과는 `NLP_rice/finetuned/...` 아래에 저장됩니다.

실행 시 config 값을 바꾸고 싶으면 Hydra override를 사용합니다.

```bash
python unlearning_methods/unlearn_npo/train.py batch_size=1 gradient_accumulation_steps=8 lr=1e-5
```

## 새 unlearning 방법론 추가 방법

새 방법론을 추가할 때는 `unlearning_methods/` 아래에 새 디렉토리를 만듭니다.

예를 들어 `abc`라는 방법론을 추가한다면 다음 구조를 만듭니다.

```text
unlearning_methods/
  unlearn_abc/
    __init__.py
    config.yaml
    dataloader.py
    loss.py
    train.py
```

구현 순서는 보통 다음과 같습니다.

1. 기존 방법론 디렉토리 중 가장 가까운 구조를 복사합니다.
2. `config.yaml`에서 method name, 데이터 경로, batch size, loss coefficient, save path를 정리합니다.
3. `dataloader.py`에서 해당 방법론이 요구하는 batch 형식을 만듭니다.
4. `loss.py`에서 batch를 받아 objective를 계산하는 함수를 구현합니다.
5. `train.py`에서 모델 로드, dataloader 생성, optimizer/trainer 구성, 저장 로직을 연결합니다.
6. 아래처럼 단일 명령으로 실행 가능해야 합니다.

```bash
python unlearning_methods/unlearn_abc/train.py
```

새 방법론의 `train.py`는 최소한 다음 역할을 포함해야 합니다.

- project root 기준 상대 경로 해석
- fine-tuned 모델 로드
- 필요하면 oracle/reference/base 모델 로드
- method-local dataloader 생성
- method-local loss 적용
- 학습 중 checkpoint 저장
- 최종 모델 저장
- 필요하면 retain/forget 평가 실행

## Fine-tuning과 평가

Unlearning 전 fine-tuned 모델을 만들 때는 루트의 `finetune.py`를 사용합니다.

```bash
python finetune.py
```

Fine-tuned 모델과 base 모델을 비교 평가할 때는 `evaluate_finetune.py`를 사용합니다.

```bash
python evaluate_finetune.py model_path=./finetuned
```

평가 config는 `config/eval_finetune.yaml`을 기본으로 사용합니다. 전체 언어를 한 번에 평가하도록 구성되어 있으며, truth ratio와 exact match 계열 지표를 함께 확인할 수 있습니다.

## Unlearning 평가

Unlearning이 적용된 모델의 retain/forget 성능은 루트의 `evaluate_util.py`로 평가합니다. 평가 config는 `config/eval.yaml` 하나를 기준으로 사용하며, 기본값은 전체 언어 평가입니다.

```bash
python evaluate_util.py model_path=./finetuned/subspace_xlingual_2e-05_forget01_5_ko
```

특정 언어만 평가하려면 `languages`를 override합니다.

```bash
python evaluate_util.py \
  model_path=./finetuned/subspace_xlingual_2e-05_forget01_5_ko \
  'languages=[en,ko,fr]'
```

단일 언어 평가도 같은 config로 처리합니다.

```bash
python evaluate_util.py \
  model_path=./finetuned/subspace_xlingual_2e-05_forget01_5_ko \
  'languages=[ko]'
```

다국어 평가 결과는 기본적으로 아래 구조로 저장됩니다.

```text
{model_path}/eval_results/multilingual_ds_sizeNone/
  en/
    eval_log.json
    eval_log_forget.json
    eval_log_aggregated.json
  ko/
    eval_log.json
    eval_log_forget.json
    eval_log_aggregated.json
  multilingual_aggregated.json
```

`Forget Quality`까지 계산하려면 99% retain 모델의 평가 결과 JSON이 필요합니다. 이때 필요한 것은 checkpoint가 아니라 retain 모델을 같은 eval pipeline으로 평가해 만든 `eval_log_aggregated.json`입니다. 경로는 `retain_result`, `retain_result_by_language`, 또는 `retain_result_template`으로 넘길 수 있습니다.

## Legacy config 주의

`config/forget*.yaml` 계열 파일은 이전 구조에서 사용하던 legacy config입니다. 현재 unlearning 실행의 기준은 각 방법론 디렉토리 안의 `config.yaml`입니다. Unlearning 평가는 언어별 `eval_<language>.yaml` 대신 `config/eval.yaml`로 통합되어 있습니다.

새로운 unlearning 실험을 추가하거나 수정할 때는 `unlearning_methods/unlearn_<method_name>/config.yaml`을 먼저 수정하세요.
