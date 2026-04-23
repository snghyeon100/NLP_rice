# Qwen 2B Fine-tuning TODO (for future unlearning comparison)

## 0) 실험 원칙 결정
- [x] **입력 포맷을 지금부터 chat template 기반으로 통일**한다.
- [x] 이번 단계는 **fine-tuning까지만 수행**하고, unlearning은 나중에 수행한다.
- [x] Qwen 체크포인트를 **`Qwen/Qwen3.5-2B-Base`로 고정**한다.
- [x] 학습 방식은 **full fine-tuning + DeepSpeed ZeRO-3 (3x16GB GPU)**로 고정한다.

## 1) 설정 파일 수정
- [x] `config/model_config.yaml`에 `qwen3_5_2b` 블록 추가
- [x] `hf_key`를 `Qwen/Qwen3.5-2B-Base`로 설정
- [x] `flash_attention2`는 우선 `false`로 시작 (환경 안정성 우선)
- [x] `gradient_checkpointing`은 `true` 권장 (메모리 절약)
- [x] `ft_model_path`를 로컬 저장 경로로 지정 (예: `../scratch/tofu_ft_qwen3_5_2b`)
- [x] `config/finetune.yaml`의 `model_family`를 `qwen3_5_2b`로 변경
- [x] `config/finetune.yaml`의 `data_path`를 로컬 데이터셋 경로로 수정 (예: `./dataset/full_merged_all_10_lang`)
- [x] `config/finetune.yaml`의 `save_dir`를 실험용 경로로 지정
- [x] `finetune.py`에서 `TrainingArguments`의 `deepspeed='config/ds_config.json'` 활성화
- [ ] 필요 시 `config/ds_config.json`에서 offload를 CPU로 변경
- [ ] `offload_optimizer.device: cpu` 적용
- [ ] `offload_param.device: cpu` 적용 (OOM일 때만 우선 적용)

## 2) 데이터 포맷(핵심) 변경
- [x] `data_module.py`의 `convert_raw_data_to_model_format`를 `tokenizer.apply_chat_template` 기반으로 변경
- [x] 메시지 포맷을 `user(question) -> assistant(answer)`로 구성
- [x] **assistant 답변 토큰에만 loss가 걸리도록 label masking 유지**
- [ ] `max_length`/truncation/padding 처리 로직이 기존 학습 루프와 호환되는지 확인

## 3) 환경/의존성 확인
- [ ] 가상환경 활성화
- [ ] `pip install -r requirements.txt` 수행
- [ ] `transformers`가 Qwen3.5를 로드할 수 있는 버전인지 확인 (이 레포는 git main 설치를 사용)

## 4) 스모크 테스트
- [ ] 샘플 1~2개로 토크나이즈 결과 점검 (프롬프트/답변 경계, label 마스킹 확인)
- [ ] `torchrun --nproc_per_node=3 finetune.py`로 짧은 러닝 테스트
- [ ] 모델 로드/forward/backward/ZeRO 초기화 에러 없는지 확인
- [ ] OOM 시 `batch_size`↓, `gradient_accumulation_steps`↑ 조정
- [ ] 여전히 OOM이면 CPU offload(`offload_optimizer`/`offload_param`) 활성화

## 5) Full fine-tuning 실행
- [ ] `torchrun --nproc_per_node=3 finetune.py` 실행
- [ ] `save_dir`에 결과 생성 확인 (`cfg.yaml`, checkpoint, 최종 모델/토크나이저 파일)
- [ ] 사용한 최종 설정 파일 스냅샷 보관

## 6) 나중 unlearning 비교를 위한 재현성 확보
- [ ] **이번 FT artifact를 기준 체크포인트로 고정**하고 별도 백업
- [ ] 실험 메모에 base model 이름 기록
- [ ] 실험 메모에 dataset 경로/버전 기록
- [ ] 실험 메모에 seed/lr/epoch/batch 설정 기록
- [ ] 실험 메모에 commit hash(또는 코드 스냅샷) 기록
- [ ] 이후 unlearning 실험에서는 **초기 FT 체크포인트를 동일하게 유지**하고 방법론만 바꿔 비교

## Out of Scope (현재 단계)
- [ ] `forget.py` / unlearning 실행은 지금 하지 않음
