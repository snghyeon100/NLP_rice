"""Loss functions for RCP-SLoRA unlearning.

Stage 1: Retain-Conflict Projected NPO gradient computation.

핵심 아이디어:
  - forget_loss (NPO)와 retain_loss (CE)의 gradient를 별도로 계산
  - 두 gradient가 충돌(dot < 0)하면 retain 방향 성분을 forget gradient에서 제거
  - retain 예시는 직접적인 학습 target이 아니라 gradient 충돌 감지 용도

기존 unlearn_npo/loss.py에서 복사 후 독립적으로 관리.
"""

import torch
import torch.nn.functional as F
from torch import nn


# ──────────────────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────────────────

def _model_device(model):
    return next(model.parameters()).device


def _batch_to_device(inputs, device):
    return tuple(tensor.to(device) for tensor in inputs)


# ──────────────────────────────────────────────────────────
# NLL / NPO (unlearn_npo/loss.py와 동일, 독립 복사)
# ──────────────────────────────────────────────────────────

def compute_batch_nll(model, inputs):
    """각 시퀀스의 NLL (sum over tokens)을 반환."""
    device = _model_device(model)
    input_ids, labels, attention_mask = _batch_to_device(inputs, device)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    logits = outputs.logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fn(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss, outputs


def compute_dpo_loss(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    """NPO loss: DPO with win_inputs=None (forget-only variant)."""
    if ref_model is None:
        raise ValueError("RCP-SLoRA requires an oracle/reference model.")

    device = _model_device(model)
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        with torch.no_grad():
            win_ref_loss, _ = compute_batch_nll(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss.to(device))

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        with torch.no_grad():
            lose_ref_loss, _ = compute_batch_nll(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss.to(device))

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


# ──────────────────────────────────────────────────────────
# Retain CE loss
# ──────────────────────────────────────────────────────────

def compute_retain_loss(model, retain_inputs):
    """retain 예시에 대한 cross-entropy loss (scalar)."""
    device = _model_device(model)
    input_ids, labels, attention_mask = _batch_to_device(retain_inputs, device)
    outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
    return outputs.loss


# ──────────────────────────────────────────────────────────
# RCP Gradient Projection (핵심)
# ──────────────────────────────────────────────────────────

def compute_rcp_gradients(model, oracle_model, inputs, cfg):
    """RCP gradient projection을 수행하고 projected gradient 목록과 로그 딕셔너리를 반환.

    Args:
        model:        학습 중인 모델 (LoRA 부착, trainable)
        oracle_model: reference 모델 (frozen)
        inputs:       [forget_inputs, retain_inputs] (collator 출력 형식)
        cfg:          Hydra config (beta, projection_lambda, projection_eps)

    Returns:
        projected_grads: list[Tensor], LoRA trainable params 순서와 동일
        log_dict:        dict with scalar log values
    """
    forget_inputs, retain_inputs = inputs
    eps = float(cfg.projection_eps)

    # ── 1. Forget loss (NPO) ──────────────────────────────
    forget_loss, _ = compute_dpo_loss(
        model=model,
        ref_model=oracle_model,
        win_inputs=None,
        lose_inputs=forget_inputs,
        beta=cfg.beta,
    )

    # ── 2. Retain loss (CE) ───────────────────────────────
    retain_loss = compute_retain_loss(model, retain_inputs)

    # ── 3. Trainable parameters (LoRA params only) ────────
    params = [p for p in model.parameters() if p.requires_grad]

    # ── 4. Forget gradient ────────────────────────────────
    g_f = torch.autograd.grad(
        forget_loss,
        params,
        retain_graph=True,   # retain_graph=True: retain_loss 계산 전 그래프 유지
        create_graph=False,
        allow_unused=True,
    )

    # ── 5. Retain gradient ────────────────────────────────
    g_r = torch.autograd.grad(
        retain_loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    # None gradient → zero tensor
    g_f = [torch.zeros_like(p) if g is None else g for g, p in zip(g_f, params)]
    g_r = [torch.zeros_like(p) if g is None else g for g, p in zip(g_r, params)]

    # ── 6. Conflict projection ────────────────────────────
    dot   = sum((gf * gr).sum() for gf, gr in zip(g_f, g_r))
    norm_f = sum((gf * gf).sum() for gf in g_f) + eps
    norm_r = sum((gr * gr).sum() for gr in g_r) + eps

    if dot < 0:
        # retain 방향 성분을 forget gradient에서 제거
        projected = [
            gf - cfg.projection_lambda * (dot / norm_r) * gr
            for gf, gr in zip(g_f, g_r)
        ]
        projection_applied = True
    else:
        # 충돌 없음 → forget gradient 그대로 사용
        projected = list(g_f)
        projection_applied = False

    # ── 7. Retain 강화 (선택 사항) ──────────────────────────
    alpha = getattr(cfg, "alpha", 0.0)
    if alpha > 0.0:
        projected = [p + alpha * gr for p, gr in zip(projected, g_r)]

    # projection_ratio: g_f와 g_r의 cosine similarity 절댓값
    projection_ratio = torch.abs(dot) / (torch.sqrt(norm_f) * torch.sqrt(norm_r) + eps)

    log_dict = {
        "forget_loss":       forget_loss.detach().item(),
        "retain_loss":       retain_loss.detach().item(),
        "grad_dot":          dot.detach().item(),
        "grad_norm_forget":  (norm_f - eps).clamp(min=0.0).sqrt().item(),
        "grad_norm_retain":  (norm_r - eps).clamp(min=0.0).sqrt().item(),
        "projection_applied": float(projection_applied),
        "projection_ratio":  projection_ratio.detach().item(),
    }

    return projected, log_dict
