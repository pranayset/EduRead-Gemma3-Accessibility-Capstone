"""Explicit training setup helpers for EduRead V3.

Importing this module does not train a model. Optimizer/scheduler builders only
initialize objects; optimizer.step(), scheduler.step(), and backward() are
never called by these helpers.
"""
from __future__ import annotations

from typing import Any, Iterable, Tuple

from .config import EduReadConfig


def collect_trainable_parameters(model: Any) -> list:
    """Return parameters marked trainable by PEFT."""
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def build_optimizer(
    model: Any,
    config: EduReadConfig | None = None,
) -> Any:
    """Initialize the V3 PagedAdamW8bit optimizer without stepping it."""
    import bitsandbytes as bnb

    cfg = config or EduReadConfig()
    cfg.validate()
    parameters = collect_trainable_parameters(model)
    if not parameters:
        raise ValueError("Model has no trainable parameters.")
    return bnb.optim.PagedAdamW8bit(
        parameters,
        lr=cfg.training.learning_rate,
        betas=cfg.training.adam_betas,
        eps=cfg.training.adam_epsilon,
        weight_decay=cfg.training.weight_decay,
        min_8bit_size=4096,
    )


def build_scheduler(
    optimizer: Any,
    config: EduReadConfig | None = None,
) -> Any:
    """Initialize the audited linear warmup/decay scheduler; does not step."""
    from transformers import get_linear_schedule_with_warmup

    cfg = config or EduReadConfig()
    cfg.validate()
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=cfg.training.total_optimizer_updates,
    )


def count_supervised_tokens(batch: Any) -> int:
    """Count shifted supervised labels as used by the V3 loss accounting."""
    import torch

    labels = torch.as_tensor(batch["labels"])
    if labels.ndim < 2:
        labels = labels.unsqueeze(0)
    return int((labels[..., 1:] != -100).sum().item())
