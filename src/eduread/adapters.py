"""PEFT LoRA adapter configuration and persistence helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import EduReadConfig


def attach_lora_adapter(model: Any, config: EduReadConfig | None = None) -> Any:
    """Prepare a quantized model and attach a fresh LoRA adapter.

    This initializes trainable adapter parameters only. It does not run
    forward/backward passes, optimizer steps, or save files.
    """
    from peft import (
        LoraConfig as PeftLoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    cfg = config or EduReadConfig()
    cfg.validate()
    if getattr(model, "is_loaded_in_4bit", False) is not True:
        raise ValueError("Expected a 4-bit quantized base model.")
    if hasattr(model, "peft_config"):
        raise ValueError("A PEFT adapter is already attached to this model.")

    prepared = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    lora = PeftLoraConfig(
        r=cfg.lora.rank,
        lora_alpha=cfg.lora.alpha,
        target_modules=list(cfg.lora.target_modules),
        lora_dropout=cfg.lora.dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )
    return get_peft_model(prepared, lora)


def save_adapter_and_processor(
    model: Any,
    processor: Any,
    destination: str | Path,
) -> Path:
    """Save adapter/processor to a caller-selected path.

    This writes model artifacts. It is deliberately not called automatically.
    """
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target, safe_serialization=True)
    processor.save_pretrained(target)
    return target


def load_adapter(
    base_model: Any,
    adapter_path: str | Path,
    *,
    is_trainable: bool = False,
) -> Any:
    """Attach an existing PEFT adapter to an already-loaded base model."""
    from peft import PeftModel

    return PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=is_trainable,
    )
