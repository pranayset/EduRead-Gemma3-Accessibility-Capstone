"""Explicit PaliGemma V3 processor and 4-bit base-model loaders.

Nothing is loaded at import time. Call the functions deliberately from a
GPU-enabled environment after accepting the model's Hugging Face terms.
"""
from __future__ import annotations

from typing import Any

from .config import EduReadConfig


def build_bitsandbytes_config(config: EduReadConfig | None = None) -> Any:
    """Create the audited NF4 BitsAndBytesConfig; does not load model weights."""
    import torch
    from transformers import BitsAndBytesConfig

    cfg = config or EduReadConfig()
    cfg.validate()
    return BitsAndBytesConfig(
        load_in_4bit=cfg.quantization.load_in_4bit,
        load_in_8bit=cfg.quantization.load_in_8bit,
        bnb_4bit_quant_type=cfg.quantization.quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=cfg.quantization.double_quant,
    )


def load_processor(config: EduReadConfig | None = None, **kwargs: Any) -> Any:
    """Load the PaliGemma processor. This downloads/reads processor assets."""
    from transformers import AutoProcessor

    cfg = config or EduReadConfig()
    cfg.validate()
    return AutoProcessor.from_pretrained(cfg.model.model_id, **kwargs)


def load_quantized_base_model(
    config: EduReadConfig | None = None,
    *,
    require_cuda: bool = True,
    **kwargs: Any,
) -> Any:
    """Load the PaliGemma 3B NF4 base model; requires an explicit call."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration

    cfg = config or EduReadConfig()
    cfg.validate()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the audited V3 loading workflow.")

    return PaliGemmaForConditionalGeneration.from_pretrained(
        cfg.model.model_id,
        quantization_config=build_bitsandbytes_config(cfg),
        torch_dtype=torch.float16,
        device_map=cfg.model.device_map,
        attn_implementation=cfg.model.attention_implementation,
        **kwargs,
    )
