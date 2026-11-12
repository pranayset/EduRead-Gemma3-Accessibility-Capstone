"""Image-conditioned inference helpers for the audited V3 model."""
from __future__ import annotations

from typing import Any, Dict

import torch


def prompt_inputs_from_processed_sample(
    sample: Dict[str, Any],
    *,
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    """Remove supervised answer tokens and prepare the stored prompt tensors."""
    labels = torch.as_tensor(sample["labels"]).reshape(1, -1)
    input_ids = torch.as_tensor(sample["input_ids"]).reshape(1, -1)
    token_types = torch.as_tensor(sample["token_type_ids"]).reshape(1, -1)
    attention = torch.as_tensor(sample["attention_mask"]).reshape(1, -1)
    pixels = torch.as_tensor(sample["pixel_values"])

    answer_positions = torch.where(labels[0] != -100)[0]
    if answer_positions.numel() == 0:
        raise ValueError("Processed sample has no supervised target boundary.")
    boundary = int(answer_positions[0].item())

    if pixels.ndim == 3:
        pixels = pixels.unsqueeze(0)
    if pixels.ndim != 4:
        raise ValueError(f"Expected batched pixel tensor, got {tuple(pixels.shape)}")

    return {
        "input_ids": input_ids[:, :boundary].to(device),
        "token_type_ids": token_types[:, :boundary].to(device),
        "attention_mask": attention[:, :boundary].to(device),
        "pixel_values": pixels.to(device),
    }


def generate_from_processed_sample(
    model: Any,
    processor: Any,
    sample: Dict[str, Any],
    *,
    max_new_tokens: int = 64,
    device: str | torch.device = "cuda",
) -> Dict[str, Any]:
    """Run greedy generation from a processed V3 sample.

    This performs inference only; it never updates model parameters.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    model.eval()
    inputs = prompt_inputs_from_processed_sample(sample, device=device)

    autocast_enabled = str(device).startswith("cuda") and torch.cuda.is_available()
    with torch.inference_mode():
        if autocast_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
        else:
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = generated[:, prompt_length:]
    text = processor.tokenizer.decode(
        new_tokens[0].tolist(),
        skip_special_tokens=True,
    ).strip()
    eos_id = processor.tokenizer.eos_token_id
    eos_reached = bool(
        new_tokens.shape[1] and eos_id is not None
        and int(new_tokens[0, -1].item()) == eos_id
    )
    return {
        "example_id": sample.get("example_id"),
        "task_type": sample.get("task_type"),
        "image_filename": sample.get("image_filename"),
        "generated_text": text,
        "generated_tokens": int(new_tokens.shape[1]),
        "eos_reached": eos_reached,
    }
