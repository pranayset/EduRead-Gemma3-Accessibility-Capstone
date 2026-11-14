"""Read-only checks for tensors that were already preprocessed in V3.

This module intentionally contains no image processor calls and no save
operations: it must not regenerate or overwrite the audited .pt tensors.
"""

import torch
from .dataset import REQUIRED_SAMPLE_KEYS


def validate_preprocessed_sample(sample: dict) -> dict:
    missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in sample]
    if missing:
        raise ValueError(f"Sample is missing required fields: {missing}")

    input_ids = sample["input_ids"]
    labels = sample["labels"]
    token_type_ids = sample["token_type_ids"]
    attention_mask = sample["attention_mask"]
    pixel_values = sample["pixel_values"]

    for name, value in (
        ("input_ids", input_ids),
        ("labels", labels),
        ("token_type_ids", token_type_ids),
        ("attention_mask", attention_mask),
        ("pixel_values", pixel_values),
    ):
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a torch.Tensor")

    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1-D; got {tuple(input_ids.shape)}")
    if labels.shape != input_ids.shape:
        raise ValueError("labels and input_ids must have identical shapes")
    if token_type_ids.shape != input_ids.shape:
        raise ValueError("token_type_ids and input_ids must have identical shapes")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask and input_ids must have identical shapes")
    if tuple(pixel_values.shape) != (3, 448, 448):
        raise ValueError(
            f"pixel_values must have shape (3, 448, 448); got {tuple(pixel_values.shape)}"
        )
    if not torch.isfinite(pixel_values).all().item():
        raise ValueError("pixel_values contains NaN or Inf")

    shifted_supervised_tokens = int((labels[1:] != -100).sum().item())
    if shifted_supervised_tokens <= 0:
        raise ValueError("Sample has no shifted supervised tokens")

    metadata_tokens = int(sample["supervised_tokens"])
    return {
        "example_id": str(sample["example_id"]),
        "task_type": str(sample["task_type"]),
        "input_length": int(input_ids.numel()),
        "supervised_tokens_metadata": metadata_tokens,
        "shifted_supervised_tokens": shifted_supervised_tokens,
        "metadata_matches_shifted": metadata_tokens == shifted_supervised_tokens,
        "pixel_shape": tuple(pixel_values.shape),
    }
