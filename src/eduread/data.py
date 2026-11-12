"""Read-only helpers for EduRead V3 raw manifests and processed tensors."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


REQUIRED_RAW_FIELDS = {
    "example_id", "category", "task_type", "equation",
    "prefix", "label", "image_filename", "image_path",
}
SUPPORTED_TASKS = {
    "mathematical_transcription",
    "accessible_explanation",
}
PROCESSED_TENSOR_FIELDS = {
    "input_ids", "token_type_ids", "attention_mask",
    "pixel_values", "labels",
}


def load_json_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON array of records without modifying the source file."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {source}")
    if not all(isinstance(item, dict) for item in records):
        raise ValueError(f"Every record in {source} must be an object")
    return records


def validate_raw_records(
    records: Iterable[Mapping[str, Any]],
    *,
    expected_count: Optional[int] = None,
) -> List[str]:
    """Return validation errors; never edits records or files."""
    errors: List[str] = []
    items = list(records)
    if expected_count is not None and len(items) != expected_count:
        errors.append(f"Expected {expected_count} records; found {len(items)}")

    seen_ids = set()
    for index, record in enumerate(items):
        missing = REQUIRED_RAW_FIELDS - set(record)
        if missing:
            errors.append(f"Record {index}: missing fields {sorted(missing)}")
            continue
        example_id = str(record["example_id"])
        if example_id in seen_ids:
            errors.append(f"Duplicate example_id: {example_id}")
        seen_ids.add(example_id)
        if record["task_type"] not in SUPPORTED_TASKS:
            errors.append(f"{example_id}: unsupported task_type {record['task_type']!r}")
        if not str(record["label"]).strip():
            errors.append(f"{example_id}: empty label")
        if not str(record["prefix"]).strip():
            errors.append(f"{example_id}: empty prompt prefix")
    return errors


def validate_processed_tensor(
    sample: Mapping[str, Any],
    *,
    expected_image_size: int = 448,
) -> List[str]:
    """Validate one loaded tensor mapping. Requires PyTorch at call time."""
    import torch

    errors = []
    missing = PROCESSED_TENSOR_FIELDS - set(sample)
    if missing:
        return [f"Missing processed fields: {sorted(missing)}"]

    sequence_keys = ("input_ids", "token_type_ids", "attention_mask", "labels")
    lengths = {key: int(torch.as_tensor(sample[key]).numel()) for key in sequence_keys}
    if len(set(lengths.values())) != 1:
        errors.append(f"Sequence length mismatch: {lengths}")

    pixels = torch.as_tensor(sample["pixel_values"])
    if tuple(pixels.shape) != (3, expected_image_size, expected_image_size):
        errors.append(f"Unexpected pixel_values shape: {tuple(pixels.shape)}")
    if not torch.is_floating_point(pixels):
        errors.append("pixel_values must be floating point")
    if not torch.isfinite(pixels).all().item():
        errors.append("pixel_values contains NaN or Inf")

    attention = torch.as_tensor(sample["attention_mask"])
    if not torch.isin(attention, torch.tensor([0, 1], dtype=attention.dtype)).all().item():
        errors.append("attention_mask contains values other than 0 and 1")

    labels = torch.as_tensor(sample["labels"])
    if not (labels != -100).any().item():
        errors.append("No supervised target tokens found")
    return errors


def load_processed_tensor(path: str | Path) -> Dict[str, Any]:
    """Load a processed tensor on CPU; caller controls the source path."""
    import torch

    return torch.load(Path(path), map_location="cpu", weights_only=True)


def resolve_image_path(
    project_root: str | Path,
    image_filename: str,
    *,
    preferred_subdirectory: Optional[str] = None,
) -> Optional[Path]:
    """Resolve an image under a project root without changing its location."""
    root = Path(project_root).resolve()
    candidates = []
    if preferred_subdirectory:
        candidates.append(root / preferred_subdirectory / image_filename)
    candidates.extend(root.rglob(image_filename))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.is_file():
            return resolved
    return None
