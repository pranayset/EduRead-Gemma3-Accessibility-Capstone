"""Read-only access to the already-processed EduRead V3 tensor dataset."""

from pathlib import Path
from collections.abc import Sequence
import json
import torch
from torch.utils.data import Dataset


REQUIRED_SAMPLE_KEYS = (
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "pixel_values",
    "labels",
    "example_id",
    "task_type",
    "image_filename",
    "supervised_tokens",
)


def load_processed_index(index_path: str | Path) -> list[dict]:
    """Read an existing JSON index. This function never writes to the index."""
    index_path = Path(index_path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Processed index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise TypeError(f"Expected a JSON list in {index_path}")
    if not all(isinstance(record, dict) for record in records):
        raise TypeError(f"Every index entry must be an object: {index_path}")
    return records


class EduReadProcessedDatasetV3(Dataset):
    """Lazy dataset: each __getitem__ reads one existing .pt file on CPU."""

    def __init__(
        self,
        records: Sequence[dict],
        processed_dir: str | Path | None = None,
        validate_keys: bool = True,
    ):
        self.records = list(records)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.validate_keys = validate_keys

    def __len__(self) -> int:
        return len(self.records)

    def _tensor_path(self, record: dict) -> Path:
        if "processed_path" not in record:
            raise KeyError("Index record is missing 'processed_path'.")

        path = Path(record["processed_path"]).expanduser()
        # Preserve absolute paths from the audited index. Resolve only relative paths.
        if not path.is_absolute():
            if self.processed_dir is None:
                raise ValueError(
                    "Relative processed_path requires processed_dir to be supplied."
                )
            path = self.processed_dir / path
        return path

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        tensor_path = self._tensor_path(record)
        if not tensor_path.is_file():
            raise FileNotFoundError(f"Missing processed tensor: {tensor_path}")

        sample = torch.load(
            tensor_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(sample, dict):
            raise TypeError(f"Expected a dictionary tensor sample: {tensor_path}")

        if self.validate_keys:
            missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in sample]
            if missing:
                raise KeyError(
                    f"Missing keys {missing} in processed tensor {tensor_path}"
                )
        return sample
