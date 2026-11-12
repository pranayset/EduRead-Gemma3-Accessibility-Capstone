"""Read-only access to the audited EduRead V3 processed dataset."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import torch
from torch.utils.data import DataLoader, Dataset

REQUIRED_SAMPLE_KEYS = (
    "input_ids", "token_type_ids", "attention_mask", "pixel_values",
    "labels", "example_id", "task_type", "image_filename", "supervised_tokens",
)
INDEX_METADATA_KEYS = ("example_id", "task_type", "image_filename", "supervised_tokens")


def load_processed_index(path: str | Path) -> List[Dict[str, Any]]:
    """Read an existing JSON processed index without changing it."""
    with Path(path).open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not all(isinstance(x, dict) for x in records):
        raise ValueError(f"Expected a JSON list of objects: {path}")
    return records


class EduReadProcessedDatasetV3(Dataset):
    """Lazy-load per-record .pt tensors referenced by the V3 index."""
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["processed_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        sample = torch.load(path, map_location="cpu", weights_only=True)
        missing = set(REQUIRED_SAMPLE_KEYS) - set(sample)
        if missing:
            raise ValueError(f"{record.get('example_id')} missing fields: {sorted(missing)}")
        for key in INDEX_METADATA_KEYS:
            if sample[key] != record[key]:
                raise ValueError(f"Index/tensor metadata mismatch for {key}: {record.get('example_id')}")
        return sample


def build_v3_dataloaders(project_root: str | Path, expected_train: int = 160, expected_eval: int = 40):
    """Build read-only datasets and batch-size-one loaders from Dataset_V3/processed."""
    root = Path(project_root) / "Dataset_V3" / "processed"
    train_records = load_processed_index(root / "train_processed_index.json")
    eval_records = load_processed_index(root / "eval_processed_index.json")
    if len(train_records) != expected_train or len(eval_records) != expected_eval:
        raise ValueError(
            f"Expected {expected_train}/{expected_eval} train/eval records; "
            f"found {len(train_records)}/{len(eval_records)}"
        )
    train_dataset = EduReadProcessedDatasetV3(train_records)
    eval_dataset = EduReadProcessedDatasetV3(eval_records)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
    return {
        "train_dataset": train_dataset, "eval_dataset": eval_dataset,
        "train_loader": train_loader, "eval_loader": eval_loader,
    }


def build_optimizer_group_token_counts(train_loader, accumulation_steps: int = 4) -> List[int]:
    """Compute shifted-label token denominators for 40 groups of four batches."""
    if accumulation_steps != 4:
        raise ValueError("Audited V3 uses four microbatches per optimizer update.")
    counts = []
    for batch in train_loader:
        count = int((batch["labels"][..., 1:] != -100).sum().item())
        if count <= 0:
            raise ValueError("A training batch has no shifted supervised tokens.")
        counts.append(count)
    if len(counts) != 160:
        raise ValueError(f"Expected 160 training batches; got {len(counts)}")
    groups = [sum(counts[i:i+4]) for i in range(0, len(counts), 4)]
    if len(groups) != 40 or any(x <= 0 for x in groups):
        raise ValueError("Invalid V3 optimizer-group token counts.")
    return groups
