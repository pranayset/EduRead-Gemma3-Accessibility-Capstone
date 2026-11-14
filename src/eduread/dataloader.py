"""V3 DataLoader construction, matching the audited batch-size-one workflow."""

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader
from .dataset import EduReadProcessedDatasetV3, load_processed_index


@dataclass(frozen=True)
class V3DataLoaders:
    train_dataset: EduReadProcessedDatasetV3
    eval_dataset: EduReadProcessedDatasetV3
    train_loader: DataLoader
    eval_loader: DataLoader


def build_v3_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 1,
    shuffle_train: bool = False,
    num_workers: int = 0,
) -> V3DataLoaders:
    """Build loaders from the two existing V3 index files.

    The default is intentionally batch_size=1, shuffle=False, num_workers=0,
    drop_last=False, with PyTorch's default collator, matching the audited
    V3 reconstruction cell. No tensor or index files are written.
    """
    processed_dir = Path(processed_dir)
    train_records = load_processed_index(
        processed_dir / "train_processed_index.json"
    )
    eval_records = load_processed_index(
        processed_dir / "eval_processed_index.json"
    )

    train_dataset = EduReadProcessedDatasetV3(train_records, processed_dir)
    eval_dataset = EduReadProcessedDatasetV3(eval_records, processed_dir)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": False,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle_train,
        **common,
    )
    eval_loader = DataLoader(
        eval_dataset,
        shuffle=False,
        **common,
    )
    return V3DataLoaders(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
