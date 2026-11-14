"""Configuration loading for the EduRead V3 extraction workflow."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os
import yaml


@dataclass(frozen=True)
class V3Config:
    processed_dir: Path
    expected_train_records: int = 160
    expected_eval_records: int = 40
    batch_size: int = 1
    shuffle_train: bool = False
    num_workers: int = 0
    accumulation_steps: int = 4
    epochs: int = 3
    expected_train_shifted_tokens: int = 3545
    expected_eval_shifted_tokens: int = 846
    model_id: str = "google/paligemma-3b-pt-448"
    quantization: str = "nf4"
    compute_dtype: str = "float16"
    double_quantization: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2.0e-5
    optimizer: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    use_reentrant: bool = False
    max_gradient_norm: float = 1.0
    warmup_steps: int = 6

    @property
    def optimizer_groups_per_epoch(self) -> int:
        return self.expected_train_records // self.accumulation_steps

    @property
    def total_optimizer_updates(self) -> int:
        return self.optimizer_groups_per_epoch * self.epochs


def load_config(path: str | Path) -> V3Config:
    """Load YAML settings; an environment variable may override the data path."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"V3 config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    paths = raw.get("paths", {})
    data = raw.get("data", {})
    loader = raw.get("dataloader", {})
    training = raw.get("training", {})
    model = raw.get("model", {})
    lora = raw.get("lora", {})

    configured_dir = paths.get("processed_dir")
    env_dir = os.environ.get("EDUREAD_V3_PROCESSED_DIR")

    selected_dir = env_dir or configured_dir

    if not selected_dir:
        raise ValueError(
            "Set EDUREAD_V3_PROCESSED_DIR or paths.processed_dir in the YAML."
        )

    processed_dir = Path(selected_dir).expanduser()

    return V3Config(
        processed_dir=processed_dir,
        expected_train_records=int(data.get("expected_train_records", 160)),
        expected_eval_records=int(data.get("expected_eval_records", 40)),
        batch_size=int(loader.get("batch_size", 1)),
        shuffle_train=bool(loader.get("shuffle_train", False)),
        num_workers=int(loader.get("num_workers", 0)),
        accumulation_steps=int(training.get("gradient_accumulation_steps", 4)),
        epochs=int(training.get("epochs", 3)),
        expected_train_shifted_tokens=int(data.get("expected_train_shifted_tokens", 3545)),
        expected_eval_shifted_tokens=int(data.get("expected_eval_shifted_tokens", 846)),
        model_id=str(model.get("model_id", "google/paligemma-3b-pt-448")),
        quantization=str(model.get("quantization", "nf4")),
        compute_dtype=str(model.get("compute_dtype", "float16")),
        double_quantization=bool(model.get("double_quantization", True)),
        lora_r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        learning_rate=float(training.get("learning_rate", 2.0e-5)),
        optimizer=str(training.get("optimizer", "paged_adamw_8bit")),
        gradient_checkpointing=bool(training.get("gradient_checkpointing", True)),
        use_reentrant=bool(training.get("use_reentrant", False)),
        max_gradient_norm=float(training.get("max_gradient_norm", 1.0)),
        warmup_steps=int(training.get("warmup_steps", 6)),
    )
