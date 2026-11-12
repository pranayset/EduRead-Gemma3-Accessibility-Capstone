"""Configuration for the audited EduRead V3 PaliGemma workflow.

Values document the completed V3 run. Creating these objects does not load a
model, access the dataset, or start training.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = "google/paligemma-3b-pt-448"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    attention_implementation: str = "eager"
    image_size: int = 448


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    quant_type: str = "nf4"
    compute_dtype: str = "float16"
    double_quant: bool = True


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 6
    gradient_accumulation_steps: int = 4
    optimizer_updates_per_epoch: int = 40
    max_gradient_norm: float = 1.0
    weight_decay: float = 0.01
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_epsilon: float = 1e-8

    @property
    def total_optimizer_updates(self) -> int:
        return self.epochs * self.optimizer_updates_per_epoch


@dataclass(frozen=True)
class DatasetConfig:
    train_records: int = 160
    evaluation_records: int = 40
    tasks: Tuple[str, ...] = (
        "mathematical_transcription",
        "accessible_explanation",
    )


@dataclass(frozen=True)
class EduReadConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    project_root: Path = Path(".")
    output_dir: Path = Path("outputs/eduread_v3_qlora")

    def validate(self) -> None:
        if self.model.model_id != "google/paligemma-3b-pt-448":
            raise ValueError("This configuration describes the audited V3 PaliGemma 3B run.")
        if self.model.image_size != 448:
            raise ValueError("Audited V3 image resolution is 448.")
        if self.quantization.quant_type != "nf4" or not self.quantization.load_in_4bit:
            raise ValueError("Audited V3 uses 4-bit NF4 quantization.")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if self.training.total_optimizer_updates != 120:
            raise ValueError("Audited V3 configuration expects 120 optimizer updates.")
