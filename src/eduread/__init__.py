"""EduRead V3 reusable workflow modules.

Importing this package does not load models, access datasets, write files,
or start training.
"""
from .config import (
    DatasetConfig,
    EduReadConfig,
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
)

__version__ = "0.2.0"

__all__ = [
    "DatasetConfig",
    "EduReadConfig",
    "LoRAConfig",
    "ModelConfig",
    "QuantizationConfig",
    "TrainingConfig",
    "__version__",
]
