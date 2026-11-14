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


# EduRead V3 read-only validation exports
from .v3_config import V3Config, load_config
from .dataset import EduReadProcessedDatasetV3, load_processed_index
from .dataloader import V3DataLoaders, build_v3_dataloaders
from .validation import validate_v3_pipeline

__all__.extend([
    "V3Config",
    "load_config",
    "EduReadProcessedDatasetV3",
    "load_processed_index",
    "V3DataLoaders",
    "build_v3_dataloaders",
    "validate_v3_pipeline",
])
