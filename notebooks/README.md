# Notebooks

This directory contains the EduRead research notebook and Colab workflow documentation.

## Notebook hygiene

Before committing notebook updates:
- Remove access tokens, credentials, personal information, and private dataset content.
- Review cell outputs for secrets and sensitive paths.
- Avoid embedding large images, model weights, or processed dataset tensors.
- Keep a clean, reproducible sequence of setup, data validation, inference, training, and evaluation steps.
- Record package versions and hardware used for each experiment.

The audited V3 workflow uses PaliGemma 3B (`google/paligemma-3b-pt-448`) with QLoRA/PEFT. The notebook is a research artifact; claims about successful training or improved accessibility should be supported by saved artifacts and measured evaluation evidence.
