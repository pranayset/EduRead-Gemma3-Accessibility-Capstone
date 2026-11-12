# EduRead Python package

This package is the planned home for reusable EduRead implementation modules. The current research workflow is primarily notebook-based; this directory should only claim functionality that has actually been extracted and tested.

## Suggested module boundaries

- `model_loader.py`: model and processor initialization.
- `inference.py`: image-plus-prompt generation.
- `data.py`: manifest loading and validation.
- `evaluation.py`: metric and human-review helpers.
- `app.py`: Gradio interface.

Add modules incrementally, with deterministic tests and documentation. Do not create placeholder APIs that imply implemented functionality.
