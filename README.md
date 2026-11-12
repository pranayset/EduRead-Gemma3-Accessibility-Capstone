# EduRead — Gemma 3 Accessibility Capstone

An educational vision-language capstone exploring mathematical image understanding and accessible descriptions for visually impaired students.

> **V3 model clarification:** The audited V3 fine-tuning workflow uses **PaliGemma 3B** (`google/paligemma-3b-pt-448`). The repository name reflects the broader capstone lineage; this V3 run should not be described as fine-tuning standalone Gemma 3 4B IT.

## Project objectives

- Convert mathematical expressions in images into spoken mathematical wording.
- Generate clear English explanations of mathematical expressions.
- Explore parameter-efficient vision-language fine-tuning using QLoRA and PEFT.
- Evaluate mathematical fidelity, completeness, clarity, and accessibility usefulness.
- Document limitations and failure cases transparently.

## Repository structure

```text
EduRead-Gemma3-Accessibility-Capstone/
├── README.md
├── readme.txt
├── LICENSE_STATUS.md
├── README.txt
├── LICENSE_STATUS.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── CITATION.cff
├── .gitignore
├── .env.example
├── pyproject.toml
├── pyproject.toml
├── requirements.txt
├── architecture.md
├── V3_DATASET_INTEGRITY_AUDIT.md
├── configs/
│   └── training_config.yaml
├── data/
│   ├── README.md
│   └── sample/
│       └── manifest.csv
├── evaluation/
│   └── README.md
├── notebooks/
│   ├── README.md
│   └── EduRead_Gemma3_Accessibility_Capstone.ipynb
├── src/
│   └── eduread/
│       ├── README.md
│       └── __init__.py
└── tests/
    └── README.md
```

## V3 model and method

| Item | Audited V3 configuration |
|---|---|
| Base model | `google/paligemma-3b-pt-448` |
| Model class | `PaliGemmaForConditionalGeneration` |
| Image resolution | 448 × 448 |
| Fine-tuning | QLoRA / PEFT |
| Quantization | 4-bit NF4, double quantization |
| Compute precision | FP16 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Training records | 160 |
| Evaluation records | 40 |
| Epochs | 3 |
| Optimizer updates | 120 |

See [`configs/training_config.yaml`](configs/training_config.yaml). It documents known audited settings and is a reference, not a guarantee that a fresh run can be reproduced without environment-specific adjustments.

## Quick start

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install compatible dependencies. PyTorch wheels depend on your CUDA/CPU environment; select the correct build rather than assuming the captured Colab CUDA build works on every machine.
4. Open the notebook in `notebooks/` or Google Colab.
5. Configure Hugging Face authentication through Colab Secrets or a local environment variable when required.
6. Review data provenance, licensing, and split integrity before training.
7. Run a baseline inference test before attempting fine-tuning.

Do not put real tokens in notebooks or commit them. The `.env.example` file is a template only.

## Dataset policy

The private V3 dataset and model artifacts are not included in this public repository. The `data/` directory contains documentation and a sample manifest schema only. Use synthetic, public, or appropriately licensed examples for public demonstrations.

See [Data documentation](data/README.md).

## Evaluation and audit

See [Evaluation guidance](evaluation/README.md) and the [V3 Dataset Integrity Audit](V3_DATASET_INTEGRITY_AUDIT.md).

The audit found strong structural integrity across 200 records, with documented limitations:
- Two exact normalized equations overlap between training and evaluation.
- Repeated mathematical content occurs in training.
- Linear algebra and logarithms have no evaluation examples.
- Some evaluation categories have very small sample sizes.

Dataset integrity does not prove generation accuracy, generalization, or formal accessibility validation. See the audit for details.

## Architecture

See [architecture.md](architecture.md) for the system architecture, data flow, model components, and security boundaries.

## Tests

Run lightweight repository tests with:

```bash
python -m pytest
```

The tests directory currently contains test-planning documentation. Add actual tests as reusable modules are implemented; do not interpret an empty test suite as successful model validation.

## Security and responsible AI

- Private datasets, raw/processed tensors, model weights, adapters, checkpoints, and credentials must remain outside Git.
- Review notebooks and outputs before committing.
- Validate generated mathematical descriptions with appropriate human review.
- Do not claim formal accessibility certification or production readiness without evidence.

See [SECURITY.md](SECURITY.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## Licensing

A license has not yet been selected. See [LICENSE_STATUS.md](LICENSE_STATUS.md). Public availability alone does not grant reuse rights.


## Python package modules

Reusable components extracted from the completed V3 notebook are in `src/eduread/`:

- `config.py`: audited V3 model, quantization, LoRA, and training settings.
- `data.py`: read-only raw-manifest and processed-tensor validation.
- `model_loader.py`: explicit PaliGemma processor and NF4 base-model loaders.
- `adapters.py`: PEFT adapter initialization and explicit save/load helpers.
- `training.py`: optimizer/scheduler builders and supervised-token accounting.
- `inference.py`: prompt-boundary handling and greedy inference.
- `evaluation.py`: text normalization and prediction similarity summaries.

Importing the package does not load model weights, read the private dataset, write artifacts, or train. Model loading, adapter initialization, optimizer construction, inference, and saving occur only when their respective functions are explicitly called. The reusable modules do not automatically run the completed training loop.

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Run the lightweight unit tests:

```bash
python -m pytest
```

The test suite covers configuration and text-evaluation helpers; it is not an end-to-end model or GPU validation.
