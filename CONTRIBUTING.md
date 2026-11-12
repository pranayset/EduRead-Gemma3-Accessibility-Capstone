# Contributing to EduRead

Thank you for your interest in EduRead, an educational vision-language accessibility capstone.

## 1. Contribution goals

Contributions should improve:
- Mathematical image understanding.
- Accessible transcription and explanations.
- Data quality and provenance.
- Evaluation methodology.
- Reproducibility, testing, and documentation.
- Responsible and inclusive user experience.

## 2. Development setup

```bash
git clone https://github.com/pranayset/EduRead-Gemma3-Accessibility-Capstone.git
cd EduRead-Gemma3-Accessibility-Capstone
python -m venv .venv
```

Activate the environment, then install the dependencies appropriate for your hardware and Python version. See `requirements.txt` and the setup notes in `README.md`.

## 3. Project structure

- `src/eduread/`: reusable Python modules.
- `notebooks/`: experiments and Colab workflows.
- `configs/`: configuration examples.
- `data/`: public manifests and schema documentation only.
- `evaluation/`: evaluation protocols and reports.
- `tests/`: deterministic tests.

## 4. Dataset and privacy rules

- Do not commit private, personal, confidential, or restricted datasets.
- Use synthetic, public, or appropriately licensed material.
- Record provenance and license information for published samples.
- Review image and annotation correctness before use.
- Do not commit raw images or processed dataset tensors unless explicitly reviewed and permitted.
- Never upload model weights, adapter files, checkpoints, or credentials.

## 5. Model and experiment reporting

The audited V3 workflow uses PaliGemma 3B with QLoRA/PEFT. Document the exact model ID, dataset version, configuration, package versions, hardware, and evaluation split for any new experiment.

Report negative results and failure cases. Do not claim improved accuracy or accessibility without measured evidence. Keep dataset integrity findings separate from model-generation quality.

## 6. Code standards

- Prefer small, readable, documented functions.
- Follow PEP 8 conventions.
- Add input validation and useful error messages.
- Avoid hard-coded secrets and machine-specific paths.
- Add deterministic tests where possible.
- Do not make tests download a large model or require a GPU by default.

## 7. Pull request process

1. Create a feature branch.
2. Make a focused change.
3. Run relevant tests and checks.
4. Update documentation and configuration examples.
5. Inspect notebook outputs and diffs for private content.
6. Submit a pull request describing motivation, changes, validation, and limitations.

## 8. Reporting issues

Include reproduction steps, expected behaviour, actual behaviour, and relevant sanitized logs. Never include access tokens, personal data, or private datasets.

## 9. Community standards

Please follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Review [SECURITY.md](SECURITY.md) for responsible vulnerability reporting.

## 10. Licensing

No project license has been selected yet. Until a license is added, do not assume that public visibility grants permission to reuse or redistribute project materials.
