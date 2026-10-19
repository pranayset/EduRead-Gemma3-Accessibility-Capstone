# EduRead — Gemma 3 Accessibility Capstone

EduRead explores adapting a Gemma 3 vision-language model to generate useful alt-text, detailed visual descriptions, and mathematical descriptions for visually impaired students.

> **Status note:** Update the results table only after you have verified each stage. The LoRA adapter is stored separately in Google Drive and is intentionally excluded from this repository.

## Goals

- Establish a baseline using the selected Gemma 3 model.
- Prepare and review image-description examples.
- Experiment with parameter-efficient fine-tuning (LoRA/PEFT).
- Evaluate grounding, usefulness, accessibility, and mathematical fidelity.
- Build an accessible demonstration interface.

## Repository structure

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── configs/
│   └── training_config.yaml
├── data/
│   ├── README.md
│   └── sample/
│       └── manifest.csv
├── evaluation/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   └── eduread/
│       └── README.md
└── tests/
    └── README.md
```

Copy your real notebook from Google Drive to `notebooks/` and your existing Python source files to `src/eduread/`. This starter package does not contain your actual notebook, source implementation, dataset, or adapter.

## Model and compute

The intended base model is `google/gemma-3-4b-it`. Review its model card and usage conditions before using it. A free Colab GPU is not guaranteed. Validate baseline inference before running fine-tuning.

## Google Colab setup

1. Open your existing notebook from Google Drive.
2. Install compatible dependencies. `requirements.txt` is a starting point; pin tested versions before publishing a reproducible environment.
3. If authentication is required, store your Hugging Face token in Colab Secrets. Do not paste it into notebook cells or commit it to GitHub.
4. Run a baseline inference test.
5. Review dataset sources, permissions, labels, and train/validation/test splits.
6. Run fine-tuning only after the baseline works and compute is available.
7. Save the trained adapter to persistent Google Drive storage and reload it to verify the saved artifact.
8. Record evaluation results and limitations.

## Adapter storage

Keep adapter weights outside this public repository. A typical PEFT save pattern is shown below; use it only if `model` is the trained PEFT-compatible model object and `processor` is the processor used by your workflow.

```python
adapter_dir = (
    "/content/drive/MyDrive/"
    "EduRead_Gemma3_Accessibility_Capstone/final_adapter"
)
model.save_pretrained(adapter_dir)
processor.save_pretrained(adapter_dir)  # if applicable
```

After saving, verify that the adapter configuration and weight files exist, then test reloading in a fresh model instance. Do not upload model weights to this repository.

## Dataset

See [`data/README.md`](data/README.md). Do not commit private student data, private/raw images, personal information, or images without suitable usage permission. The sample manifest is only a schema example, not a dataset.

## Evaluation

See [`evaluation/README.md`](evaluation/README.md). Report actual results only. Include failure cases such as missed labels, incorrect symbols, hallucinated details, or ambiguous diagrams.

## Running the app

Copy your real app implementation into `src/eduread/` and document the tested launch command here. For example, `python -m eduread.app` is only a template and may not match your current code.

## Results status

| Stage | Status | Evidence |
|---|---|---|
| Baseline inference | Update after verifying | Notebook output or test record |
| Dataset preparation | Update after verifying | Dataset summary and provenance |
| LoRA fine-tuning | Training reported complete; adapter saved in Drive | Verify saved files and reload before claiming reproducibility |
| Evaluation | Update after running | Metrics plus human review |
| Accessible app | Update after testing | Tested launch instructions |

## Limitations

A vision-language model can misread symbols, labels, units, chart axes, and spatial relationships. Important mathematical content should be verified by a trusted source or educator. Do not claim formal accessibility validation unless it has been performed.

## Licensing and hosting

This starter repository intentionally does not include a license file or a hosted demo URL.
