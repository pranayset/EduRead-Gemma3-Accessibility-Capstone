
# EduRead Gemma 3 - System Architecture

## 1. Project Overview

EduRead Gemma 3 is an accessibility-focused vision-language
AI capstone project.

Its objective is to explore how vision-language models can
generate meaningful image descriptions and mathematical
descriptions to support visually impaired students.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Base Model | PaliGemma 3B |
| Model Framework | PyTorch |
| Model Library | Hugging Face Transformers |
| Fine-Tuning | PEFT / LoRA |
| Dataset Processing | Python / Pandas |
| Image Processing | Pillow |
| Interface | Gradio |
| Development | Google Colab |
| Model Storage | Google Drive |
| Version Control | Git / GitHub |

## 3. Architecture Components

### 3.1 Input Layer

Accepts educational images containing:
- Mathematical equations
- Diagrams
- Charts
- Educational illustrations
- Other learning materials

### 3.2 Image Preprocessing

Processes images into the format expected by
the vision-language model.

### 3.3 Model Processing

Uses PaliGemma 3B and its associated processor
to understand visual and textual information.

### 3.4 LoRA Fine-Tuning

Uses Parameter-Efficient Fine-Tuning (PEFT)
to adapt selected model components.

Training configuration includes:

- LoRA rank: 8
- LoRA alpha: 16
- LoRA dropout: 0.05
- Target modules: q_proj and v_proj

### 3.5 Output Generation

The adapted model explores generation of:

- Image Alt-Text
- Educational image descriptions
- Mathematical descriptions

### 3.6 Evaluation Layer

Evaluates generated outputs for:

- Description relevance
- Accuracy
- Completeness
- Clarity
- Accessibility usefulness

### 3.7 User Interface

Gradio provides an interface for submitting
educational images and reviewing generated descriptions.

## 4. Data Flow

1. User submits an educational image.
2. Image preprocessing prepares the input.
3. PaliGemma Processor converts the input.
4. The adapted vision-language model generates output.
5. Generated descriptions are evaluated.
6. Gradio displays the resulting descriptions.

## 5. Repository Structure

    EduRead-Gemma3-Accessibility-Capstone/
    |
    |-- README.md
    |-- CONTRIBUTING.md
    |-- architecture.md
    |-- requirements.txt
    |-- .gitignore
    |
    |-- configs/
    |   |-- training_config.yaml
    |
    |-- data/
    |   |-- README.md
    |   |-- sample/
    |       |-- manifest.csv
    |
    |-- evaluation/
    |   |-- README.md
    |
    |-- notebooks/
    |   |-- EduRead_Gemma3_Accessibility_Capstone.ipynb
    |   |-- README.md
    |
    |-- src/
    |   |-- eduread/
    |       |-- README.md
    |
    |-- tests/
        |-- README.md

## 6. Model Storage

Trained adapters and checkpoints are stored
separately from the public GitHub repository.

Google Drive location:

    /content/drive/MyDrive/
    EduRead_Gemma3_Accessibility_Capstone/final_adapter

Model weights and private training artifacts
must not be committed to GitHub.

## 7. Security and Responsible AI

- Do not commit Hugging Face access tokens.
- Do not publish private datasets.
- Use appropriately licensed educational images.
- Validate generated descriptions.
- Clearly communicate model limitations.
- Do not assume generated mathematical descriptions
  are always correct.

## 8. Future Improvements

- Improve mathematical diagram understanding.
- Expand synthetic educational datasets.
- Improve Alt-Text quality.
- Add accessibility-focused evaluation metrics.
- Improve Gradio user experience.
- Explore multilingual educational descriptions.
