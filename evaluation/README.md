# Evaluation protocol

Record the model/checkpoint, dataset version, split sizes, hardware, package versions, prompt, decoding settings, and evaluation date for each run.

## Evaluation dimensions

- Mathematical fidelity: symbols, variables, operations, units, and relationships.
- Completeness: whether all important expression components are covered.
- Grounding: whether claims are supported by the input image.
- Clarity: readable language and logical reading order.
- Accessibility usefulness: whether the explanation communicates the expression without relying on visual notation.
- Failure analysis: omissions, hallucinations, OCR errors, ambiguity, and premature stopping.

Automatic metrics should not replace human review for mathematical correctness and accessibility.

## V3 dataset limitations

The V3 integrity audit found two exact normalized equations shared between training and evaluation, training-content redundancy, and no evaluation examples for linear algebra or logarithms. Probability, statistics, and trigonometry have very small evaluation samples. Report these limitations when presenting results.

See [V3 Dataset Integrity Audit](../V3_DATASET_INTEGRITY_AUDIT.md).

## Run record template

| Run | Model/checkpoint | Dataset/split | Hardware | Generation settings | Metrics | Human review | Notes |
|---|---|---|---|---|---|---|---|
| Baseline | TBD | TBD | TBD | TBD | Not recorded here | TBD | Add verified results |
| V3 adapter | PaliGemma 3B + V3 adapter | V3 evaluation | See run log | Greedy, max 64 tokens in audited comparison | See notebook | Required | Do not infer quality from training loss alone |
