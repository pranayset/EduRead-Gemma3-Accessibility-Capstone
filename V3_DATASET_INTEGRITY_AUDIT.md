# EduRead Gemma 3 Accessibility Capstone
## V3 Dataset Integrity Audit Report

**Audit date:** November 2026  
**Audit scope:** V3 raw data, processed tensors, source images, targets, category coverage, duplication, train/evaluation separation, and tokenizer alignment  
**Audit mode:** Read-only diagnostic audit  
**Audit coverage:** Steps 1–71F

---

## 1. Executive summary

The V3 dataset contains 200 task records: 160 training records and 40 evaluation records. Each mathematical example is represented by two task records: mathematical transcription (T1) and accessible explanation (T2).

The audit found strong structural and processing integrity. All 200 processed records passed the major tensor, metadata, and target-alignment checks. All 200 source images were found and readable. No exact image-file duplicates were detected across the training and evaluation splits.

The audit also identified limitations:
- Two normalized mathematical equations occur in both training and evaluation, affecting four evaluation records.
- Repeated mathematical content and near-duplicate targets exist within the training split.
- Linear algebra and logarithms have training examples but no evaluation examples.
- Probability, statistics, and trigonometry have very small evaluation samples.
- Some evaluation vocabulary does not occur in the training accessible-explanation targets, although tokenizer round-trip checks passed.

**Overall conclusion:** V3 is structurally sound for continued controlled experimentation and portfolio demonstration, with documented content-diversity and evaluation-coverage limitations. Dataset integrity does not establish model-generation quality.

## 2. Dataset overview

| Item | Training | Evaluation | Total |
|---|---:|---:|---:|
| Records | 160 | 40 | 200 |
| Mathematical transcription | 80 | 20 | 100 |
| Accessible explanation | 80 | 20 | 100 |
| Unique image paths | 80 | 20 | 100 |
| Processed tensor files | 160 | 40 | 200 |

T1 is mathematical transcription; T2 is accessible explanation. Each T1/T2 pair shares the same source image.

## 3. Raw dataset integrity

The raw JSON files contained 160 training records and 40 evaluation records.

| Validation | Result |
|---|---|
| Unique training IDs | 160 |
| Unique evaluation IDs | 40 |
| Duplicate IDs within splits | 0 |
| Missing or invalid IDs | 0 |
| Cross-split ID overlap | 0 |
| T1/T2 pairing | Valid |
| T1/T2 equation consistency | Valid |
| T1/T2 image-filename consistency | Valid |
| Task-specific prompt validation | 200/200 passed |

The two task prompts were consistent with their intended purposes: spoken mathematical transcription without symbolic notation, and clear English explanations for visually impaired students.

### Image paths

The raw JSON image paths used the stale prefix `/content/gdrive/`, while the active Drive mount used `/content/drive/`. All 200 paths were correctable by resolving the mount prefix; no source image remained missing or unresolved. The audit did not modify the JSON paths.

## 4. Source image integrity

All 200 source images were found and readable. All were PNG files.

Observed dimensions:

| Dimensions | Images |
|---|---:|
| 1260 × 421 | 120 |
| 921 × 423 | 58 |
| 1074 × 421 | 20 |
| 957 × 423 | 2 |

Image modes were RGB (160) and RGBA (40). No unreadable or invalid images were found.

Cross-split checks found:
- Shared image filenames: 0
- Shared normalized image paths: 0
- Exact SHA256 image matches: 0

The processed image tensors had shape `(3, 448, 448)`, dtype `float32`, finite values, and observed range -1 to 1. All three RGB channels were pixel-identical, indicating grayscale image content stored in three channels. No constant images were found.

These checks establish file and tensor validity, not universal visual readability or semantic correctness.

## 5. Processed tensor integrity

The processed dataset contained 160 training tensor files and 40 evaluation tensor files. All 200 records contained the expected fields:

- `input_ids`
- `token_type_ids`
- `attention_mask`
- `pixel_values`
- `labels`
- `example_id`
- `task_type`
- `image_filename`
- `supervised_tokens`

All records passed required-field, sequence-length, metadata, image-tensor, and numerical-finiteness checks.

| Item | Training | Evaluation |
|---|---:|---:|
| Supervised tokens | 3,545 | 846 |
| Ignored tokens | 170,560 | 42,640 |
| Invalid labels below -100 | 0 | 0 |
| Empty supervised targets | 0 | 0 |

Supervised label boundaries were consistent:
- Mathematical transcription: position 1064
- Accessible explanation: position 1068

All attention masks contained valid binary values. No zero-padding positions were found. Token-type IDs matched the supervised-label mask across all records.

## 6. Raw target versus processed label alignment

The raw JSON labels were tokenized using the tokenizer saved with the final adapter and compared with the supervised token IDs in each processed tensor, including the expected EOS token.

| Dataset | Checked | Passed | Failed |
|---|---:|---:|---:|
| Training | 160 | 160 | 0 |
| Evaluation | 40 | 40 | 0 |
| **Total** | **200** | **200** | **0** |

**Alignment pass rate: 100%.**

This confirms consistent transfer of raw target text into processed supervised labels. It does not independently establish mathematical correctness or pedagogical quality.

## 7. Category distribution

| Category | Training | Evaluation | Total |
|---|---:|---:|---:|
| Algebra | 32 | 8 | 40 |
| Calculus | 22 | 8 | 30 |
| Geometry | 30 | 6 | 36 |
| Integrals | 10 | 4 | 14 |
| Linear algebra | 2 | 0 | 2 |
| Logarithms | 4 | 0 | 4 |
| Physics | 16 | 8 | 24 |
| Probability | 16 | 2 | 18 |
| Statistics | 16 | 2 | 18 |
| Trigonometry | 12 | 2 | 14 |
| **Total** | **160** | **40** | **200** |

All 10 training categories contained both task types. All 8 categories represented in evaluation also contained both task types, with equal T1/T2 counts.

### Categories absent from evaluation

**Linear algebra**
- EQ100_T1 and EQ100_T2
- Equation: `det(A)=ad-bc`

**Logarithms**
- EQ027_T1 and EQ027_T2: `log₂(8)=3`
- EQ008_T1 and EQ008_T2: `log₁₀(100)=2`

These six records have no corresponding category coverage in evaluation.

Probability, statistics, and trigonometry each have only one evaluation record per task, so category-specific conclusions for these topics are especially sensitive to individual examples.

## 8. Target quality and formatting checks

The audit reviewed all 80 training and 20 evaluation accessible-explanation targets.

Training accessible targets:
- Minimum: 12 words
- Maximum: 37 words
- Average: approximately 22.93 words

Evaluation accessible targets:
- Minimum: 16 words
- Maximum: 45 words
- Average: approximately 26.5 tokens

Across the 100 accessible-explanation targets, automated screening found:
- Empty targets: 0
- Non-string targets: 0
- Unexpected control characters: 0
- Leading/trailing whitespace: 0
- Consecutive repeated words: 0
- Unexpected mathematical notation under the refined screening rule: 0
- Formatting issues under the checks performed: 0

Targets contained one to three sentences. These automated checks support formatting consistency but cannot prove complete mathematical or instructional quality.

## 9. Duplicate and redundancy audit

### Accessible explanations

The training accessible-explanation review identified one exact duplicate target group and nine near-duplicate pairs at the selected similarity threshold. Some pairs concerned related but distinct equations, so textual similarity alone does not establish an incorrect duplicate.

### Mathematical transcription

The case-sensitive review identified six duplicate target groups involving 12 records. The repeated content included:
- Circle area
- Circle circumference
- Population variance
- A trigonometric identity
- Chain rule
- Newton's second law

Visual review showed that the paired images represented the same mathematical content, although the image files and layouts differed.

The 12 records represent 7.5% of the 160 training records. Ten were transcription records and two were accessible-explanation records.

**Interpretation:** The training split contains mathematical-content redundancy, particularly in transcription. This is a dataset-diversity limitation, not evidence of corrupted records.

## 10. Training/evaluation mathematical-content overlap

Two unique normalized equations appeared in both splits:

1. `∫x² dx = x³/3 + C`
   - Training: EQ037_T1 and EQ037_T2
   - Evaluation: EQ004_T1 and EQ004_T2

2. `A = 1/2 bh`
   - Training: EQ074_T1 and EQ074_T2
   - Evaluation: EQ015_T1 and EQ015_T2

This affects four evaluation records.

There were no exact cross-split image-file matches, but different image files can still depict the same mathematical expression. The triangle-area pair EQ015 and EQ074 was visually inspected and showed the same mathematical content.

After excluding the four records affected by exact normalized equation overlap:
- Remaining evaluation records: 36
- Remaining unique equation IDs: 18
- Mathematical transcription: 18
- Accessible explanation: 18

This 36-record subset is unaffected by the exact normalized equation matches identified in this audit. It is not guaranteed to be free from broader conceptual similarity.

## 11. Tokenizer and vocabulary integrity

The tokenizer was loaded locally from the final adapter directory.

- Tokenizer class: `GemmaTokenizer`
- Vocabulary size: 257,153
- Unknown token: `<unk>`
- Unknown token ID: 3

All 200 target strings passed exact tokenizer decode round-trip checks:
- Accessible explanations: 100/100
- Mathematical transcriptions: 100/100
- Unknown-token substitutions detected: 0

The evaluation accessible-explanation vocabulary contained words not found in the training accessible-explanation targets. Coverage was approximately 77.93% in the original accessible-only comparison and approximately 78.62% when both training task types were included. The tokenizer represented the examined unseen words and full evaluation sentences without unknown-token substitutions.

Vocabulary coverage is not the same as model understanding or generation accuracy.

## 12. Visual verification of training-only categories

Three source images were inspected:

- EQ100_T1: `det(A)=ad-bc`
- EQ027_T1: `log₂(8)=3`
- EQ008_T1: `log₁₀(100)=2`

All three images visually matched their dataset equations. Their transcription and accessible-explanation targets were mathematically consistent with the displayed expressions.

**Visual verification: 3/3 passed.**

This verifies the inspected examples individually; it does not provide evaluation coverage for linear algebra or logarithms.

## 13. Consolidated results

| Audit area | Result |
|---|---|
| Raw JSON record integrity | PASS |
| Unique IDs and split ID separation | PASS |
| T1/T2 pairing and prompt consistency | PASS |
| Source image availability/readability | PASS |
| Exact cross-split image duplicates | None found |
| Processed tensor structure and numerical validity | PASS |
| Attention and token-type/label-mask alignment | PASS |
| Raw target versus processed labels | PASS — 100% |
| Category/task balance | PASS |
| Accessible-target formatting checks | PASS |
| Tokenizer round-trip | PASS — 100% |
| Training duplicate content | Redundancy identified |
| Exact train/evaluation equation separation | Two overlapping equations |
| Linear algebra evaluation coverage | Missing |
| Logarithms evaluation coverage | Missing |
| Small category evaluation samples | Limitation identified |
| Training-only visual checks | PASS — 3/3 |

## 14. Recommendations for a future dataset revision

These are recommendations only; no changes were made during this read-only audit.

1. Add evaluation examples for linear algebra and logarithms.
2. Remove or replace the two exact equation overlaps if a strictly expression-independent evaluation set is required.
3. Review repeated training content, especially the six duplicate transcription groups.
4. Increase evaluation sample sizes for probability, statistics, and trigonometry.
5. Expand mathematical diversity within existing categories.
6. Replace environment-specific image paths with portable project-relative paths.
7. Continue human review of image readability, equation correctness, and accessibility quality.
8. Report full evaluation results separately from results on the 36-record subset without exact equation overlap.

## 15. Scope and limitations

This was a read-only diagnostic audit. No dataset files, model weights, adapter files, training configurations, or notebook content were modified. No retraining or GPU execution was performed as part of this audit.

The audit does not establish:
- That every image is readable to every user.
- That every target is pedagogically optimal.
- That the dataset covers every mathematical concept.
- That the model generates correct explanations for unseen expressions.
- That the evaluation set is free from all conceptual similarity to training data.
- That the model is production-ready.

Model-generation quality must be evaluated separately from dataset integrity.

## 16. Final conclusion

The EduRead V3 dataset demonstrates strong structural and processing integrity. All 200 records passed the major record, image, tensor, prompt, label, and tokenizer checks performed in this audit.

The dataset nevertheless has documented limitations in mathematical-content separation, training redundancy, and evaluation coverage.

**Final assessment: Structurally sound, with documented content-diversity and evaluation-coverage limitations.**

The dataset is suitable for continued controlled experimentation and portfolio demonstration, provided these limitations are disclosed when presenting evaluation results.

**Audit completed through Step 71F.**  
**Audit mode:** Read-only  
**Dataset modifications:** None  
**Model retraining:** None  
**GPU use during audit:** None
