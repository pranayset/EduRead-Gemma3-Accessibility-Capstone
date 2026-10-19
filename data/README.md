# Dataset guidance

Do not commit private/raw images, personal information, or images without suitable usage permission. The repository ignores `data/raw/`, `data/processed/`, and `data/private/` by default.

Suggested fields for a reviewed manifest:
- `image_path`
- `alt_text`
- `detailed_description`
- `math_description`
- `source`
- `license`
- `split` (train, validation, test)

Review all labels for correctness and accessibility. The sample manifest is a schema example, not a dataset.
