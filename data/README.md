# Dataset documentation

The private V3 dataset is stored outside this public repository. This directory is intended for dataset documentation, manifests, and explicitly reviewed public samples only.

## Data handling rules

- Do not commit private, personal, confidential, or restricted data.
- Do not commit raw or processed V3 dataset images or tensor files.
- Use synthetic, public, or appropriately licensed examples for public demonstrations.
- Record source, license, collection method, and any transformations for published samples.
- Review image-target alignment and mathematical correctness before using examples.

## Suggested manifest fields

- `example_id`
- `image_path`
- `task_type`
- `category`
- `prefix`
- `label`
- `source`
- `license`
- `split` (train, validation, evaluation)

The sample manifest is a schema example, not a copy of the private V3 dataset.

See [V3 Dataset Integrity Audit](../V3_DATASET_INTEGRITY_AUDIT.md) for the audit findings and limitations.
