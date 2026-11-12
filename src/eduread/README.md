# EduRead V3 reusable Python modules

These modules extract reusable pieces from the completed V3 notebook. They do not run automatically when imported.

## V3 processed dataset and DataLoader

`data.py` reads the existing files under `Dataset_V3/processed/`:

- `train_processed_index.json` (160 records)
- `eval_processed_index.json` (40 records)
- Per-example `.pt` tensors referenced by those indexes

It lazy-loads each tensor on CPU using `weights_only=True`, checks index metadata against tensor metadata, and reconstructs the notebook's DataLoaders with batch size 1, no shuffle, zero workers, and no dropped batches. It does not render images, call the processor, regenerate tensors, or rewrite indexes.

Example (read-only dataset reconstruction):

```python
from eduread.data import build_v3_dataloaders, build_optimizer_group_token_counts

parts = build_v3_dataloaders(
    "/content/drive/MyDrive/EduRead_Gemma3_Accessibility_Capstone"
)
groups = build_optimizer_group_token_counts(parts["train_loader"], accumulation_steps=4)

print(len(parts["train_dataset"]))  # 160
print(len(parts["eval_dataset"]))   # 40
print(len(groups))                  # 40 optimizer groups
```

## Three-epoch V3 training

`training.py` contains:

- `create_v3_optimizer`: PagedAdamW8bit over the 72 LoRA tensors (921,600 trainable parameters).
- `create_v3_scheduler`: linear warmup/decay, 6 warmup steps and 120 total optimizer updates.
- `evaluate_v3_loss`: evaluation without gradient updates.
- `run_v3_training`: callable three-epoch workflow.

The extracted training loop follows the notebook's schedule: 3 epochs, 160 microbatches per epoch, groups of four microbatches, 40 optimizer updates per epoch, FP16 CUDA autocast, gradient clipping at 1.0, evaluation after each epoch, and epoch checkpoint/metrics saving.

**Training is disabled by default.** Calling `run_v3_training(...)` without `execute=True` returns a plan and performs no forward/backward pass, optimizer/scheduler step, or output write. Explicit opt-in requires CUDA, the expected V3 LoRA state, a fresh optimizer and scheduler, and an empty output directory. Existing checkpoint directories are never overwritten.

The entry point currently supports a fresh three-epoch run only. It does not restore optimizer/scheduler state or resume a partially completed run. Do not invoke it when your goal is simply to use or evaluate the already-trained adapter.

Example (dry run; no training):

```python
from eduread.training import run_v3_training

plan = run_v3_training(
    model=model_v3,
    processor=processor_v3,
    train_loader=parts["train_loader"],
    eval_loader=parts["eval_loader"],
    optimizer=optimizer_v3,
    scheduler=scheduler_v3,
    group_token_counts=groups,
    output_dir="/path/to/a/new/empty/output",
)
# execute=False by default; no training or file writes.
print(plan)
```

To intentionally start a new training run, a caller must deliberately pass `execute=True` after reviewing the code and confirming that the destination is a new empty directory. This is not needed for the completed V3 project.

## Model identity

The audited V3 run uses `google/paligemma-3b-pt-448` and `PaliGemmaForConditionalGeneration`, not standalone `google/gemma-3-4b-it`.

## Privacy

Keep private V3 manifests, image files, processed tensors, model weights, adapters, optimizer state, checkpoints, and credentials outside Git.
