"""CPU-only validation of the existing EduRead V3 processed dataset."""

from collections import Counter
from pathlib import Path
import torch

from .dataloader import build_v3_dataloaders
from .preprocessing import validate_preprocessed_sample
from .v3_validation_training import (
    V3TrainingPlan,
    calculate_optimizer_group_token_counts,
)
from .v3_validation_evaluation import summarize_evaluation_loader


def validate_v3_pipeline(
    processed_dir: str | Path,
    expected_train_records: int = 160,
    expected_eval_records: int = 40,
    expected_train_shifted_tokens: int = 3545,
    expected_eval_shifted_tokens: int = 846,
    accumulation_steps: int = 4,
) -> dict:
    """Validate existing indexes/tensors without writing or training."""
    errors = []
    loaders = build_v3_dataloaders(
        processed_dir=processed_dir,
        batch_size=1,
        shuffle_train=False,
        num_workers=0,
    )

    if len(loaders.train_dataset) != expected_train_records:
        errors.append(
            f"Expected {expected_train_records} training records; "
            f"found {len(loaders.train_dataset)}."
        )
    if len(loaders.eval_dataset) != expected_eval_records:
        errors.append(
            f"Expected {expected_eval_records} evaluation records; "
            f"found {len(loaders.eval_dataset)}."
        )

    task_counts_train = Counter()
    task_counts_eval = Counter()
    train_shifted_tokens = 0
    eval_shifted_tokens = 0
    checked_train = 0
    checked_eval = 0
    metadata_mismatches = []

    # Explicitly access every processed tensor on CPU.
    for dataset, task_counts, split in (
        (loaders.train_dataset, task_counts_train, "train"),
        (loaders.eval_dataset, task_counts_eval, "eval"),
    ):
        for index in range(len(dataset)):
            sample = dataset[index]
            result = validate_preprocessed_sample(sample)
            task_counts[result["task_type"]] += 1

            if not result["metadata_matches_shifted"]:
                metadata_mismatches.append(
                    {
                        "split": split,
                        "example_id": result["example_id"],
                        "metadata": result["supervised_tokens_metadata"],
                        "shifted": result["shifted_supervised_tokens"],
                    }
                )

            if split == "train":
                train_shifted_tokens += result["shifted_supervised_tokens"]
                checked_train += 1
            else:
                eval_shifted_tokens += result["shifted_supervised_tokens"]
                checked_eval += 1

    if train_shifted_tokens != expected_train_shifted_tokens:
        errors.append(
            f"Training shifted-token total expected "
            f"{expected_train_shifted_tokens}; found {train_shifted_tokens}."
        )
    if eval_shifted_tokens != expected_eval_shifted_tokens:
        errors.append(
            f"Evaluation shifted-token total expected "
            f"{expected_eval_shifted_tokens}; found {eval_shifted_tokens}."
        )
    if metadata_mismatches:
        errors.append(
            f"{len(metadata_mismatches)} sample metadata counts do not match "
            "the shifted-label count. Review the original notebook's "
            "supervised_tokens convention before changing any data."
        )

    group_token_counts = calculate_optimizer_group_token_counts(
        loaders.train_loader,
        accumulation_steps=accumulation_steps,
    )
    expected_groups = expected_train_records // accumulation_steps
    if len(group_token_counts) != expected_groups:
        errors.append(
            f"Expected {expected_groups} optimizer groups; "
            f"found {len(group_token_counts)}."
        )
    if sum(group_token_counts) != train_shifted_tokens:
        errors.append("Optimizer-group token counts do not sum to training tokens.")

    eval_summary = summarize_evaluation_loader(loaders.eval_loader)
    if eval_summary["evaluation_examples"] != expected_eval_records:
        errors.append("Evaluation DataLoader batch count is incorrect.")

    # Inspect one collated batch for the audited 1x3x448x448 image shape.
    first_batch = next(iter(loaders.train_loader))
    if tuple(first_batch["pixel_values"].shape) != (1, 3, 448, 448):
        errors.append(
            "First training batch pixel_values shape is not (1, 3, 448, 448)."
        )

    plan = V3TrainingPlan(
        micro_batch_size=1,
        gradient_accumulation_steps=accumulation_steps,
        epochs=3,
        expected_training_examples=expected_train_records,
    )

    return {
        "status": "PASSED" if not errors else "FAILED",
        "execution_mode": "CPU-only data validation",
        "training_records": checked_train,
        "evaluation_records": checked_eval,
        "training_batches": len(loaders.train_loader),
        "evaluation_batches": len(loaders.eval_loader),
        "training_task_counts": dict(task_counts_train),
        "evaluation_task_counts": dict(task_counts_eval),
        "training_shifted_supervised_tokens": train_shifted_tokens,
        "evaluation_shifted_supervised_tokens": eval_shifted_tokens,
        "optimizer_groups": len(group_token_counts),
        "microbatches_per_group": accumulation_steps,
        "optimizer_group_token_counts": group_token_counts,
        "training_plan": {
            "groups_per_epoch": plan.groups_per_epoch,
            "epochs": plan.epochs,
            "planned_total_updates": plan.total_optimizer_updates,
            "executed_updates": 0,
        },
        "metadata_mismatches": metadata_mismatches,
        "errors": errors,
        "model_loaded": False,
        "forward_pass_run": False,
        "backward_pass_run": False,
        "optimizer_step_run": False,
        "adapter_modified": False,
        "dataset_modified": False,
    }
