"""Read-only training-plan reconstruction; no optimizer or model steps."""

from dataclasses import dataclass
from itertools import islice
from collections.abc import Iterable
import torch


@dataclass(frozen=True)
class V3TrainingPlan:
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    epochs: int = 3
    expected_training_examples: int = 160
    learning_rate: float = 2.0e-5
    optimizer_name: str = "paged_adamw_8bit"
    max_gradient_norm: float = 1.0
    warmup_steps: int = 6

    @property
    def groups_per_epoch(self) -> int:
        if self.expected_training_examples % self.gradient_accumulation_steps:
            raise ValueError("Training examples must divide evenly into accumulation groups.")
        return self.expected_training_examples // self.gradient_accumulation_steps

    @property
    def total_optimizer_updates(self) -> int:
        return self.groups_per_epoch * self.epochs


def iter_microbatch_groups(loader: Iterable, accumulation_steps: int = 4):
    """Yield sequential groups of four batches; does not call backward/step."""
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    iterator = iter(loader)
    while True:
        group = list(islice(iterator, accumulation_steps))
        if not group:
            break
        if len(group) != accumulation_steps:
            raise ValueError(
                f"Incomplete accumulation group: got {len(group)}, "
                f"expected {accumulation_steps}"
            )
        yield group


def shifted_token_count(batch: dict) -> int:
    """Count labels used by the V3 causal next-token objective."""
    labels = batch["labels"]
    return int((labels[..., 1:] != -100).sum().item())


def calculate_optimizer_group_token_counts(
    train_loader,
    accumulation_steps: int = 4,
) -> list[int]:
    """Recreate the 40 token denominators from sequential groups of 4.

    This is a CPU data audit only. It does not create an optimizer or train.
    """
    counts = []
    for group in iter_microbatch_groups(train_loader, accumulation_steps):
        count = sum(shifted_token_count(batch) for batch in group)
        if count <= 0:
            raise ValueError("An optimizer group has no shifted supervised tokens.")
        counts.append(count)
    return counts


def validate_training_plan(plan: V3TrainingPlan) -> dict:
    groups = plan.groups_per_epoch
    return {
        "micro_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "groups_per_epoch": groups,
        "epochs": plan.epochs,
        "total_optimizer_updates": plan.total_optimizer_updates,
        "learning_rate": plan.learning_rate,
        "optimizer": plan.optimizer_name,
        "training_execution": "NOT RUN",
    }


def run_v3_training(*args, execute: bool = False, **kwargs):
    """Safety stop: this extraction package does not run training."""
    if not execute:
        raise RuntimeError(
            "Training is disabled in the extraction workflow. "
            "No model, optimizer, adapter, or checkpoint was changed."
        )
    raise NotImplementedError(
        "Training execution is intentionally not implemented in this "
        "read-only extraction. Use the audited notebook training cells only "
        "after a separately approved training plan."
    )
