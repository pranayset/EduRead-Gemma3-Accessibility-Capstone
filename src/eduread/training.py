"""Opt-in callable implementation of the audited EduRead V3 training loop.

Importing this module does not train. run_v3_training defaults to execute=False
and returns a plan without forward/backward passes, optimizer steps, or writes.
"""
from __future__ import annotations
import json
import math
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List
import torch

MODEL_KEYS = ("input_ids", "token_type_ids", "attention_mask", "pixel_values", "labels")


def create_v3_optimizer(model):
    """Initialize PagedAdamW8bit over the audited LoRA tensors only."""
    import bitsandbytes as bnb
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(named) != 72 or sum(p.numel() for _, p in named) != 921600:
        raise ValueError("Expected the audited 72 LoRA tensors / 921,600 parameters.")
    if any("lora_" not in n.lower() for n, _ in named):
        raise ValueError("Unexpected non-LoRA trainable parameter.")
    return bnb.optim.PagedAdamW8bit(
        [p for _, p in named], lr=2e-5, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01, min_8bit_size=4096,
    )


def create_v3_scheduler(optimizer):
    """Initialize the linear schedule (6 warmup, 120 total steps)."""
    from transformers import get_linear_schedule_with_warmup
    return get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=6, num_training_steps=120
    )


def evaluate_v3_loss(model, eval_loader, device="cuda"):
    """Token-weighted evaluation, without gradients or optimizer updates."""
    model.eval()
    weighted_loss = 0.0
    tokens = 0
    examples = 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {k: batch[k].to(device) for k in MODEL_KEYS}
            count = int((inputs["labels"][..., 1:] != -100).sum().item())
            if count <= 0:
                raise ValueError("Evaluation batch has no shifted supervised tokens.")
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                outputs = model(**inputs, use_cache=False, num_items_in_batch=count)
            loss = float(outputs.loss.item())
            if not math.isfinite(loss):
                raise FloatingPointError("Non-finite evaluation loss.")
            weighted_loss += loss * count
            tokens += count
            examples += int(inputs["labels"].shape[0])
            del outputs, inputs
    if examples != 40 or tokens != 846:
        raise ValueError(f"Expected V3 eval totals 40/846; got {examples}/{tokens}.")
    return {"evaluation_examples": examples, "evaluation_tokens": tokens,
            "evaluation_loss": weighted_loss / tokens}


def _run_epoch(model, processor, train_loader, eval_loader, optimizer, scheduler,
               group_counts, epoch, output_root, device):
    checkpoint = output_root / "checkpoints" / f"epoch_{epoch}"
    metrics_file = output_root / "logs" / f"epoch_{epoch}_metrics.json"
    if checkpoint.exists() or metrics_file.exists():
        raise FileExistsError(f"Refusing to overwrite epoch {epoch} artifacts.")

    model.train()
    iterator = iter(train_loader)
    weighted_loss = 0.0
    total_tokens = 0
    group_losses, grad_norms, rates = [], [], []
    started = time.time()

    for group_index in range(40):
        batches = list(islice(iterator, 4))
        if len(batches) != 4:
            raise ValueError("Each V3 optimizer update requires four microbatches.")
        denominator = int(group_counts[group_index])
        actual = sum(int((b["labels"][..., 1:] != -100).sum().item()) for b in batches)
        if actual != denominator or denominator <= 0:
            raise ValueError(f"Shifted token denominator mismatch at group {group_index + 1}.")

        optimizer.zero_grad(set_to_none=True)
        group_loss = 0.0
        for micro_index, batch in enumerate(batches, 1):
            inputs = {k: batch[k].to(device) for k in MODEL_KEYS}
            with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
                outputs = model(**inputs, use_cache=False, num_items_in_batch=denominator)
            loss = outputs.loss
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, group {group_index + 1}, microbatch {micro_index}.")
            value = float(loss.detach().item())
            loss.backward()
            group_loss += value
            del outputs, loss, inputs

        trainable = [p for p in model.parameters() if p.requires_grad]
        if len(trainable) != 72 or any(p.grad is None for p in trainable):
            raise RuntimeError("Expected gradients for all 72 LoRA tensors.")
        if not all(torch.isfinite(p.grad).all().item() for p in trainable):
            raise FloatingPointError("Non-finite LoRA gradient.")
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        if not torch.isfinite(torch.as_tensor(norm)).item():
            raise FloatingPointError("Non-finite gradient norm.")

        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        group_losses.append(group_loss)
        grad_norms.append(float(norm))
        rates.append(lr_used)
        weighted_loss += group_loss * denominator
        total_tokens += denominator
        if (group_index + 1) % 5 == 0:
            print(f"Epoch {epoch}/3 update {group_index + 1}/40 | loss {group_loss:.6f} | grad {float(norm):.6f} | lr {lr_used:.10f}")
        del batches, trainable, norm

    if total_tokens != 3545 or any(p.grad is not None for p in model.parameters() if p.requires_grad):
        raise RuntimeError("Unexpected V3 training-token total or uncleared gradients.")
    train_loss = weighted_loss / total_tokens
    if not math.isfinite(train_loss):
        raise FloatingPointError("Non-finite training loss.")

    evaluation = evaluate_v3_loss(model, eval_loader, device=device)
    model.train()
    checkpoint.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(checkpoint, safe_serialization=True)
    processor.save_pretrained(checkpoint)
    torch.save({
        "epoch": epoch, "global_step": epoch * 40,
        "training_loss": train_loss, "evaluation_loss": evaluation["evaluation_loss"],
        "training_tokens": total_tokens, "evaluation_tokens": evaluation["evaluation_tokens"],
        "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
    }, checkpoint / "training_state.pt")

    metrics = {
        "epoch": epoch, "optimizer_updates": 40, "global_step": epoch * 40,
        "training_examples": 160, "training_tokens": total_tokens,
        "training_loss": train_loss, **evaluation,
        "learning_rate_final": float(optimizer.param_groups[0]["lr"]),
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "use_reentrant": False, "elapsed_seconds": time.time() - started,
        "group_losses": group_losses, "gradient_norms": grad_norms,
        "learning_rates_used": rates,
    }
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def run_v3_training(*, model: Any, processor: Any, train_loader: Any,
                    eval_loader: Any, optimizer: Any, scheduler: Any,
                    group_token_counts: List[int], output_dir: str | Path,
                    execute: bool = False, device: str = "cuda") -> Dict[str, Any]:
    """Run the original three epochs only after explicit execute=True."""
    if not execute:
        return {"executed": False, "planned_epochs": 3,
                "planned_optimizer_updates": 120,
                "reason": "Training is disabled by default; no model or file operations performed."}

    if not torch.cuda.is_available() or not device.startswith("cuda"):
        raise RuntimeError("Audited V3 training requires CUDA.")
    if len(train_loader) != 160 or len(eval_loader) != 40 or len(group_token_counts) != 40:
        raise ValueError("Expected 160 train batches, 40 eval batches, and 40 token denominators.")
    if len(optimizer.state) != 0 or scheduler.last_epoch != 0:
        raise RuntimeError("Fresh optimizer/scheduler required; resume is not implemented.")
    if not getattr(model, "is_gradient_checkpointing", False):
        raise RuntimeError("V3 gradient checkpointing must be enabled.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    if len(trainable) != 72 or sum(p.numel() for p in trainable) != 921600:
        raise RuntimeError("Model trainable state does not match audited V3 LoRA.")
    if any(p.grad is not None for p in trainable):
        raise RuntimeError("Existing gradients detected; refusing to start.")

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoints", "logs", "final_adapter", "evaluation"):
        (destination / name).mkdir(exist_ok=True)

    results = []
    for epoch in range(1, 4):
        results.append(_run_epoch(
            model, processor, train_loader, eval_loader, optimizer, scheduler,
            group_token_counts, epoch, destination, device
        ))

    final_adapter = destination / "final_adapter"
    if any(final_adapter.iterdir()):
        raise FileExistsError(f"Refusing to overwrite final adapter directory: {final_adapter}")
    model.save_pretrained(final_adapter, safe_serialization=True)
    processor.save_pretrained(final_adapter)
    return {"executed": True, "epochs": results, "global_step": 120}
