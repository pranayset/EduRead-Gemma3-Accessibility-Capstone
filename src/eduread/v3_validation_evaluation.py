"""Read-only evaluation-set accounting; no model forward pass is performed."""

from .v3_validation_training import shifted_token_count


def summarize_evaluation_loader(eval_loader) -> dict:
    examples = 0
    shifted_tokens = 0
    for batch in eval_loader:
        examples += 1
        shifted_tokens += shifted_token_count(batch)
    return {
        "evaluation_examples": examples,
        "evaluation_shifted_supervised_tokens": shifted_tokens,
        "model_evaluation_run": False,
    }
