"""Evaluation helpers for EduRead generation outputs."""
from __future__ import annotations

import re
import string
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List


def normalize_text(text: str) -> str:
    value = str(text).lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", value).strip()


def compare_prediction(
    generated_text: str,
    reference_text: str,
) -> Dict[str, Any]:
    """Return exact match and character similarity for one prediction."""
    generated = normalize_text(generated_text)
    reference = normalize_text(reference_text)
    return {
        "exact_match": generated == reference,
        "character_similarity": SequenceMatcher(
            None, generated, reference
        ).ratio(),
    }


def summarize_predictions(
    predictions: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Summarize prediction records containing generated/reference text."""
    items: List[Dict[str, Any]] = []
    for item in predictions:
        metrics = compare_prediction(
            item.get("generated_text", ""),
            item.get("reference_text", ""),
        )
        items.append({**item, **metrics})

    if not items:
        return {"examples": 0, "exact_matches": 0, "mean_character_similarity": 0.0}

    return {
        "examples": len(items),
        "exact_matches": sum(bool(x["exact_match"]) for x in items),
        "mean_character_similarity": sum(
            float(x["character_similarity"]) for x in items
        ) / len(items),
        "records": items,
    }
