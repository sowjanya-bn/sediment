"""
CR application layer.

apply_correction: applies a single CR to an entry view (non-destructive copy).
get_effective_entry: folds all CRs for an entry in timestamp order.

Originals are never mutated. CRs are stored as separate records linked by
entry_id + entry_hash — this keeps state_direction corrections as snapshot
annotations that do not foreclose relational computation across entries later.
"""
import json


def apply_correction(entry: dict, cr: dict) -> dict:
    """Return a new entry dict with the correction applied.

    Only the field targeted by cr_type is modified. All other fields are
    preserved exactly. previous_value and corrected_value are JSON-encoded
    strings in the CR record.
    """
    result = dict(entry)
    cr_type = cr["cr_type"]
    corrected = json.loads(cr["corrected_value"])

    if cr_type == "relabel":
        result["emotion_label"] = corrected

    elif cr_type == "valence_nudge":
        result["valence"] = max(-1.0, min(1.0, float(corrected)))

    elif cr_type == "confidence_override":
        score = max(0.0, min(1.0, float(corrected)))
        result["confidence_score"] = score
        result["low_confidence"] = score < 0.5

    elif cr_type == "state_direction_edit":
        # Stored as a snapshot annotation. Does not prevent future relational
        # computation — the original entry record is unchanged.
        result["state_direction"] = corrected

    elif cr_type == "theme_edit":
        result["themes"] = list(corrected)

    return result


def get_effective_entry(entry: dict, corrections: list[dict]) -> dict:
    """Apply all corrections for an entry in timestamp order.

    Returns the final effective view. The original entry dict is not modified.
    Corrections with a mismatched entry_hash are skipped — they were made
    against a different version of the entry.
    """
    result = dict(entry)
    for cr in sorted(corrections, key=lambda c: c["timestamp"]):
        if cr["entry_hash"] != entry.get("entry_hash"):
            continue
        result = apply_correction(result, cr)
    return result
