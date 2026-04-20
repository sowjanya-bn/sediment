import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import anthropic
import ollama as _ollama

from .prompt import SYSTEM_PROMPT, STRICT_SYSTEM_PROMPT, build_user_message
from .schema import (
    validate_record, EMOTION_LABELS, STATE_DIRECTIONS, THEMES,
    INPUT_TYPES, UNCERTAINTY_PHRASES, SCHEMA_VERSION,
)

# --- Logging setup ---

_SEDIMENT_DIR = Path(os.path.expanduser("~/.sediment"))
_LOG_FILE = _SEDIMENT_DIR / "extraction.log"


def _get_logger() -> logging.Logger:
    _SEDIMENT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sediment.extraction")
    if not logger.handlers:
        handler = logging.FileHandler(_LOG_FILE)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --- Input analysis ---

_FLAT_INPUTS = {"ok", "fine", "meh", "okay", "k", "yeah", "yep", "nope", "no", "yes"}


def is_short_or_flat(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 10:
        return True
    if stripped.lower() in _FLAT_INPUTS:
        return True
    return False


def uncertainty_penalty(text: str) -> float:
    """Return a confidence penalty (0.0–0.3) based on uncertainty phrase density."""
    lower = text.lower()
    hits = sum(1 for phrase in UNCERTAINTY_PHRASES if phrase in lower)
    if hits == 0:
        return 0.0
    if hits == 1:
        return 0.1
    if hits == 2:
        return 0.2
    return 0.3


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_input_default(entry_id: str, timestamp: str, raw_text: str, entry_hash: str) -> dict:
    return {
        "id": entry_id,
        "timestamp": timestamp,
        "schema_version": SCHEMA_VERSION,
        "raw_text": raw_text,
        "input_type": "affective",
        "valence": 0.0,
        "arousal": "low",
        "emotion_label": "indeterminate",
        "themes": ["uncertainty"],
        "intensity": "low",
        "salient_focus": "no clear focus",
        "state_direction": "indeterminate",
        "low_confidence": True,
        "confidence_score": 0.1,
        "entry_hash": entry_hash,
    }


def _force_defaults(result: dict) -> dict:
    """Fill missing or invalid fields with safe defaults after validation failure."""
    defaults = {
        "input_type": "affective",
        "valence": 0.0,
        "arousal": "low",
        "emotion_label": "indeterminate",
        "themes": ["uncertainty"],
        "intensity": "low",
        "salient_focus": "no clear focus",
        "state_direction": "indeterminate",
        "low_confidence": True,
        "confidence_score": 0.1,
    }
    for key, val in defaults.items():
        if result.get(key) is None:
            result[key] = val
    # Clamp numeric ranges
    try:
        result["valence"] = max(-1.0, min(1.0, float(result["valence"])))
    except (TypeError, ValueError):
        result["valence"] = 0.0
    try:
        result["confidence_score"] = max(0.0, min(1.0, float(result["confidence_score"])))
    except (TypeError, ValueError):
        result["confidence_score"] = 0.1
    # Fix out-of-vocabulary values
    if result.get("input_type") not in INPUT_TYPES:
        result["input_type"] = "affective"
    if result.get("emotion_label") not in EMOTION_LABELS:
        result["emotion_label"] = "indeterminate"
    if result.get("state_direction") not in STATE_DIRECTIONS:
        result["state_direction"] = "indeterminate"
    if result.get("arousal") not in {"low", "medium", "high"}:
        result["arousal"] = "low"
    if result.get("intensity") not in {"low", "medium", "high"}:
        result["intensity"] = "low"
    # Fix themes
    themes = result.get("themes")
    if not isinstance(themes, list) or len(themes) == 0:
        result["themes"] = ["uncertainty"]
    else:
        valid = [t for t in themes if t in THEMES]
        result["themes"] = valid if valid else ["uncertainty"]
    # Fix salient_focus word count
    if len(str(result.get("salient_focus", "")).strip().split()) < 3:
        result["salient_focus"] = "no clear focus"
    return result


# --- LLM call ---

def _strip_fences(content: str) -> str:
    if content.startswith("```"):
        lines = content.splitlines()
        return "\n".join(line for line in lines if not line.startswith("```")).strip()
    return content


def _call_llm_anthropic(raw_text: str, strict: bool = False) -> dict:
    client = anthropic.Anthropic()
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0.1,
        system=system,
        messages=[{"role": "user", "content": build_user_message(raw_text)}],
    )
    return json.loads(_strip_fences(message.content[0].text.strip()))


def _call_llm_ollama(raw_text: str, model: str, strict: bool = False) -> dict:
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    response = _ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": build_user_message(raw_text)},
        ],
        options={"temperature": 0.1},
    )
    return json.loads(_strip_fences(response.message.content.strip()))


def _call_llm(raw_text: str, strict: bool = False, backend: str = "anthropic", ollama_model: str = "llama3.2") -> dict:
    if backend == "ollama":
        return _call_llm_ollama(raw_text, model=ollama_model, strict=strict)
    return _call_llm_anthropic(raw_text, strict=strict)


# --- Main extract function ---

def extract(raw_text: str, backend: str = "anthropic", ollama_model: str = "llama3.2") -> dict:
    logger = _get_logger()
    entry_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry_hash = _sha256(raw_text)
    snippet = raw_text[:80].replace("\n", " ")

    if is_short_or_flat(raw_text):
        record = short_input_default(entry_id, timestamp, raw_text, entry_hash)
        logger.info(
            "id=%s snippet=%r validation_result=short_input low_confidence=True",
            entry_id, snippet,
        )
        return record

    # Compute uncertainty penalty before LLM call — applied to model's self-reported score
    penalty = uncertainty_penalty(raw_text)

    validation_result = "fail"
    try:
        result = _call_llm(raw_text, strict=False, backend=backend, ollama_model=ollama_model)
        is_valid, err = validate_record({
            **result,
            "id": entry_id,
            "timestamp": timestamp,
            "raw_text": raw_text,
            "entry_hash": entry_hash,
            "schema_version": SCHEMA_VERSION,
        })
        if is_valid:
            validation_result = "pass"
        else:
            logger.info(
                "id=%s snippet=%r validation_result=retry err=%s",
                entry_id, snippet, err,
            )
            result = _call_llm(raw_text, strict=True, backend=backend, ollama_model=ollama_model)
            is_valid2, err2 = validate_record({
                **result,
                "id": entry_id,
                "timestamp": timestamp,
                "raw_text": raw_text,
                "entry_hash": entry_hash,
                "schema_version": SCHEMA_VERSION,
            })
            if is_valid2:
                validation_result = "pass"
            else:
                result = _force_defaults(result)
                result["low_confidence"] = True
                validation_result = "fail"
                logger.info(
                    "id=%s snippet=%r validation_result=fail low_confidence=True err=%s",
                    entry_id, snippet, err2,
                )
    except Exception as e:
        logger.error("id=%s snippet=%r extraction_error=%s", entry_id, snippet, str(e))
        result = _force_defaults({})
        result["low_confidence"] = True
        validation_result = "fail"

    result["id"] = entry_id
    result["timestamp"] = timestamp
    result["raw_text"] = raw_text
    result["entry_hash"] = entry_hash
    result["schema_version"] = SCHEMA_VERSION

    # Apply uncertainty phrase penalty to model's self-reported confidence
    if penalty > 0:
        raw_score = result.get("confidence_score", 0.5)
        result["confidence_score"] = max(0.0, round(raw_score - penalty, 2))

    # Metacognitive entries: cap confidence — patterns ≠ present affective state
    if result.get("input_type") == "metacognitive":
        result["confidence_score"] = min(result.get("confidence_score", 0.5), 0.75)

    result["low_confidence"] = result.get("confidence_score", 0.0) < 0.5 or validation_result == "fail"

    low_conf = result.get("low_confidence", False)
    logger.info(
        "id=%s input_type=%s snippet=%r validation_result=%s low_confidence=%s penalty=%.1f",
        entry_id, result.get("input_type", "?"), snippet, validation_result, low_conf, penalty,
    )

    return result
