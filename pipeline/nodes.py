from datetime import datetime, timezone
from uuid import uuid4

from extractor.core import (
    _call_llm,
    _force_defaults,
    _get_logger,
    _sha256,
    is_short_or_flat,
    short_input_default,
    uncertainty_penalty,
)

from extractor.schema import SCHEMA_VERSION, validate_record

from .state import SedimentState


def ingest_metadata(state: SedimentState) -> SedimentState:
    raw_text = state["raw_text"]
    entry_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry_hash = _sha256(raw_text)
    snippet = raw_text[:80].replace("\n", " ")
    return {
        "entry_id": entry_id,
        "timestamp": timestamp,
        "entry_hash": entry_hash,
        "snippet": snippet,
    }


def precheck_input(state: SedimentState) -> SedimentState:
    return {
        "is_short_or_flat": is_short_or_flat(state["raw_text"]),
    }


def short_input_default_node(state: SedimentState) -> SedimentState:
    record = short_input_default(
        state["entry_id"],
        state["timestamp"],
        state["raw_text"],
        state["entry_hash"],
    )
    logger = _get_logger()
    logger.info(
        "id=%s snippet=%r validation_result=short_input low_confidence=True",
        state["entry_id"],
        state["snippet"],
    )
    return {
        "record": record,
        "validation_result": "short_input",
    }


def compute_penalty(state: SedimentState) -> SedimentState:
    return {
        "penalty": uncertainty_penalty(state["raw_text"]),
    }


def extract_initial(state: SedimentState) -> SedimentState:
    result = _call_llm(
        state["raw_text"],
        strict=False,
        backend=state.get("backend", "anthropic"),
        ollama_model=state.get("ollama_model", "llama3.2"),
    )
    return {
        "llm_result": result,
        "strict_retry_used": False,
    }


def validate_initial(state: SedimentState) -> SedimentState:
    candidate = {
        **state["llm_result"],
        "id": state["entry_id"],
        "timestamp": state["timestamp"],
        "raw_text": state["raw_text"],
        "entry_hash": state["entry_hash"],
        "schema_version": SCHEMA_VERSION,
    }
    is_valid, err = validate_record(candidate)
    logger = _get_logger()

    if is_valid:
        return {
            "validation_passed": True,
            "validation_error": "",
            "validation_result": "pass",
        }

    logger.info(
        "id=%s snippet=%r validation_result=retry err=%s",
        state["entry_id"],
        state["snippet"],
        err,
    )
    return {
        "validation_passed": False,
        "validation_error": err,
        "validation_result": "retry",
    }


def extract_strict(state: SedimentState) -> SedimentState:
    result = _call_llm(
        state["raw_text"],
        strict=True,
        backend=state.get("backend", "anthropic"),
        ollama_model=state.get("ollama_model", "llama3.2"),
    )
    return {
        "llm_result": result,
        "strict_retry_used": True,
    }


def validate_strict(state: SedimentState) -> SedimentState:
    candidate = {
        **state["llm_result"],
        "id": state["entry_id"],
        "timestamp": state["timestamp"],
        "raw_text": state["raw_text"],
        "entry_hash": state["entry_hash"],
        "schema_version": SCHEMA_VERSION,
    }
    is_valid, err = validate_record(candidate)
    logger = _get_logger()

    if is_valid:
        return {
            "validation_passed": True,
            "validation_error": "",
            "validation_result": "pass",
        }

    logger.info(
        "id=%s snippet=%r validation_result=fail low_confidence=True err=%s",
        state["entry_id"],
        state["snippet"],
        err,
    )
    return {
        "validation_passed": False,
        "validation_error": err,
        "validation_result": "fail",
    }


def fallback_defaults(state: SedimentState) -> SedimentState:
    record = _force_defaults(state.get("llm_result", {}))
    record["low_confidence"] = True
    return {
        "record": record,
    }


def handle_extraction_error(state: SedimentState) -> SedimentState:
    logger = _get_logger()
    logger.error(
        "id=%s snippet=%r extraction_error=%s",
        state.get("entry_id", "?"),
        state.get("snippet", ""),
        state.get("extraction_error", "unknown"),
    )
    record = _force_defaults({})
    record["low_confidence"] = True
    return {
        "record": record,
        "validation_result": "fail",
    }


def finalize_record(state: SedimentState) -> SedimentState:
    result = state.get("record") or dict(state["llm_result"])

    result["id"] = state["entry_id"]
    result["timestamp"] = state["timestamp"]
    result["raw_text"] = state["raw_text"]
    result["entry_hash"] = state["entry_hash"]
    result["schema_version"] = SCHEMA_VERSION

    penalty = state.get("penalty", 0.0)
    if penalty > 0:
        raw_score = result.get("confidence_score", 0.5)
        result["confidence_score"] = max(0.0, round(raw_score - penalty, 2))

    if result.get("input_type") == "metacognitive":
        result["confidence_score"] = min(result.get("confidence_score", 0.5), 0.75)

    validation_result = state.get("validation_result", "fail")
    result["low_confidence"] = (
        result.get("confidence_score", 0.0) < 0.5 or validation_result == "fail"
    )

    logger = _get_logger()
    logger.info(
        "id=%s input_type=%s snippet=%r validation_result=%s low_confidence=%s penalty=%.1f",
        state["entry_id"],
        result.get("input_type", "?"),
        state["snippet"],
        validation_result,
        result.get("low_confidence", False),
        penalty,
    )

    return {
        "record": result,
    }


def run_extract_initial(state: SedimentState) -> SedimentState:
    try:
        return extract_initial(state)
    except Exception as e:
        return {"extraction_error": str(e), "validation_result": "fail"}


def run_extract_strict(state: SedimentState) -> SedimentState:
    try:
        return extract_strict(state)
    except Exception as e:
        return {"extraction_error": str(e), "validation_result": "fail"}