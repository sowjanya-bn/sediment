from typing import Any, Literal, Optional, TypedDict


class SedimentState(TypedDict, total=False):
    raw_text: str
    backend: Literal["anthropic", "ollama"]
    ollama_model: str

    entry_id: str
    timestamp: str
    entry_hash: str
    snippet: str

    is_short_or_flat: bool
    penalty: float

    llm_result: dict[str, Any]
    record: dict[str, Any]

    validation_passed: bool
    validation_error: str
    validation_result: Literal["short_input", "pass", "retry", "fail"]

    strict_retry_used: bool
    extraction_error: str