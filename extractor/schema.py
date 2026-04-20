from typing import Literal
from pydantic import BaseModel, field_validator, model_validator

SCHEMA_VERSION = "1.1"

# trust removed — low-signal fallback replaced by indeterminate
EMOTION_LABELS = {
    "joy", "sadness", "anger", "fear", "anticipation",
    "neutral", "flat", "ambivalent", "indeterminate",
}

THEMES = {
    "uncertainty", "overwhelm", "purpose", "traction",
    "self_trust", "isolation", "momentum", "creative_satisfaction",
}

# indeterminate = unified uncertainty output, not a separate mechanism per field
STATE_DIRECTIONS = {
    "gaining_clarity", "losing_direction", "building_momentum",
    "stuck", "regaining_confidence", "drifting", "indeterminate",
}

INPUT_TYPES = {"affective", "metacognitive"}

AROUSAL_VALUES = {"low", "medium", "high"}
INTENSITY_VALUES = {"low", "medium", "high"}

# Phrases that signal genuine uncertainty — used to down-rank confidence in extract.py
UNCERTAINTY_PHRASES = (
    "not sure",
    "i keep",
    "i don't know why",
    "i guess",
    "i'm not sure",
    "maybe",
    "kind of",
    "sort of",
    "i can't tell",
    "i wonder",
    "i don't know",
)


class SignalRecord(BaseModel):
    id: str
    timestamp: str
    schema_version: str
    raw_text: str
    input_type: str                          # affective | metacognitive
    valence: float
    arousal: Literal["low", "medium", "high"]
    emotion_label: str
    themes: list[str]
    intensity: Literal["low", "medium", "high"]
    salient_focus: str
    state_direction: str
    low_confidence: bool
    confidence_score: float
    entry_hash: str

    @field_validator("input_type")
    @classmethod
    def valid_input_type(cls, v: str) -> str:
        if v not in INPUT_TYPES:
            raise ValueError(f"input_type '{v}' not in vocabulary: {INPUT_TYPES}")
        return v

    @field_validator("valence")
    @classmethod
    def valence_range(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"valence must be between -1.0 and 1.0, got {v}")
        return v

    @field_validator("confidence_score")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence_score must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("emotion_label")
    @classmethod
    def valid_emotion(cls, v: str) -> str:
        if v not in EMOTION_LABELS:
            raise ValueError(f"emotion_label '{v}' not in vocabulary: {EMOTION_LABELS}")
        return v

    @field_validator("themes")
    @classmethod
    def valid_themes(cls, v: list[str]) -> list[str]:
        if not (1 <= len(v) <= 3):
            raise ValueError(f"themes must have 1-3 items, got {len(v)}")
        invalid = set(v) - THEMES
        if invalid:
            raise ValueError(f"themes {invalid} not in vocabulary: {THEMES}")
        return v

    @field_validator("state_direction")
    @classmethod
    def valid_state_direction(cls, v: str) -> str:
        if v not in STATE_DIRECTIONS:
            raise ValueError(f"state_direction '{v}' not in vocabulary: {STATE_DIRECTIONS}")
        return v

    @field_validator("salient_focus")
    @classmethod
    def focus_length(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) < 3:
            raise ValueError(f"salient_focus must be at least 3 words, got {len(words)}: '{v}'")
        if len(words) > 6:
            raise ValueError(f"salient_focus must be under 6 words, got {len(words)}: '{v}'")
        return v

    @model_validator(mode="after")
    def low_confidence_matches_score(self) -> "SignalRecord":
        object.__setattr__(self, "low_confidence", self.confidence_score < 0.5)
        return self


def validate_record(data: dict) -> tuple[bool, str]:
    """Validate a dict against SignalRecord schema. Returns (is_valid, error_message)."""
    try:
        SignalRecord(**data)
        return True, ""
    except Exception as e:
        return False, str(e)
