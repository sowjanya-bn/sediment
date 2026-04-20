"""
Correction record schema (v1.1+).

Corrections are surgical and field-level. A correction to one field
does not invalidate the rest of the record. Corrections are stored
independently — originals are never mutated.

state_direction is corrected here as a snapshot annotation. The longer-term
design is for state_direction to be computed relationally across entries.
This CR layer does not foreclose that transition: corrections are stored as
separate records linked by entry_id + entry_hash, not embedded in entries.
"""
from pydantic import BaseModel, field_validator

# indeterminate is a schema outcome, not a CR action
CR_TYPES = {
    "relabel",             # correct emotion_label
    "valence_nudge",       # adjust valence score
    "confidence_override", # flip or adjust confidence
    "state_direction_edit",# correct trajectory label
    "theme_edit",          # add or remove individual themes
}

# field_target values by cr_type — used for validation
_CR_FIELD_TARGETS = {
    "relabel": "emotion_label",
    "valence_nudge": "valence",
    "confidence_override": "confidence_score",
    "state_direction_edit": "state_direction",
    "theme_edit": "themes",
}


class CorrectionRecord(BaseModel):
    cr_id: str
    entry_id: str
    entry_hash: str
    schema_version: str        # must match the entry's schema_version
    cr_type: str               # from CR_TYPES
    field_target: str          # explicit field being corrected
    previous_value: str        # JSON-encoded (use json.dumps for scalars too)
    corrected_value: str       # JSON-encoded
    timestamp: str
    note: str | None = None

    @field_validator("cr_type")
    @classmethod
    def valid_cr_type(cls, v: str) -> str:
        if v not in CR_TYPES:
            raise ValueError(f"cr_type '{v}' not in: {CR_TYPES}")
        return v

    @field_validator("field_target")
    @classmethod
    def valid_field_target(cls, v: str) -> str:
        valid = set(_CR_FIELD_TARGETS.values())
        if v not in valid:
            raise ValueError(f"field_target '{v}' not in: {valid}")
        return v

    @field_validator("schema_version")
    @classmethod
    def schema_version_present(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("schema_version is required on correction records")
        return v


def validate_correction(data: dict) -> tuple[bool, str]:
    try:
        CorrectionRecord(**data)
        return True, ""
    except Exception as e:
        return False, str(e)
