"""
Tests for sediment extraction logic.
All tests mock the Anthropic API — no real API calls are made.
"""
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from extractor.extract import extract, is_short_or_flat, uncertainty_penalty, _sha256
from extractor.schema import validate_record, SignalRecord, EMOTION_LABELS, STATE_DIRECTIONS
from corrections.schema import validate_correction, CR_TYPES
from corrections.apply import apply_correction, get_effective_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(**overrides) -> dict:
    """Return a minimal valid LLM response dict with optional field overrides."""
    base = {
        "input_type": "affective",
        "valence": 0.3,
        "arousal": "medium",
        "emotion_label": "anticipation",
        "themes": ["purpose", "momentum"],
        "intensity": "medium",
        "salient_focus": "new project taking shape",
        "state_direction": "building_momentum",
        "low_confidence": False,
        "confidence_score": 0.85,
    }
    base.update(overrides)
    return base


def _make_record(**overrides) -> dict:
    """Return a minimal valid full record (including system-assigned fields)."""
    base = {
        "id": "test-id",
        "timestamp": "2026-04-20T10:00:00Z",
        "schema_version": "1.1",
        "raw_text": "some reflective text here",
        "input_type": "affective",
        "valence": 0.3,
        "arousal": "medium",
        "emotion_label": "anticipation",
        "themes": ["purpose"],
        "intensity": "medium",
        "salient_focus": "new project taking shape",
        "state_direction": "building_momentum",
        "low_confidence": False,
        "confidence_score": 0.85,
        "entry_hash": "abc" * 20,
    }
    base.update(overrides)
    return base


def _mock_message(response_dict: dict) -> MagicMock:
    """Wrap a dict in a fake anthropic Message object."""
    content_block = MagicMock()
    content_block.text = json.dumps(response_dict)
    msg = MagicMock()
    msg.content = [content_block]
    return msg


# ---------------------------------------------------------------------------
# 1. Same input → same hash always
# ---------------------------------------------------------------------------

class TestEntryHash:
    def test_same_input_same_hash(self):
        text = "I feel overwhelmed and uncertain about everything."
        assert _sha256(text) == _sha256(text)

    def test_different_inputs_different_hash(self):
        assert _sha256("hello world") != _sha256("goodbye world")

    def test_hash_is_sha256(self):
        text = "some reflective thought"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert _sha256(text) == expected

    @patch("extractor.extract._call_llm")
    def test_extracted_record_has_stable_hash(self, mock_llm):
        mock_llm.return_value = _make_llm_response()
        text = "I feel overwhelmed and uncertain about everything."
        record = extract(text)
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert record["entry_hash"] == expected_hash

    @patch("extractor.extract._call_llm")
    def test_two_extractions_same_text_same_hash(self, mock_llm):
        mock_llm.return_value = _make_llm_response()
        text = "Feeling stuck but curious."
        r1 = extract(text)
        r2 = extract(text)
        assert r1["entry_hash"] == r2["entry_hash"]


# ---------------------------------------------------------------------------
# 2. Short / flat input returns low_confidence=True
# ---------------------------------------------------------------------------

class TestShortFlatInput:
    def test_empty_string(self):
        record = extract("")
        assert record["low_confidence"] is True

    def test_very_short_input(self):
        record = extract("ok")
        assert record["low_confidence"] is True

    def test_flat_word_fine(self):
        record = extract("fine")
        assert record["low_confidence"] is True

    def test_flat_word_meh(self):
        record = extract("meh")
        assert record["low_confidence"] is True

    def test_nine_char_input(self):
        record = extract("123456789")
        assert record["low_confidence"] is True

    def test_is_short_or_flat_helper(self):
        assert is_short_or_flat("ok") is True
        assert is_short_or_flat("meh") is True
        assert is_short_or_flat("fine") is True
        assert is_short_or_flat("short") is True  # < 10 chars
        assert is_short_or_flat("This is a longer, meaningful sentence.") is False

    @patch("extractor.extract._call_llm")
    def test_long_input_not_low_confidence_by_default(self, mock_llm):
        mock_llm.return_value = _make_llm_response(confidence_score=0.9, low_confidence=False)
        record = extract("I have been thinking deeply about the nature of my work and purpose.")
        assert record["low_confidence"] is False


# ---------------------------------------------------------------------------
# 3. Out-of-vocabulary values trigger low_confidence
# ---------------------------------------------------------------------------

class TestOutOfVocabulary:
    @patch("extractor.extract._call_llm")
    def test_bad_emotion_label_triggers_retry_and_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(emotion_label="excitement")
        mock_llm.return_value = bad_response
        record = extract("I am super excited about this new project and can't wait to start.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_trust_is_no_longer_valid(self, mock_llm):
        bad_response = _make_llm_response(emotion_label="trust")
        mock_llm.return_value = bad_response
        record = extract("I am super excited about this new project and can't wait to start.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_bad_theme_triggers_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(themes=["happiness"])
        mock_llm.return_value = bad_response
        record = extract("I am super excited about this new project and can't wait to start.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_bad_state_direction_triggers_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(state_direction="improving")
        mock_llm.return_value = bad_response
        record = extract("I am working hard and things are going well today for once.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_bad_arousal_triggers_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(arousal="very_high")
        mock_llm.return_value = bad_response
        record = extract("Everything is moving so fast I can barely keep up with the pace.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_retry_succeeds_clears_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(emotion_label="excitement")
        good_response = _make_llm_response()
        mock_llm.side_effect = [bad_response, good_response]
        record = extract("I am excited about the project direction and next steps planned.")
        assert record["low_confidence"] is False


# ---------------------------------------------------------------------------
# 4. Schema validation — v1.1
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_record_passes(self):
        is_valid, err = validate_record(_make_record())
        assert is_valid is True
        assert err == ""

    def test_missing_input_type(self):
        data = _make_record()
        del data["input_type"]
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "input_type" in err

    def test_invalid_input_type(self):
        data = _make_record(input_type="emotional")
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "input_type" in err

    def test_missing_required_field(self):
        data = _make_record()
        del data["schema_version"]
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "schema_version" in err

    def test_missing_multiple_fields(self):
        data = {"id": "abc123", "raw_text": "some text"}
        is_valid, err = validate_record(data)
        assert is_valid is False

    def test_valence_out_of_range(self):
        data = _make_record(valence=2.5)
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "valence" in err

    def test_confidence_out_of_range(self):
        data = _make_record(confidence_score=1.5)
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "confidence_score" in err

    def test_too_many_themes(self):
        data = _make_record(themes=["purpose", "momentum", "uncertainty", "overwhelm"])
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "themes" in err

    def test_salient_focus_too_long(self):
        data = _make_record(salient_focus="one two three four five six seven")
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "salient_focus" in err

    def test_low_confidence_forced_true_when_score_low(self):
        data = _make_record(confidence_score=0.3, low_confidence=False)
        is_valid, _ = validate_record(data)
        assert is_valid is True
        record = SignalRecord(**data)
        assert record.low_confidence is True

    def test_low_confidence_forced_false_when_score_high(self):
        # Model self-reports low_confidence=True but score is 0.8 — must be corrected
        data = _make_record(confidence_score=0.8, low_confidence=True)
        is_valid, _ = validate_record(data)
        assert is_valid is True
        record = SignalRecord(**data)
        assert record.low_confidence is False


# ---------------------------------------------------------------------------
# 5. New v1.1 vocabulary
# ---------------------------------------------------------------------------

class TestNewVocabulary:
    def test_indeterminate_is_valid_emotion_label(self):
        data = _make_record(emotion_label="indeterminate", confidence_score=0.3)
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_neutral_is_valid_emotion_label(self):
        data = _make_record(emotion_label="neutral")
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_flat_is_valid_emotion_label(self):
        data = _make_record(emotion_label="flat")
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_ambivalent_is_valid_emotion_label(self):
        data = _make_record(emotion_label="ambivalent")
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_trust_is_invalid_emotion_label(self):
        data = _make_record(emotion_label="trust")
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "emotion_label" in err

    def test_indeterminate_is_valid_state_direction(self):
        data = _make_record(state_direction="indeterminate", confidence_score=0.3)
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_emotion_labels_complete(self):
        expected = {"joy", "sadness", "anger", "fear", "anticipation",
                    "neutral", "flat", "ambivalent", "indeterminate"}
        assert EMOTION_LABELS == expected

    def test_state_directions_include_indeterminate(self):
        assert "indeterminate" in STATE_DIRECTIONS
        assert "drifting" in STATE_DIRECTIONS  # not removed


# ---------------------------------------------------------------------------
# 6. Confidence score — derivation and penalty
# ---------------------------------------------------------------------------

class TestConfidenceScoreRule:
    @patch("extractor.extract._call_llm")
    def test_low_confidence_score_sets_flag(self, mock_llm):
        mock_llm.return_value = _make_llm_response(confidence_score=0.4, low_confidence=False)
        record = extract("I feel a strange mix of emotions I cannot quite identify right now.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_high_confidence_score_keeps_flag_false(self, mock_llm):
        mock_llm.return_value = _make_llm_response(confidence_score=0.8, low_confidence=False)
        record = extract("I feel deeply satisfied with the progress I made today on this project.")
        assert record["low_confidence"] is False

    @patch("extractor.extract._call_llm")
    def test_model_mismatch_corrected(self, mock_llm):
        # Model reports low_confidence=True but confidence_score=0.8 — must be fixed
        mock_llm.return_value = _make_llm_response(confidence_score=0.8, low_confidence=True)
        record = extract("I feel deeply satisfied with the progress I made today on this project.")
        assert record["low_confidence"] is False
        assert record["confidence_score"] == 0.8


class TestUncertaintyPenalty:
    def test_no_phrases_no_penalty(self):
        assert uncertainty_penalty("I feel great about everything today.") == 0.0

    def test_one_phrase_penalty(self):
        # "i guess" matches exactly one phrase
        assert uncertainty_penalty("I guess it's fine.") == 0.1

    def test_two_phrases_penalty(self):
        # "maybe" + "kind of" = 2 hits
        assert uncertainty_penalty("Maybe it's kind of okay.") == 0.2

    def test_three_phrases_max_penalty(self):
        # "maybe" + "kind of" + "sort of" = 3 hits
        assert uncertainty_penalty("Maybe it's kind of sort of fine.") == 0.3

    def test_case_insensitive(self):
        assert uncertainty_penalty("NOT SURE about this.") == 0.1

    @patch("extractor.extract._call_llm")
    def test_penalty_applied_to_confidence_score(self, mock_llm):
        mock_llm.return_value = _make_llm_response(confidence_score=0.8)
        record = extract("I guess this project is going somewhere eventually.")
        # "i guess" → penalty 0.1; 0.8 - 0.1 = 0.7
        assert record["confidence_score"] == pytest.approx(0.7, abs=0.01)

    @patch("extractor.extract._call_llm")
    def test_penalty_clamped_at_zero(self, mock_llm):
        mock_llm.return_value = _make_llm_response(confidence_score=0.2)
        record = extract("Not sure, maybe, kind of, I guess it's fine.")
        assert record["confidence_score"] >= 0.0


# ---------------------------------------------------------------------------
# 7. Input type routing
# ---------------------------------------------------------------------------

class TestInputTypeRouting:
    def test_valid_affective(self):
        data = _make_record(input_type="affective")
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_valid_metacognitive(self):
        data = _make_record(input_type="metacognitive")
        is_valid, err = validate_record(data)
        assert is_valid is True, err

    def test_invalid_input_type_rejected(self):
        data = _make_record(input_type="emotional")
        is_valid, err = validate_record(data)
        assert is_valid is False

    @patch("extractor.extract._call_llm")
    def test_metacognitive_confidence_capped(self, mock_llm):
        mock_llm.return_value = _make_llm_response(
            input_type="metacognitive", confidence_score=0.95
        )
        record = extract("I keep starting projects and abandoning them before they're done.")
        assert record["confidence_score"] <= 0.75

    @patch("extractor.extract._call_llm")
    def test_affective_confidence_not_capped(self, mock_llm):
        mock_llm.return_value = _make_llm_response(
            input_type="affective", confidence_score=0.95
        )
        record = extract("I feel deeply satisfied with what I accomplished today.")
        assert record["confidence_score"] == pytest.approx(0.95, abs=0.01)

    @patch("extractor.extract._call_llm")
    def test_extracted_record_includes_input_type(self, mock_llm):
        mock_llm.return_value = _make_llm_response(input_type="affective")
        record = extract("I feel overwhelmed by everything on my plate right now.")
        assert record["input_type"] in {"affective", "metacognitive"}


# ---------------------------------------------------------------------------
# 8. Correction layer — schema
# ---------------------------------------------------------------------------

class TestCorrectionSchema:
    def _make_cr(self, **overrides) -> dict:
        base = {
            "cr_id": "cr-001",
            "entry_id": "entry-001",
            "entry_hash": "abc" * 20,
            "schema_version": "1.1",
            "cr_type": "relabel",
            "field_target": "emotion_label",
            "previous_value": json.dumps("sadness"),
            "corrected_value": json.dumps("ambivalent"),
            "timestamp": "2026-04-20T10:00:00Z",
        }
        base.update(overrides)
        return base

    def test_valid_cr_passes(self):
        is_valid, err = validate_correction(self._make_cr())
        assert is_valid is True, err

    def test_invalid_cr_type_rejected(self):
        is_valid, err = validate_correction(self._make_cr(cr_type="indeterminate"))
        assert is_valid is False
        assert "cr_type" in err

    def test_indeterminate_not_a_cr_type(self):
        assert "indeterminate" not in CR_TYPES

    def test_schema_version_required(self):
        is_valid, err = validate_correction(self._make_cr(schema_version=""))
        assert is_valid is False
        assert "schema_version" in err

    def test_all_cr_types_valid(self):
        field_map = {
            "relabel": "emotion_label",
            "valence_nudge": "valence",
            "confidence_override": "confidence_score",
            "state_direction_edit": "state_direction",
            "theme_edit": "themes",
        }
        for cr_type, field in field_map.items():
            is_valid, err = validate_correction(
                self._make_cr(cr_type=cr_type, field_target=field)
            )
            assert is_valid is True, f"{cr_type}: {err}"

    def test_note_is_optional(self):
        cr = self._make_cr()
        cr.pop("note", None)
        is_valid, err = validate_correction(cr)
        assert is_valid is True, err


# ---------------------------------------------------------------------------
# 9. Correction layer — apply
# ---------------------------------------------------------------------------

class TestApplyCorrection:
    def _entry(self, **overrides) -> dict:
        base = {
            "entry_hash": "hash-abc",
            "emotion_label": "sadness",
            "valence": -0.6,
            "confidence_score": 0.85,
            "low_confidence": False,
            "state_direction": "losing_direction",
            "themes": ["uncertainty", "overwhelm"],
        }
        base.update(overrides)
        return base

    def _cr(self, cr_type, corrected, entry_hash="hash-abc", timestamp="t1") -> dict:
        return {
            "cr_type": cr_type,
            "entry_hash": entry_hash,
            "corrected_value": json.dumps(corrected),
            "timestamp": timestamp,
        }

    def test_relabel(self):
        result = apply_correction(self._entry(), self._cr("relabel", "ambivalent"))
        assert result["emotion_label"] == "ambivalent"
        assert result["valence"] == -0.6  # untouched

    def test_valence_nudge(self):
        result = apply_correction(self._entry(), self._cr("valence_nudge", -0.3))
        assert result["valence"] == pytest.approx(-0.3)
        assert result["emotion_label"] == "sadness"  # untouched

    def test_valence_clamped_high(self):
        result = apply_correction(self._entry(), self._cr("valence_nudge", 2.5))
        assert result["valence"] == 1.0

    def test_valence_clamped_low(self):
        result = apply_correction(self._entry(), self._cr("valence_nudge", -2.5))
        assert result["valence"] == -1.0

    def test_confidence_override_high(self):
        result = apply_correction(self._entry(), self._cr("confidence_override", 0.9))
        assert result["confidence_score"] == pytest.approx(0.9)
        assert result["low_confidence"] is False

    def test_confidence_override_low_syncs_flag(self):
        result = apply_correction(self._entry(), self._cr("confidence_override", 0.3))
        assert result["confidence_score"] == pytest.approx(0.3)
        assert result["low_confidence"] is True

    def test_confidence_override_does_not_touch_other_fields(self):
        result = apply_correction(self._entry(), self._cr("confidence_override", 0.2))
        assert result["emotion_label"] == "sadness"
        assert result["state_direction"] == "losing_direction"

    def test_state_direction_edit(self):
        result = apply_correction(self._entry(), self._cr("state_direction_edit", "stuck"))
        assert result["state_direction"] == "stuck"
        assert result["emotion_label"] == "sadness"  # untouched

    def test_theme_edit(self):
        result = apply_correction(self._entry(), self._cr("theme_edit", ["isolation"]))
        assert result["themes"] == ["isolation"]
        assert result["emotion_label"] == "sadness"  # untouched

    def test_original_entry_not_mutated(self):
        entry = self._entry()
        apply_correction(entry, self._cr("relabel", "joy"))
        assert entry["emotion_label"] == "sadness"

    def test_hash_mismatch_skips_cr(self):
        entry = self._entry()
        crs = [self._cr("relabel", "joy", entry_hash="WRONG")]
        result = get_effective_entry(entry, crs)
        assert result["emotion_label"] == "sadness"

    def test_stacked_crs_applied_in_order(self):
        entry = self._entry()
        crs = [
            self._cr("relabel", "flat", timestamp="t1"),
            self._cr("state_direction_edit", "stuck", timestamp="t2"),
        ]
        result = get_effective_entry(entry, crs)
        assert result["emotion_label"] == "flat"
        assert result["state_direction"] == "stuck"
        assert result["valence"] == -0.6  # untouched

    def test_later_cr_overrides_earlier(self):
        entry = self._entry()
        crs = [
            self._cr("relabel", "flat", timestamp="t1"),
            self._cr("relabel", "neutral", timestamp="t2"),
        ]
        result = get_effective_entry(entry, crs)
        assert result["emotion_label"] == "neutral"
