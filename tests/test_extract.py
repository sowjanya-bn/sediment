"""
Tests for sediment extraction logic.
All tests mock the Anthropic API — no real API calls are made.
"""
import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from extractor.extract import extract, is_short_or_flat, _sha256
from extractor.schema import validate_record, SignalRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm_response(**overrides) -> dict:
    """Return a minimal valid LLM response dict with optional field overrides."""
    base = {
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
        record1 = extract(text)
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert record1["entry_hash"] == expected_hash

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
        # "123456789" is exactly 9 chars → short
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
        bad_response = _make_llm_response(emotion_label="excitement")  # not in vocab
        mock_llm.return_value = bad_response  # both calls return bad response
        record = extract("I am super excited about this new project and can't wait to start.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_bad_theme_triggers_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(themes=["happiness"])  # not in vocab
        mock_llm.return_value = bad_response
        record = extract("I am super excited about this new project and can't wait to start.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_bad_state_direction_triggers_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(state_direction="improving")  # not in vocab
        mock_llm.return_value = bad_response
        record = extract("I am working hard and things are going well today for once.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_bad_arousal_triggers_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(arousal="very_high")  # not in vocab
        mock_llm.return_value = bad_response
        record = extract("Everything is moving so fast I can barely keep up with the pace.")
        assert record["low_confidence"] is True

    @patch("extractor.extract._call_llm")
    def test_retry_succeeds_clears_low_confidence(self, mock_llm):
        bad_response = _make_llm_response(emotion_label="excitement")
        good_response = _make_llm_response()  # valid
        mock_llm.side_effect = [bad_response, good_response]
        record = extract("I am excited about the project direction and next steps planned.")
        assert record["low_confidence"] is False


# ---------------------------------------------------------------------------
# 4. Schema validation catches missing fields
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def test_valid_record_passes(self):
        data = {
            "id": "abc123",
            "timestamp": "2026-04-20T10:00:00Z",
            "schema_version": "1.0",
            "raw_text": "some text",
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
        is_valid, err = validate_record(data)
        assert is_valid is True
        assert err == ""

    def test_missing_required_field(self):
        data = {
            "id": "abc123",
            "timestamp": "2026-04-20T10:00:00Z",
            # schema_version missing
            "raw_text": "some text",
            "valence": 0.3,
            "arousal": "medium",
            "emotion_label": "anticipation",
            "themes": ["purpose"],
            "intensity": "medium",
            "salient_focus": "new project",
            "state_direction": "building_momentum",
            "low_confidence": False,
            "confidence_score": 0.85,
            "entry_hash": "abc" * 20,
        }
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "schema_version" in err

    def test_missing_multiple_fields(self):
        data = {"id": "abc123", "raw_text": "some text"}
        is_valid, err = validate_record(data)
        assert is_valid is False

    def test_valence_out_of_range(self):
        data = {
            "id": "x", "timestamp": "2026-04-20T10:00:00Z", "schema_version": "1.0",
            "raw_text": "t", "valence": 2.5, "arousal": "low", "emotion_label": "joy",
            "themes": ["purpose"], "intensity": "low", "salient_focus": "clear goal set",
            "state_direction": "gaining_clarity", "low_confidence": False,
            "confidence_score": 0.9, "entry_hash": "abc",
        }
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "valence" in err

    def test_confidence_out_of_range(self):
        data = {
            "id": "x", "timestamp": "2026-04-20T10:00:00Z", "schema_version": "1.0",
            "raw_text": "t", "valence": 0.0, "arousal": "low", "emotion_label": "joy",
            "themes": ["purpose"], "intensity": "low", "salient_focus": "clear goal set",
            "state_direction": "gaining_clarity", "low_confidence": False,
            "confidence_score": 1.5, "entry_hash": "abc",
        }
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "confidence_score" in err

    def test_too_many_themes(self):
        data = {
            "id": "x", "timestamp": "2026-04-20T10:00:00Z", "schema_version": "1.0",
            "raw_text": "t", "valence": 0.0, "arousal": "low", "emotion_label": "joy",
            "themes": ["purpose", "momentum", "uncertainty", "overwhelm"],
            "intensity": "low", "salient_focus": "clear goal set",
            "state_direction": "gaining_clarity", "low_confidence": False,
            "confidence_score": 0.9, "entry_hash": "abc",
        }
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "themes" in err

    def test_salient_focus_too_long(self):
        data = {
            "id": "x", "timestamp": "2026-04-20T10:00:00Z", "schema_version": "1.0",
            "raw_text": "t", "valence": 0.0, "arousal": "low", "emotion_label": "joy",
            "themes": ["purpose"], "intensity": "low",
            "salient_focus": "one two three four five six seven",  # 7 words
            "state_direction": "gaining_clarity", "low_confidence": False,
            "confidence_score": 0.9, "entry_hash": "abc",
        }
        is_valid, err = validate_record(data)
        assert is_valid is False
        assert "salient_focus" in err

    def test_low_confidence_forced_true_when_score_low(self):
        data = {
            "id": "x", "timestamp": "2026-04-20T10:00:00Z", "schema_version": "1.0",
            "raw_text": "t", "valence": 0.0, "arousal": "low", "emotion_label": "joy",
            "themes": ["purpose"], "intensity": "low", "salient_focus": "clear goal set",
            "state_direction": "gaining_clarity", "low_confidence": False,
            "confidence_score": 0.3, "entry_hash": "abc",
        }
        is_valid, _ = validate_record(data)
        assert is_valid is True
        record = SignalRecord(**data)
        assert record.low_confidence is True


# ---------------------------------------------------------------------------
# 5. Confidence score < 0.5 sets low_confidence=True in extract()
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
