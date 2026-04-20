"""
Benchmark / regression tests for extraction quality.

These tests make real LLM API calls to validate that extraction stays consistent
as models, prompts, or schema change. They are skipped when ANTHROPIC_API_KEY is
not set (CI without secrets) or when running with --no-benchmark.

Run with:
    pytest tests/test_benchmark.py -v
    pytest tests/test_benchmark.py -v --backend ollama --ollama-model phi4

Each test defines directional expectations — not exact values, since LLMs vary
slightly across runs. The goal is to catch meaningful regressions (wrong valence
sign, wrong arousal tier, confidence drift) rather than enforce pixel-perfect output.
"""
import os
import pytest

from extractor.extract import extract
from extractor.schema import EMOTION_LABELS, THEMES, STATE_DIRECTIONS

# ---------------------------------------------------------------------------
# Setup: skip if no API key and not using ollama
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--backend", default="anthropic", choices=["anthropic", "ollama"])
    parser.addoption("--ollama-model", default="phi4")


@pytest.fixture(scope="session")
def backend(request):
    return request.config.getoption("--backend", default="anthropic")


@pytest.fixture(scope="session")
def ollama_model(request):
    return request.config.getoption("--ollama-model", default="phi4")


@pytest.fixture(autouse=True)
def require_api(backend):
    if backend == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(text, backend, ollama_model):
    return extract(text, backend=backend, ollama_model=ollama_model)


def assert_valid_schema(r):
    """Every record must be schema-complete with valid vocab values."""
    assert r["emotion_label"] in EMOTION_LABELS, f"Bad emotion_label: {r['emotion_label']}"
    assert r["state_direction"] in STATE_DIRECTIONS, f"Bad state_direction: {r['state_direction']}"
    assert r["arousal"] in {"low", "medium", "high"}
    assert r["intensity"] in {"low", "medium", "high"}
    assert 1 <= len(r["themes"]) <= 3
    assert all(t in THEMES for t in r["themes"]), f"Bad themes: {r['themes']}"
    assert -1.0 <= r["valence"] <= 1.0
    assert 0.0 <= r["confidence_score"] <= 1.0
    assert 3 <= len(r["salient_focus"].split()) <= 6, f"Bad salient_focus: '{r['salient_focus']}'"


# ---------------------------------------------------------------------------
# Benchmark cases
# ---------------------------------------------------------------------------

class TestNegativeHighArousal:
    """'Everything is on fire' — should be clearly negative, high energy."""

    TEXT = "Everything is on fire and nobody seems to care. I've been trying to fix this for three days."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_negative(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] < -0.3, f"Expected clearly negative valence, got {r['valence']}"

    def test_arousal_not_low(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["arousal"] in {"medium", "high"}, f"Expected medium/high arousal, got {r['arousal']}"

    def test_high_confidence(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["confidence_score"] >= 0.6, f"Expected confident extraction, got {r['confidence_score']}"

    def test_not_low_confidence(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["low_confidence"] is False


class TestPositiveMomentum:
    """'Finally shipped the thing' — positive, done, momentum."""

    TEXT = "Finally shipped the thing. Took forever but it's done and it actually works."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_positive(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] > 0.2, f"Expected positive valence, got {r['valence']}"

    def test_state_direction_forward(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["state_direction"] in {"building_momentum", "gaining_clarity", "regaining_confidence"}, \
            f"Expected forward direction, got {r['state_direction']}"

    def test_not_low_confidence(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["low_confidence"] is False


class TestFlatLowSignal:
    """'Today was okay I guess' — flat, should flag low confidence."""

    TEXT = "Today was okay I guess. Nothing really happened."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_low_confidence(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["low_confidence"] is True, "Flat input should be flagged low_confidence"

    def test_valence_near_neutral(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert -0.4 <= r["valence"] <= 0.4, f"Expected near-neutral valence, got {r['valence']}"


class TestAmbiguousDirection:
    """'Starting things and not finishing' — negative lean, uncertain direction."""

    TEXT = "I keep starting things and not finishing them. Not sure if that's growth or just chaos."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_not_positive(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] <= 0.2, f"Expected non-positive valence, got {r['valence']}"

    def test_state_direction_not_forward(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["state_direction"] not in {"building_momentum", "regaining_confidence"}, \
            f"Ambiguous rant should not show strong forward direction, got {r['state_direction']}"


class TestPurposeLoss:
    """'Work feels pointless' — negative, low arousal, purpose theme."""

    TEXT = "I don't know why I'm doing any of this anymore. The work feels pointless even when it goes well."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_negative(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] < -0.3, f"Expected negative valence, got {r['valence']}"

    def test_purpose_theme_present(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert "purpose" in r["themes"], f"Expected 'purpose' theme, got {r['themes']}"

    def test_state_direction_negative(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["state_direction"] in {"losing_direction", "drifting", "stuck"}, \
            f"Expected negative direction, got {r['state_direction']}"


class TestIsolation:
    """'Nobody on the team gets it' — frustration, isolation theme."""

    TEXT = "Nobody on the team gets what I'm trying to do. I've explained it three times now."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_negative(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] < 0, f"Expected negative valence, got {r['valence']}"

    def test_isolation_or_overwhelm_theme(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert any(t in r["themes"] for t in {"isolation", "overwhelm"}), \
            f"Expected isolation/overwhelm theme, got {r['themes']}"


class TestCreativeSatisfaction:
    """'Building something for fun' — positive, creative_satisfaction."""

    TEXT = "Spent the whole afternoon just building something for fun. No deadline, no ticket. Remembered why I love this."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_positive(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] > 0.3, f"Expected positive valence, got {r['valence']}"

    def test_creative_satisfaction_theme(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert "creative_satisfaction" in r["themes"], \
            f"Expected 'creative_satisfaction' theme, got {r['themes']}"

    def test_not_low_confidence(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["low_confidence"] is False


class TestRegainingConfidence:
    """'Solved something I thought was beyond me' — positive trajectory."""

    TEXT = "I was really doubting myself last week but today I solved something I thought was beyond me."

    def test_schema_valid(self, backend, ollama_model):
        assert_valid_schema(run(self.TEXT, backend, ollama_model))

    def test_valence_positive(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["valence"] > 0.2, f"Expected positive valence, got {r['valence']}"

    def test_state_direction_upward(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["state_direction"] in {"regaining_confidence", "building_momentum", "gaining_clarity"}, \
            f"Expected upward direction, got {r['state_direction']}"

    def test_not_low_confidence(self, backend, ollama_model):
        r = run(self.TEXT, backend, ollama_model)
        assert r["low_confidence"] is False
