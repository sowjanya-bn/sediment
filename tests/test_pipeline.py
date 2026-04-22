from unittest.mock import patch

from pipeline.graph import build_sediment_graph


def _make_llm_response(**overrides):
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


class TestPipelineGraph:
    def test_short_input_routes_to_default(self):
        graph = build_sediment_graph()
        state = graph.invoke({"raw_text": "ok", "backend": "anthropic", "ollama_model": "llama3.2"})
        record = state["record"]
        assert record["low_confidence"] is True
        assert record["emotion_label"] == "indeterminate"

    @patch("extractor.core._call_llm")
    def test_valid_first_pass(self, mock_llm):
        mock_llm.return_value = _make_llm_response()
        graph = build_sediment_graph()
        state = graph.invoke({
            "raw_text": "I feel focused and hopeful about what I am building.",
            "backend": "anthropic",
            "ollama_model": "llama3.2",
        })
        record = state["record"]
        assert record["low_confidence"] is False

    @patch("extractor.core._call_llm")
    def test_retry_then_success(self, mock_llm):
        bad = _make_llm_response(emotion_label="excitement")
        good = _make_llm_response()
        mock_llm.side_effect = [bad, good]

        graph = build_sediment_graph()
        state = graph.invoke({
            "raw_text": "I feel focused and hopeful about what I am building.",
            "backend": "anthropic",
            "ollama_model": "llama3.2",
        })
        record = state["record"]
        assert record["low_confidence"] is False

    @patch("extractor.core._call_llm")
    def test_retry_then_fallback(self, mock_llm):
        bad = _make_llm_response(emotion_label="excitement")
        mock_llm.side_effect = [bad, bad]

        graph = build_sediment_graph()
        state = graph.invoke({
            "raw_text": "I feel focused and hopeful about what I am building.",
            "backend": "anthropic",
            "ollama_model": "llama3.2",
        })
        record = state["record"]
        assert record["low_confidence"] is True
        assert record["emotion_label"] == "indeterminate"