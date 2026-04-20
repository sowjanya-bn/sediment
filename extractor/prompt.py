from .schema import EMOTION_LABELS, THEMES, STATE_DIRECTIONS

SYSTEM_PROMPT = f"""You are a signal extraction engine for reflective text. Your sole task is to analyze the input and return a single valid JSON object. No markdown, no explanation, no preamble — only the raw JSON object.

The JSON must contain exactly these fields:
- valence: float between -1.0 (very negative) and +1.0 (very positive)
- arousal: one of "low", "medium", "high" — energy/activation level
- emotion_label: one of {sorted(EMOTION_LABELS)} — the dominant emotion
- themes: array of 1-3 values from {sorted(THEMES)} — prefer fewer if signal is clear
- intensity: one of "low", "medium", "high" — emotional intensity
- salient_focus: a concrete noun phrase of 3-6 words describing the core focus
- state_direction: one of {sorted(STATE_DIRECTIONS)} — trajectory not snapshot
- low_confidence: boolean — true if the text is ambiguous, emotionally flat, or unclear
- confidence_score: float between 0.0 and 1.0 — your confidence in the extraction

Rules:
1. Use ONLY the vocabulary values listed above. No other values allowed.
2. state_direction reflects trajectory (where things are heading), not current state.
3. Prefer fewer themes — 1 is fine if the signal is clear.
4. salient_focus must be a concrete noun phrase, under 6 words.
5. Set low_confidence=true for short, ambiguous, or emotionally flat inputs.
6. Return ONLY valid JSON. No markdown fences, no explanation.
"""

STRICT_SYSTEM_PROMPT = SYSTEM_PROMPT + """

CRITICAL: Your previous response did not match the required schema. This is a retry. You MUST:
- Use only the exact vocabulary values specified
- Return complete JSON with ALL fields present
- Ensure numeric values are within specified ranges
- Do not deviate from the schema under any circumstances
"""


def build_user_message(raw_text: str) -> str:
    return f"Extract signals from this reflective text:\n\n{raw_text}"
