# CLAUDE.md — Rant Signal Extraction Module

## Project Summary

Build a local-first, standalone text → structured signal extraction module.
Takes free-form reflective text ("rants") as input and returns a stable, consistent
structured record. This is phase one only. No correlation engine, no UI, no insights layer.

The goal is consistency over time, not perfect interpretation.
Extraction should be slightly boring by design — patterns emerge from accumulation, not cleverness.

---

## Core Philosophy

> Be boring in extraction, so patterns can be interesting later.

- Consistency > correctness
- Store everything; infer carefully
- Schema evolves slowly and deliberately
- This module must stand alone — no hard dependency on Daypilot

---

## Extraction Method

Use an LLM API call (Anthropic Claude) for extraction. This handles messy,
unstructured language reliably. The output schema is enforced deterministically
on the response — the intelligence is in the extraction, the stability is in the schema.

LLM calls must use low temperature (0.0–0.2) to ensure consistent outputs.
Consistency across runs is more important than creativity. Without this,
the "consistency > correctness" principle breaks silently.

If extraction confidence is low on any field, flag the entry with a `low_confidence: true`
field rather than silently guessing. Do not block ingestion — flag and store.

---

## Output Schema (v1 — lock this)

Every entry produces the following fields. No nulls — always return the closest
valid value from the controlled vocabulary.

```json
{
  "id": "uuid",
  "timestamp": "ISO8601",
  "schema_version": "1.0",
  "raw_text": "original input, always stored verbatim",
  "valence": 0.0,
  "arousal": "low | medium | high",
  "emotion_label": "see vocabulary",
  "themes": ["max 2-3 from fixed set"],
  "intensity": "low | medium | high",
  "salient_focus": "3-6 word concrete phrase",
  "state_direction": "see vocabulary",
  "low_confidence": false,
  "confidence_score": 0.85,
  "entry_hash": "sha256(raw_text)"
}
```

### Field Notes

- `valence`: float from -1.0 (negative) to +1.0 (positive). Based on Russell's Circumplex Model.
- `arousal`: energy level of the emotional state, independent of valence. High = activated (anxious, excited). Low = deactivated (sad, calm).
- `intensity`: strength of emotional expression, not energy level. Arousal = energy. Intensity = strength. A low-arousal state can still have high intensity (e.g. deep grief). Do not conflate these.
- `emotion_label`: human-readable surface tag. Drawn from Plutchik-inspired set.
- `themes`: maximum 2-3. Do not exceed. Choose closest match from fixed vocabulary.
- `salient_focus`: short, concrete phrase — what the rant is actually about, not an abstract summary.
- `state_direction`: captures movement, not feeling. Which way is the writer's state trending?
- `low_confidence`: set to true if the input is ambiguous enough that field values are uncertain.
- `confidence_score`: optional float 0.0–1.0. Anchoring guide:
  - 0.8–1.0 → clear emotional signal and themes
  - 0.5–0.8 → somewhat ambiguous but interpretable
  - below 0.5 → vague, flat, or unclear input — set `low_confidence: true`
- `salient_focus`: if this field drifts toward abstraction, prefer a concrete noun phrase tied to real-world context ("job interview prep anxiety" not "existential performance concern").
- `timestamp`: assigned at ingestion time. Do not infer from text content.

---

## Controlled Vocabulary (v1)

### Emotion Labels
```
joy
sadness
anger
fear
anticipation
trust
```

### Themes
```
uncertainty
overwhelm
purpose
traction
self_trust
isolation
momentum
creative_satisfaction
```

Do not add themes without evidence from real data. If a theme repeatedly
forces itself into the closest-match category, that is the signal to expand.
Version the schema when expanding (see schema_version field).

### State Direction
```
gaining_clarity
losing_direction
building_momentum
stuck
regaining_confidence
drifting
```

State direction reflects movement, not feeling. Ask: which way is this going?

---

## Extraction Rules

1. Always return a value for every field — no nulls, no skipped fields.
2. If unsure, choose the closest category. Do not invent new ones.
3. Themes: maximum 2-3. Prefer fewer if the signal is clear.
4. `salient_focus` must be short and concrete, not abstract ("presentation deadline stress" not "cognitive burden around future obligations").
5. `state_direction` reflects trajectory, not current emotion.
6. Set `low_confidence: true` for genuinely ambiguous inputs rather than forcing false precision.
7. Raw text is always stored verbatim, regardless of content.

### Theme Selection Priority

Themes should reflect:
1. Primary source of tension or focus
2. Secondary contributing context (optional)

Avoid tagging surface-level mentions if a deeper theme is clearer.
Example: rant mentions "meeting" but is fundamentally about overwhelm → tag `overwhelm`, not a work-surface label.

### State Direction Heuristic

State direction must be inferred from language patterns, not emotional tone:
- Movement words: getting, losing, starting, avoiding, building
- Contrast within text: "I was X but now Y"
- Trajectory implied by frustration or progress

If no clear movement is present → default to `stuck` or `drifting`.
Do not confuse state direction with emotion label — they are independent.

### Validation (Hard Requirement)

All outputs must be validated against schema before storage.
If validation fails:
- Retry once with a stricter prompt — explicitly remind the model to use only allowed vocabulary values
- If still invalid → store with `low_confidence: true` and log the error

LLMs will drift. This guardrail is non-negotiable.

---

## Storage

- SQLite, local
- Single `entries` table
- Always retain `raw_text` — schema may evolve and entries may need reprocessing
- `schema_version` field on every record enables future migration
- Index on `timestamp` for time-range queries

### Reprocessing (First-Class Requirement)

The system must support re-running extraction on existing entries when the schema evolves.
Do not overwrite original records silently. When reprocessing:
- Store the new extraction alongside the original, or
- Migrate explicitly with version tracking

This is how the module stays viable as vocabulary and schema mature.

### Suggested Schema

```sql
CREATE TABLE entries (
  id TEXT PRIMARY KEY,
  timestamp TEXT NOT NULL,
  schema_version TEXT NOT NULL DEFAULT '1.0',
  raw_text TEXT NOT NULL,
  valence REAL,
  arousal TEXT,
  emotion_label TEXT,
  themes TEXT,         -- JSON array stored as string
  intensity TEXT,
  salient_focus TEXT,
  state_direction TEXT,
  low_confidence INTEGER DEFAULT 0,
  confidence_score REAL,
  entry_hash TEXT
);
```

---

## Export

Include a JSON export function from day one.
Supports export by entry ID, date range, or full dump.
Output format: newline-delimited JSON or single JSON array.
This keeps the module genuinely standalone and makes future integrations
a data handshake, not a code dependency.

---

## Research Foundation

This schema is grounded in established research:

- **Russell's Circumplex Model of Affect** — valence/arousal axes as the numeric backbone
- **Plutchik's Wheel of Emotions** — basis for the emotion label vocabulary
- **LIWC (Linguistic Inquiry and Word Count)** — inspiration for theme categories (social, cognitive load, affect, personal concerns)
- **Experience Sampling Method (ESM)** literature — validates the approach: repeated lightweight capture + consistent schema → meaningful personal patterns over time

Key lesson from ESM research: schema stability matters more than schema sophistication.
Do not change the vocabulary mid-stream without versioning.

---

## Observability (Required)

Log every extraction run. No heavy infrastructure — flat log file is sufficient.
Each log entry should capture:
- Raw input
- Extracted output
- Validation result (pass / fail / retry)
- `low_confidence` flag

This enables debugging and schema refinement without instrumenting the database.

---

## Schema Freeze Window

Do not modify the schema or vocabulary until at least 20–30 entries are collected.
Early inconsistencies should be tolerated to preserve comparability across entries.
Resist the urge to fix the schema before you have real signal. Over-tuning early
destroys longitudinal value.

---

- No correlation engine
- No dashboards or visualisation
- No insights or summary layer
- No Daypilot-specific integration
- No UI beyond CLI for testing

---

## Success Criteria

- Same input style → consistent output shape every time
- Multiple entries → comparable, queryable signals
- Vocabulary does not drift across entries
- Feels slightly imperfect but stable
- A new entry can be processed and stored in under 3 seconds

---

## Nice to Have (only if trivial)

- Confidence score per field (separate from binary `low_confidence` flag)
- Debug output showing why a tag was chosen
- Simple CLI: `extract "text here"` → prints structured record; support both raw JSON and pretty-printed output
- `entry_hash`: sha256 of raw_text — enables deduplication, reprocessing tracking, and debugging without additional state. Must be computed before extraction and remain stable across reprocessing. Same input → same hash always.

### Edge Case: Empty or Ultra-Short Input

If input is extremely short or non-expressive (e.g. "ok", "fine", "meh"):
- Set `low_confidence: true`
- Default to neutral valence (0.0), low arousal, minimal or no themes

Do not hallucinate meaning into thin input.

---

## Project Structure (suggested)

```
/
├── CLAUDE.md               ← this file
├── extractor/
│   ├── __init__.py
│   ├── extract.py          ← core extraction logic (LLM call + schema enforcement)
│   ├── schema.py           ← vocabulary definitions, validation
│   └── prompt.py           ← extraction prompt template
├── storage/
│   ├── __init__.py
│   ├── db.py               ← SQLite interface
│   └── export.py           ← JSON export utilities
├── cli.py                  ← simple CLI for testing
├── tests/
│   └── test_extract.py     ← basic consistency tests
└── requirements.txt
```

---

## Prompt Guidance for Extraction

The extraction prompt should instruct the model to:

1. Return only valid JSON matching the schema
2. Use only vocabulary from the controlled sets
3. Prefer fewer themes over forcing extra ones
4. Treat `state_direction` as trajectory, not snapshot
5. Set `low_confidence: true` for short, ambiguous, or emotionally flat inputs
6. Keep `salient_focus` under 6 words and concrete

Enforce schema on the response side — validate against vocabulary before storing.
Reject and flag any response that does not parse or contains out-of-vocabulary values.
