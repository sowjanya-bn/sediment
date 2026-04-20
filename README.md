# sediment

A local-first text → structured signal extraction module. Takes free-form reflective text ("rants") and returns stable, consistent structured records stored in SQLite.

> Be boring in extraction, so patterns can be interesting later.

Consistency over correctness. Schema evolves slowly. Everything is stored verbatim.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Anthropic backend** — set your API key:
```bash
export ANTHROPIC_API_KEY=your_key
```

**Ollama backend** — install [Ollama](https://ollama.com), then pull a model:
```bash
ollama pull phi4
ollama pull llama3.2
ollama pull mistral
```

---

## Usage

### Single entry

```bash
# Pretty-printed output (default)
python cli.py "I can't focus on anything today"

# Raw JSON
python cli.py "I can't focus on anything today" --json
```

### Backends

```bash
# Anthropic Claude (default)
python cli.py "your text here"

# Ollama with phi4 (recommended local model)
python cli.py "your text here" --backend ollama --ollama-model phi4

# Ollama with llama3.2
python cli.py "your text here" --backend ollama --ollama-model llama3.2
```

### Batch processing from CSV

CSV must have a `text` column (or any column — first column used as fallback).

```bash
python cli.py --input-csv rants.csv --output-jsonl results.jsonl
python cli.py --input-csv rants.csv --output-jsonl results.jsonl --backend ollama --ollama-model phi4
```

Progress prints to stderr. Each entry is also saved to SQLite.

### Export

```bash
# Export all entries as NDJSON
python cli.py --export-all

# Export since a date
python cli.py --export-since 2026-01-01

# Pipe to a file
python cli.py --export-all > all_entries.jsonl
```

---

## Output schema (v1.1)

Every entry produces the following fields. No nulls — always returns the closest valid value.

| Field | Type | Description |
|---|---|---|
| `id` | uuid | Unique entry ID |
| `timestamp` | ISO8601 | Assigned at ingestion, never inferred from text |
| `schema_version` | string | Schema version (`1.1`) |
| `raw_text` | string | Verbatim input, always stored |
| `input_type` | affective / metacognitive | Affective = feelings; metacognitive = patterns/observations about self |
| `valence` | float | -1.0 (negative) to +1.0 (positive) |
| `arousal` | low / medium / high | Energy/activation level |
| `emotion_label` | string | Dominant emotion (see vocabulary) |
| `themes` | array | 1–3 themes from fixed vocabulary |
| `intensity` | low / medium / high | Strength of emotional expression |
| `salient_focus` | string | 3–6 word concrete phrase — what it's actually about |
| `state_direction` | string | Trajectory (see vocabulary) |
| `low_confidence` | bool | Always derived from `confidence_score < 0.5` |
| `confidence_score` | float | 0.0–1.0; down-ranked by uncertainty phrases and metacognitive routing |
| `entry_hash` | sha256 | sha256(raw_text) — stable across reprocessing |

### Controlled vocabulary

**emotion_label:** `joy` `sadness` `anger` `fear` `anticipation` `neutral` `flat` `ambivalent` `indeterminate`

**themes:** `uncertainty` `overwhelm` `purpose` `traction` `self_trust` `isolation` `momentum` `creative_satisfaction`

**state_direction:** `gaining_clarity` `losing_direction` `building_momentum` `stuck` `regaining_confidence` `drifting` `indeterminate`

`indeterminate` is the unified uncertainty output for both `emotion_label` and `state_direction` — used when no signal is clear enough to classify. It is a schema outcome, not a fallback.

---

## Storage

SQLite database at `~/.sediment/sediment.db`. Extraction log at `~/.sediment/extraction.log`.

Raw text is always retained — entries can be reprocessed if schema evolves.

---

## Tests

```bash
# Unit tests (no API calls, uses mocks)
pytest tests/test_extract.py -v

# Benchmark / regression tests (real API calls)
pytest tests/test_benchmark.py -v

# Benchmarks with Ollama
pytest tests/test_benchmark.py -v --backend ollama --ollama-model phi4

# All tests
pytest -v
```

Benchmark tests are skipped automatically if `ANTHROPIC_API_KEY` is not set.

Each benchmark defines directional expectations per input (valence sign, expected themes, state direction, confidence tier) — not exact values. Re-run after any prompt, schema, or model change to catch regressions.

---

## Corrections (CR layer)

Field-level corrections to extracted records. A correction to one field does not affect the rest of the record — originals are never mutated.

| CR type | Field corrected |
|---|---|
| `relabel` | `emotion_label` |
| `valence_nudge` | `valence` |
| `confidence_override` | `confidence_score` (also syncs `low_confidence`) |
| `state_direction_edit` | `state_direction` |
| `theme_edit` | `themes` |

Each CR record carries `entry_hash` — corrections made against a different version of the entry are ignored when replaying. `schema_version` is required on every CR record.

`state_direction` corrections are snapshot annotations. The design keeps them as separate records (not embedded in entries) so that future relational computation across entries can supersede them without migration.

---

## Project structure

```
sediment/
├── extractor/
│   ├── extract.py      — core: LLM call, validation, retry, fallback, logging
│   ├── schema.py       — Pydantic model, vocabulary, validators
│   └── prompt.py       — system prompt and strict retry prompt
├── storage/
│   ├── db.py           — SQLite interface
│   └── export.py       — JSON export utilities
├── corrections/
│   ├── schema.py       — CorrectionRecord model, CR_TYPES
│   ├── db.py           — corrections table, save/query
│   └── apply.py        — apply_correction, get_effective_entry
├── cli.py              — CLI entry point
├── tests/
│   ├── test_extract.py     — unit tests (mocked)
│   └── test_benchmark.py   — regression tests (real API calls)
├── rants.csv           — sample inputs for batch testing
└── requirements.txt
```

---

## Model recommendations

| Model | Backend | Schema compliance | Speed | Notes |
|---|---|---|---|---|
| claude-haiku-4-5 | Anthropic | Excellent | Fast | Default, most reliable |
| phi4 | Ollama | Good | Medium | Best local option |
| llama3.2 | Ollama | Good | Fast | Slightly conservative confidence scores |
| mistral | Ollama | Poor | Fast | Ignores salient_focus constraints |
