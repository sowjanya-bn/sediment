#!/usr/bin/env python3
"""
sediment CLI

Usage:
  python cli.py "your rant text here"            # pretty-printed output
  python cli.py "your rant text here" --json     # raw JSON
  python cli.py --export-all                     # export all as NDJSON to stdout
  python cli.py --export-since 2026-01-01        # export since date as NDJSON
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from extractor.extract import extract
from storage.db import save_entry
from storage.export import export_all, export_by_date_range, to_ndjson


def pretty_print(record: dict) -> None:
    lines = [
        f"ID:             {record['id']}",
        f"Timestamp:      {record['timestamp']}",
        f"Valence:        {record['valence']:+.2f}",
        f"Arousal:        {record['arousal']}",
        f"Emotion:        {record['emotion_label']}",
        f"Themes:         {', '.join(record['themes'])}",
        f"Intensity:      {record['intensity']}",
        f"Salient focus:  {record['salient_focus']}",
        f"State dir:      {record['state_direction']}",
        f"Confidence:     {record['confidence_score']:.2f}",
        f"Low confidence: {record['low_confidence']}",
        f"Entry hash:     {record['entry_hash'][:16]}...",
        f"Schema:         {record['schema_version']}",
    ]
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="sediment: extract structured signals from reflective text"
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Free-form reflective text to extract signals from",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output raw JSON instead of pretty-printed format",
    )
    parser.add_argument(
        "--export-all",
        action="store_true",
        help="Export all entries as NDJSON to stdout",
    )
    parser.add_argument(
        "--export-since",
        metavar="DATE",
        help="Export entries since DATE (ISO8601, e.g. 2026-01-01) as NDJSON to stdout",
    )
    parser.add_argument(
        "--input-csv",
        metavar="FILE",
        help="CSV file to process in batch (uses 'text' column, or first column if absent)",
    )
    parser.add_argument(
        "--output-jsonl",
        metavar="FILE",
        help="Output file for batch results (JSONL). Required with --input-csv",
    )
    parser.add_argument(
        "--backend",
        choices=["anthropic", "ollama"],
        default="anthropic",
        help="LLM backend to use (default: anthropic)",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        metavar="MODEL",
        help="Ollama model name (default: llama3.2)",
    )

    args = parser.parse_args()

    if args.export_all:
        entries = export_all()
        print(to_ndjson(entries))
        return

    if args.export_since:
        today = datetime.now(timezone.utc).date().isoformat()
        entries = export_by_date_range(args.export_since, today)
        print(to_ndjson(entries))
        return

    if args.input_csv:
        if not args.output_jsonl:
            print("Error: --output-jsonl is required with --input-csv", file=sys.stderr)
            sys.exit(1)
        with open(args.input_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Use 'text' column if present, otherwise first column
            col = "text" if reader.fieldnames and "text" in reader.fieldnames else (reader.fieldnames[0] if reader.fieldnames else None)
            if col is None:
                print("Error: CSV has no columns", file=sys.stderr)
                sys.exit(1)
        out_path = Path(args.output_jsonl)
        processed = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for i, row in enumerate(rows, 1):
                text = row.get(col, "").strip()
                if not text:
                    continue
                record = extract(text, backend=args.backend, ollama_model=args.ollama_model)
                save_entry(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                print(f"[{i}/{len(rows)}] {record['id']} — {record['emotion_label']} ({record['confidence_score']:.2f})", file=sys.stderr)
        print(f"Done. {processed} entries written to {out_path}", file=sys.stderr)
        return

    if not args.text:
        parser.print_help()
        sys.exit(1)

    record = extract(args.text, backend=args.backend, ollama_model=args.ollama_model)
    save_entry(record)

    if args.output_json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
    else:
        pretty_print(record)


if __name__ == "__main__":
    main()
