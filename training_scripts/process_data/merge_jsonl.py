#!/usr/bin/env python3
"""
Script to merge multiple JSONL files and extract specific fields.
Outputs a JSON file with response_text, genui_json, and identifiers.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dictionaries."""
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num} in {file_path}: {e}")
    return records


def extract_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the required fields from a record."""
    # Define the fields to keep (identifiers + response_text + genui_json)
    fields_to_keep = [
        'ui_id',
        'response_id',
        'query_id',
        'intent',
        'tags',
        'intent_bucket',
        'response_text',
        'genui_json'
    ]

    extracted = {}
    for field in fields_to_keep:
        if field in record:
            value = record[field]
            # Convert genui_json to string if it's a dict/object
            if field == 'genui_json' and isinstance(value, (dict, list)):
                extracted[field] = json.dumps(value, ensure_ascii=False)
            else:
                extracted[field] = value

    return extracted


def merge_jsonl_files(input_files: List[str], output_file: str) -> None:
    """Merge multiple JSONL files and write to a single JSON file."""
    all_records = []

    for input_file in input_files:
        print(f"Processing: {input_file}")
        records = load_jsonl(input_file)
        print(f"  Found {len(records)} records")

        for record in records:
            extracted = extract_fields(record)
            all_records.append(extracted)

    print(f"\nTotal records merged: {len(all_records)}")

    # Write to output JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


    print(f"Output written to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge multiple JSONL files into a single JSON file with selected fields.'
    )
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Input JSONL file(s) to merge'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output JSON file path'
    )

    args = parser.parse_args()

    # Validate input files exist
    for input_file in args.input:
        if not Path(input_file).exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

    merge_jsonl_files(args.input, args.output)


if __name__ == '__main__':
    main()