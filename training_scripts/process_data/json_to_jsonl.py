#!/usr/bin/env python3
"""
Convert a JSON array file to JSONL format.
Each element of the array becomes a separate line in the output.
"""

import json
import argparse
import os


def json_to_jsonl(input_path, output_path=None):
    """Convert JSON array file to JSONL format.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSONL file (optional, defaults to input with .jsonl extension)
    """
    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = base + '.jsonl'

    # Read JSON array
    print(f"Reading JSON from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("WARNING: Input is not a JSON array. Wrapping single object in array.")
        data = [data]

    # Write JSONL
    print(f"Writing JSONL to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"Converted {len(data)} items from JSON array to JSONL format.")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert JSON array to JSONL format')
    parser.add_argument('input_file', help='Path to input JSON file')
    parser.add_argument('-o', '--output', help='Path to output JSONL file (default: same as input with .jsonl extension)')
    args = parser.parse_args()

    json_to_jsonl(args.input_file, args.output)
