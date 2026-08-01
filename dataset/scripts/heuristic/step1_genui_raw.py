#!/usr/bin/env python3
"""
Script to clean genui_json values by removing all characters preceding '{"root"'.
The '{"root"' pattern appears just after '\nassistant\n' in the input.

Usage: python fix_genui_raw.py <input_file>
"""

import json
import sys
import os


def clean_genui_json(value: str) -> str:
    """
    Clean the genui_json value by finding '{"root"' and returning everything from that point.
    The pattern '{"root"' typically appears after '\nassistant\n'.
    """
    # Find the position of '{"root"' in the string
    root_start = value.find('{"root"')

    if root_start != -1:
        # Return everything from '{"root"' onwards
        return value[root_start:]
    else:
        # If '{"root"' is not found, return the original value
        # This might indicate malformed data
        return value


def process_file(input_path: str, output_path: str) -> None:
    """
    Process the input JSONL file and write cleaned data to output file.
    Reads all content first to avoid data loss when input and output are the same file.
    """
    processed_count = 0
    modified_count = 0

    # Read all lines first to avoid data loss when input == output
    with open(input_path, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Process and write to output
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)

                # Check if genui_json field exists
                if 'genui_json' in record:
                    original_value = record['genui_json']
                    cleaned_value = clean_genui_json(original_value)

                    if original_value != cleaned_value:
                        modified_count += 1
                        record['genui_json'] = cleaned_value

                # Write the (possibly modified) record
                outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue

    print(f"Processing complete!")
    print(f"  Total records processed: {processed_count}")
    print(f"  Records modified: {modified_count}")
    print(f"  Output written to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_genui_raw.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Use the same file for input and output
    process_file(input_file, input_file)
