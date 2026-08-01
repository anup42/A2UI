#!/usr/bin/env python3
"""
Script to rename 'prediction' keys to 'genui_json' in a JSONL file and rename the file to genui.jsonl.
Usage: python step0_rename.py <path_to_jsonl_file>
"""

import json
import sys
import os


def rename_prediction_key(file_path):
    """
    Read a JSONL file, rename 'prediction' keys to 'genui_json',
    and write the modified content back.

    Args:
        file_path: Path to the JSONL file to process
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    # Read all lines and process
    modified_lines = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                modified_lines.append(line)
                continue

            try:
                data = json.loads(line)

                # Rename 'prediction' key to 'genui_json' if it exists
                if 'prediction' in data:
                    data['genui_json'] = data.pop('prediction')

                modified_lines.append(json.dumps(data, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                modified_lines.append(line)

    # Write back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modified_lines))

    # Rename the file to genui.jsonl
    dir_path = os.path.dirname(file_path)
    output_path = os.path.join(dir_path, 'genui.jsonl')
    os.rename(file_path, output_path)

    print(f"Successfully processed {file_path}")
    print(f"Renamed 'prediction' keys to 'genui_json'")
    print(f"File renamed to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python step0_rename.py <path_to_jsonl_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    rename_prediction_key(file_path)
