#!/usr/bin/env python3
"""
Script to convert "genui_json" field from JSON string to JSON object in a JSONL file.

This script handles cases where the genui_json field contains:
- Escaped JSON strings (with \" instead of ")
- Malformed JSON that needs extraction and parsing
- Arrays that should be objects (merges array elements into single object)
- Pathological recursive patterns (truncates excessive nesting)
- Uses json-repair library to fix LLM-generated malformed JSON

Usage:
    python fix_genui_json_strings.py <input_file>

Example:
    python fix_genui_json_strings.py dataset/data/runs/golden50_qwen3_0.6/genui.jsonl

Note: The file is modified in-place (same file is used for input and output).
"""

import json
import re
import sys
from pathlib import Path

try:
    from json_repair import repair_json
except ImportError:
    print("Installing json-repair library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "json-repair", "-q"])
    from json_repair import repair_json


def truncate_recursive_pattern(value: str, max_repetitions: int = 10) -> str:
    """
    Detect and truncate pathological recursive patterns in JSON strings.

    Handles patterns like: "default": {"card": {"type": "Card", "props": {}, "default": {"card": ...
    """
    # Pattern to detect recursive "default": {"card": or similar structures
    recursive_pattern = r'("default":\s*\{"card":\s*\{"type":\s*"Card")'
    matches = list(re.finditer(recursive_pattern, value))

    if len(matches) > max_repetitions:
        # Find where the excessive recursion starts (after max_repetitions)
        cut_point = matches[max_repetitions].start()

        # Truncate at that point and close properly
        truncated = value[:cut_point]

        # Find the last complete structure and close it properly
        # Remove trailing incomplete structures
        truncated = truncated.rstrip(',').rstrip()

        # Count open braces and close them
        open_braces = truncated.count('{') - truncated.count('}')
        truncated += '}' * open_braces

        return truncated

    return value


def try_parse_with_repair(value: str) -> tuple[any, bool]:
    """
    Try to parse a string as JSON, using json-repair if standard parsing fails.

    Returns:
        Tuple of (parsed_value, success)
    """
    # First try standard JSON parsing
    try:
        parsed = json.loads(value)
        return parsed, True
    except json.JSONDecodeError:
        pass

    # Try truncating recursive patterns
    truncated = truncate_recursive_pattern(value)
    if truncated != value:
        try:
            parsed = json.loads(truncated)
            return parsed, True
        except json.JSONDecodeError:
            pass

    # Try json-repair with different options
    try:
        # Try with return_objects=True first
        repaired = repair_json(value, return_objects=True)
        if repaired is not None and isinstance(repaired, (dict, list)):
            return repaired, True
    except Exception:
        pass

    try:
        # Try with return_objects=False (returns JSON string)
        repaired_str = repair_json(value, return_objects=False)
        if repaired_str:
            parsed = json.loads(repaired_str)
            return parsed, True
    except Exception:
        pass

    # Try json-repair on truncated version
    try:
        repaired = repair_json(truncated, return_objects=True)
        if repaired is not None and isinstance(repaired, (dict, list)):
            return repaired, True
    except Exception:
        pass

    return None, False


def normalize_genui_json(parsed_value):
    """
    Normalize the parsed genui_json value:
    - If it's an array with dict elements, extract/merge into a single object
    - If it's already an object, return as-is
    """
    if isinstance(parsed_value, dict):
        return parsed_value

    if isinstance(parsed_value, list) and len(parsed_value) > 0:
        # If array contains dicts, merge them into a single object
        # Priority: later elements override earlier ones for duplicate keys
        result = {}
        for item in parsed_value:
            if isinstance(item, dict):
                result.update(item)
        return result if result else parsed_value

    return parsed_value


def convert_genui_json(input_path: str, output_path: str) -> dict:
    """
    Read a JSONL file, convert genui_json from string to JSON object, and write to output.
    Reads all content first to avoid data loss when input and output are the same file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file

    Returns:
        Dictionary with statistics about the conversion
    """
    stats = {
        "total_lines": 0,
        "successful_conversions": 0,
        "failed_conversions": 0,
        "already_parsed": 0,
        "skipped_empty": 0,
        "no_genui_json": 0,
        "arrays_converted": 0
    }

    input_file = Path(input_path)
    output_file = Path(output_path)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Read all lines first to avoid data loss when input == output
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    # Process and write to output
    with open(output_file, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(lines, 1):
            stats["total_lines"] += 1
            line = line.strip()

            # Skip empty lines
            if not line:
                stats["skipped_empty"] += 1
                continue

            try:
                # Parse the line as JSON
                record = json.loads(line)

                # Check if genui_json field exists
                if "genui_json" not in record:
                    outfile.write(line + '\n')
                    stats["no_genui_json"] += 1
                    continue

                genui_value = record["genui_json"]

                # Check if already a dict/list (not a string)
                if not isinstance(genui_value, str):
                    # Normalize if it's an array
                    normalized = normalize_genui_json(genui_value)
                    if isinstance(normalized, dict) and isinstance(genui_value, list):
                        record["genui_json"] = normalized
                        stats["arrays_converted"] += 1
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats["already_parsed"] += 1
                    continue

                # genui_value is a string - try to parse it
                parsed_genui, success = try_parse_with_repair(genui_value)

                if success and parsed_genui is not None:
                    # Normalize the parsed value (convert arrays to objects if needed)
                    normalized = normalize_genui_json(parsed_genui)
                    record["genui_json"] = normalized
                    if isinstance(parsed_genui, list) and isinstance(normalized, dict):
                        stats["arrays_converted"] += 1
                    outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stats["successful_conversions"] += 1
                else:
                    # Could not parse - keep original
                    outfile.write(line + '\n')
                    stats["failed_conversions"] += 1
                    print(f"  Line {line_num}: Could not parse genui_json")

            except json.JSONDecodeError as e:
                # Line itself is not valid JSON
                stats["failed_conversions"] += 1
                print(f"  Line {line_num}: Line is not valid JSON - {e}")
                outfile.write(line + '\n')

    return stats


def print_stats(stats: dict):
    """Print conversion statistics."""
    print("\n" + "=" * 50)
    print("Conversion Statistics")
    print("=" * 50)
    print(f"Total lines processed: {stats['total_lines']}")
    print(f"Successful conversions: {stats['successful_conversions']}")
    print(f"Arrays converted to objects: {stats['arrays_converted']}")
    print(f"Already parsed (no conversion needed): {stats['already_parsed']}")
    print(f"No genui_json field: {stats['no_genui_json']}")
    print(f"Failed conversions: {stats['failed_conversions']}")
    print(f"Skipped empty lines: {stats['skipped_empty']}")

    success_rate = (stats['successful_conversions'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    print("=" * 50)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print(f"\nUsage: python {sys.argv[0]} <input_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Use the same file for input and output (in-place editing)
    output_path = input_path

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path} (same as input - in-place editing)")

    stats = convert_genui_json(input_path, output_path)
    print_stats(stats)

    if stats["failed_conversions"] > 0:
        print(f"\nWarning: {stats['failed_conversions']} lines could not be converted.")
        print("These lines contain malformed genui_json that cannot be parsed as valid JSON.")
        print("The original lines have been preserved in the output file.")

    print(f"\nSuccessfully converted {stats['successful_conversions']} records.")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
