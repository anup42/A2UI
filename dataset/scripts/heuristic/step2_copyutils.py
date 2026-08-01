#!/usr/bin/env python3
"""
Script to copy utility files (queries.jsonl, responses.jsonl, golden50_url_mapping.jsonl)
from a util folder to a target input folder.

Usage: python step2_copyutils.py <util_folder> <input_folder>
"""

import shutil
import sys
import os


FILES_TO_COPY = [
    "queries.jsonl",
    "responses.jsonl",
    "golden50_url_mapping.jsonl"
]


def copy_files_to_folder(util_folder: str, input_folder: str):
    """
    Copy the utility files from util folder to the input folder.

    Args:
        util_folder: Path to the source folder containing utility files
        input_folder: Path to the target folder where files will be copied
    """
    if not os.path.exists(util_folder):
        print(f"Error: Util folder not found: {util_folder}")
        sys.exit(1)

    if not os.path.exists(input_folder):
        print(f"Error: Input folder not found: {input_folder}")
        sys.exit(1)

    if not os.path.isdir(util_folder):
        print(f"Error: Util path is not a directory: {util_folder}")
        sys.exit(1)

    if not os.path.isdir(input_folder):
        print(f"Error: Input path is not a directory: {input_folder}")
        sys.exit(1)

    copied_count = 0
    for filename in FILES_TO_COPY:
        source_path = os.path.join(util_folder, filename)
        dest_path = os.path.join(input_folder, filename)

        if not os.path.exists(source_path):
            print(f"Warning: Source file not found: {source_path}")
            continue

        shutil.copy2(source_path, dest_path)
        print(f"Copied: {filename} -> {input_folder}")
        copied_count += 1

    print(f"\nSuccessfully copied {copied_count}/{len(FILES_TO_COPY)} files to: {input_folder}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python step2_copyutils.py <util_folder> <input_folder>")
        sys.exit(1)

    util_folder = sys.argv[1]
    input_folder = sys.argv[2]
    copy_files_to_folder(util_folder, input_folder)
