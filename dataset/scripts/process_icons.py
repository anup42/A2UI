#!/usr/bin/env python3
"""
Script to process SVG icons in genui.jsonl:
1. Finds all SVG icon URLs in the JSONL file
2. Downloads missing icons to the assets folder
3. Replaces all SVG URLs with local asset paths

Asset folder format: <response_id>_<N>_<icon_name>.svg
JSONL icon URL format: <someurl><icon_name>.svg
"""

import json
import os
import re
import ssl
import urllib.request
from pathlib import Path

# Create SSL context that doesn't verify certificates (for downloading public icons)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


def extract_icon_name_from_url(url):
    """Extract icon name from a URL like <someurl><icon_name>.svg"""
    if not isinstance(url, str):
        return None
    # Only match HTTP URLs, not local asset paths
    if not url.startswith('http'):
        return None
    match = re.search(r'/([^/]+)\.svg$', url)
    if match:
        return match.group(1)
    return None


def find_svg_icon_names_in_assets(assets_dir):
    """
    Find all SVG files in the assets directory and return a set of icon names.
    Asset format: <response_id>_<number>_<icon_name>.svg
    """
    svg_icon_names = set()
    if not os.path.exists(assets_dir):
        return svg_icon_names

    for filename in os.listdir(assets_dir):
        if filename.endswith('.svg'):
            # Extract icon_name from format: r_XXXXXX_XX_N_icon-name.svg
            match = re.match(r'r_\d+_\d+_\d+_(.+)\.svg$', filename)
            if match:
                icon_name = match.group(1)
                svg_icon_names.add(icon_name)

    return svg_icon_names


def find_missing_icons_recursive(obj, existing_icon_names, missing_icons, seen_urls, response_id=""):
    """
    Recursively traverse any JSON object and find SVG URLs without matching assets.
    """
    if isinstance(obj, dict):
        current_response_id = obj.get('response_id', response_id)

        for key, value in obj.items():
            if isinstance(value, str):
                if value.endswith('.svg') and value.startswith('http'):
                    icon_name = extract_icon_name_from_url(value)
                    if icon_name and icon_name not in existing_icon_names:
                        if value not in seen_urls:
                            seen_urls.add(value)
                            missing_icons.append({
                                'url': value,
                                'icon_name': icon_name,
                                'response_id': current_response_id
                            })
            else:
                find_missing_icons_recursive(value, existing_icon_names, missing_icons, seen_urls, current_response_id)

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, str):
                if item.endswith('.svg') and item.startswith('http'):
                    icon_name = extract_icon_name_from_url(item)
                    if icon_name and icon_name not in existing_icon_names:
                        if item not in seen_urls:
                            seen_urls.add(item)
                            missing_icons.append({
                                'url': item,
                                'icon_name': icon_name,
                                'response_id': response_id
                            })
            else:
                find_missing_icons_recursive(item, existing_icon_names, missing_icons, seen_urls, response_id)


def download_svg(url, output_path):
    """Download an SVG file from URL and save it to output_path."""
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response:
            svg_content = response.read().decode('utf-8')

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)

        return True
    except Exception as e:
        print(f"  Error downloading {url}: {e}")
        return False


def replace_svg_recursive(obj, svg_files_map, stats):
    """
    Recursively traverse any JSON object and replace SVG URLs with local asset paths.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                icon_name = extract_icon_name_from_url(value)
                if icon_name:
                    stats['found'] += 1
                    if icon_name in svg_files_map:
                        stats['replaced'] += 1
                        obj[key] = f"../assets/{svg_files_map[icon_name]}"
            else:
                replace_svg_recursive(value, svg_files_map, stats)

    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                icon_name = extract_icon_name_from_url(item)
                if icon_name:
                    stats['found'] += 1
                    if icon_name in svg_files_map:
                        stats['replaced'] += 1
                        obj[i] = f"../assets/{svg_files_map[icon_name]}"
            else:
                replace_svg_recursive(item, svg_files_map, stats)


def process_jsonl_file(jsonl_path, assets_dir):
    """
    Process the JSONL file: download missing icons and replace all SVG URLs.
    """
    # Step 1: Find existing icon names in assets
    existing_icon_names = find_svg_icon_names_in_assets(assets_dir)
    print(f"Found {len(existing_icon_names)} existing SVG icon names in assets folder")

    # Step 2: Find missing icons
    missing_icons = []
    seen_urls = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                continue

            find_missing_icons_recursive(data, existing_icon_names, missing_icons, seen_urls)

    # Remove duplicates based on icon_name
    unique_missing = {}
    for icon in missing_icons:
        if icon['icon_name'] not in unique_missing:
            unique_missing[icon['icon_name']] = icon

    missing_icons = list(unique_missing.values())
    print(f"Found {len(missing_icons)} unique missing SVG icons")

    # Step 3: Download missing icons with proper naming
    if missing_icons:
        print("-" * 50)
        print("Downloading missing icons...")

        icon_counter = {}
        for icon in missing_icons:
            response_id = icon.get('response_id', 'unknown')
            icon_name = icon['icon_name']

            if response_id not in icon_counter:
                icon_counter[response_id] = 1
            counter = icon_counter[response_id]
            icon_counter[response_id] += 1

            # Format: r_XXXXXX_XX_N_icon-name.svg
            response_prefix = response_id if response_id != 'unknown' else 'missing'
            output_filename = f"{response_prefix}_{counter:02d}_{icon_name}.svg"
            output_path = os.path.join(assets_dir, output_filename)

            print(f"Downloading: {icon_name} -> {output_filename}")

            if download_svg(icon['url'], output_path):
                existing_icon_names.add(icon_name)  # Add to set for replacement

    # Step 4: Build svg_files_map for replacement
    svg_files_map = {}
    for filename in os.listdir(assets_dir):
        if filename.endswith('.svg'):
            match = re.match(r'r_\d+_\d+_\d+_(.+)\.svg$', filename)
            if match:
                icon_name = match.group(1)
                svg_files_map[icon_name] = filename

    print(f"\nTotal available icon names for replacement: {len(svg_files_map)}")

    # Step 5: Replace SVG URLs in JSONL
    print("-" * 50)
    print("Replacing SVG URLs with local asset paths...")

    stats = {'found': 0, 'replaced': 0}
    modified_lines = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                modified_lines.append(line)
                continue

            replace_svg_recursive(data, svg_files_map, stats)
            modified_lines.append(json.dumps(data, ensure_ascii=False))

    # Write back the modified content
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(modified_lines))
        if modified_lines:
            f.write('\n')

    return stats['found'], stats['replaced']


def main():
    import sys

    # Default path
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = r"C:\Users\adarsh.ag\Desktop\GenUI-LM\dataset\data\runs\golden50_trained"

    folder_path = Path(folder_path)

    if not folder_path.exists():
        print(f"Error: Folder does not exist: {folder_path}")
        sys.exit(1)

    assets_dir = folder_path / "assets"

    # Try multiple possible JSONL file names in order of preference
    possible_jsonl_names = ["genui_fixed_local.jsonl", "genui_fixed.jsonl", "genui.jsonl"]
    jsonl_path = None

    for name in possible_jsonl_names:
        candidate = folder_path / name
        if candidate.exists():
            jsonl_path = candidate
            break

    if jsonl_path is None:
        print(f"Error: No JSONL file found in {folder_path}")
        print(f"Looked for: {', '.join(possible_jsonl_names)}")
        sys.exit(1)

    # Create assets directory if it doesn't exist
    if not assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created assets directory: {assets_dir}")

    print(f"Processing: {jsonl_path}")
    print(f"Assets directory: {assets_dir}")
    print("=" * 60)

    icons_found, icons_replaced = process_jsonl_file(jsonl_path, assets_dir)

    print("=" * 60)
    print("Final Statistics:")
    print(f"  Total SVG URLs found in JSONL: {icons_found}")
    print(f"  SVG URLs successfully replaced: {icons_replaced}")
    if icons_found > 0:
        print(f"  Replacement rate: {icons_replaced/icons_found*100:.1f}%")


if __name__ == "__main__":
    main()
