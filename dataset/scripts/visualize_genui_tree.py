#!/usr/bin/env python3
"""
GenUI JSON Tree Visualizer

Generates an interactive tree visualization from genui.jsonl file.
Uses nested HTML structure with expandable/collapsible nodes.

Usage:
    python visualize_genui_tree.py --query_id q_000001 --jsonl_path /path/to/genui.jsonl --output output.html
"""

import argparse
import json
import sys
import subprocess
import webbrowser
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


ELEMENT_TYPES = {
    'Stack': '#3b82f6', 'List': '#60a5fa', 'Card': '#10b981',
    'Text': '#f97316', 'Formula': '#84cc16', 'CodeBlock': '#a855f7',
    'ConsoleLog': '#9333ea', 'EmailPreview': '#f472b6', 'Table': '#ec4899',
    'Chart': '#fb7185', 'Image': '#22d3ee', 'Icon': '#06b6d4',
    'Video': '#f59e0b', 'AudioPlayer': '#d97706', 'Divider': '#6b7280',
    'Button': '#8b5cf6', 'Tabs': '#c026d3', 'Modal': '#7c3aed',
    'TextField': '#14b8a6', 'CheckBox': '#0d9488', 'ChoicePicker': '#0f766e',
    'Slider': '#115e59', 'DateTimeInput': '#134e4a',
}


def find_entry_by_query_id(jsonl_path: str, query_id: str) -> Optional[Dict]:
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get('query_id') == query_id:
                    return entry
            except json.JSONDecodeError:
                pass
    return None


def build_element_tree(elements: Dict[str, Dict], root_id: str) -> Dict[str, Any]:
    def build_node(element_id: str, visited: set) -> Optional[Dict[str, Any]]:
        if element_id in visited or element_id not in elements:
            return None
        visited.add(element_id)
        elem = elements[element_id]
        children = []
        for child_id in elem.get('children', []):
            child_node = build_node(child_id, visited.copy())
            if child_node:
                children.append(child_node)
        return {
            'id': element_id, 'type': elem.get('type', 'Unknown'),
            'props': elem.get('props', {}), 'on': elem.get('on', {}),
            'children': children
        }
    return build_node(root_id, set())


def get_color_for_type(elem_type: str) -> str:
    return ELEMENT_TYPES.get(elem_type, '#6b7280')


def generate_tree_html(tree: Dict[str, Any], depth: int = 0) -> str:
    elem_id = tree['id']
    elem_type = tree['type']
    color = get_color_for_type(elem_type)
    children = tree.get('children', [])
    has_children = len(children) > 0

    display_id = elem_id[:30] + '...' if len(elem_id) > 30 else elem_id

    children_html = ''
    if has_children:
        children_html = '<div class="children">' + ''.join(generate_tree_html(child, depth + 1) for child in children) + '</div>'

    arrow = '▼' if has_children else '·'
    expand_class = ' expanded' if depth < 2 else ''

    return f'''
    <div class="tree-node{expand_class}" data-has-children="{str(has_children).lower()}">
        <div class="node-row" onclick="toggleNode(this)">
            <span class="arrow">{arrow}</span>
            <span class="node-type" style="background-color: {color};">{elem_type}</span>
            <span class="node-id">{display_id}</span>
            {f'<span class="child-count">{len(children)} children</span>' if has_children else ''}
        </div>
        {children_html}
    </div>
    '''


def generate_html(tree: Dict[str, Any], query_id: str, entry: Dict) -> str:
    tree_content = generate_tree_html(tree)

    def count_elements(node: Dict) -> Dict[str, int]:
        counts = {'total': 1, 'by_type': {node['type']: 1}}
        for child in node.get('children', []):
            child_counts = count_elements(child)
            counts['total'] += child_counts['total']
            for t, c in child_counts['by_type'].items():
                counts['by_type'][t] = counts['by_type'].get(t, 0) + c
        return counts

    stats = count_elements(tree)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GenUI Tree - {query_id}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border-color: #475569;
            --accent: #3b82f6;
            --line-color: #475569;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .header {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem 2rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border-color);
        }}

        .header h1 {{
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }}

        .header h1::before {{
            content: '🌳';
        }}

        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }}

        .metadata-item {{
            background: var(--bg-tertiary);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            font-size: 0.875rem;
        }}

        .metadata-item .label {{
            color: var(--text-secondary);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }}

        .metadata-item .value {{
            font-weight: 500;
            color: var(--accent);
        }}

        .stats {{
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}

        .stat-badge {{
            background: var(--bg-tertiary);
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .stat-badge .count {{
            font-weight: 700;
            color: var(--accent);
        }}

        .tree-container {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid var(--border-color);
            overflow-x: auto;
        }}

        .tree-node {{
            margin-left: 0;
        }}

        .tree-node .children {{
            margin-left: 1.5rem;
            padding-left: 1rem;
            border-left: 1px dashed var(--line-color);
            display: none;
        }}

        .tree-node.expanded > .children {{
            display: block;
        }}

        .tree-node.expanded > .node-row > .arrow {{
            transform: rotate(90deg);
        }}

        .node-row {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            margin: 0.25rem 0;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.15s;
        }}

        .node-row:hover {{
            background: var(--bg-tertiary);
        }}

        .arrow {{
            width: 16px;
            height: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: var(--text-secondary);
            transition: transform 0.2s;
            flex-shrink: 0;
        }}

        .tree-node[data-has-children="false"] > .node-row > .arrow {{
            visibility: hidden;
        }}

        .node-type {{
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            color: white;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            flex-shrink: 0;
        }}

        .node-id {{
            font-weight: 500;
            color: var(--text-primary);
            font-size: 0.85rem;
        }}

        .child-count {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: var(--bg-tertiary);
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            margin-left: auto;
        }}

        .legend {{
            margin-top: 2rem;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }}

        .legend h3 {{
            font-size: 0.875rem;
            margin-bottom: 0.75rem;
            color: var(--text-secondary);
        }}

        .legend-items {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.8rem;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 2px;
        }}

        .controls {{
            margin-bottom: 1rem;
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .control-btn {{
            background: var(--bg-tertiary);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.8rem;
            transition: all 0.2s;
        }}

        .control-btn:hover {{
            background: var(--border-color);
            color: var(--text-primary);
        }}

        .info-panel {{
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.85rem;
        }}

        .info-panel strong {{
            color: var(--accent);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>GenUI Tree Visualization</h1>
            <div class="metadata">
                <div class="metadata-item"><div class="label">Query ID</div><div class="value">{query_id}</div></div>
                <div class="metadata-item"><div class="label">UI ID</div><div class="value">{entry.get('ui_id', 'N/A')}</div></div>
                <div class="metadata-item"><div class="label">Response ID</div><div class="value">{entry.get('response_id', 'N/A')}</div></div>
                <div class="metadata-item"><div class="label">Intent</div><div class="value">{entry.get('intent', 'N/A')}</div></div>
                <div class="metadata-item"><div class="label">Created At</div><div class="value">{entry.get('created_at', 'N/A')}</div></div>
            </div>
            <div class="stats">
                <div class="stat-badge"><span class="count">{stats['total']}</span><span>total elements</span></div>
                {"".join(f'<div class="stat-badge"><span class="count">{count}</span><span>{t}</span></div>' for t, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]))}
            </div>
        </header>

        <div class="controls">
            <button class="control-btn" onclick="expandAll()">Expand All</button>
            <button class="control-btn" onclick="collapseAll()">Collapse All</button>
            <button class="control-btn" onclick="expandLevel(1)">Expand 1 Level</button>
            <button class="control-btn" onclick="expandLevel(2)">Expand 2 Levels</button>
            <button class="control-btn" onclick="expandLevel(3)">Expand 3 Levels</button>
        </div>

        <div class="info-panel">
            <strong>💡 Tip:</strong> Click on any node to expand/collapse its children. Nodes with children show a ▼ arrow.
        </div>

        <div class="tree-container">
            {tree_content}
        </div>

        <div class="legend">
            <h3>Element Types</h3>
            <div class="legend-items">
                {"".join(f'<div class="legend-item"><div class="legend-color" style="background: {c}"></div><span>{t}</span></div>' for t, c in ELEMENT_TYPES.items())}
            </div>
        </div>
    </div>

    <script>
        function toggleNode(row) {{
            const node = row.parentElement;
            node.classList.toggle('expanded');
        }}

        function expandAll() {{
            document.querySelectorAll('.tree-node[data-has-children="true"]').forEach(node => {{
                node.classList.add('expanded');
            }});
        }}

        function collapseAll() {{
            document.querySelectorAll('.tree-node').forEach(node => {{
                node.classList.remove('expanded');
            }});
        }}

        function expandLevel(level) {{
            collapseAll();
            document.querySelectorAll('.tree-node[data-has-children="true"]').forEach((node, index) => {{
                const depth = getNodeDepth(node);
                if (depth < level) {{
                    node.classList.add('expanded');
                }}
            }});
        }}

        function getNodeDepth(node) {{
            let depth = 0;
            let parent = node.parentElement;
            while (parent && parent.classList.contains('tree-container') === false) {{
                if (parent.classList.contains('tree-node')) {{
                    depth++;
                }}
                parent = parent.parentElement;
            }}
            return depth;
        }}

        // Expand first 2 levels by default
        document.addEventListener('DOMContentLoaded', () => {{
            expandLevel(2);
        }});
    </script>
</body>
</html>'''
    return html


def main():
    parser = argparse.ArgumentParser(description='Generate HTML tree visualization from genui.jsonl')
    parser.add_argument('-q', '--query_id', required=True, help='Query ID to visualize')
    parser.add_argument('-j', '--jsonl_path', default='/home/adarsh_ag/k_anup/dataset/data/runs/dataset_gemma4_50k_cyclic_v1/genui.jsonl', help='Path to genui.jsonl')
    parser.add_argument('-o', '--output', default='genui_tree.html', help='Output HTML file')
    parser.add_argument('--no-open', action='store_true', help='Do not open browser')

    args = parser.parse_args()

    print(f"Searching for query_id: {args.query_id}")
    print(f"Reading from: {args.jsonl_path}")

    entry = find_entry_by_query_id(args.jsonl_path, args.query_id)
    if not entry:
        print(f"Error: No entry found for query_id '{args.query_id}'", file=sys.stderr)
        sys.exit(1)

    genui_json = entry.get('genui_json')
    if not genui_json:
        print("Error: No genui_json found", file=sys.stderr)
        sys.exit(1)

    elements = genui_json.get('elements', {})
    root_id = genui_json.get('root')

    if not root_id:
        print("Error: No root element found", file=sys.stderr)
        sys.exit(1)

    print(f"Building tree from root: {root_id}")
    tree = build_element_tree(elements, root_id)

    if not tree:
        print(f"Error: Could not build tree from root '{root_id}'", file=sys.stderr)
        sys.exit(1)

    html = generate_html(tree, args.query_id, entry)

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Generated: {output_path.absolute()}")
    abs_path = output_path.absolute()
    print(f"🚀 Opening in browser...")
    print(f"📓 JupyterLab: Right-click file → Open With → Browser")
    print(f"   Notebook: from IPython.display import IFrame; IFrame('{abs_path}', width=1400, height=800)")

    if not args.no_open:
        try:
            webbrowser.open(f"file://{abs_path}")
        except Exception:
            try:
                subprocess.run(['xdg-open', str(abs_path)], check=True)
            except Exception:
                print(f"  Note: Open manually: {abs_path}")


if __name__ == '__main__':
    main()
