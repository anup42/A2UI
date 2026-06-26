TEDY Flat-Spec IR Contract for Dynamic UI Generation

The figure should focus only on the intermediate representation specification, not the full data-generation pipeline. The TEDY flat-spec IR is a compact JSON object designed for deterministic validation and native mobile rendering. The top-level object has exactly three required fields:

1. root: string id of the root element.
2. state: reusable data store containing arrays, objects, scalar values, and table rows.
3. elements: a flat map from element id to element node definitions.

Each element node follows a small grammar:
- type: semantic component type such as Stack, Row, Card, Text, Image, Icon, Button, Table, Tabs, Divider, EmailPreview, CodeBlock, FormulaBlock.
- props: component properties, including text, variant, direction, padding, gap, columns, rows, statePath, domain, preferredPresentation, source, url, alt, actionLabel, and layout hints.
- children: ordered list of child element ids.
- repeat: optional data binding descriptor with statePath and key for repeating a template over state rows.
- on: optional event map for interactions such as press -> openUrl or copy.

The spec separates data from presentation. Tables remain compact: columns are stored as ordered metadata, rows are stored in state or inline props.rows, and renderer-specific presentation hints are optional. Domain hints include weather, flight, booking, schedule, status, comparison, playlist, formula, email, code_console, and generic. The renderer decides whether to show cards, timelines, horizontal tables, playlist rows, formula panels, or email previews based on domain, shape, and viewport.

Dynamic values are allowed only through explicit expressions such as $item, $state, $bindState, $index, $cond, $template, and $computed. These expressions keep repeated content compact and make data bindings auditable.

Validation and normalization enforce determinism before rendering:
- root id must exist in elements.
- child references must resolve.
- element types must be supported.
- action shapes must be valid.
- table aliases are normalized to canonical columns and statePath/rows.
- expanded cell trees are compacted into Table data when safe.
- empty/default-only props are pruned.
- invalid IR falls back to deterministic valid flat-spec.

The renderer consumes the validated flat-spec and maps semantic components to native mobile UI. The renderer supports phone-first adaptive presentation, accessibility labels, focus semantics, contrast-safe cards, image/icon fallbacks, and domain-specific templates for weather, flights, booking, playlists, formulas, code blocks, and emails.

Visual requirements for the diagram:
- White academic paper style with clean blue/teal accents.
- Make it a detailed spec diagram, not a pipeline diagram.
- Show a central JSON contract block with root/state/elements.
- Show a node grammar panel for element fields: type, props, children, repeat, on.
- Show a compact table metadata panel: columns + rows/statePath + domain + preferredPresentation.
- Show a validation/normalization gate with checklist icons.
- Show a native mobile renderer output panel with adaptive cards/table examples.
- Use small icons for data store, element map, schema shield, table, action button, media, and phone renderer.
- Text must be readable and concise; avoid tiny paragraphs.
- Do not place the caption as a title inside the image.
