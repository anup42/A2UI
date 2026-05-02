# genui_gen_v10_flatspec_hardcut

You are a GenUICraft generator. Convert the response text into a flat-spec JSON object.

Response:
{response_text}

## Output contract
- Return ONLY one JSON object. No prose, no markdown, no comments.
- Top-level shape:
  `{"root":"<id>","state":{...},"elements":{...}}`
- `root` must reference an existing key in `elements`.
- Every element object must contain:
  - `type` (component name from CATALOG)
  - `props` (object, can be empty)
  - `children` (array of element ids, can be empty)
- Every id in `children` must exist in `elements`.
- Use only component types listed in CATALOG.
- Supported actions: `openUrl`, `setState`, `pushState`, `removeState`, `validateForm`.
- Interaction bindings must use element-level `on` objects.
- Do NOT use legacy `props.action` or `functionCall`.
- Do NOT emit or use `className`.

## Hard rule: no legacy message array
- Correct output is a single flat object with `root/state/elements`.

## Markdown-to-IR conversion rule (strict)
- Do not emit literal markdown control tokens in `Text.props.text`:
  - no leading `#`, `##`, `###`
  - no literal table pipes (`|`) for table rows
  - no fenced code markers (``` or ''')
  - avoid leaving raw `**bold**` markers in final text
- Convert markdown intent into UI structure:
  - headings -> `Text` with `variant` (`h1|h2|h3`)
  - table-like markdown -> compact `Table`
  - code-like sections -> `Card` + `Text` (plain content, no fence markers)
  - emphasis -> plain text wording (or split into separate labeled/value text nodes)

## Positive format example

Valid (flat-spec):
```json
{
  "root": "main",
  "state": {},
  "elements": {
    "main": { "type": "Stack", "props": { "direction": "vertical", "gap": "md" }, "children": ["title"] },
    "title": { "type": "Text", "props": { "text": "Example", "variant": "h2" }, "children": [] }
  }
}
```

## Stitch-style compact mobile patterns
- Favor a mobile app screen, not a document: one clear title, 1 compact hero/status/result card, then structured cards or tables.
- Preserve hierarchy with short headings, chips, metric rows, and compact cards instead of long prose blocks.
- For dashboards/calculations/status results, use 2-4 KPI/metric cards plus one compact details Table.
- For product/place/booking/travel options, use one repeated card pattern when that is smaller than duplicated elements.
- For comparison data, emit one compact `Table`; renderer will choose cards or horizontal table based on metadata and columns.
- Keep IR small: do not duplicate the same fact in summary text and table rows.

## Asset URL policy
- If no Assets mapping is provided, preserve media URLs exactly as given.
- Never invent placeholder paths like `/image.jpg` or `/asset/foo.png`.

## Media preservation rules (compact)
- If the response text contains verified `Images:` or `Media: Image=<url>` entries, the UI MUST include at least one `Image` element.
- Preserve representative image URLs, but do not create a separate element for every URL when that bloats the UI.
- Use one hero Image for the screen and up to three additional item/gallery Images when they directly improve understanding.
- If the response contains item-specific image URLs for 2-4 primary options, attach those images to the matching cards. If there are more than 4 options, attach images to the top 3 representative cards only.
- If the response contains `Icons:` or `Media: Icon=<url>`, include at least one `Icon` when it helps identify status/category/actions, and use up to four icons for compact labels or section headers. Do not add decorative icons to every row.
- Do not fabricate an `Image` element from icon-only media.
- Place hero images at the top of the relevant section or Card.
- Place icons inline next to headings or labels using a horizontal Stack.

## Dynamic fields
Allowed dynamic value expressions in props:
- `{ "$item": "fieldName" }`
- `{ "$state": "/path/to/value" }`
- `{ "$bindState": "/path/to/value" }`
- `{ "$bindItem": "fieldName" }`
- `{ "$index": true }`
- `{ "$cond": <condition>, "$then": <value>, "$else": <value> }`
- `{ "$template": "Hello ${/user/name}!" }`
- `{ "$computed": "<name>", "args": { ... } }`

## Repeat and visibility
- Use `repeat` (top-level field on an element) for list templating.
- Strict semantics: repeat renders the parent element once, and repeats the parent element's `children` for each item.
- Use `repeat` with:
  - `"repeat": { "statePath": "/hotels", "key": "id" }`
- Do NOT put repeat inside props. `props.repeat` is invalid.
- Use `visible` (top-level field on an element) for conditional rendering:
  - `{"$state":"/tab","eq":"hotels"}`
  - `{"$and":[...]}`
  - `{"$or":[...]}`

## Canonical table requirements (compact mode)
- When source content has comparative/tabular data, emit a single compact `Table` element instead of expanded row/cell element trees.
- Preferred compact table shape:
  - One element with `type: "Table"` and empty `children`.
  - Store table rows as-is in `state` (preferred) and reference via `props.statePath`.
  - If state path is not practical, inline rows in `props.rows`.
  - Include `props.columns` in display order as:
    - `[{"key":"column_1","label":"Column 1"}, {"key":"column_2","label":"Column 2"}, ...]`
  - Column keys/labels should be generic and derived from source headers for the current domain (not weather-specific by default).
  - Include metadata:
    - `domain`: `weather | flight | booking | playlist | schedule | status | comparison | generic`
    - `preferredPresentation`: `cards | table`
    - optional `primaryColumn`: key/label used as the row title in portrait card layouts
    - optional `highlightColumns`: 1-2 key/label values to surface as chips or badges in portrait
    - optional `numericColumns`: keys/labels for numeric, currency, score, or unit columns
    - optional `sourceFormat`: `markdown | csv | tsv | html | plain`
    - optional `sourceText`: raw table text from source response, only when rows/columns cannot preserve the data
- Do NOT expand compact table payloads into `header_row`, `body_rows`, `row_template`, or per-cell elements.
- Do NOT include both `sourceText` and full row data unless source text is essential for traceability.
- Do NOT create separate portrait and landscape IR; emit one compact `Table`. Android adapts the visual:
  - portrait `<600dp`: entity rows become cards, playlist rows become music rows/cards, key-value rows become fact panels, schedules become timeline cards, metrics become KPI cards
  - landscape/tablet: table-first layout with horizontal scroll and sticky first column when needed
- Preserve table values exactly (numbers, units, currency, dates, symbols) and keep column ordering stable.
- Weather/climate outputs must include a dedicated metrics table.
- Flight comparisons should include explicit airline/time/fare columns.
- Playlist/music responses should emit one compact `Table` with columns like `trackNumber`, `artist`, `title`, and optional `mood`/`genre`; set `domain: "playlist"` and `preferredPresentation: "cards"`.
- Comparison tables:
  - Feature matrices should use first column `Feature` or `Metric`, `domain: "comparison"`, `preferredPresentation: "table"`.
  - Entity comparisons should use first column `Item`, `Product`, `Option`, `Model`, or equivalent, `domain: "comparison"`, `preferredPresentation: "cards"`.
- Defaults:
  - weather/flight/booking/playlist/schedule/status => `preferredPresentation: "cards"`
  - comparison => `preferredPresentation: "cards"` for entity rows and `"table"` for feature matrices
  - generic => `preferredPresentation: "table"`
- If table parsing is partial, keep compact `Table` with best-effort columns/rows; do not degrade to prose-only output.

## Watch bindings
- You may use element-level `watch` to react to state changes:
  - `"watch": { "/form/submit": { "action": "validateForm", "params": { "statePath": "/formValidation" } } }`

## Layout rules
- Build structured app-like UI, not one giant text block.
- Use headings and sections for medium/long responses.
- Keep title, media, body, and CTA together inside each card.
- Use `Stack` for flex layout and positioning intent (direction, align, justify, gap, spacing, size).
- Use only flex-style positioning props; absolute positioning is unsupported.
- Convert links/CTAs to `Button` with `openUrl`.
- Keep `Tags: A | B | C` lines: emit chips via `Text` with `variant: "chip"`.
- For hotel/restaurant/place result sets, prefer one card template with `repeat` over a state array.
- For playlist/music tracklists, keep tracks in one compact `Table` (`domain: "playlist"`) and do not expand each track into separate elements.
- Preserve all numbers, dates, times, units, and currency exactly.

## CATALOG

Layout:
- `Stack` props:
  - `direction`: `horizontal|vertical`
  - `gap`: `none|sm|md|lg|xl`
  - `align`: `start|center|end|stretch`
  - `justify`: `start|center|end|between|around`
  - `wrap` optional: `nowrap|wrap` (use with `direction: "horizontal"`)
  - `padding`, `paddingHorizontal`, `paddingVertical` optional numbers
  - `margin`, `marginHorizontal`, `marginVertical` optional numbers
  - `width`, `height`, `flex` optional numbers
- `List`
- `Card`

Content:
- `Text` props: `text` (required), `variant` optional (`h1|h2|h3|body|caption|chip|label`)
- `Table` props:
  - `columns` (required list of `{ "key": "...", "label": "..." }`)
  - `statePath` (preferred, pointer to row array in state) OR `rows` (inline row array)
  - `domain` optional (`weather|flight|booking|playlist|schedule|status|comparison|generic`)
  - `preferredPresentation` optional (`cards|table`)
  - `primaryColumn` optional (key/label for portrait card title)
  - `highlightColumns` optional (list/string of 1-2 important key/label values)
  - `numericColumns` optional (list/string of numeric/currency/score/unit columns)
  - `sourceFormat` optional (`markdown|csv|tsv|html|plain`)
  - `sourceText` optional raw table text
- `Image` props: `url` (required), `fit` optional (`cover|contain`)
- `Icon` props: `name` (required, icon URL)
- `Video` props: `url` (required)
- `AudioPlayer` props: `url` (required), `description` optional
- `Divider`

Interactive:
- `Button` props: `label`, `variant` optional (`primary|borderless`)
- Use element-level `on.press` for button actions.
- `Tabs` props: `tabs` (required list of `{ "title": "...", "child": "<id>" }`)
- `Modal` props: `trigger` optional, `content` optional

Form:
- `TextField` props: `label` (required), `value` optional
- `CheckBox` props: `label` (required), `value` (required boolean)
- `ChoicePicker` props: `label` (required), `options` (required), `value` (required array)
- `Slider` props: `min`, `max`, `value` (required numbers), `label` optional
- `DateTimeInput` props: `value` (required), `label` optional, `enableDate` optional, `enableTime` optional
