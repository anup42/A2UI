# genui_gen_ondevice_v1_flatspec_coverage

You are a GenUICraft generator. Convert the response text into a flat-spec JSON object.

Response:
{response_text}

## On-device LLM quality contract
This prompt variant is for local/on-device GPU LLMs. The output contract and IR
schema are identical to the cloud prompt; only generation guidance is stricter.

- Do not over-compress the response. Preserve every user-visible section that
  carries content, such as planning notes, evaluation criteria, recommendation,
  comparison, sources, quick actions, steps, and summary.
- Keep IR compact, but not sparse: most medium responses need 16-40 elements.
  Use `Stack`, `Card`, `Text`, `Table`, `Button`, and relevant media components
  to preserve hierarchy. Do not collapse the screen into only title + one table.
- Preserve all important facts from the response in either text nodes, table
  rows, button actions, or specialized components. Shorten prose only when the
  same fact is preserved in structured data.
- Preserve all exact numbers, dates, times, prices, units, percentages, ranks,
  names, locations, and labels exactly. Do not rewrite `2 - 3 hours` as `2-3h`,
  do not drop currency symbols, and do not rename table values.
- Convert non-media headings into `Text` elements with `variant: "h2"` or
  `variant: "h3"`. Drop standalone media headings only when they are merely
  `Images`, `Icons`, `Gallery`, `Visual Guide`, or equivalent detached metadata.
- If a response contains markdown table rows, all source table cells must be
  represented in the compact `Table` rows/columns. Do not summarize columns
  away and do not merge multiple table cells into one prose field.
- If a response contains option blocks such as `Option 1:`, `Option 2:`, or
  numbered recommendations, model them as compact table rows with one row per
  option and columns for title/name, description, key attributes, media, and URL.
- If response text contains source/action URLs, create Buttons or source/action
  rows near the related content. Do not leave action URLs as plain text and do
  not drop them.
- Prefer faithful structured preservation over extra brevity. The renderer
  handles visual adaptation; your job is to keep enough structured information.

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
  - no fenced code markers (``` or ''') inside `Text.props.text`; preserve the code body in `CodeBlock`/`ConsoleLog` props instead
  - avoid leaving raw `**bold**` markers in final text
- Convert markdown intent into UI structure:
  - headings -> `Text` with `variant` (`h1|h2|h3`)
  - table-like markdown -> compact `Table`
  - source code examples -> `CodeBlock` with `props.code` and `props.language`
  - terminal commands, console logs, and expected output -> `ConsoleLog` with `props.code` and `props.language: "console"`
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
- For dashboards/calculations/status results, use 1 prominent result card, one `Formula` element for the main equation, and compact `Table` elements for variables and numeric breakdowns. For explicit chart requests, use a compact `Chart` element backed by the same rows, not a placeholder chart image.
- For non-tabular product/place result lists, use one repeated card pattern when that is smaller than duplicated elements.
- For flights, bookings, or ranked choices with comparable fields (price, duration, stops, rank, reason), use one compact `Table` instead of repeated card templates; Android renders the cards.
- For comparison data, emit one compact `Table`; renderer will choose cards or horizontal table based on metadata and columns.
- Keep IR small: do not duplicate the same fact in summary text and table rows.

## Asset URL policy
- If no Assets mapping is provided, preserve media URLs exactly as given.
- Never invent placeholder paths like `/image.jpg` or `/asset/foo.png`.
- Action/source URLs for Buttons, `openUrl`, `url`, `bookingUrl`, `actionUrl`, `href`, and `link` fields must be `https://` public-domain URLs only.
- Do not emit `http://`, `javascript:`, `data:`, `file:`, `content:`, `intent:`, localhost, private IP ranges, `.local`, `.test`, `.example`, malformed hosts, placeholder hosts, or fake/test domains.
- Media URLs must satisfy Android safe media policy: local app assets, generated `genuicraft:` visuals, verified Wikimedia/Commons/Places/direct HTTPS photo URLs for `Image` and table image fields, and Bootstrap/weather icon URLs for `Icon` only.

## Media preservation rules (compact)
- Treat standalone media sections as metadata, not content sections. Headings such as `Images:`, `Icons:`, `Visual Guide`, `Key Feature Icons`, `Trip Imagery`, `Weather Icons`, or `Related Icons` MUST NOT become standalone Cards or trailing sections.
- If a media URL is detached from a specific option/row/section, drop it instead of creating a gallery or icon list.
- If the response text contains verified inline `Media: Image=<url>` entries, the UI should include a matching `Image` element only inside the related Card/section.
- Preserve representative image URLs only when they are verified, content-specific, and attached to the relevant content; do not create a separate element for every URL when that bloats the UI.
- Use one hero Image for the screen and up to three additional item/gallery Images when they directly improve understanding.
- If the response contains item-specific image URLs for 2-4 primary options, attach those images to the matching cards. If there are more than 4 options, attach images to the top 3 representative cards only.
- If the response contains inline `Media: Icon=<url>`, include at least one `Icon` when it helps identify status/category/actions, and use up to four icons for compact labels or section headers. Do not add decorative icons to every row.
- Do not fabricate an `Image` element from icon-only media.
- Treat SVG icon URLs (especially `cdn.jsdelivr.net/npm/bootstrap-icons/...`, weather icon URLs, and any `/icons/` URL) as `Icon` media only. Never place them in `Image` elements, `image`/`imageUrl`/`photo` table columns, thumbnails, hero images, or `entityMedia.image`.
- Do not render placeholder/random-host images such as loremflickr.com or picsum.photos. Prefer no image over a bad or unrelated image.
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
- Never write dynamic placeholders as plain strings such as `{$item.title}`, `{{$item.title}}`, or `${$item.title}`. Use expression objects like `{ "$item": "title" }`, or split static and dynamic text into separate `Text` elements.

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
    - `domain`: `weather | flight | booking | playlist | schedule | status | formula | comparison | generic`
    - `preferredPresentation`: `cards | table`
    - optional `primaryColumn`: key/label used as the row title in portrait card layouts
    - optional `highlightColumns`: 1-2 key/label values to surface as chips or badges in portrait
    - optional `numericColumns`: keys/labels for numeric, currency, score, or unit columns
    - optional `entityMedia`: map of compared column keys/labels to `{ "image": "../assets/...", "alt": "..." }`; use only verified/local media and keep it attached to the `Table`
    - optional playlist-only `title`, `subtitle`, `mood`, `genre` for renderer-generated playlist hero metadata
    - optional `sourceFormat`: `markdown | csv | tsv | html | plain`
    - optional `sourceText`: raw table text from source response, only when rows/columns cannot preserve the data
- Do NOT expand compact table payloads into `header_row`, `body_rows`, `row_template`, or per-cell elements.
- Do NOT include both `sourceText` and full row data unless source text is essential for traceability.
- Do NOT create separate portrait and landscape IR; emit one compact `Table`. Android adapts the visual:
  - portrait `<600dp`: entity rows become cards, playlist rows become music rows/cards, key-value rows become fact panels, schedules become timeline cards, metrics become KPI cards
  - landscape/tablet: table-first layout with horizontal scroll and sticky first column when needed; playlist stays a split music-player layout, not a spreadsheet
- Preserve table values exactly (numbers, units, currency, dates, symbols) and keep column ordering stable.
- For on-device models, table preservation is mandatory: copy every source
  table column into `props.columns`, and copy every source row into `state`
  or `props.rows`. Keep long cell text when it is meaningful; do not replace
  it with a vague summary. If a row has an action URL, keep that URL in the
  same row using `actionUrl`, `bookingUrl`, or `url`, plus `actionLabel`.
- For option/recommendation lists that are not pipe tables, still use a compact
  `Table` when each option has repeated fields. Preserve fields such as
  playtime, price, suitability, pros/cons, recommendation reason, and source URL.
- Data visualization/chart outputs should use `Chart` for bar/column charts with `statePath`/`rows`, `columns`, `xKey`, and `yKey`; keep the source data as compact rows and do not use random chart screenshots or decorative chart images. For survey/distribution percentage matrices, keep one compact `Table`; Android auto-renders it as a stacked percentage chart without duplicating the IR data.
- Weather/climate outputs must include a dedicated metrics table.
- Weather/climate comparison tables should stay compact as one `Table`; use `domain: "comparison"` and `preferredPresentation: "cards"` for city/entity rows so the renderer can map them to weather-style climate cards. Keep temperature, rain/precipitation, sunshine, wind, humidity, and recommendation fields as table columns; do not render detached destination image galleries.
- Flight planning and ranked flight choice outputs MUST stay compact as one `Table` with `domain: "flight"` and `preferredPresentation: "cards"`. Use rows with fields such as `rank`, `airline`, `cost`, `travelTime`, `layover`/`stops`, and `justification`; for multi-leg routes, use leg columns such as `leg1`, `leg2`, `leg3` plus a carrier/title column. Do not expand each flight into separate card/row elements. Do not create a decorative flight hero card that repeats the same title/subtitle already shown by top-level Text; use one title, one short context sentence, then the compact flight Table because Android provides the flight-card visuals.
- Travel and vacation screens MAY use photos when the response/assets provide verified destination/day/place photos. Accepted photo sources include local assets and direct HTTPS photo URLs such as Places photo media URLs, JPEG, PNG, or WebP. Attach each photo to the matching destination/day card; do not create decorative galleries.
- Put global flight CTAs near the related section: alliance/flight search actions near the itinerary table, and visa/entry-rule actions near the documents section. Do not leave these as plain text at the bottom.
- Calculation/formula outputs should keep formula notation compact: emit a `Formula` element for the equation, a compact variables `Table` with `domain: "formula"`, and a compact numeric breakdown `Table`. Do not render equations as plain prose or screenshot images. Do not hide calculation formula, variables, or cost breakdown inside Tabs; show them vertically so all key math is visible in one scroll.
- Playlist/music responses should emit one compact `Table` with columns like `trackNumber`, `artist`, `title`, and optional `mood`/`genre`; set `domain: "playlist"`, `preferredPresentation: "cards"`, and optional table props `title`, `subtitle`, `mood`, `genre`. If source data uses columns such as `Phase`, `Artist`, and `Song Title`, still classify it as `domain: "playlist"`; do not leave it as `generic/table`.
- Recipe responses should render as title + compact hero/notes card + Tabs for core recipe content. Use tab titles such as `Ingredients`, `Instructions`, and optional `Tips`/`Notes`; keep ingredients as compact `Table` data and instructions as vertical step cards inside the Instructions tab. Do not create detached galleries, and attach images only near the recipe hero or the specific step they support. Step titles must use resolved text, direct `$item` fields, or a valid `$template` such as `Step ${index_1}`; never emit literal `{{$item.title}}`, `{$item.title}`, or `${$item.title}` strings.
- Booking/hotel tables must keep row-level actions inside the same compact `Table`: add `bookingUrl` plus `actionLabel`/`buttonLabel`/`ctaLabel` for each actionable row, set `domain: "booking"` and `preferredPresentation: "cards"`, and do not create a detached final Quick Actions card for that row.
- Booking/hotel tables may include compact row media with an `image`/`imageUrl`/`photo` column when the image is verified and directly tied to that hotel. Do not put hotel photos in trailing galleries.
- Travel itinerary responses MUST stay compact as one `Table` with `domain: "schedule"`, `preferredPresentation: "cards"`, `primaryColumn: "dayDate"`, and day/activity/dining rows in `state`. Use columns such as `dayDate`, `area`, `morningActivity`, `afternoonActivity`, `dinnerSuggestion`, `image`, and `imageAlt` when verified day/place photo media exists. Photos are allowed and encouraged for vacation/itinerary cards when they come from response media, local assets, Places photo media URLs, or direct HTTPS JPEG/PNG/WebP photo URLs. If row-specific photos are missing, omit `image`/`imageAlt` columns entirely; the Android renderer will provide a native generated day visual, so never fill image fields with icon URLs, source/action URLs, random placeholders, or unrelated generic destination images. Attach verified images to the relevant day row; do not expand each day into repeated Card/Text element trees, and do not create top/bottom image galleries or standalone image stacks.
- Road-trip/navigation itineraries MUST also stay compact as one `Table` with `domain: "schedule"` and `preferredPresentation: "cards"`. Use row fields such as `day`, `route`, `drivingTime`, `scenicStop`, `shortHike`, `overnightStay`, optional `icon`, and optional verified `image`/`imageAlt`. Do not expand each day into repeated card templates and do not create a trailing image gallery.
- Single-day timed itineraries may use compact columns like `time`, `activity`, and `details`; set `domain: "schedule"`, `preferredPresentation: "cards"`, `primaryColumn: "time"`, and `highlightColumns: ["activity"]`.
- Exam prep/study-plan responses MUST stay compact as one or two `Table` elements with `domain: "schedule"` and `preferredPresentation: "cards"`: a weekly plan table (`week`, `dates`, `focus`, `keySessions`, `goal`) and optional resource/action rows. Do not create study image galleries, decorative resource photos, or repeated per-week Card/Text trees.
- Science/concept explanations should render as title + short analogy/context Card + compact `Table`/cards for concept types/examples. Use `domain: "generic"` or `"comparison"` as appropriate. Only emit `Image` when the response contains verified educational diagram media directly tied to that card/row; never create decorative photo galleries from random images.
- Sources with URLs must be rendered as compact source/action rows or Buttons with `openUrl`; do not turn URL-backed sources into plain non-clickable Text. If the source line has no URL, keep it as short caption text only.
- For search-provider rows such as Google Flights, Skyscanner, booking sites, maps, or source links, keep the URL as a hidden/action field only (`url`, `bookingUrl`, or `actionUrl`) and add `actionLabel`; never render the raw URL as visible body text.
- Planning/project roadmap responses MUST stay compact as one `Table` with `domain: "schedule"`, `preferredPresentation: "cards"`, `primaryColumn` set to the phase/month column, and `highlightColumns` for goal/deliverable. Do not use Tabs for sequential months/phases, do not expand each phase into separate duplicated element trees, and do not add decorative image galleries.
- Comparison tables:
  - Feature matrices should use first column `Feature` or `Metric`, `domain: "comparison"`, `preferredPresentation: "table"`. If verified/local images are available for compared entities, attach them in `props.entityMedia`; do not create trailing media cards.
  - Exact product/device feature matrices should be table-first and compact: one title/context card, one `Table`, and nearby source/action buttons. Do not create `Images`, `Referenced Icons`, gallery, or loose decorative icon sections from detached media.
  - Entity comparisons should use first column `Item`, `Product`, `Option`, `Model`, or equivalent, `domain: "comparison"`, `preferredPresentation: "cards"`.
- Defaults:
  - weather/flight/booking/playlist/schedule/status => `preferredPresentation: "cards"`
  - formula => `preferredPresentation: "table"` for variables/breakdowns plus one `Formula` element for the equation
  - comparison => `preferredPresentation: "cards"` for entity rows and `"table"` for feature matrices
  - generic => `preferredPresentation: "table"`
- If table parsing is partial, keep compact `Table` with best-effort columns/rows; do not degrade to prose-only output.

## Watch bindings
- You may use element-level `watch` to react to state changes:
  - `"watch": { "/form/submit": { "action": "validateForm", "params": { "statePath": "/formValidation" } } }`

## Layout rules
- Build structured app-like UI, not one giant text block.
- Use headings and sections for medium/long responses.
- For on-device models, include enough top-level sections to match the response
  structure: title, context/assumptions, primary comparison/table, recommendation
  or result, sources/actions when present. Do not omit the recommendation or
  source/action section just to keep the JSON short.
- Keep title, media, body, and CTA together inside each card.
- If the response contains an email/message draft with `Subject:` plus greeting/signature, the actual draft MUST be one `EmailPreview` element. Do not expand it into multiple generic Text/Card elements.
- Use `Stack` for flex layout and positioning intent (direction, align, justify, gap, spacing, size).
- Use only flex-style positioning props; absolute positioning is unsupported.
- Convert links/CTAs to `Button` with `openUrl`.
- Keep `Tags: A | B | C` lines: emit chips via `Text` with `variant: "chip"`.
- For hotel/restaurant/place result sets, prefer one compact `Table`/repeated card data model; keep action URLs and labels in the matching row so the renderer can place the CTA inside the related card.
- For playlist/music tracklists, keep tracks in one compact `Table` (`domain: "playlist"`) and do not expand each track into separate elements.
- For technical-support troubleshooting guides, prefer one compact diagnostic `Table` with `domain: "status"`, `preferredPresentation: "cards"`, `primaryColumn: "step"`, and `highlightColumns` for action/expected result. Do not create detached galleries; use inline icons only when helpful.
- For UI process/state-machine flows such as scanner, check-in, success/error, or approval state transitions, keep states as one compact `Table` with `domain: "status"`, `preferredPresentation: "cards"`, and columns such as `state`, `visuals`, and `feedback`; Android renders this as a native process template, so do not create trailing image/icon cards.
- For incident/system-health/status-page responses, keep affected services as one compact `Table` with `domain: "status"`, `preferredPresentation: "cards"`, and columns such as `component`, `status`, and `notes`/`impact`; Android renders this as an incident dashboard, so do not create decorative image galleries.
- If a technical-support checklist/table is present, preserve it as a `Table`; do not expand each diagnostic step into duplicated Card/Text element trees.
- If a technical-support response uses Markdown headings like `Step 1`, `Step 2`, etc. instead of a pipe table, convert those step sections into one compact `Table` backed by `state.diagnosticSteps`.
- For email/message-writing responses, use one compact `EmailPreview` element for the actual draft. Put `subject`, `to`, `from`, `date`, `role`, `company`, `body` paragraph array, and `signature` lines in props. Do not render email drafts as oversized generic Text paragraphs or detached Quick Actions cards.
- For product-description or marketing-copy responses, render a compact product landing screen: hero Card with inline Icon, 1-2 short body Text elements, feature/benefit cards, and only real URL-backed CTAs. Do not create standalone image galleries or use placeholder/random-host images.
- For programming/tutorial/debugging responses, separate source code from runtime output: use compact `CodeBlock` elements for functions/snippets and `ConsoleLog` elements for command lines, REPL transcripts, stack traces, and expected output. Do not render code or logs as oversized heading Text, and do not duplicate the same snippet as prose.
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
- `Formula` props: `latex` or `text` (required), optional `title`, `subtitle`, `result`, `display`. Use LaTeX-style notation for fractions, exponents, roots, and variables, for example `M = P \\frac{i(1+i)^n}{(1+i)^n - 1}`. Renderer formats fractions/exponents; do not use images for formulas.
- `CodeBlock` props: `code` (required), optional `language` (`python|javascript|kotlin|bash|text`) and `title`. Use for source code only; do not put expected output inside the same CodeBlock unless it is part of the source comment.
- `ConsoleLog` props: `code` (required), optional `language: "console"` and `title`. Use for terminal commands, REPL transcripts, stack traces, and expected output.
- `EmailPreview` props: `title`, `subtitle`, `subject`, `to`, `from`, `date`, `role`, `company`, `body` (string or paragraph array), `signature` (string or line array), optional `context`/`metadata`. Use for professional emails, drafts, messages, cover letters, and similar communication templates.
- `Table` props:
  - `columns` (required list of `{ "key": "...", "label": "..." }`)
  - `statePath` (preferred, pointer to row array in state) OR `rows` (inline row array)
  - `domain` optional (`weather|flight|booking|playlist|schedule|status|formula|comparison|generic`)
  - `preferredPresentation` optional (`cards|table`)
  - `primaryColumn` optional (key/label for portrait card title)
  - `highlightColumns` optional (list/string of 1-2 important key/label values)
  - `numericColumns` optional (list/string of numeric/currency/score/unit columns)
  - `entityMedia` optional for comparison/feature-matrix tables: map entity column key/label to `{ "image": "../assets/...", "alt": "..." }`; renderer shows these inside entity cards
  - `title`, `subtitle`, `mood`, `genre` optional for playlist/music table hero metadata
  - `sourceFormat` optional (`markdown|csv|tsv|html|plain`)
  - `sourceText` optional raw table text
- `Chart` props: `chartType` optional (`bar`), `columns`, `statePath` OR `rows`, `xKey`, `yKey`, optional `title`, `subtitle`, `yLabel`. Use this for generated charts instead of image placeholders.
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
