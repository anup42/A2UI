# genui_gen_ondevice_v2_flatspec_compact

Convert the response text into one compact mobile flat-spec JSON object.

Response:
{response_text}

## Priority rules
- Return ONLY JSON. No prose, markdown, comments, or code fences.
- Preserve important facts exactly: names, dates, times, prices, units, percentages, ranks, URLs, labels, and table cells.
- Build a mobile app screen, not a document: title, concise context/result card, structured sections, compact tables/cards, nearby actions/sources.
- Keep IR compact but complete. Medium responses usually need 12-35 elements. Do not collapse to only title + one table when sections/actions/sources exist.
- Preserve response headings as `Text` with `variant: h2|h3`, except detached media-only headings such as Images, Icons, Gallery, Visual Guide, Related Icons.
- Do not emit raw markdown control tokens in text: no `#`, `**bold**`, table pipes for table rows, fenced markers, or literal dynamic strings like `{$item.title}`.

## Output contract
Top-level shape: `{"root":"<id>","state":{...},"elements":{...}}`.
- `root` must exist in `elements`.
- Every element: `type`, `props`, `children`.
- Every child id must exist.
- Use only the catalog below.
- Use element-level `on.press` for actions. Supported actions: `openUrl`, `setState`, `pushState`, `removeState`, `validateForm`.
- Do not output legacy message arrays, `className`, `props.action`, or `functionCall`.

## Catalog
Layout: `Stack`, `Card`, `List`, `Divider`.
- `Stack.props`: `direction` vertical|horizontal, `gap` none|sm|md|lg|xl, optional `align`, `justify`, `wrap`, `padding`, `paddingHorizontal`, `paddingVertical`, `margin`, `width`, `height`, `flex`.
Content: `Text`, `Table`, `Chart`, `Formula`, `Image`, `Icon`, `CodeBlock`, `ConsoleLog`, `EmailPreview`, `Video`, `AudioPlayer`.
- `Text.props`: `text`, optional `variant` h1|h2|h3|body|caption|chip|label.
- `Table.props`: `columns`, `statePath` or `rows`, optional `domain`, `preferredPresentation`, `primaryColumn`, `highlightColumns`, `numericColumns`, `entityMedia`, `title`, `subtitle`, `mood`, `genre`, `sourceFormat`, `sourceText`.
- `Chart.props`: `chartType`, `columns`, `statePath` or `rows`, `xKey`, `yKey`, optional title/subtitle/yLabel.
- `Formula.props`: `latex` or `text`, optional title/subtitle/result/display.
- `Image.props`: `url`, optional `fit` cover|contain. `Icon.props`: `name`.
- `CodeBlock.props`: `code`, optional language/title. `ConsoleLog.props`: `code`, optional language/title.
- `EmailPreview.props`: title, subtitle, subject, to, from, date, role, company, body, signature, context, metadata.
Interactive/form: `Button`, `Tabs`, `Modal`, `TextField`, `CheckBox`, `ChoicePicker`, `Slider`, `DateTimeInput`.
- `Button.props`: `label`, optional `variant` primary|borderless.
- `Tabs.props.tabs`: list of `{ "title": "...", "child": "elementId" }`.

## Tables and repeated data
- When the response has comparative, tabular, option-list, ranking, itinerary, recipe ingredient, status, or numeric breakdown data, emit one compact `Table`; do not expand rows/cells into element trees.
- Preferred table: rows in `state`, `props.statePath` points to them, `props.columns` preserves source order and all source columns.
- Use generic column keys derived from labels, not weather-only names.
- Do not include both full rows and `sourceText` unless rows cannot preserve the table. Prefer rows.
- Optional metadata: `domain` weather|flight|booking|playlist|schedule|status|formula|comparison|generic; `preferredPresentation` cards|table; `primaryColumn`; `highlightColumns`; `numericColumns`.
- Renderer adapts portrait/landscape. Do not create separate portrait and landscape IR.

## Domain routing
- Weather/climate: title + compact summary/current card + forecast/metrics `Table` with `domain: weather`, `preferredPresentation: cards`. Preserve temp, rain, wind, humidity, recommendation.
- Flight: one compact `Table` with `domain: flight`, `preferredPresentation: cards`; fields like rank, airline, route/legs, depart/arrive, duration, stops, cost, reason, actionUrl/actionLabel.
- Booking/hotel/restaurant/place: compact `Table` with `domain: booking`, `preferredPresentation: cards`; keep row-level image only if verified and directly tied to that row; keep booking/action URL and label in the same row.
- Playlist/music: one compact `Table` with `domain: playlist`, `preferredPresentation: cards`; fields trackNumber, title, artist, duration, mood/genre; optional table title/subtitle/mood/genre.
- Schedule/itinerary/planning/study/event: one compact `Table` with `domain: schedule`, `preferredPresentation: cards`; primary column is day/time/month/phase; attach verified row images only to matching rows.
- Status/support/process: one compact `Table` with `domain: status`, `preferredPresentation: cards`; fields step/state/component/status/action/expected.
- Comparison: feature matrices use first column Feature/Metric, `domain: comparison`, `preferredPresentation: table`; entity comparisons use Item/Product/Option first column and `preferredPresentation: cards`.
- Formula/calculation: show a `Formula`, a variables `Table`, and a breakdown `Table`; preserve equation and values exactly.
- Data visualization: use `Chart` for explicit chart requests when rows have numeric x/y data; also keep the same data as compact rows/table when useful.
- Recipe: title + notes/hero card + Tabs for Ingredients/Instructions when appropriate; ingredients as Table, instructions as step Table/cards. Never emit unresolved `$item` strings.
- Email/message/cover letter: use one `EmailPreview` for the draft, not generic paragraphs.
- Code/tutorial/debugging: use `CodeBlock` for source code and `ConsoleLog` for terminal output, stack traces, commands, REPL logs.
- Product/marketing copy: hero card + concise benefit/feature table/cards + URL-backed CTAs only.

## Media and URLs
- If Assets mapping is provided, use only local paths from that mapping for media. Do not emit remote media URLs for images/icons.
- If an asset's original URL contains random/placeholder hosts such as `loremflickr.com`, `picsum.photos`, `placehold.co`, `placeholder`, or `.example`, do not use that local path in the IR.
- If no verified image exists, omit the image; renderer can use native placeholders. Prefer no image over bad/random media.
- Never invent local paths like `/image.jpg` or `/asset/foo.png`.
- SVG/bootstrap/weather icon URLs are `Icon` only, never `Image` or table photo fields.
- Do not create trailing galleries/icon lists from detached media sections. Attach verified media to the relevant hero/card/table row, or drop it.
- URLs must be actions/sources near related content. Use `Button` with `openUrl`, or row fields `url`, `actionUrl`, `bookingUrl` plus `actionLabel`. Do not show raw URLs as body text.
- Allow only public `https://` URLs for actions. Reject `http`, `javascript`, `data`, `file`, `content`, localhost, private IPs, `.local`, `.test`, `.example`.

## Dynamic fields
Use expression objects only: `{ "$item": "field" }`, `{ "$state": "/path" }`, `{ "$bindState": "/path" }`, `{ "$bindItem": "field" }`, `{ "$index": true }`, `{ "$cond": ..., "$then": ..., "$else": ... }`, `{ "$template": "Step ${index_1}" }`, `{ "$computed": "name", "args": {...} }`.
`repeat` and `visible` are top-level element fields, not props.

## Final checklist before output
- Valid JSON object with `root/state/elements` only.
- All ids resolve; no dangling children.
- Tables preserve all source columns/rows.
- Headings, recommendations, actions, and sources are not dropped.
- No raw markdown, raw URLs-as-text, detached media cards, fake media paths, or placeholder image assets.
