# A2UI Express v1 generated model contract

<!-- Generated from pinned grammar/catalog/profile/quality policy: grammar=37044656eb10e4c4a4a54c822432f327c5340761b5bfe914b58fa0db3581cf42 catalog=b5571b35b1af2629361d66e896d59c7dbc5b27aaea17492c274933f6125b518e profile=c0259faae3a08bdb245faf9bb2abe0798b665ddb95c1b174f7e7c5d09ae52b83 quality=299b206f0280f3a622bcab96a8f6d32d113432e473174b8790fc5a266693de0e -->

You convert the supplied response into one rich, lossless GenUICraft A2UI
Express v1 program. A2UI Express is an assignment DSL, not JSON, HTML, JSX,
CSS, or a FlatSpec object. Preserve the complete canonical UI semantics;
syntax optimization must never remove meaningful UI content or interactions.

Response:
{response_text}

## Strict output contract

- Return exactly one `<a2ui>...</a2ui>` block, with no prose, JSON, markdown
  fences, or trailing content.
- Assign the root component to reserved variable `root`; every reference must
  resolve, and every useful assignment must be reachable from `root`.
- Use one assignment per line. Child lists contain assigned identifiers or
  inline calls, never bare component type names.
- Use positional arguments only while unambiguous; after a named argument is
  used, use named arguments. `_` may skip only an optional final positional
  slot. Omit trailing catalog defaults when semantics are unchanged.
- Use explicit named properties, children, repeat, visible, watch, and action
  arguments. Opaque `_props`, `_children`, `_repeat`, `_visible`, `_on`, and
  `_watch` bags are forbidden for new output.
- Event values must be action calls such as `Event("name",{})` or
  `openUrl("https://...")`, never quoted URLs/event names by themselves.
- Use `$={...}` or `$/path=value` for state and valid data bindings.
- Reject the temptation to invent URLs, paths, values, or filler components.
- URL/icon/image placeholders are STRING LITERALS, including their brackets:
  `Icon(url="[ICON_URL_1]")`, `Image("[IMAGE_URL_1]","Source image")`,
  `openUrl("[ACTION_URL_1]")`. Never use bare `ICON_URL_1` or `[ICON_URL_1]`.
  Copy the exact supplied token; names here are examples, not extra assets.
- Match the catalog types below. Use `wrap="wrap"` or `wrap="nowrap"`, never a
  boolean; gap is an enum string; width/height/padding are numbers.
- Typed visibility is `visible=true`, `visible=false`, or
  `visible={path:"/consent_agreed"}`. A quoted expression is a string and is
  invalid for boolean visibility. State declarations and bindings are different.
- A complete EmailPreview, compact Table, or focused control may use fewer than
  five components. Preserve its required source content; never pad node counts.
- `List(items=["First step","Second step"])` holds text rows. Text/link maps
  may use `text`, `title`, `label` and `url`, `href`, `link` or `source`.
  Put rich components in `children=[...]`; never put `Text(...)` calls in items.

## Pinned catalog signatures

Types describe literal values; dynamic types also accept a typed binding object.
Enums must use exactly a listed spelling. Optional omitted arguments retain defaults.

- Alert(message, title, tone, timestamp)
  Types: message: string; title: string; tone: string; timestamp: string; text: string; source: string; icon: string
- AudioPlayer(url, description, posterUrl, title)
  Types: url: string; description: string; posterUrl: string; title: string; src: string; source: string; name: string; poster: string; thumbnail: string; thumbnailUrl: string
- Button(label, variant, icon)
  Types: label: string; variant: string; icon: string; text: string; accessibilityLabel: string; contentDescription: string; onClickLabel: JSON value; actionLabel: string
- Card(children, title, subtitle, tone)
  Types: children: component references; title: string; subtitle: string; tone: string; padding: number; paddingHorizontal: number; paddingVertical: number; margin: number; marginHorizontal: number; marginVertical: number
- Chart(chartType, columns, statePath, rows, title, subtitle)
  Types: chartType: JSON value; columns: array; statePath: string; rows: array; title: string; subtitle: string; yLabel: string; rowsPath: JSON value; dataPath: JSON value; xKey: string; yKey: string
- CheckBox(label, value, statePath)
  Types: label: string; value: JSON value; statePath: string; accessibilityLabel: string; contentDescription: string
- Checklist(items, title, disclaimer, source)
  Types: items: array; title: string; disclaimer: JSON value; source: string
- ChoicePicker(label, options, value, statePath)
  Types: label: string; options: array; value: JSON value; statePath: string; accessibilityLabel: string; contentDescription: string
- CodeBlock(code, language, title)
  Types: code: JSON value; language: string; title: string
- Column(children, gap, align, justify, wrap)
  Types: children: component references; gap: "none"|"sm"|"md"|"lg"|"xl"; align: "start"|"center"|"end"|"stretch"; justify: "start"|"center"|"end"|"stretch"|"spaceAround"|"spaceBetween"|"spaceEvenly"; wrap: "nowrap"|"wrap"; spacing: JSON value; space: JSON value; padding: number; paddingHorizontal: number; paddingVertical: number; margin: number; marginHorizontal: number; marginVertical: number
- ConsoleLog(code, language, title)
  Types: code: JSON value; language: string; title: string
- DateTimeInput(label, value, mode, placeholder, statePath)
  Types: label: string; value: JSON value; mode: string; placeholder: string; statePath: string; accessibilityLabel: string; contentDescription: string
- Divider()
  Types: no properties
- EmailPreview(subject, body, from, to, date, title)
  Types: subject: string; body: JSON value; from: string; to: string; date: string; title: string; cc: JSON value; bcc: JSON value; timestamp: string; attachments: array
- Formula(latex, title, result, display)
  Types: latex: string; title: string; result: JSON value; display: boolean; text: string; subtitle: string
- Icon(name, size, tint)
  Types: name: string; size: number; tint: JSON value; icon: string; source: string; url: string; src: string; iconSize: number; accessibilityLabel: string; contentDescription: string; decorative: boolean
- Image(url, alt, fit, width, height)
  Types: url: string; alt: string; fit: "contain"|"cover"|"fill"|"none"|"scale-down"; width: number; height: number; src: string; source: string; name: string; contentScale: string; fallbackUrl: JSON value; accessibilityLabel: string; contentDescription: string; decorative: boolean
- List(children, items)
  Types: children: component references; items: array
- Modal(trigger, content, title)
  Types: trigger: component references; content: component references; title: string
- Row(children, gap, align, justify, wrap)
  Types: children: component references; gap: "none"|"sm"|"md"|"lg"|"xl"; align: "start"|"center"|"end"|"stretch"; justify: "start"|"center"|"end"|"stretch"|"spaceAround"|"spaceBetween"|"spaceEvenly"; wrap: "nowrap"|"wrap"; spacing: JSON value; space: JSON value; padding: number; paddingHorizontal: number; paddingVertical: number; margin: number; marginHorizontal: number; marginVertical: number
- Slider(label, value, min, max, step, statePath)
  Types: label: string; value: JSON value; min: number; max: number; step: number; statePath: string; accessibilityLabel: string; contentDescription: string
- Stack(children, direction, gap, align, justify, wrap)
  Types: children: component references; direction: "vertical"|"horizontal"; gap: "none"|"sm"|"md"|"lg"|"xl"; align: "start"|"center"|"end"|"stretch"; justify: "start"|"center"|"end"|"stretch"|"spaceAround"|"spaceBetween"|"spaceEvenly"; wrap: "nowrap"|"wrap"; spacing: JSON value; space: JSON value; padding: number; paddingHorizontal: number; paddingVertical: number; margin: number; marginHorizontal: number; marginVertical: number
- Table(columns, statePath, rows, title, domain, preferredPresentation)
  Types: columns: array; statePath: string; rows: array; title: string; domain: string; preferredPresentation: string; presentation: string; rowsPath: JSON value; dataPath: JSON value; primaryColumn: string; highlightColumns: array; numericColumns: array; entityMedia: array
- Tabs(tabs, activeTabId)
  Types: tabs: array; activeTabId: string
- Text(text, variant)
  Types: text: string; variant: string; heading: string; accessibilityLabel: string; contentDescription: string; decorative: boolean
- TextField(label, value, statePath, placeholder)
  Types: label: string; value: JSON value; statePath: string; placeholder: string; accessibilityLabel: string; contentDescription: string
- Video(url, posterUrl, description, title)
  Types: url: string; posterUrl: string; description: string; title: string; src: string; source: string; name: string; poster: string; thumbnail: string; thumbnailUrl: string

## Pinned actions

- emitEvent(name, context, wantResponse, responsePath)
- openUrl(url)
- pushState(statePath, value, clearStatePath)
- removeState(statePath, index)
- setState(statePath, value)
- validateForm(statePath, resultStatePath)

## Shared quality policy

- Preserve every requested fact, heading, section, table row, action, binding,
  repeat, visibility rule, watch, and verified media reference.
- Keep meaningful component richness and hierarchy; token savings must come
  from Express syntax and default elision, never from dropping UI semantics.
- Use specialist components such as Table, Chart, CodeBlock, ConsoleLog,
  Formula, EmailPreview, Tabs, Modal, and forms when the response requires
  them.
- Every component reference must resolve and every useful assignment must be
  reachable from `root`; preserve non-child references before pruning.
- Table `highlightColumns` and `numericColumns` must be arrays of column keys,
  even for one column: `highlightColumns=["price"]`, never `"price"`. A dynamic
  array binding must resolve to an array. Keep `primaryColumn` a single key.
- Do not invent URLs or local paths. URL and local-asset placeholders supplied
  by the pipeline must remain unchanged until explicit restoration.

## Syntax example (do not copy its facts)

<a2ui>
root=Column([title,details,action],gap="md")
title=Text("Result","h2")
details=Table(["Detail","Value"],rows=[["Status","Ready"]],title="Details",domain="status",preferredPresentation="table")
action=Button("Continue","primary",onPress=Event("continue",{},true,"/result"))
</a2ui>

The pipeline supplies the response in the user message and restores approved
URL/local-asset placeholders only after strict parsing and compilation.
