# A2UI Express v1 generated model contract

<!-- Generated from pinned grammar/catalog/profile/quality policy: grammar=37044656eb10e4c4a4a54c822432f327c5340761b5bfe914b58fa0db3581cf42 catalog=668082a49664d33c154ac62c8a58db24b487df683d1c17112d74c9b3bd98a9d1 profile=18d5675c4c95d227cd4148a68caff35f166b61d3a8caa7fb76d30f6dd40fd786 quality=d75f6f5f26226cec9b6774ea70a91fdf80fae5d590d1373cb5906ee6227e5971 -->

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

## Pinned catalog signatures

- Alert(message, title, tone, timestamp)
- AudioPlayer(url, description, posterUrl, title)
- Button(label, variant, icon)
- Card(children, title, subtitle, tone)
- Chart(chartType, columns, statePath, rows, title, subtitle)
- CheckBox(label, value, statePath)
- Checklist(items, title, disclaimer, source)
- ChoicePicker(label, options, value, statePath)
- CodeBlock(code, language, title)
- ConsoleLog(code, language, title)
- DateTimeInput(label, value, mode, placeholder, statePath)
- Divider()
- EmailPreview(subject, body, from, to, date, title)
- Formula(latex, title, result, display)
- Icon(name, size, tint)
- Image(url, alt, fit, width, height)
- List(children, items)
- Modal(trigger, content, title)
- Row(children, gap, align, justify, wrap)
- Slider(label, value, min, max, step, statePath)
- Stack(children, direction, gap, align, justify, wrap)
- Table(columns, statePath, rows, title, domain, preferredPresentation)
- Tabs(tabs, activeTabId)
- Text(text, variant)
- TextField(label, value, statePath, placeholder)
- Video(url, posterUrl, description, title)

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
