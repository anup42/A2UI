- Preserve every requested fact, heading, section, table row, action, binding,
  repeat, visibility rule, watch, and verified media reference.
- Keep meaningful component richness and hierarchy; token savings must come
  from Express syntax and default elision, never from dropping UI semantics.
- Chart types: bar, column, line, area, groupedbar, stackedbar, stackedarea,
  scatter, pie, donut, combo, radar, bubble, funnel, treemap (flat weights), box.
  Bind xKey and all intended measures. `series` is an array of objects with
  yKey, label, optional unit, axis (left/right), and type (required for combo:
  column/bar/line/area/scatter). Explicit series override yKey. Without series,
  grouped/stacked/line/area/radar use numeric non-X columns; single-value chart
  types use yKey or the second column. Combo requires explicit series; dual axes
  require both left and right assignments. Use xType category (source order),
  number (numeric spacing), or time (ISO date/time spacing); no implicit sorting.
  Scatter/bubble require numeric X. Bubble requires sizeKey. Box requires
  boxKeys with min/q1/median/q3/max column keys for supplied statistics.
  Bar defaults horizontal; column/groupedbar/stackedbar default vertical and
  accept orientation horizontal/vertical. Pie/donut/funnel/treemap require one
  nonnegative series and positive total. Radar requires at least three
  comparable nonnegative categories. Stacks require complete values; other
  Cartesian nulls remain gaps. Preserve exact units and qualifiers; never invent
  data or coerce an unsupported request into a different chart. Limit one chart
  to 512 rows and 16 series; split larger supplied data into labeled charts.
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
