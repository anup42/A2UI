- Preserve every requested fact, heading, section, table row, action, binding,
  repeat, visibility rule, watch, and verified media reference.
- Keep meaningful component richness and hierarchy; token savings must come
  from Express syntax and default elision, never from dropping UI semantics.
- Treat bullets as source formatting, not a requirement to emit `List`.
  When list items describe entities with multiple fields (such as hotels,
  flights, products, or events), extract the supplied fields into one record
  per entity. Keep its separately readable fields together under that entity's
  `Card` title, in source order, rather than one undifferentiated Text/List item.
  Preserve compact `Table` records with `rows` or `statePath` and
  `preferredPresentation="cards"` where the domain's native card route uses
  tables; do not expand those compact records into duplicate component trees.
  Merely wrapping the unchanged bullet in a Card is not field extraction.
- Keep each value, qualifier, unit, date, action, and source/media reference
  attached to its original entity and section. Preserve all supplied fields,
  including details that do not fit common columns; do not swap values across
  records, infer missing values, invent facts or unsupported field meanings,
  duplicate the original bullet beside its extracted fields, or add empty
  cards to increase component counts. Source-grounded column labels may name
  fields; they must not imply information absent from the response.
- Plain steps, checklists, and unstructured text items remain `List`; do not
  force them into cards. Feature-by-entity comparison matrices remain compact
  `Table` data with `preferredPresentation="table"`.

  Field extraction example (illustrative facts only; never copy them):
  Source:
  Hotels:
  - Hotel: Cedar; City: Pune; Price: INR 4200/night; Rating: 4.6/5
  - Hotel: Elm; City: Kochi; Price: INR 5300/night; Rating: 4.8/5

  <a2ui>
  root=Column([heading,cedar,elm])
  heading=Text("Hotels","h2")
  cedar=Card([c1,c2,c3],title="Hotel: Cedar")
  c1=Text("City: Pune")
  c2=Text("Price: INR 4200/night")
  c3=Text("Rating: 4.6/5")
  elm=Card([e1,e2,e3],title="Hotel: Elm")
  e1=Text("City: Kochi")
  e2=Text("Price: INR 5300/night")
  e3=Text("Rating: 4.8/5")
  </a2ui>

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
