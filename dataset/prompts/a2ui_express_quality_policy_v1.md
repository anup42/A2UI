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
