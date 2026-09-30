# Native Chart contract v2.1

`Chart` now draws the requested chart instead of converting every subtype into
a single horizontal bar series. The Android app and GenUICraft SDK share the
same implementation and the Python training/generation gate models that data
contract. `renderer_capabilities.json` is version 2.1.0; effective semantics are
5.4.3 and training preparation is 2.2.0, invalidating older prepared caches.

| chartType | Rendering and required data |
|---|---|
| bar, column | One Y measure; horizontal/vertical by default, respectively |
| groupedbar | Separate bars for each series, with one shared scale |
| stackedbar | Signed stacks with separate positive/negative totals |
| line, area | Multiple series, optional right axis, gaps for missing values |
| stackedarea | Multiple complete series stacked above/below zero |
| scatter | Numeric X spacing and one or more Y series |
| pie, donut | One nonnegative measure, positive total; category color key |
| combo | Explicit series with column/bar/line/area/scatter marks and left/right axes |
| radar | At least three comparable, nonnegative category values per series |
| bubble | Numeric X/Y and an explicit nonnegative sizeKey; circle area encodes size |
| funnel | One nonnegative measure; each stage's top width encodes its value |
| treemap | Flat nonnegative weights; rectangle area encodes weight |
| box | Supplied min/q1/median/q3/max statistics bound through boxKeys |

Common properties retain their existing positions: `Chart(chartType, columns,
statePath, rows, title, subtitle)`. Rows can also come from `rowsPath`, `dataPath`,
or `data`. Explicit `xKey`/`yKey` must resolve; a misspelled key never silently
selects another column. The visible scrollable data key retains all supplied
columns, including values or caveats that are not plotted.

`series` is an array of `{yKey, label?, unit?, axis?, type?}`. It overrides yKey.
Without it, line/area/grouped/stacked/radar charts use every numeric non-X column;
single-measure types use yKey or the second column. Combo requires series and a
mark type on every series. Dual-axis aliases also require both axis assignments.
Units label the display; they never convert or manufacture values.

`xType` defaults to category (numeric for scatter/bubble). Number and ISO time
preserve spacing; time axes use UTC labels with intraday precision. Rows and connecting segments retain source order, with no
implicit sort or aggregation. Horizontal bars use categorical positions.
Zero has zero length; negative bars extend below/left of zero. Missing Cartesian
values remain gaps, while stacked/weight/radar/box charts require complete data.

`boxKeys` maps the five named statistics to columns. The renderer validates their
ordering rather than calculating unspecified quartiles. Bubble size needs its
own column. Unsupported types, invalid values, incomplete bindings, more than
512 rows or more than 16 series produce a diagnostic placeholder and fail the
effective chart contract; they do not become bar charts. Pie/donut/funnel/treemap
do not accept negative weights. Treemap does not infer a hierarchy.

Example using supplied measurements:

```text
<a2ui>
root=Chart("combo",["week","hours","quiz"],rows=[["W1",8,78],["W2",10,90]],xKey="week",title="Study progress",series=[{yKey:"hours",label:"Study",type:"column",axis:"left",unit:"h"},{yKey:"quiz",label:"Quiz",type:"line",axis:"right",unit:"%"}])
</a2ui>
```

Generation uses the refreshed Express contract and shared Stage 3 fidelity
guidance. Do not resume a saved generation run under a different fingerprint.
The deployed model's frozen prompt and weights are not rewritten by a renderer
update; a future model release needs training with the new contract.

Existing v11/v11s remain byte-identical. Preparation re-evaluates their actual
resolved chart data, so compatible targets become eligible without relabeling.
Missing semantics and the 59 reviewed text joins need Stage 3 regeneration into
a new reviewed release. A chart-contract pass alone does not establish source
factuality, complete source fidelity, split safety, or tokenizer eligibility.

Validation uses one shared JSON fixture corpus in Python and both Kotlin
modules, including gaps, signed stacks, unequal numeric/time intervals, dual
scales, invalid bindings and unsupported values. Native Robolectric bitmap tests
exercise the production Canvas drawing functions for all 16 families. These are
local render tests, not a claim of physical-device or deployed-model validation.
