# v11 chart support and revalidation — 2026-10-01

Native Chart now supports all 16 chart families represented by the v11 subtype
inventory. Both the Android app and GenUICraft SDK implement the drawing paths;
the schema, Express catalogs, generated prompts, Python evidence and training
preparation use the same contract. See the
[complete chart contract](../../../dataset/docs/chart_contract_v2_1.md).

The original issue was a runtime mismatch: only bar/column were admitted, and
the renderer extracted one Y column into horizontal bars. That lost grouped or
stacked series and could not represent lines, numeric scatter spacing, pies or
separate axes. The implementation now preserves signed values, null gaps,
numeric/time spacing and explicit series/axis bindings. Every supplied data
column remains readable beneath the plot, including unplotted shares/caveats.

## Effect on existing data

The read-only scan covered all 111,350 v11s rows, including 1,440 rows with Chart.
The same new-data suffix is present in v11. Raw training/validation SHA-256 values
still match the supplied release manifests; no source or target was rewritten.

| Result | Rows |
|---|---:|
| Originally held for unsupported chart subtype | 1,183 |
| Now drawable with the supplied bindings | 1,062 |
| Newly clear of both chart and reviewed text holds | **1,059** |
| Drawable but still held for an unsafe text join | 3 |
| Originally held chart rows still needing semantic repair | 121 |
| Additional chart rows caught by stricter data validation | 6 |
| All chart rows with incomplete/invalid data | 127 |
| Original text-boundary holds, unchanged | 59 |
| Distinct remaining held rows (one overlap) | **185** |

The 1,059 newly admitted rows comprise 1,037 train and 22 validation examples.
The original 1,238 distinct holds therefore become 185 after these checks:
1,238 − 1,059 + 6. This is preparation eligibility under these review gates;
other source-fidelity, tokenizer, split and training checks still apply.

Remaining component issues include 88 combo charts without explicit series,
23 of those lacking dual-axis assignments, 25 invalid numeric X bindings,
14 invalid scalar values, two box plots without statistic bindings and two
bubble charts without size bindings. Counts overlap; the JSON records every
affected component and row. These are supported chart types with insufficient
or invalid supplied data, not a reason to exclude correctly specified charts.

Missing bindings and unsafe joined text need regeneration through Stage 3 into
a new reviewed release. The renderer does not invent measurements, quartiles,
axis assignments or source separators. The original release and frozen trained
model prompt remain intact.

## Evidence and reproduction

- [Row-level revalidation](revalidation_release.json) includes unchanged target
  hashes, original split hashes, preparation results and remaining diagnostics.
- [Validation record](validation.json) records the tested implementation and
  suite results. Shared fixtures cover all families and invalid-input boundaries.
- [Native plot overview](native_chart_overview.png) shows production Canvas
  rendering through Robolectric native graphics. The full Compose card also
  adds titles, legends, axis/unit labels, accessibility text and the data key.

```powershell
python training/scripts/revalidate_v11_charts.py --dataset-dir <original-or-restored-v11s> --output-report <new-output.json>
python -m pytest dataset/tests/test_rich_chart_contract.py training/tests/test_v11_release_review.py -q
```

The audit refuses to overwrite a report or write inside the dataset directory,
and refuses a report if its chart implementation changes while it runs. New
generation uses the refreshed prompt; saved runs with an older fingerprint must
use a fresh output directory. An outdated app table-routing test was aligned
with the existing explicit-table behavior and its SDK counterpart; table
runtime behavior was not changed.

Validation here covers source compilation, local JVM/native drawing and data
admission. No physical-device install, model retraining, proprietary Bixby build,
new model-driven sample generation or AAR publication was performed.
