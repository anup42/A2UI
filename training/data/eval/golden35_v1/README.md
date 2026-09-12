# Golden35 v1

This is the explicitly approved **35-case, 35-unique-source** strict-valid subset of the historical 50-case reference run. It is named Golden35 throughout active training/evaluation use. The original 50-case source files remain untouched for provenance; they are not the active evaluation cohort.

Membership is frozen by source/query identity, not by taking the next 35 rows that pass a future parser. The approved IDs are stored in `benchmark_manifest.json` and in `ir_training.data.golden35_subset.APPROVED_SOURCE_IDS`. The manifest binds the original GenUI and response files, frozen output bytes, every accepted source/target identity, and all 15 excluded sources and reasons. Excluded responses remain reserved from training using both original and URL-placeholder-normalized hashes.

All 35 accepted reference targets pass native Express, production wire-schema, root-reachability and semantic round-trip checks. No excluded target was manually edited or replaced. Category counts are recorded in the manifest and should not be mistaken for the original cohort's coverage. Report metrics with denominator 35 and benchmark identity `golden35_v1`; old 50-case aggregates are not directly comparable.

Normal GPU-host preparation should consume this frozen artifact through the active Golden35 config. To reproduce its materialization from the pinned original source files in the repository, choose a new output directory:

```bash
python training/scripts/create_golden35_subset.py --output-dir /path/to/new-golden35-v1
```

The builder uses explicit approved membership and refuses changed original source bytes or a missing approved valid case. `.gitattributes` preserves the historical source's CRLF checkout bytes and the frozen artifact's LF bytes on Windows and Linux. The new output manifest records its byte hash; compare that hash with this frozen revision before using any rebuilt artifact. Prompt or codec changes may require an intentional new benchmark version rather than silently updating the fixed reference.

The validation API is `ir_training.data.golden35_subset.validate_subset_rows(rows, manifest)`. It supports target-assignment and row reordering during strict preparation, while enforcing source/target semantic identity and the fixed 35-member cohort. The caller separately verifies raw or prepared output bytes against their corresponding manifest.
