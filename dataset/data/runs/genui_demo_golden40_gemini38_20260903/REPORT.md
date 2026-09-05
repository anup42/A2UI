# GenUI Demo Golden 40 Manual Android Run

Run ID: `genui_demo_golden40_gemini38_20260903`

## Configuration and execution

- Execution: real visible GenUI Demo flow, one prompt at a time on each device.
- Response model: `gemini-3.8-flash`.
- IR model: `gemini-3.8-flash`.
- Gemini route: Vertex AI Express API key.
- MCP: enabled.
- Flip8 (`R3GL203AKSF`, `SM-F776U`, Android 17, 1080x2520): ordinals 1-20.
- Fold7 (`R3CY30QFWLP`, `SM-F966B`, Android 16, 1080x2520): ordinals 21-40.
- The two device lanes ran concurrently; cases remained serial within each lane.

The API key value is not stored in this run. Configuration evidence records only whether a key was present.

## Outcome

| Metric | Result |
|---|---:|
| Planned/completed | 40 / 40 |
| Pipeline successes with strict-valid A2UI Express IR | 32 |
| Strict-validation failures | 8 |
| Successful native replay captures | 32 / 32 |
| Fully reviewed and accepted | 2 |
| Successful cases retained with known issues | 30 |
| Issue records | 55 (39 errors, 16 warnings) |

Only `s17_short` and `s18_full` passed pipeline, native rendering, prompt-content checks, and visual review without a recorded issue. All other outputs remain preserved as failure/known-issue golden evidence.

## Pair-by-pair status

| Pair | Scenario | Full | Short |
|---|---|---|---|
| s01 | Bengaluru weather | Rendered; issues | Rendered; issues |
| s02 | One-way flights | Rendered; issues | Rendered; issues |
| s03 | Return flights | Strict-validation failure | Strict-validation failure |
| s04 | Bengaluru hotels | Strict-validation failure | Strict-validation failure |
| s05 | Mysuru hotels | Strict-validation failure | Rendered; issues |
| s06 | Indiranagar restaurants | Rendered; issues | Rendered; issues |
| s07 | Koramangala vegetarian restaurants | Rendered; issues | Strict-validation failure |
| s08 | Mysuru attractions | Strict-validation failure | Rendered; issues |
| s09 | Bengaluru attractions | Rendered; issues | Rendered; issues |
| s10 | India technology news | Rendered; issues | Rendered; issues |
| s11 | Bengaluru local news | Rendered; issues | Rendered; issues |
| s12 | Mushroom risotto | Strict-validation failure | Rendered; issues |
| s13 | Headphone comparison | Rendered; issues | Rendered; issues |
| s14 | Walking chart/table | Rendered; issues | Rendered; issues |
| s15 | Workshop schedule | Rendered; issues | Rendered; issues |
| s16 | Packing checklist | Rendered; issues | Rendered; issues |
| s17 | Bill split | Rendered; issues | **Accepted** |
| s18 | Inventory | **Accepted** | Rendered; issues |
| s19 | Python lists vs tuples | Rendered; issues | Rendered; issues |
| s20 | Email preview | Rendered; issues | Rendered; issues |

## Strict-validation failures

The following eight cases ended with the same app error: `step output failed strict validation after repair. No fallback IR was rendered.`

- `s03_full`, `s03_short`: return flights.
- `s04_full`, `s04_short`: Bengaluru hotels.
- `s05_full`: Mysuru hotels.
- `s07_short`: vegetarian restaurants.
- `s08_full`: Mysuru attractions.
- `s12_full`: mushroom risotto.

These cases have prompt, result JSON, failure screenshot, UI hierarchy (except the harvested `s01_short`, which is not a failure), and filtered device log. The app did not persist response or IR payloads for Stage 3 failures, so those artifacts are intentionally absent rather than fabricated.

## Integration results

| Integration | Pipeline result | Live-data result |
|---|---|---|
| Weather | 2/2 rendered | 2/2 live via Open-Meteo |
| Flights | 2/4 rendered | One-way pair used live flight data; both return-flight cases failed strict IR validation |
| Hotels | 1/4 rendered | The one success used live hotel data; three cases failed strict IR validation |
| Restaurants | 3/4 rendered | **0 live**: the app logged that the Google Maps/Places API key was not configured and fell back to normal Stage 2 |
| Places | 3/4 rendered | **0 live**: the app logged that the Google Maps/Places API key was not configured and fell back to normal Stage 2 |
| News | 4/4 rendered | 4/4 live via NewsData |
| Non-live scenarios | 17/18 rendered | MCP correctly routed these to ordinary LLM generation |

Restaurants and Places therefore demonstrate fallback rendering, not working live integrations. Their generated addresses, ratings, links, and imagery must not be treated as verified current provider data.

## High-impact content and render issues

- Requested item counts were ignored in seven cases: flight, hotel, and news prompts asking for 3 or 5 items rendered 8 cards.
- `s01_short` requested only temperature, condition, and rain chance, but rendered extra metrics/sources/actions and clipped weather values.
- `s09_short` rendered tourist attractions with restaurant/fork-and-knife presentation.
- `s13_full` and `s13_short` clip the rightmost headphone table content.
- `s14_full` renders the bar chart as a text/code block and leaks a media URL; `s14_short` adds a malformed table even though the prompt says chart only.
- `s15_full` and `s15_short` overflow schedule columns beyond the right edge.
- `s16_full` renders category headers but no 12 checklist items; `s16_short` renders an empty checklist card.
- `s17_full` clips formulas and leaves a stray `Num` fragment.
- `s18_short` omits the requested total stock value.
- `s19_full` corrupts `fruits.append("cherry")` into `https://fruits.append("cherry")`; both s19 tables overflow horizontally.
- Full-height stitched images show five tile-overlap artifacts; viewport screenshots are retained separately for comparison.

Exact per-case issue types and messages are in `issues.jsonl` and `visual_review.jsonl`.

## Latency and reliability

- Measured cases: 39 (the harvested `s01_short` has `elapsed_ms=0` and is excluded).
- Mean: 158.0 seconds.
- Median: 124.6 seconds.
- P95: 470.3 seconds.
- Maximum: 497.9 seconds (`s07_full`).
- Eight cases logged Vertex Express HTTP 429 / resource-exhausted retries; 30 matching retry/error lines were retained.
- No production-app fatal exception or ANR occurred during the 40 golden scenario executions.

Setup-only evidence is preserved under `attempts/` and `diagnostics/`: a Fold7 notification-permission overlay, an initial non-durable preference setup that reverted to Azure defaults, an instrumentation teardown `DeadObjectException` before Fold7's lane, and a Flip8 shell `uiautomator dump` collision. None is counted as a scenario outcome.

## Artifact map

- `scenarios.jsonl`: exact 40 prompts and pair metadata.
- `queries.jsonl`, `responses.jsonl`, `genui.jsonl`: standard dataset records.
- `execution_results.jsonl`: merged per-case device results.
- `issues.jsonl`: pipeline plus review issues.
- `visual_review.jsonl`: manual full-height and viewport review decisions.
- `android_device_rendered/`: 32 full-height and 32 viewport native-render screenshots plus capture manifest.
- `native_render_checks.jsonl`: 32/32 successful native renderer checks.
- `cases/<pair>_<slug>/<variant>/`: prompt, response, IR, actual visible-flow screenshot, replay screenshots, UI hierarchy, filtered log, and metadata.
- `contact_sheets/`: four labeled viewport overview sheets.
- `diagnostics/`: final device metadata, package-focused log snapshots, and screenshots.
- `attempts/`: setup and retry evidence excluded from canonical outcomes.
- `SHA256SUMS.txt`: integrity hashes for the finalized run.

The earlier direct-pipeline partial experiment is isolated in `genui_demo_golden40_gemini38_20260903_instrumented_partial` and is not included in this manual visible-app golden run.
