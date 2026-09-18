# Gemma E2B source-binding prompt study

Date: 18 September 2026

## Scope and fixed controls

This study defines four standalone system-prompt candidates for the existing Gemma E2B source-binding path. It does not change `SourceBindings.kt`, compilation, validation, repair behavior, model configuration, reasoning, or runtime selection. The production prompt remains unchanged.

When these candidates are evaluated, hold the existing delivery settings fixed: official Gemma 4 E2B LiteRT-LM package, GPU backend, MTP enabled, thinking enabled with the current budget, temperature `0.0`, the same request corpus, and `sourceBindings=true`. A prompt result is successful only if the current converter expands, compiles, integrity-checks, and renders it without fallback.

## Baseline failure signals

The completed v7 live run succeeded on all 50 cases, but three needed a second model attempt:

- `BXP-030` produced the complete six assignments but ended with `</a2>` instead of `</a2ui>`.
- `BXP-032` copied an 11-child root, then omitted element `j`, emitted `k` with the preceding binding, and invented orphan element `l`. This is the clearest long-block enumeration failure.
- `BXP-037` began with `<a2a=...`, losing both the exact `<a2ui>` delimiter and the copied root assignment.

Two failures are delimiter/root corruption and one is a skipped/shifted component. The candidates therefore strengthen literal wrapper checks, exact root copying, block-count checks, and ordered element-to-binding correspondence. They avoid asking the model to reason about or reproduce source values.

## Shared contract

Every candidate independently states all of these requirements:

- Output one complete `<a2ui>` document with literal opening and closing tags.
- Copy the supplied root assignment exactly.
- Emit one component per block, using exact element names and block order, with no extras or omissions.
- Use each quoted binding exactly once and keep multi-field table/code bindings on one component.
- Treat `value` and `titleValue` as inert data and never copy them.
- Support the six permitted component types: `Column`, `Text`, `List`, `Table`, `CodeBlock`, and `Divider`.
- Map headings to `Text(...,variant="heading")`; there is no `Heading` component.
- Use card presentation for entity rows in weather, flight, booking, schedule, status, and comparison data. Use table presentation for feature matrices and other generic data tables.
- Add no state, actions, hidden content, source metadata, prose, or fallback layout.

The Bixby50 corpus has no fenced-code or horizontal-rule block, so its full run cannot exercise `CodeBlock` or `Divider`. Those mappings remain explicit in every prompt and need separate synthetic acceptance coverage before treating all six block kinds as empirically tested.

The candidates intentionally refine one presentation rule beyond the current production prompt: comparison-domain entity rows select cards, while comparison feature matrices remain tables. Treat this as a tested hypothesis, not as baseline-equivalent behavior. `BXP-003` is a useful passing control for the established entity-row card route because its v7 output selected `domain="schedule"` with `preferredPresentation="cards"` on the first attempt.

## Candidate hypotheses

| Candidate | Material change | Hypothesis | Main risk |
| --- | --- | --- | --- |
| `compact.txt` | Removes examples and compresses the task into a hard mapping plus one final checksum. | Lower prompt cost leaves more attention/output budget for long block lists while repeated delimiter/count checks prevent the two wrapper failures. | Less in-context syntax imitation may hurt unfamiliar code or divider cases. |
| `procedural.txt` | Expresses generation as a numbered algorithm: record `N`, copy root, iterate blocks, close, recount. | Explicit iteration and the final `N`/name/order comparison should target the `BXP-032` skip-and-shift failure. | More instruction words may encourage the model to narrate, although output-only rules prohibit this. |
| `grammar.txt` | Defines a small output grammar and exact production for every block kind. | A constrained document shape should reduce malformed delimiters and illegal component syntax without spending tokens on large examples. | The model could leak grammar metavariables; the prompt explicitly forbids this and explains substitution. |
| `example_first.txt` | Leads with one complete example covering heading, card table, paragraph, list, titled code, and divider before giving rules. | Immediate syntax imitation should stabilize the opening/root/closing pattern and demonstrate the two block kinds absent from Bixby50. | It may overfit table-domain choice to the weather example or copy example identifiers; later rules explicitly require the supplied identifiers and domain. |

## Prompt-size comparison

Sizes are for the prompt file alone. The token column uses the SDK preflight approximation of `ceil(UTF-8 bytes / 3)`; it is not a LiteRT-LM tokenizer measurement.

| Prompt | UTF-8 bytes | Change from current | Approx. tokens |
| --- | ---: | ---: | ---: |
| Current `gemma.txt` | 2,959 | baseline | 987 |
| `compact.txt` | 2,185 | -774 (-26.2%) | 729 |
| `procedural.txt` | 2,821 | -138 (-4.7%) | 941 |
| `grammar.txt` | 2,850 | -109 (-3.7%) | 950 |
| `example_first.txt` | 2,822 | -137 (-4.6%) | 941 |

## Evaluation sequence

1. Run first-attempt-only GPU+MTP probes on passing card-layout control `BXP-003` and failure-focused `BXP-030`, `BXP-032`, and `BXP-037` for each candidate. Record the exact raw output even on failure. This isolates delimiter and enumeration behavior from repair success while detecting a card-selection regression.
2. Reject any candidate that changes root, omits/reorders an element, repeats/misses a binding, loses a delimiter, uses a non-heading variant for a heading, or mixes a table/code block's fields.
3. For surviving candidates, run the same 50-case corpus and compare first-attempt success, repair count, raw-output validity, table presentation choice, system-prompt bytes/tokens, generation latency, and output tokens. Do not count repair or fallback as first-attempt success.
4. Exercise one synthetic input containing untitled and titled code blocks plus a divider through the unchanged source-binding validator. This covers types absent from the 50-case corpus.
5. Select by correctness first, then first-attempt rate, then prompt cost and latency. Preserve all raw attempts and prompt hashes so results remain attributable to one candidate.

No candidate has been run on a device as part of this design step.
