# Why all Bixby50 cases used source-text fallback

This report distinguishes three different operations. The raw E2B model
program is compiled first. A bounded output-only repair then tries only safe
framing normalization. If both are rejected, the SDK discards the generated
program and builds a new A2UI document from exact source blocks. Therefore the
final **50/50 rendered** result does not mean 50 model programs were repaired:
it means **0 model programs were repairable and 50 exact-source fallbacks were
built**.

## Exact outcome

| Gate | Accepted | Meaning |
|---|---:|---|
| Raw strict compiler | **0/50** | Every generated Express program was invalid. |
| Current bounded output-only repair | **0/50** | BOM/fence/final-close normalization was insufficient for every case. |
| Aggressive compiler-salvage ceiling | **1/50** | BXP-005 could be made parseable only after 13 graph/layout mutations. |
| Aggressive salvage passing source integrity | **0/50** | BXP-005 still had 25 source/graph integrity findings. |
| Exact-source fallback compiled | **50/50** | New documents were built from source blocks. |
| Exact-source fallback integrity issues | **0** | The final documents passed the current mechanical gate. |
| Exact-source fallback renderer issues | **0** | The final replay rendered all 50 on device. |

If “after repair” means the final fallback documents, no compile, canonical
graph, source-coverage, table, scroll, or renderer issue was recorded. One
semantic presentation limitation remained: **BXP-027** used generic Text/List
nodes for an email example and received the v5.4 `missing_special_role=email`
cap. The malformed model program itself was replaced, not repaired.

## What the bounded repair actually tried

The production repairer has only three framing corrections: remove one leading
BOM, unwrap one exact Markdown code fence, and add the final `>` to a trailing
`</a2ui`. In this captured run, **0** raw outputs had a leading BOM, **0** had
an exact fence wrapper, and **0** were otherwise-complete programs needing only
that final closing character. Consequently, every output-only attempt returned
no valid candidate.

The repairer deliberately cannot append omitted program content, truncate an
ambiguous suffix, close arbitrary strings or delimiters, rename properties,
replace semantic enum values, wrap scalar children, invent components, or
delete graph references. Those edits can alter visible facts or model intent.

## First compiler rejection for every model output

The compiler reports the first deterministic failure, so these categories are
mutually exclusive but not exhaustive. Later corruption can be hidden behind
the first parse error.

| First rejection | Count | Cases |
|---|---:|---|
| incomplete/trailing envelope | 30 | BXP-003, BXP-004, BXP-006, BXP-009, BXP-013, BXP-016, BXP-018, BXP-019, BXP-023, BXP-024, BXP-027, BXP-028, BXP-029, BXP-030, BXP-031, BXP-032, BXP-033, BXP-034, BXP-035, BXP-036, BXP-037, BXP-039, BXP-040, BXP-042, BXP-043, BXP-044, BXP-045, BXP-046, BXP-048, BXP-050 |
| unclosed expression | 9 | BXP-001, BXP-007, BXP-011, BXP-014, BXP-015, BXP-017, BXP-025, BXP-041, BXP-047 |
| unbalanced delimiter | 4 | BXP-012, BXP-022, BXP-038, BXP-049 |
| unknown property | 2 | BXP-002, BXP-008 |
| unsupported prefix | 2 | BXP-020, BXP-026 |
| invalid gap enum | 1 | BXP-005 |
| children type | 1 | BXP-010 |
| empty/invalid children | 1 | BXP-021 |

The dominant failure is not a missing closing tag by itself. Of 29 generations
that stopped at `max_new_tokens`, 28 ended with an incomplete/trailing envelope
and BXP-026 started with a corrupted `< 2ui>` prefix. Adding `</a2ui>` cannot
restore omitted components, source text, or references. The remaining 21 cases
stopped through a closing sentinel or native/EOS stop and were still invalid,
so increasing the output limit alone cannot fix the run.

Stop-reason counts:

- `max_new_tokens`: **29**
- `closing_sentinel`: **13**
- `eos_or_native_stop`: **8**

## Why broader repair was rejected

The current production repair only removes a leading BOM, unwraps one exact
Markdown fence, or normalizes a malformed final closing token. It does not add
missing components, close arbitrary strings, delete suffixes, rename arbitrary
properties, change enum values, or prune graph references. Those operations
can make a program parse while silently changing the answer.

An offline upper-bound probe tested more aggressive, sample-independent rules:
case-only property normalization, singleton-child wrapping, gap normalization,
layout-default substitution, and removal of dangling child references. Only
**BXP-005** compiled. It required 13 changes, including five deleted dangling
references. The resulting program still produced **25 integrity
findings**:

- 18 repeated-child findings and unreachable elements `ac` through `ah`;
- missing citations `[1]`, `[5]`, `[2]`, and `[3]`;
- changed facts such as `5:00 -> 5:0:0` and `11:55 -> 11:555`;
- missing and invented wording; and
- changed visible wording order or multiplicity.

Its v5.4 score was only **40.0**, content-unit fidelity **0.2699**, exact
number/date/unit F-beta **0.3205**, and root reachability **0.8286**. The final
SDK correctly classifies this candidate as `SOURCE_TEXT_FALLBACK` when source
text is supplied; it cannot be mistaken for a repaired model result.

Raw numeric substring retention provides another warning signal. Forty-nine
sources contained numeric literals. Only **3/49** raw outputs retained all of
them, **42/49** retained some, and **4/49** retained none; mean diagnostic
recall was **0.519**. This is only a substring
diagnostic, because invalid programs cannot receive a reliable semantic score.

## Per-case result

`Upper-bound probe` reports whether the aggressive offline probe reached strict
compiler validity. `Numeric recall` is diagnostic only. Refusal codes are:

- **R1:** incomplete/trailing document; repair cannot append omitted content or
  delete an ambiguous suffix.
- **R2:** malformed grammar/delimiters; repair cannot guess quote, bracket, or
  expression placement.
- **R3:** catalog/property/enum corruption; repair cannot rename or replace
  semantic values.
- **R4:** graph-shape corruption; repair cannot wrap, invent, or prune
  references.
- **R5:** invalid prefix; candidates must start exactly with `<a2ui>`.

Every final result was independently built from exact source blocks. `Fallback
score/cap` is a representation-quality result for that replacement document;
it is not a score for the rejected model program.

| Case | Stop | First AAR rejection | Confirmed raw defects | Refusal | Upper-bound probe | Numeric recall | Final fallback; score/cap |
|---|---|---|---|---|---|---:|---|
| [BXP-001](../20260921_e2b_mobile_full50/native/BXP-001/output.express) | closing_sentinel | unclosed expression | Unclosed root=Column(, invalid &lt;a,b&gt;, a malformed final table, and a &lt;/a2ui&lt;/a2ui&gt; tail. | R2 | not compiler-valid | 0.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-001/recovered.output.express); 98.157; no active cap |
| [BXP-002](../20260921_e2b_mobile_full50/native/BXP-002/output.express) | closing_sentinel | unknown property | Unknown Column property 铺; an assignment also lacks a left-hand ID, Gap is invalid, and Text calls are free-standing. | R3 | not compiler-valid | 0.750 | [SOURCE_TEXT_FALLBACK](cases/BXP-002/recovered.output.express); 95.866; no active cap |
| [BXP-003](../20260921_e2b_mobile_full50/native/BXP-003/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in a long 111… repetition loop. | R1 | not compiler-valid | 0.714 | [SOURCE_TEXT_FALLBACK](cases/BXP-003/recovered.output.express); 98.697; no active cap |
| [BXP-004](../20260921_e2b_mobile_full50/native/BXP-004/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated Hungary tokens. | R1 | not compiler-valid | 0.474 | [SOURCE_TEXT_FALLBACK](cases/BXP-004/recovered.output.express); 96.860; no active cap |
| [BXP-005](../20260921_e2b_mobile_full50/native/BXP-005/output.express) | closing_sentinel | invalid gap enum | Invalid gap 🚚; later gaps, child forms, properties, references, and Button syntax are also invalid. | R3/R4 | compiled, integrity rejected | 0.917 | [SOURCE_TEXT_FALLBACK](cases/BXP-005/recovered.output.express); 97.156; no active cap |
| [BXP-006](../20260921_e2b_mobile_full50/native/BXP-006/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output degenerates into hundreds of repeated = lines. | R1 | not compiler-valid | 0.706 | [SOURCE_TEXT_FALLBACK](cases/BXP-006/recovered.output.express); 97.165; no active cap |
| [BXP-007](../20260921_e2b_mobile_full50/native/BXP-007/output.express) | eos_or_native_stop | unclosed expression | Corrupt Table column/state expression and unclosed syntax; invalid Gap and children syntax remain later. | R2 | not compiler-valid | 0.769 | [SOURCE_TEXT_FALLBACK](cases/BXP-007/recovered.output.express); 91.754; no active cap |
| [BXP-008](../20260921_e2b_mobile_full50/native/BXP-008/output.express) | closing_sentinel | unknown property | Unknown case-sensitive Gap property; corrupted gap values and Table column structure remain later. | R3 | not compiler-valid | 0.118 | [SOURCE_TEXT_FALLBACK](cases/BXP-008/recovered.output.express); 95.250; no active cap |
| [BXP-009](../20260921_e2b_mobile_full50/native/BXP-009/output.express) | max_new_tokens | incomplete/trailing envelope | Malformed root=ColumnRY… followed by repeated component tokens/~3 degeneration and no close. | R1 | not compiler-valid | 1.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-009/recovered.output.express); 97.386; no active cap |
| [BXP-010](../20260921_e2b_mobile_full50/native/BXP-010/output.express) | closing_sentinel | children type | Card(children=a/e/h) uses scalar children; later Column and final expressions are also malformed. | R4 | not compiler-valid | 0.333 | [SOURCE_TEXT_FALLBACK](cases/BXP-010/recovered.output.express); 96.005; no active cap |
| [BXP-011](../20260921_e2b_mobile_full50/native/BXP-011/output.express) | eos_or_native_stop | unclosed expression | Top-level Column has no ID; Table and Button calls are malformed and properties/gaps are invalid. | R2 | not compiler-valid | 0.448 | [SOURCE_TEXT_FALLBACK](cases/BXP-011/recovered.output.express); 93.217; no active cap |
| [BXP-012](../20260921_e2b_mobile_full50/native/BXP-012/output.express) | closing_sentinel | unbalanced delimiter | Unbalanced expressions begin with Column(/… and mix anonymous and non-catalog calls. | R2 | not compiler-valid | 0.500 | [SOURCE_TEXT_FALLBACK](cases/BXP-012/recovered.output.express); 96.420; no active cap |
| [BXP-013](../20260921_e2b_mobile_full50/native/BXP-013/output.express) | max_new_tokens | incomplete/trailing envelope | One block closes, a second &lt;a2ui&gt; opens, then bullet repetition continues without a closing envelope. | R1 | not compiler-valid | 0.429 | [SOURCE_TEXT_FALLBACK](cases/BXP-013/recovered.output.express); 97.127; no active cap |
| [BXP-014](../20260921_e2b_mobile_full50/native/BXP-014/output.express) | closing_sentinel | unclosed expression | Unclosed argument lists include a corrupt gap argument and malformed Table objects. | R2 | not compiler-valid | 0.444 | [SOURCE_TEXT_FALLBACK](cases/BXP-014/recovered.output.express); 95.989; no active cap |
| [BXP-015](../20260921_e2b_mobile_full50/native/BXP-015/output.express) | eos_or_native_stop | unclosed expression | Malformed Table column objects and component expressions; invalid gaps/components remain behind the first error. | R2 | not compiler-valid | 0.375 | [SOURCE_TEXT_FALLBACK](cases/BXP-015/recovered.output.express); 97.092; no active cap |
| [BXP-016](../20260921_e2b_mobile_full50/native/BXP-016/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated aeae… tokens. | R1 | not compiler-valid | 0.714 | [SOURCE_TEXT_FALLBACK](cases/BXP-016/recovered.output.express); 95.499; no active cap |
| [BXP-017](../20260921_e2b_mobile_full50/native/BXP-017/output.express) | eos_or_native_stop | unclosed expression | Invalid component token and malformed Button expression with undefined references. | R2 | not compiler-valid | 0.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-017/recovered.output.express); 96.507; no active cap |
| [BXP-018](../20260921_e2b_mobile_full50/native/BXP-018/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output repeats 3, quotes, and Hungary. | R1 | not compiler-valid | 0.875 | [SOURCE_TEXT_FALLBACK](cases/BXP-018/recovered.output.express); 96.899; no active cap |
| [BXP-019](../20260921_e2b_mobile_full50/native/BXP-019/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated =af tokens. | R1 | not compiler-valid | 0.778 | [SOURCE_TEXT_FALLBACK](cases/BXP-019/recovered.output.express); 96.810; no active cap |
| [BXP-020](../20260921_e2b_mobile_full50/native/BXP-020/output.express) | closing_sentinel | unsupported prefix | Opening tag is &lt;a2a2ui&gt;; the body also contains malformed properties and calls. | R5 | not compiler-valid | 0.875 | [SOURCE_TEXT_FALLBACK](cases/BXP-020/recovered.output.express); 97.322; no active cap |
| [BXP-021](../20260921_e2b_mobile_full50/native/BXP-021/output.express) | closing_sentinel | empty/invalid children | Numeric/scalar children replace non-empty references; duplicate IDs/cycles and an invalid gap also remain. | R4 | not compiler-valid | 1.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-021/recovered.output.express); 95.162; no active cap |
| [BXP-022](../20260921_e2b_mobile_full50/native/BXP-022/output.express) | closing_sentinel | unbalanced delimiter | Mixed Express/JSON fragments, children!, stray tokens/braces, and a second &lt;a2ui&gt; block. | R2 | not compiler-valid | 0.727 | [SOURCE_TEXT_FALLBACK](cases/BXP-022/recovered.output.express); 96.461; no active cap |
| [BXP-023](../20260921_e2b_mobile_full50/native/BXP-023/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated af/ag identifiers. | R1 | not compiler-valid | 0.286 | [SOURCE_TEXT_FALLBACK](cases/BXP-023/recovered.output.express); 96.356; no active cap |
| [BXP-024](../20260921_e2b_mobile_full50/native/BXP-024/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated identifier tuples. | R1 | not compiler-valid | 0.700 | [SOURCE_TEXT_FALLBACK](cases/BXP-024/recovered.output.express); 97.760; no active cap |
| [BXP-025](../20260921_e2b_mobile_full50/native/BXP-025/output.express) | eos_or_native_stop | unclosed expression | The root line interleaves a malformed Column call with object data and never closes correctly. | R2 | not compiler-valid | 1.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-025/recovered.output.express); 94.866; no active cap |
| [BXP-026](../20260921_e2b_mobile_full50/native/BXP-026/output.express) | max_new_tokens | unsupported prefix | Opening tag is &lt; 2ui&gt;; malformed Table syntax and bullet repetition follow with no close. | R5/R1 | not compiler-valid | 0.059 | [SOURCE_TEXT_FALLBACK](cases/BXP-026/recovered.output.express); 95.701; no active cap |
| [BXP-027](../20260921_e2b_mobile_full50/native/BXP-027/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated af identifiers. | R1 | not compiler-valid | 0.100 | [SOURCE_TEXT_FALLBACK](cases/BXP-027/recovered.output.express); 78.000; missing_special_role(email), cap 0.78 |
| [BXP-028](../20260921_e2b_mobile_full50/native/BXP-028/output.express) | max_new_tokens | incomplete/trailing envelope | Malformed state opener $/{… plus repeated identifiers and no closing envelope. | R1 | not compiler-valid | 0.111 | [SOURCE_TEXT_FALLBACK](cases/BXP-028/recovered.output.express); 97.295; no active cap |
| [BXP-029](../20260921_e2b_mobile_full50/native/BXP-029/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output degenerates into repeated bullet characters. | R1 | not compiler-valid | 0.231 | [SOURCE_TEXT_FALLBACK](cases/BXP-029/recovered.output.express); 97.025; no active cap |
| [BXP-030](../20260921_e2b_mobile_full50/native/BXP-030/output.express) | closing_sentinel | incomplete/trailing envelope | A complete &lt;/a2ui&gt; is followed by literal title; deleting the suffix still leaves state only and no root. | R1 | not compiler-valid | 0.750 | [SOURCE_TEXT_FALLBACK](cases/BXP-030/recovered.output.express); 97.189; no active cap |
| [BXP-031](../20260921_e2b_mobile_full50/native/BXP-031/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated identifier sequences. | R1 | not compiler-valid | 0.083 | [SOURCE_TEXT_FALLBACK](cases/BXP-031/recovered.output.express); 97.706; no active cap |
| [BXP-032](../20260921_e2b_mobile_full50/native/BXP-032/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in a long 000… repetition loop. | R1 | not compiler-valid | 0.476 | [SOURCE_TEXT_FALLBACK](cases/BXP-032/recovered.output.express); 94.735; no active cap |
| [BXP-033](../20260921_e2b_mobile_full50/native/BXP-033/output.express) | max_new_tokens | incomplete/trailing envelope | Malformed state/path expression followed by hundreds of Z tokens and no close. | R1 | not compiler-valid | 0.833 | [SOURCE_TEXT_FALLBACK](cases/BXP-033/recovered.output.express); 92.813; no active cap |
| [BXP-034](../20260921_e2b_mobile_full50/native/BXP-034/output.express) | max_new_tokens | incomplete/trailing envelope | Repeated quoted bullet values leave the final string/list unfinished. | R1 | not compiler-valid | 0.690 | [SOURCE_TEXT_FALLBACK](cases/BXP-034/recovered.output.express); 95.734; no active cap |
| [BXP-035](../20260921_e2b_mobile_full50/native/BXP-035/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated identifier sequences. | R1 | not compiler-valid | 0.810 | [SOURCE_TEXT_FALLBACK](cases/BXP-035/recovered.output.express); 96.260; no active cap |
| [BXP-036](../20260921_e2b_mobile_full50/native/BXP-036/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output degenerates into hundreds of repeated = lines. | R1 | not compiler-valid | 0.889 | [SOURCE_TEXT_FALLBACK](cases/BXP-036/recovered.output.express); 96.757; no active cap |
| [BXP-037](../20260921_e2b_mobile_full50/native/BXP-037/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in a repeated _1 sequence. | R1 | not compiler-valid | 0.636 | [SOURCE_TEXT_FALLBACK](cases/BXP-037/recovered.output.express); 97.782; no active cap |
| [BXP-038](../20260921_e2b_mobile_full50/native/BXP-038/output.express) | closing_sentinel | unbalanced delimiter | State array/object delimiters are mismatched; no valid root UI remains behind the first error. | R2 | not compiler-valid | n/a | [SOURCE_TEXT_FALLBACK](cases/BXP-038/recovered.output.express); 97.510; no active cap |
| [BXP-039](../20260921_e2b_mobile_full50/native/BXP-039/output.express) | max_new_tokens | incomplete/trailing envelope | The output degenerates almost immediately into $/$/… and has no close. | R1 | not compiler-valid | 0.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-039/recovered.output.express); 96.856; no active cap |
| [BXP-040](../20260921_e2b_mobile_full50/native/BXP-040/output.express) | eos_or_native_stop | incomplete/trailing envelope | Malformed pseudo-Express followed by repeated Hindi text; native stop occurs without &lt;/a2ui&gt;. | R1 | not compiler-valid | 0.462 | [SOURCE_TEXT_FALLBACK](cases/BXP-040/recovered.output.express); 97.230; no active cap |
| [BXP-041](../20260921_e2b_mobile_full50/native/BXP-041/output.express) | eos_or_native_stop | unclosed expression | Unclosed children list, malformed Table/Column calls, and bare q immediately before the close. | R2 | not compiler-valid | 0.550 | [SOURCE_TEXT_FALLBACK](cases/BXP-041/recovered.output.express); 96.215; no active cap |
| [BXP-042](../20260921_e2b_mobile_full50/native/BXP-042/output.express) | max_new_tokens | incomplete/trailing envelope | Unclosed root=Column( and nested Tabs construction followed by _t repetition. | R1 | not compiler-valid | 0.000 | [SOURCE_TEXT_FALLBACK](cases/BXP-042/recovered.output.express); 97.538; no active cap |
| [BXP-043](../20260921_e2b_mobile_full50/native/BXP-043/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in hundreds of Z lines. | R1 | not compiler-valid | 0.485 | [SOURCE_TEXT_FALLBACK](cases/BXP-043/recovered.output.express); 94.786; no active cap |
| [BXP-044](../20260921_e2b_mobile_full50/native/BXP-044/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated identifier tuples. | R1 | not compiler-valid | 0.062 | [SOURCE_TEXT_FALLBACK](cases/BXP-044/recovered.output.express); 96.413; no active cap |
| [BXP-045](../20260921_e2b_mobile_full50/native/BXP-045/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated Hungary tokens. | R1 | not compiler-valid | 0.750 | [SOURCE_TEXT_FALLBACK](cases/BXP-045/recovered.output.express); 93.182; no active cap |
| [BXP-046](../20260921_e2b_mobile_full50/native/BXP-046/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in repeated identifier tuples. | R1 | not compiler-valid | 0.500 | [SOURCE_TEXT_FALLBACK](cases/BXP-046/recovered.output.express); 97.049; no active cap |
| [BXP-047](../20260921_e2b_mobile_full50/native/BXP-047/output.express) | eos_or_native_stop | unclosed expression | Malformed data arrays contain an unkeyed ],[{…}] fragment and a broken storage entry; no valid root remains. | R2 | not compiler-valid | 0.762 | [SOURCE_TEXT_FALLBACK](cases/BXP-047/recovered.output.express); 94.996; no active cap |
| [BXP-048](../20260921_e2b_mobile_full50/native/BXP-048/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output ends in a long 666… repetition loop. | R1 | not compiler-valid | 0.571 | [SOURCE_TEXT_FALLBACK](cases/BXP-048/recovered.output.express); 94.720; no active cap |
| [BXP-049](../20260921_e2b_mobile_full50/native/BXP-049/output.express) | closing_sentinel | unbalanced delimiter | Numerous missing parentheses, duplicate assignments, and extra closing delimiters. | R2 | not compiler-valid | 0.200 | [SOURCE_TEXT_FALLBACK](cases/BXP-049/recovered.output.express); 97.428; no active cap |
| [BXP-050](../20260921_e2b_mobile_full50/native/BXP-050/output.express) | max_new_tokens | incomplete/trailing envelope | No closing envelope; the output repeats malformed =Column(ar(ar) statements. | R1 | not compiler-valid | 0.500 | [SOURCE_TEXT_FALLBACK](cases/BXP-050/recovered.output.express); 97.405; no active cap |

## Residual quality after source fallback

All 50 replacement documents were strict-valid, canonical-valid, fully root
reachable, and had content coverage **1.0**. All 50 rendered; all **23/23**
table cases retained complete columns; device replay recorded no renderer
issue. The fallback v5.4 representation score averaged
**95.922**, with range
**78.000–98.697**.

Only **BXP-027** had an active semantic cap. Its source includes a “Harmless
example email,” while deterministic fallback emitted ordinary Text/List nodes
instead of an email-specific semantic role. It scored **78.0**, capped at
**0.78** for `missing_special_role=email`. It still compiled, preserved source
content, was fully reachable, rendered, and recorded `issues=[]`. This is a
presentation/semantics gap rather than content loss or renderer failure.

The other general limitation is that source fallback chooses deterministic
headings, paragraphs, lists, tables, code blocks, and dividers. It preserves
the answer safely but does not reconstruct the richer layout that the failed
model generation intended. There is no reference IR for this holdout corpus,
so mechanical coverage does not prove equivalence to an intended design, and
`issues=[]` is not a human visual, accessibility, or design-quality judgment.

## Complete machine-readable evidence

- [Per-case CSV](repair_failure_analysis.csv) includes exact compiler messages,
  confirmed raw defects, repair refusal codes, why repair stopped,
  aggressive-probe errors/change rules, numeric diagnostics, and final
  fallback score/cap/render status.
- [Structured JSON](repair_failure_analysis.json) contains the aggregate counts,
  all 50 records, the complete 13-change BXP-005 mutation list, and all 25
  integrity findings, plus final fallback quality evidence.
- [Raw model report](../20260921_e2b_mobile_full50/REPORT.md) retains generation,
  GPU/MTP, token-limit, and original compiler evidence.
- [Final recovery report](REPORT.md) retains the 50/50 device-render evidence.

## Practical conclusion

The problem is primarily model-output quality, not a renderer failure. This
captured run combines truncation/token degeneration, malformed Express grammar,
catalog hallucinations, broken graph references, and factual mutations. A more
permissive parser would at best make one case compile and would still accept
incorrect content. The current fallback behavior is therefore the safe result
for all 50 cases, with the BXP-027 email-role presentation limitation recorded
above. Improving the model/export or producing a substantially more compact
constrained output is required before these cases can be counted as
model-generated A2UI successes.
