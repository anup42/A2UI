# Full source-structure and Table-use census

This complete pass inspected the final task source in all **150,292 training and 910 validation records**, excluding fixed few-shot text. It joined each record to its structural-audit result and verified every target SHA-256 at the corresponding split/line. Both complete source-file hashes match the frozen inventory. The final pass took **69.86 seconds** in one process; no model or dataset modifications were made.

## Table-heavy targets are substantially source-driven

**125,572 training sources (83.55%) contain a Markdown-like table separator**, and **141,900 (94.42%) contain the weaker signal of pipe-separated data lines outside detected code fences**. Therefore, the 147,555 training targets containing `Table` (98.18%) are not, by themselves, evidence that the model is being taught to invent tables for arbitrary prose.

### Exact cross-tab: source separator versus target Table

All entries below are strict-valid targets; the all-parsed-target cross-tab is identical for this corpus.

| Training source | Target has Table | Target has no Table | Total |
|---|---:|---:|---:|
| Has Markdown-like separator | 125,418 | 154 | 125,572 |
| No detected separator | 22,137 | 2,583 | 24,720 |
| Total | 147,555 | 2,737 | 150,292 |

| Validation source | Target has Table | Target has no Table | Total |
|---|---:|---:|---:|
| Has Markdown-like separator | 758 | 1 | 759 |
| No detected separator | 131 | 20 | 151 |
| Total | 889 | 21 | 910 |

The separator heuristic has meaningful false negatives. For example, train line 5 contains pipe-delimited bank/APY/minimum-balance/fee records without a dash separator; line 10 contains a pipe-delimited travel-budget breakdown. Both are genuinely table-like source content even though the strict separator detector says “no.”

Using the broader pipe-row signal, **141,429 training targets with Table also have pipe-separated source lines**; only **6,126** Table targets lack that broader signal. Conversely, 471 sources with pipe-separated lines have no Table target. This still does not adjudicate correctness: lists can legitimately become card-presented Tables, and source tables can legitimately become native cards. Review preserved facts, rows and rendered layout rather than enforcing source/target syntax equality.

**Improvement implication:** test and expand source-format diversity if deployment needs free-form paragraphs, letters, natural unstructured lists, code-containing answers, or standalone media layouts. Do not indiscriminately remove Table targets or add Image nodes solely to equalize component percentages against Golden35. An image can be carried by a Table row or another property's binding, so standalone `Image` counts are not image-retention scores.

## Source markup census

| Detected feature | Training | Validation |
|---|---:|---:|
| Markdown-style heading | 149,624 | 906 |
| Bullet-list line | 141,979 | 848 |
| Numbered-list line | 1,451 | 9 |
| Markdown-like table separator outside fences | 125,572 | 759 |
| Pipe-separated line outside fences | 141,900 | 858 |
| Fenced code block | 242 | 2 |
| Unclosed detected fence at end | 0 | 0 |
| Media/Image/Video/Audio/Icon declaration line | 149,093 | 903 |
| Named `Action: [Button: ...]` line | 144,598 | 873 |
| Markdown image syntax | 2 | 0 |
| Markdown link syntax | 257 | 0 |

This describes an overwhelmingly structured, heading/list/media-declaration source format. Counts are **markup matches, not semantic intent labels**. The named-action regex here is anchored to a declaration line and is intentionally narrower than the semantic-review script's broader action-label extraction; its count need not equal that script's 144,657 training rows with named actions.

## Raw URL versus placeholder modes are mixed

| Source mode | Training | Validation |
|---|---:|---:|
| Literal HTTP(S) scheme, no tested typed placeholder | 76,899 | 451 |
| Typed placeholder, no literal HTTP(S) scheme | 72,580 | 452 |
| Both | 134 | 3 |
| Neither | 679 | 4 |

These four rows partition the entire source population. About half the corpus uses literal URL schemes and about half uses typed placeholders. “Literal HTTP(S)” detects the scheme text; it does **not** establish a valid, safe or reachable URL, and can match a URL example in a tutorial.

| Typed source placeholder presence | Training | Validation |
|---|---:|---:|
| Image URL/asset | 4,785 | 38 |
| Icon URL/asset | 72,230 | 453 |
| Action URL | 69,454 | 435 |
| Source URL | 35,091 | 245 |
| Media URL/asset | 195 | 2 |
| Generic URL | 0 | 0 |

These typed counts overlap. They do not measure total image/media demand because other rows contain literal URLs or different media declarations. Recover the original maps and standardize source/target URL handling as one bound transformation. Do not assume missing maps can be reconstructed safely by global token renaming or invented destinations.

## Unicode-block presence, not language classification

47,258 training sources and 285 validation sources contain at least one non-ASCII character. The remaining 103,034 / 625 sources are ASCII-only.

| Tested character range present in a source | Training | Validation |
|---|---:|---:|
| Latin extended | 33,603 | 209 |
| Greek | 27,790 | 171 |
| Box drawing | 27,119 | 161 |
| Block elements | 11,246 | 65 |
| Currency symbols | 186 | 1 |

No characters were found in the other tested ranges: Cyrillic, Hebrew, Arabic, Devanagari, Bengali, Gurmukhi, Gujarati, Tamil, Telugu, Kannada, Malayalam, Sinhala, Thai, Hangul, Hiragana, Katakana, CJK ideographs, general punctuation, the defined emoji/dingbat ranges, or U+FFFD. The exact ranges are in `summary.json`; this is not a complete Unicode script taxonomy.

**Do not call the Greek/box-drawing counts multilingual coverage.** These ranges can be triggered by mojibake such as the already-confirmed corrupted bullet/dash/accent sequences. Names, technical symbols, transliteration and damaged strings also prevent character presence from proving a document's language. If multilingual generation is required, curate and evaluate independently verified language-specific examples rather than relying on these counts.

## Method and limits

The table detector looks for a whole line with at least two pipe-separated dash columns, each with at least three dashes and optional Markdown alignment colons. A second metric accepts weaker pipe-delimited data lines. A simple backtick/tilde fence recognizer excludes detected code blocks only from the explicitly named table/pipe metrics. Other markup regexes may also match literal examples inside code. This is not a full Markdown parser.

Source formatting, component presence and character ranges are descriptive signals, not quality scores. They cannot prove a target preserved all facts, selected the best layout, or bound a usable asset. This census does not resolve the missing provenance, Golden contamination, semantic omissions, or template-bound tokenization limitations from the main report.

Reproduce with `python training/scripts/audits/full_data_source_structure_20260913.py`. `summary.json` contains the script hash, frozen input hashes, every census count, both cross-tab variants, separator-count histograms, exact Unicode ranges and bounded source-line examples. No training files are written.
