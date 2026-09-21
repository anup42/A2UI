# Bixby50 safe A2UI recovery and renderer report

Evaluation date: 21 September 2026. In this evaluation, the recovery path turns
each rejected generation into a typed A2UI document from the exact source blocks.
On the connected Samsung SM-F776U, **50/50 fallback documents compiled and
rendered, with zero renderer issues**. This does not change the underlying model result: its
raw output remains **0/50 strict valid**, and the bounded syntax repair remains
**0/50** on this corpus.

## Outcome

| Measure | Result |
|---|---:|
| Raw E2B model output strict valid | **0/50** |
| Safe syntax repair accepted | **0/50** |
| Rejected model outputs | **50/50** |
| Exact-source typed fallback compiled | **50/50** |
| Exact-source typed fallback rendered | **50/50** |
| Renderer failures / recorded issues | **0 / 0** |
| Tables with all checked columns observed | **23/23** |
| Cases reaching observed vertical end | **50/50** |
| Device captures / vertical swipes | **227 / 110** |
| Model calls during replay | **0** |
| Batch wall elapsed | **294.209 s** |
| Mean recorded case replay | **5.826 s/case** |

The recovered output is usable as a lossless presentation fallback when model
output cannot be trusted. It preserves source order and exact table cells,
headings, paragraphs, lists, code blocks and dividers. It uses a deterministic
generic layout, so it does not recover the model's intended rich layout or
domain-specific composition.

## Repair policy

`compileWithRepair(input, sourceText)` follows three explicit outcomes:

1. `NONE`: strict output compiles and passes source-content integrity.
2. `STRUCTURAL`: only an outer code fence, a leading UTF-8 BOM, or a malformed
   final `</a2ui` token is normalized; the result must pass the same compiler and
   content checks.
3. `SOURCE_TEXT_FALLBACK`: rejected output is discarded and exact typed source
   blocks are compiled into deterministic A2UI.

The repair deliberately does not truncate tails, prune dangling graph nodes,
drop unknown properties, invent IDs/root assignments, fuzzy-match content, or
mutate values. Those operations can make malformed syntax parse by deleting or
changing visible information. In this corpus, even the closest aggressive
salvage candidate still failed source integrity, so it remains rejected.

## Recovered-corpus score

The v5.4 evaluator scored the deterministic fallback corpus separately:

| Measure | Result |
|---|---:|
| Strict schema valid | **50/50** |
| Canonical semantic valid | **50/50** |
| Fully root reachable | **50/50** |
| Mean source content coverage | **100%** |
| Mean generation reward v5.4 | **95.92/100** |
| Score range | **78.00–98.70** |

This is a source-only holdout with no reference IR. The score measures the
fallback representation and source fidelity. It does not measure the E2B model,
fact correctness, or equivalence to a human-designed UI. See the
[aggregate metrics](scoring/aggregate_metrics.json) and
[per-case results](per_case.csv).

## Device rendering

The final replay used run `mobile_gpu_mtp_full50_repair_20260921_r5` on Samsung
SM-F776U, Android API 37. All raw source hashes match the previously archived
[full E2B run](../20260921_e2b_mobile_full50/REPORT.md). The device generated
the fallback through the AAR, compiled it, rendered it, scrolled every case to
an observed vertical end, and checked table headers or representative cells.

Representative final-build screenshots:

- [BXP-001 weather table](screenshots/BXP-001.png)
- [BXP-005 urban transport](screenshots/BXP-005.png)
- [BXP-030 retirement savings](screenshots/BXP-030.png)
- [BXP-050 ocean and monsoon science](screenshots/BXP-050.png)

The complete 657-file, 82,184,391-byte capture set remains on the connected
device and in the ignored host run folder. The committed package keeps every
recovered Express/JSON document and per-case replay result, plus representative
screenshots. This avoids duplicating 80 MB of repeated hierarchy captures.

## Performance boundary

No inference was rerun, so recovery adds no new model throughput measurement.
The same archived model run remains: median decode **18.24 tokens/s**, weighted
decode **18.30 tokens/s**, and MTP draft acceptance **68.52%**. The 294.209 s
replay time is compiler/render/capture automation time and must not be compared
to token generation speed.

## Per-case result

Every row had raw strict validity `false`, safe structural validity `false`,
recovery kind `SOURCE_TEXT_FALLBACK`, content coverage `1.0`, render status
`rendered`, zero missing checked table columns, and zero renderer issues.

| Case | Domain | Recovery | v5.4 | Tables | V-swipes | Replay s |
|---|---|---|---:|---:|---:|---:|
| [BXP-001](cases/BXP-001/recovered.output.express) | Weather | SOURCE_TEXT_FALLBACK | 98.16 | 1 | 1 | 3.874 |
| [BXP-002](cases/BXP-002/recovered.output.express) | Air quality | SOURCE_TEXT_FALLBACK | 95.87 | 0 | 1 | 1.961 |
| [BXP-003](cases/BXP-003/recovered.output.express) | Rail travel | SOURCE_TEXT_FALLBACK | 98.70 | 1 | 2 | 7.465 |
| [BXP-004](cases/BXP-004/recovered.output.express) | Airline baggage | SOURCE_TEXT_FALLBACK | 96.86 | 0 | 2 | 3.084 |
| [BXP-005](cases/BXP-005/recovered.output.express) | Urban transport | SOURCE_TEXT_FALLBACK | 97.16 | 0 | 2 | 3.108 |
| [BXP-006](cases/BXP-006/recovered.output.express) | Road trips | SOURCE_TEXT_FALLBACK | 97.16 | 1 | 2 | 9.414 |
| [BXP-007](cases/BXP-007/recovered.output.express) | Sightseeing | SOURCE_TEXT_FALLBACK | 91.75 | 1 | 1 | 4.186 |
| [BXP-008](cases/BXP-008/recovered.output.express) | Restaurants | SOURCE_TEXT_FALLBACK | 95.25 | 1 | 1 | 6.021 |
| [BXP-009](cases/BXP-009/recovered.output.express) | Accommodation | SOURCE_TEXT_FALLBACK | 97.39 | 0 | 2 | 3.055 |
| [BXP-010](cases/BXP-010/recovered.output.express) | Consumer audio | SOURCE_TEXT_FALLBACK | 96.00 | 0 | 1 | 1.909 |
| [BXP-011](cases/BXP-011/recovered.output.express) | Smartphones | SOURCE_TEXT_FALLBACK | 93.22 | 1 | 1 | 9.898 |
| [BXP-012](cases/BXP-012/recovered.output.express) | Productivity software | SOURCE_TEXT_FALLBACK | 96.42 | 1 | 1 | 8.062 |
| [BXP-013](cases/BXP-013/recovered.output.express) | Bank deposits | SOURCE_TEXT_FALLBACK | 97.13 | 1 | 3 | 7.714 |
| [BXP-014](cases/BXP-014/recovered.output.express) | Foreign exchange | SOURCE_TEXT_FALLBACK | 95.99 | 0 | 1 | 1.890 |
| [BXP-015](cases/BXP-015/recovered.output.express) | Cricket | SOURCE_TEXT_FALLBACK | 97.09 | 0 | 1 | 1.881 |
| [BXP-016](cases/BXP-016/recovered.output.express) | Streaming entertainment | SOURCE_TEXT_FALLBACK | 95.50 | 1 | 1 | 4.067 |
| [BXP-017](cases/BXP-017/recovered.output.express) | Books | SOURCE_TEXT_FALLBACK | 96.51 | 0 | 1 | 1.868 |
| [BXP-018](cases/BXP-018/recovered.output.express) | Music charts | SOURCE_TEXT_FALLBACK | 96.90 | 0 | 1 | 1.893 |
| [BXP-019](cases/BXP-019/recovered.output.express) | Video games | SOURCE_TEXT_FALLBACK | 96.81 | 1 | 1 | 4.180 |
| [BXP-020](cases/BXP-020/recovered.output.express) | Space missions | SOURCE_TEXT_FALLBACK | 97.32 | 0 | 1 | 2.496 |
| [BXP-021](cases/BXP-021/recovered.output.express) | Archaeology | SOURCE_TEXT_FALLBACK | 95.16 | 0 | 2 | 3.023 |
| [BXP-022](cases/BXP-022/recovered.output.express) | Gardening | SOURCE_TEXT_FALLBACK | 96.46 | 0 | 2 | 3.035 |
| [BXP-023](cases/BXP-023/recovered.output.express) | Pet care | SOURCE_TEXT_FALLBACK | 96.36 | 0 | 2 | 3.010 |
| [BXP-024](cases/BXP-024/recovered.output.express) | Waste management | SOURCE_TEXT_FALLBACK | 97.76 | 0 | 3 | 5.430 |
| [BXP-025](cases/BXP-025/recovered.output.express) | Language learning | SOURCE_TEXT_FALLBACK | 94.87 | 1 | 2 | 5.288 |
| [BXP-026](cases/BXP-026/recovered.output.express) | Artificial intelligence | SOURCE_TEXT_FALLBACK | 95.70 | 1 | 2 | 9.164 |
| [BXP-027](cases/BXP-027/recovered.output.express) | Cybersecurity | SOURCE_TEXT_FALLBACK | 78.00 | 0 | 3 | 5.598 |
| [BXP-028](cases/BXP-028/recovered.output.express) | Digital privacy | SOURCE_TEXT_FALLBACK | 97.29 | 0 | 4 | 7.951 |
| [BXP-029](cases/BXP-029/recovered.output.express) | Income tax | SOURCE_TEXT_FALLBACK | 97.02 | 1 | 3 | 7.799 |
| [BXP-030](cases/BXP-030/recovered.output.express) | Retirement savings | SOURCE_TEXT_FALLBACK | 97.19 | 1 | 3 | 7.634 |
| [BXP-031](cases/BXP-031/recovered.output.express) | Property due diligence | SOURCE_TEXT_FALLBACK | 97.71 | 0 | 3 | 6.089 |
| [BXP-032](cases/BXP-032/recovered.output.express) | Health insurance | SOURCE_TEXT_FALLBACK | 94.73 | 1 | 3 | 7.682 |
| [BXP-033](cases/BXP-033/recovered.output.express) | Public health | SOURCE_TEXT_FALLBACK | 92.81 | 0 | 3 | 5.449 |
| [BXP-034](cases/BXP-034/recovered.output.express) | Nutrition | SOURCE_TEXT_FALLBACK | 95.73 | 1 | 2 | 6.006 |
| [BXP-035](cases/BXP-035/recovered.output.express) | Fitness | SOURCE_TEXT_FALLBACK | 96.26 | 0 | 3 | 5.981 |
| [BXP-036](cases/BXP-036/recovered.output.express) | Mental wellbeing | SOURCE_TEXT_FALLBACK | 96.76 | 0 | 3 | 5.437 |
| [BXP-037](cases/BXP-037/recovered.output.express) | School education | SOURCE_TEXT_FALLBACK | 97.78 | 1 | 3 | 7.588 |
| [BXP-038](cases/BXP-038/recovered.output.express) | Careers | SOURCE_TEXT_FALLBACK | 97.51 | 0 | 1 | 1.810 |
| [BXP-039](cases/BXP-039/recovered.output.express) | Small-business compliance | SOURCE_TEXT_FALLBACK | 96.86 | 0 | 3 | 5.545 |
| [BXP-040](cases/BXP-040/recovered.output.express) | Consumer rights | SOURCE_TEXT_FALLBACK | 97.23 | 0 | 3 | 6.623 |
| [BXP-041](cases/BXP-041/recovered.output.express) | International travel rules | SOURCE_TEXT_FALLBACK | 96.22 | 1 | 2 | 11.585 |
| [BXP-042](cases/BXP-042/recovered.output.express) | Electric vehicles | SOURCE_TEXT_FALLBACK | 97.54 | 0 | 3 | 5.302 |
| [BXP-043](cases/BXP-043/recovered.output.express) | Home solar energy | SOURCE_TEXT_FALLBACK | 94.79 | 0 | 3 | 5.355 |
| [BXP-044](cases/BXP-044/recovered.output.express) | Home appliances | SOURCE_TEXT_FALLBACK | 96.41 | 1 | 3 | 7.607 |
| [BXP-045](cases/BXP-045/recovered.output.express) | Urban climate | SOURCE_TEXT_FALLBACK | 93.18 | 1 | 4 | 9.907 |
| [BXP-046](cases/BXP-046/recovered.output.express) | Biotechnology | SOURCE_TEXT_FALLBACK | 97.05 | 0 | 4 | 7.714 |
| [BXP-047](cases/BXP-047/recovered.output.express) | Battery technology | SOURCE_TEXT_FALLBACK | 95.00 | 1 | 2 | 11.107 |
| [BXP-048](cases/BXP-048/recovered.output.express) | History and heritage | SOURCE_TEXT_FALLBACK | 94.72 | 0 | 3 | 5.426 |
| [BXP-049](cases/BXP-049/recovered.output.express) | Traditional arts | SOURCE_TEXT_FALLBACK | 97.43 | 1 | 3 | 13.425 |
| [BXP-050](cases/BXP-050/recovered.output.express) | Ocean and monsoon science | SOURCE_TEXT_FALLBACK | 97.41 | 1 | 3 | 9.715 |

## Validation and artifacts

- **328 JVM tests passed** across 35 suites; zero failures, errors or skips.
- Release AAR 0.3.0 SHA-256: `3d3ef9ca998496e7568458e8d7b38fd6cf66f740f96c42693d33e6642f809e51`.
- Installed reference APK SHA-256: `660c5df75c124b486f393020dee9b34e6a63ac504584c9e837656dcf66dc160e`.
- Device summary: [replay_summary.json](device/replay_summary.json).
- Full compact evidence index: [checksums.json](checksums.json).
