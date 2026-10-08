# Fold7 visual review progress

Snapshot: 2026-10-08T22:24:04.851521+00:00. Denominator: **50**.

| Status | Cases |
|---|---:|
| visually accepted | 23 |
| renderable needs work | 0 |
| renderable unreviewed | 22 |
| no renderable output | 5 |
| generation pending | 0 |

Current-model renderable documents: **45/50**. This is preparation coverage, not visual acceptance.

Current cases without renderer input: BXP-027, BXP-028, BXP-033, BXP-042, BXP-043. Guard-cutoff records are not labelled model hallucinations.

Acceptance covers rendering of the selected document. Model/source gaps remain separate. Fidelity warning counts are diagnostic, especially for state-bound tables.

Confirmed model findings are recorded only from explicit reviewer observations in [review_findings.json](review_findings.json); all original notes remain linked below. The historical train-fixture hash failure is a separate validation finding.

| Case | Domain | Visual status | Current evidence |
|---|---|---|
| BXP-001 | Weather | accepted | [Review](BXP-001/review.json) · [Prepared document](../20261008_streaming_rendering/native/fold7_fp16_mtp_20261008/BXP-001_on/final.express) |
| BXP-002 | Air quality | accepted | [Review](BXP-002/review.json) · [Prepared document](native_generation/visual_002_current_20261009/BXP-002/a2ui.json) |
| BXP-003 | Rail travel | accepted | [Review](BXP-003/review.json) · [Prepared document](../20261008_streaming_rendering/native/fold7_fp16_mtp_20261008/BXP-003_on/final.express) |
| BXP-004 | Airline baggage | accepted | [Review](BXP-004/review.json) · [Prepared document](../20261008_streaming_rendering/native/fold7_fp16_mtp_20261008/BXP-004_on/final.express) |
| BXP-005 | Urban transport | accepted | [Review](BXP-005/review.json) · [Prepared document](native_generation/visual_prep_005_010_20261009/BXP-005/a2ui.json) |
| BXP-006 | Road trips | accepted | [Review](BXP-006/review.json) · [Prepared document](native_generation/visual_prep_005_010_20261009/BXP-006/a2ui.json) |
| BXP-007 | Sightseeing | accepted | [Review](BXP-007/review.json) · [Prepared document](native_generation/visual_prep_005_010_20261009/BXP-007/a2ui.json) |
| BXP-008 | Restaurants | accepted | [Review](BXP-008/review.json) · [Prepared document](../20261008_streaming_rendering/native/fold7_fp16_mtp_20261008/BXP-008_on/final.a2ui.json) |
| BXP-009 | Accommodation | accepted | [Review](BXP-009/review.json) · [Prepared document](native_generation/visual_prep_005_010_20261009/BXP-009/a2ui.json) |
| BXP-010 | Consumer audio | accepted | [Review](BXP-010/review.json) · [Prepared document](native_generation/visual_prep_005_010_20261009/BXP-010/a2ui.json) |
| BXP-011 | Smartphones | accepted | [Review](BXP-011/review.json) · [Prepared document](../20261008_streaming_rendering/native/fold7_fp16_mtp_20261008/BXP-011_on/final.a2ui.json) |
| BXP-012 | Productivity software | accepted | [Review](BXP-012/review.json) · [Prepared document](native_generation/visual_prep_012_015_20261009/BXP-012/a2ui.json) |
| BXP-013 | Bank deposits | accepted | [Review](BXP-013/review.json) · [Prepared document](native_generation/visual_prep_012_015_20261009/BXP-013/a2ui.json) |
| BXP-014 | Foreign exchange | accepted | [Review](BXP-014/review.json) · [Prepared document](native_generation/visual_prep_012_015_20261009/BXP-014/a2ui.json) |
| BXP-015 | Cricket | accepted | [Review](BXP-015/review.json) · [Prepared document](native_generation/visual_prep_012_015_20261009/BXP-015/a2ui.json) |
| BXP-016 | Streaming entertainment | accepted | [Review](BXP-016/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-016/a2ui.json) |
| BXP-017 | Books | accepted | [Review](BXP-017/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-017/a2ui.json) |
| BXP-018 | Music charts | accepted | [Review](BXP-018/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-018/a2ui.json) · Preserved failed before baseline: allTableColumnsObserved, renderedWithoutIssues |
| BXP-019 | Video games | accepted | [Review](BXP-019/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-019/a2ui.json) |
| BXP-020 | Space missions | accepted | [Review](BXP-020/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-020/a2ui.json) |
| BXP-021 | Archaeology | accepted | [Review](BXP-021/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-021/a2ui.json) |
| BXP-022 | Gardening | accepted | [Review](BXP-022/review.json) · [Prepared document](native_generation/visual_prep_016_022_20261009/BXP-022/a2ui.json) |
| BXP-023 | Pet care | accepted | [Review](BXP-023/review.json) · [Prepared document](native_generation/visual_prep_023_027_20261009/BXP-023/a2ui.json) |
| BXP-024 | Waste management | pending | Not reviewed · [Prepared document](native_generation/visual_prep_023_027_20261009/BXP-024/a2ui.json) |
| BXP-025 | Language learning | pending | Not reviewed · [Prepared document](native_generation/visual_prep_023_027_20261009/BXP-025/a2ui.json) |
| BXP-026 | Artificial intelligence | pending | Not reviewed · [Prepared document](native_generation/visual_prep_023_027_20261009/BXP-026/a2ui.json) |
| BXP-027 | Cybersecurity | not reviewable | Not reviewed · [Generation result](native_generation/visual_prep_023_027_20261009/BXP-027/result.json) |
| BXP-028 | Digital privacy | not reviewable | Not reviewed · [Generation result](native_generation/visual_prep_028_034_20261009/BXP-028/result.json) |
| BXP-029 | Income tax | pending | Not reviewed · [Prepared document](native_generation/visual_prep_028_034_20261009/BXP-029/a2ui.json) |
| BXP-030 | Retirement savings | pending | Not reviewed · [Prepared document](native_generation/visual_prep_028_034_20261009/BXP-030/a2ui.json) |
| BXP-031 | Property due diligence | pending | Not reviewed · [Prepared document](native_generation/visual_prep_028_034_20261009/BXP-031/a2ui.json) |
| BXP-032 | Health insurance | pending | Not reviewed · [Prepared document](native_generation/visual_prep_028_034_20261009/BXP-032/a2ui.json) |
| BXP-033 | Public health | not reviewable | Not reviewed · [Generation result](native_generation/visual_prep_028_034_20261009/BXP-033/result.json) |
| BXP-034 | Nutrition | pending | Not reviewed · [Prepared document](native_generation/visual_prep_028_034_20261009/BXP-034/a2ui.json) |
| BXP-035 | Fitness | pending | Not reviewed · [Prepared document](native_generation/visual_prep_035_039_20261009/BXP-035/a2ui.json) |
| BXP-036 | Mental wellbeing | pending | Not reviewed · [Prepared document](native_generation/visual_prep_035_039_20261009/BXP-036/a2ui.json) |
| BXP-037 | School education | pending | Not reviewed · [Prepared document](native_generation/visual_prep_035_039_20261009/BXP-037/a2ui.json) |
| BXP-038 | Careers | pending | Not reviewed · [Prepared document](native_generation/visual_prep_035_039_20261009/BXP-038/a2ui.json) |
| BXP-039 | Small-business compliance | pending | Not reviewed · [Prepared document](native_generation/visual_prep_035_039_20261009/BXP-039/a2ui.json) |
| BXP-040 | Consumer rights | pending | Not reviewed · [Prepared document](native_generation/visual_prep_040_044_20261009/BXP-040/a2ui.json) |
| BXP-041 | International travel rules | pending | Not reviewed · [Prepared document](native_generation/visual_prep_040_044_20261009/BXP-041/a2ui.json) |
| BXP-042 | Electric vehicles | not reviewable | Not reviewed · [Generation result](native_generation/visual_prep_040_044_20261009/BXP-042/result.json) |
| BXP-043 | Home solar energy | not reviewable | Not reviewed · [Generation result](native_generation/visual_prep_040_044_20261009/BXP-043/result.json) |
| BXP-044 | Home appliances | pending | Not reviewed · [Prepared document](native_generation/visual_prep_040_044_20261009/BXP-044/a2ui.json) |
| BXP-045 | Urban climate | pending | Not reviewed · [Prepared document](native_generation/visual_prep_045_050_20261009/BXP-045/a2ui.json) |
| BXP-046 | Biotechnology | pending | Not reviewed · [Prepared document](native_generation/visual_prep_045_050_20261009/BXP-046/a2ui.json) |
| BXP-047 | Battery technology | pending | Not reviewed · [Prepared document](native_generation/visual_prep_045_050_20261009/BXP-047/a2ui.json) |
| BXP-048 | History and heritage | pending | Not reviewed · [Prepared document](native_generation/visual_prep_045_050_20261009/BXP-048/a2ui.json) |
| BXP-049 | Traditional arts | pending | Not reviewed · [Prepared document](native_generation/visual_prep_045_050_20261009/BXP-049/a2ui.json) |
| BXP-050 | Ocean and monsoon science | pending | Not reviewed · [Prepared document](native_generation/visual_prep_045_050_20261009/BXP-050/a2ui.json) |

Explicit before-render waivers preserve the failed baseline receipt and source hashes. Only the two named render/column checks may be waived for comparison; every after check must pass. These baseline diagnostics are not marked as passing checks or stale accepted evidence.

Refresh after later reviews with:

```powershell
python GenUICraft/validation/20261009_fold7_visual_review/refresh_progress.py
```

The script reads existing reviews/receipts/native-generation inventory and writes only these progress reports and the source listing. It performs no model inference, device action, SDK build, or edits to review annotations/generated UI.
