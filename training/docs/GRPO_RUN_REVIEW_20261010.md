# E2B GRPO: evidence review and corrective experiment controls

Reviewed run: `e2b_v11_grpo_20261009_f9e1e784_full200_v1`.
Run revision: `f9e1e7845ddf65dc807e47166c2e368466dbee43`.
Code reviewed against: `7898fd48cb7cbdb457477637152ab0ec62e68f81`, branch `new_ir_changes_20260331`.
Uploaded archive SHA-256: `9373dbe047dfd6c84bef35960f987f5f664a2f4c14baf5fe724638b740b596ad`.
All 142 entries in the archive's SHA256SUMS.txt were verified. No model was trained or run during this review.

## Assessment

GRPO is updating the QAT LoRA adapter; this is not the earlier zero-learning failure. However, the evidence does not establish a reliable improvement over SFT. Severe length filtering, unstable development quality, short audit coverage, and costly validation are the first issues to fix. Do not extend the same run blindly or deploy the final step-200 adapter.

### Development results

The following scores use all 32 Golden32 rows. There are only 31 unique sources; one source is deliberately repeated. Golden32 is the checkpoint-selection development set, not an independent test set.

| Checkpoint | Strict valid | V5.4 quality (0–100) | CAP | Fully root reachable |
| --- | ---: | ---: | ---: | ---: |
| Saved SFT starting comparison | 24/32 | 51.44 | 0.5356 | 43.75% |
| GRPO 50 | 27/32 | 45.18 | 0.4634 | 25.00% |
| GRPO 100 | 19/32 | 38.00 | 0.3922 | 28.13% |
| GRPO 150, selected | 24/32 | 53.70 | 0.5663 | 50.00% |
| GRPO 200 | 21/32 | 40.01 | 0.4138 | 25.00% |

On unique sources, selected-step quality is 55.4317 versus 53.1038 for SFT: +2.3279 points. Nine sources improve, nine regress, and thirteen are unchanged. A paired, source-level bootstrap (20,000 resamples, seed 42) gives a descriptive 95% interval of [-9.1193, +13.7663] points. Source-context, prompt-token, and metric-fingerprint bindings match in this comparison. This interval includes zero and is additionally selection-biased because this cohort selected step 150. It is not confirmatory evidence. The saved SFT outputs are not a fresh step-zero evaluation inside this GRPO run.

Selected-checkpoint final cohorts: Golden35 is 24/35 strict-valid (68.57%), quality 53.4109; Bixby50 is 45/50 (90%), quality 86.1455. Matched SFT baselines for these cohorts are absent from the bundle. Bixby50 is source-only: its source-fidelity score is useful, but reference-IR metrics are not directly comparable. Neither result proves an improvement over SFT.

## What works

All four rank health traces contain 200 updates: 19 warming-up records and 181 passed-window records each, with no reported health issues. Gradients and sampled adapter changes are nonzero and no nonfinite parameters/gradients are reported. Correct within-prompt reward-diversity auditing is already present at the run revision. Keep that fix and the isolated sampling configuration, EOS/closing-tag handling, masked synthetic terminal token, QAT wrappers, and seed/adapter/export provenance checks.

The checkpoint selector correctly retained step 150 rather than the worse final state. Merge/export receipts report successful package construction; Android/native evaluation was explicitly deferred. A successful exporter receipt is not proof of on-device generation quality or numerical parity.

## Confirmed issues

### 1. Most training examples were excluded by the reference length limit

`run/data_audit.json -> prepared_counts` records 10,149/200,034 training rows retained (5.0736%) and 215/4,084 validation rows (5.2644%). All 189,885 and 3,869 respective exclusions at this stage are `sequence_too_long`. The earlier structural-filter counts are different and must not be mistaken for the rows actually trained on.

The run uses a 4,096-token total reference-sequence limit although the original SFT recipe was 6,144 tokens. The pipeline also mistakenly uses that total-sequence limit as GRPO's prompt limit and adds the completion budget to it in its context check. Reference sequence = prompt + reference answer; rollout length = prompt + sampled answer + masked sentinel. These are separate constraints.

The patch separates these budgets, defaults the GRPO preparation sequence to 6,144, and keeps prompt/output budgets at 5,120/2,048: the rollout context bound is 7,169, within E2B's 8,192. It fails preparation below 50% train/validation retention by default. This threshold is an engineering safeguard, not an empirically optimal curriculum. Actual retention at 6,144 must be measured on the GPU host; the excluded/raw datasets are not in this archive. No rows are silently truncated or altered.

### 2. Optimizer health does not imply semantic quality

Step 50 increases validity but lowers quality and reachability. Step 200 is worse than both step 150 and the SFT comparison. Training reward rises in later windows, but that alone does not establish generalization or reward hacking. Low-gradient or NaN fixes are not the primary remaining problem.

The patch generates a fresh QAT-on Golden32 baseline before the first optimizer update, using the shared evaluator and preserving training RNG states. A new synchronized development callback supports patience-based stopping. Automatic pipeline promotion requires >0.5 unique-source quality points of gain and no drop in unique-source strict validity, CAP, or full root reachability. The gate is hash-bound to the configuration, selected adapter, baseline, and selected metrics. It is a conservative development screen, not a significance test. The existing scalar checkpoint selector remains unchanged: if its best checkpoint fails a structural guard, promotion stops rather than silently selecting another policy.

### 3. Wrapper failures have different causes

At step 150, eight invalid rows comprise four outputs that hit 2,048 tokens, two duplicate-ID failures, and two occurrences of the same source with a leading `Hungary` prefix before an otherwise complete envelope. At step 200, seven rows hit the limit and four have duplicate IDs. Golden35 has six limit failures, three missing openers, and two duplicate IDs. Bixby50 has three missing-openers/early-EOS cases, one contaminated prefix, and one syntax/string failure; none are token-limit failures.

Some capped outputs contain repeated component/child sequences. A larger cap alone may prolong loops; it cannot repair duplicate IDs, missing openers, or contaminated prefixes. Do not blacklist `Hungary`, strip arbitrary prose, append a fabricated close tag, or reward repaired output. Probe first-token behavior and compare QAT-on/off with identical prompts on the host. The bundle's statement that q_012038 fails at every checkpoint is incorrect: it is valid in the saved SFT baseline.

### 4. Audit interpretation and coverage were incomplete

The 1,024 startup reward records (256/rank) cover optimizer steps 0–31 only. Actual details live at `breakdown.qat_reward`, not at the top level. There are 958/1,024 strict-valid candidates (93.5547%); the bundled analysis reporting unknown strict validity/empty reasons missed that nested structure. Common recorded penalties include content-unit fidelity, output precision, visible-content fidelity, numbers, and semantic-role coverage. These are startup observations, not a late-run census.

The patch adds a complete first-local-batch audit every 25 steps on every rank, separately for train/eval, with candidate IDs, termination reasons, and unchanged nested scorer evidence. Existing startup audits remain intact. `review_grpo_run.py` reads the actual schema and separates failure categories. `completions/clipped_ratio` means completion truncation, not optimizer/PPO ratio clipping; the archive README confuses these.

### 5. Full stochastic validation consumes substantial time

The run log's final 215-row RL validation pass takes about 18,850 seconds, versus about 33 hours for the full training job including evaluation pauses. Repeating full sampled validation is expensive and is not the greedy deployment selector.

The new pipeline defaults periodic RL validation to 32 deterministic, source-deduplicated rows, stratified by intent and prompt-length bucket. Original validation files remain unchanged and selected indices/source IDs/full-file hashes are recorded. `validation_max_rows=0` restores full periodic validation. Golden32 selection and final holdouts are not reduced. This reduces evaluated row count; no wall-clock speedup has been benchmarked for the patch.

## Follow-up experiments, not silently enabled fixes

Start a fresh, matched pilot from the original verified SFT adapter. Keep `qat_source_fidelity_v1`, temperature 0.8, beta=0, group size four, and the QAT/export contract unchanged. Compare 1e-6 versus 5e-7 learning rate on the same prepared data and fixed development protocol; do not mix a data change and LR change into an unsupported causal attribution.

Only after fixing coverage, consider completion-only SFT rehearsal or a genuine frozen QAT SFT-reference KL ablation to control drift. Disabling the active adapter does not produce the required QAT SFT reference. Neither objective change is implemented here. Calibrate any new reward on training-only candidate contrasts (numbers, tables, references, omissions, duplicate IDs), not by tuning repeatedly on Golden35/Bixby50.

For long cases, compare both baseline and candidate at a separately declared 2,816-token cap (5,120+2,816+1=7,937), keeping primary scores at the original cap. Determine whether extra tokens finish faithful UIs or extend repetition. Longer-term grammar/ID-constrained generation requires matching the likelihood calculation to constrained sampling. Compact source/table references require a versioned preparation–SFT–renderer contract, not a GRPO-only target rewrite.

## Reproduction and validation

From repository root:

```sh
python training/scripts/review_grpo_run.py /path/to/extracted_bundle --output /tmp/grpo_review.json
python -m pytest -q training/tests/test_grpo_experiment.py
python -m compileall -q training
```

Review environment: 23 new CPU tests passed; Python compilation and whitespace-diff checks passed. Tests include actual pure pipeline-config assembly, artifact parsing, retention/budget boundaries, deterministic validation selection, gate integrity, and simulated callback/RNG/audit events. They do not execute real TRL/Transformers training or GPU/DDP generation. Full repository regression tests, pinned-dependency checks, real four-GPU callback/collective validation, peak-memory checks, learning evaluation, and native QA remain for the training host. No post-fix accuracy gain is claimed.

Read `training/docs/CODEX_GRPO_NEXT_RUN.md` before launching.
