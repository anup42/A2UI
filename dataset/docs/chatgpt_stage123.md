# Generating fresh three-stage examples with ChatGPT

Run from the repository root with the existing GenUI ChatGPT sign-in. The runner checks the installed Codex CLI, account, model and reasoning effort at startup. It uses GPT-6.2 Sol at medium reasoning for queries, source answers and A2UI Express targets.

```powershell
python -B dataset/scripts/run_chatgpt_stage123.py --output-dir C:/genui/runs/sol10k --target 10000 --workers 8 --shard-size 32
```

Start the read-only dashboard in a separate terminal.

```powershell
python -B dataset/scripts/chatgpt_stage123_status.py --output-dir C:/genui/runs/sol10k --port 8784
```

Open [the local dashboard](http://127.0.0.1:8784). The target counts Stage 3 submissions. Eight workers run by default. Use `--workers 32` for up to 32 concurrent subscription turns. Each group finishes all three stages before the next group starts. If some queries or source answers fail screening, up to 5% additional source slots can replace them within the same Stage 3 target. Final training admission depends on each generated label passing validation.

Stage 1 varies domains and task shapes, with extra entity-list examples, numerical questions with independently computed answers, and references with supplied destinations. Stage 2 checks source quality, required results and URL provenance. A confirmed quality failure can receive one corrective attempt. Stage 3 runs the standard Express parser, v5.4 validation, reference binding and fidelity checks. Golden and Bixby sources are excluded. Downstream accuracy requires a separate training run and held-out evaluation.

The source journal is `sources.sqlite`. Each completed group publishes an integrity-checked `evidence.tar.gz` containing the sources, receipts, decisions and the accepted training candidates in `p/pilot_accepted.jsonl` when generation took place. Stage 3 dashboard totals update after each group is validated. Exact and close lexical query duplicates and exact source duplicates are screened across this run.

Use the same command, output directory, seed and code snapshot to continue an interrupted run. Completed work is reused. An uncertain subscription turn requires review before replay. Keep at least 512 MiB free for the next group. For a long run, use an isolated checkout so subsequent edits cannot change its evaluator.

### Switch an existing run to consecutive stages

At a checkpoint with no active requests or unfinished label groups, run the phased coordinator against the same output directory. Use the same model, source generator and evaluator snapshot. The coordinator verifies the original plan and completed archives. It reuses saved queries and answers, and finishes the initial groups up to the exact transition count.

```powershell
python -B dataset/scripts/run_chatgpt_stage123_phased.py --output-dir C:/genui/runs/sol10k --switch-after 1000 --workers 32 --label-size 256
```

For a target of 10,000 label submissions, the coordinator then generates 9,000 primary queries. It prepares a small pool of reserve query slots based on observed source losses, bounded by the original replacement allowance. Stage 2 draws on those reserves only as needed to reach 9,000 eligible source answers. Stage 3 starts after the full answer set passes source screening. It submits 9,000 labels in consecutive groups of up to 256 for validation and archival. Held labels count toward submissions.

The phased Stage 2 prompt puts the common instructions first and the individual task last. Cache reuse depends on the provider. The existing dashboard reads the same progress file, and label totals update after each archived group. Use the phased command for subsequent resumes. The transition counts are in `phased_sources.json`, and the completed query and answer phases have separate integrity receipts.

Phased source turns have a 15-minute completion deadline. When an individual query or answer reaches it, the runner journals the skipped request and holds its slots. Other workers continue. Interrupted requests whose delivery is uncertain require review.

For a paused bulk Stage 1 run, use the same output directory and settings. Offline reconciliation checks the saved replies against their original prompts, applies the original query and duplicate checks, and records each recovered or held slot. Supply one `--skip-unresolved-query` for each reviewed request to skip. The command writes a `phased_query_recovery_*.json` receipt and exits. Then use the normal phased command above to continue generation.

```powershell
python -B dataset/scripts/run_chatgpt_stage123_phased.py --output-dir C:/genui/runs/sol10k --switch-after 1000 --workers 32 --label-size 256 --reconcile-queries --skip-unresolved-query q-1666-4-0 --skip-unresolved-query q-4594-4-0
```

The two keys in this example belong to the November 22 run. Use the keys reported by the journal for another run. Reconciliation keeps the original schedule and binds the timeout-aware runtime in a separate receipt for a run started with the earlier phased scheduler.
