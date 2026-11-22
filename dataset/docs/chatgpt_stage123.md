# Generating fresh three-stage examples with ChatGPT

Run from the repository root with the existing GenUI ChatGPT sign-in. The runner checks the installed Codex CLI, account, model and reasoning effort at startup. It uses GPT-6.2 Sol at medium reasoning for queries, source answers and A2UI Express targets.

```powershell
python -B dataset/scripts/run_chatgpt_stage123.py --output-dir C:/genui/runs/sol10k --target 10000 --workers 8 --shard-size 32
```

Start the read-only dashboard in a separate terminal.

```powershell
python -B dataset/scripts/chatgpt_stage123_status.py --output-dir C:/genui/runs/sol10k --port 8784
```

Open [the local dashboard](http://127.0.0.1:8784). The target counts Stage 3 submissions. Up to eight subscription turns run concurrently. Each group finishes all three stages before the next group starts. If some queries or source answers fail screening, up to 5% additional source slots can replace them within the same Stage 3 target. Final training admission depends on each generated label passing validation.

Stage 1 varies domains and task shapes, with extra entity-list examples, numerical questions with independently computed answers, and references with supplied destinations. Stage 2 checks source quality, required results and URL provenance. A confirmed quality failure can receive one corrective attempt. Stage 3 runs the standard Express parser, v5.4 validation, reference binding and fidelity checks. Golden and Bixby sources are excluded. Downstream accuracy requires a separate training run and held-out evaluation.

The source journal is `sources.sqlite`. Each completed group publishes an integrity-checked `evidence.tar.gz` containing the sources, receipts, decisions and the accepted training candidates in `p/pilot_accepted.jsonl` when generation took place. Stage 3 dashboard totals update after each group is validated. Exact and close lexical query duplicates and exact source duplicates are screened across this run.

Use the same command, output directory, seed and code snapshot to continue an interrupted run. Completed work is reused. An uncertain subscription turn requires review before replay. Keep at least 512 MiB free for the next group. For a long run, use an isolated checkout so subsequent edits cannot change its evaluator.
