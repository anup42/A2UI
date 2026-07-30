#!/usr/bin/env python3
"""Host CLI for the immutable Single-Codex GenUI Judge Benchmark v2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


DATASET_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATASET_ROOT / "src"))

from pipeline.genui_judge import (  # noqa: E402
    analyze_benchmark,
    append_judgment_pass,
    build_benchmark_selection,
    build_adjudication_packets,
    build_blinded_packets,
    build_repeat_analysis,
    blind_workspace_status,
    finalize_groundtruth,
    import_blind_batch,
    import_blind_adjudication_batch,
    initialize_blind_workspace,
    precompute_selection_v5_4,
    prepare_blind_batch,
    prepare_blind_adjudication_batch,
    protocol_fingerprint,
    rescore_selected_v5_4,
    seal_implementation_provenance,
    sync_frozen_protocol_to_blind_workspace,
    write_selection_coverage_audit,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _completed_packet_ids(root: Path) -> set[str]:
    return {
        str(row["packet_id"])
        for row in _read_jsonl(root / "judgments_by_packet.jsonl")
    }


def _schedule(root: Path) -> list[dict[str, Any]]:
    return sorted(
        _read_jsonl(root / "judge_schedule.jsonl"),
        key=lambda row: int(row["schedule_position"]),
    )


def _status(root: Path) -> dict[str, Any]:
    schedule = _schedule(root)
    complete = _completed_packet_ids(root)
    completed_positions = [
        int(row["schedule_position"])
        for row in schedule
        if row["packet_id"] in complete
    ]
    first_incomplete = next(
        (
            row
            for row in schedule
            if str(row["packet_id"]) not in complete
        ),
        None,
    )
    return {
        "scheduled_packet_count": len(schedule),
        "completed_packet_count": len(complete),
        "remaining_packet_count": len(schedule) - len(complete),
        "pilot_completed": all(
            str(row["packet_id"]) in complete for row in schedule[:32]
        ),
        "frozen_protocol_exists": (
            root / "frozen_judge_protocol.json"
        ).exists(),
        "first_incomplete": first_incomplete,
        "highest_completed_position": (
            max(completed_positions) if completed_positions else None
        ),
        "protocol_fingerprint": protocol_fingerprint(),
    }


def _freeze_protocol(root: Path) -> dict[str, Any]:
    destination = root / "frozen_judge_protocol.json"
    if destination.exists():
        raise FileExistsError(destination)
    status = _status(root)
    if not status["pilot_completed"]:
        raise ValueError("all 32 pilot packets must be complete before freeze")
    source = root / "judge_protocol.json"
    protocol = json.loads(source.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != protocol_fingerprint():
        raise ValueError("pilot protocol fingerprint does not match code")
    value = {
        **protocol,
        "frozen_after_pilot": True,
        "pilot_packet_count": 32,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "judge_instructions_sha256": _hash_file(
            root / "judge_instructions.md"
        ),
        "pilot_judgments_sha256": _hash_file(
            root / "judgments_by_packet.jsonl"
        ),
        "packet_schema_sha256": _hash_file(
            DATASET_ROOT / "schema" / "genui_codex_packet_v2.schema.json"
        ),
        "judgment_schema_sha256": _hash_file(
            DATASET_ROOT
            / "schema"
            / "genui_codex_judgment_v2.schema.json"
        ),
        "formula_fingerprint": hashlib.sha256(
            json.dumps(
                {
                    "formulas": protocol["formulas"],
                    "weights": protocol["weights"],
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    _write_json(destination, value)
    pointer = root / "blind_judge_workspace.json"
    if pointer.exists():
        value["blind_workspace_sync"] = (
            sync_frozen_protocol_to_blind_workspace(root)
        )
    return value


def _prepare_batch(
    root: Path,
    *,
    start: int,
    count: int,
) -> dict[str, Any]:
    if start < 0 or count <= 0 or count > 20:
        raise ValueError("batch start/count must select 1 to 20 packets")
    schedule = _schedule(root)
    if start >= len(schedule):
        raise ValueError("batch start is outside the judging schedule")
    selected = schedule[start : start + count]
    if start >= 32 and not (root / "frozen_judge_protocol.json").exists():
        raise ValueError("freeze the pilot protocol before post-pilot batches")
    protocol_path = (
        root / "frozen_judge_protocol.json"
        if (root / "frozen_judge_protocol.json").exists()
        else root / "judge_protocol.json"
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_fingerprint") != protocol_fingerprint():
        raise ValueError("rubric fingerprint drift")
    instructions_path = root / "judge_instructions.md"
    instructions_hash = _hash_file(instructions_path)
    expected_instructions_hash = protocol.get(
        "judge_instructions_sha256"
    )
    if (
        expected_instructions_hash is not None
        and instructions_hash != expected_instructions_hash
    ):
        raise ValueError("judge instructions drift")
    completed = _completed_packet_ids(root)
    packet_root = root / "packets"
    rows = []
    for row in selected:
        packet_id = str(row["packet_id"])
        rows.append(
            {
                **row,
                "already_complete": packet_id in completed,
                "screenshot_only_packet": str(
                    packet_root / "screenshot_only" / f"{packet_id}.json"
                ),
                "source_conditioned_packet": str(
                    packet_root / "source_conditioned" / f"{packet_id}.json"
                ),
            }
        )
    batch_dir = root / "judge_batches"
    path = batch_dir / f"batch_{start:04d}_{start + len(rows) - 1:04d}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("protocol_fingerprint") != protocol_fingerprint():
            raise ValueError("existing batch has a different rubric")
        return existing
    value = {
        "schema_version": "genui_single_codex_judge_batch.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "start_position": start,
        "end_position_inclusive": start + len(rows) - 1,
        "packet_count": len(rows),
        "protocol_path": str(protocol_path),
        "judge_instructions_path": str(instructions_path),
        "judge_instructions_sha256": instructions_hash,
        "protocol_fingerprint": protocol_fingerprint(),
        "reload_rubric_for_this_batch": True,
        "ordered_passes": ["screenshot_only", "source_conditioned"],
        "packets": rows,
    }
    _write_json(path, value)
    return value


def _write_handoff(root: Path, milestone: int) -> Path:
    if milestone <= 0 or milestone % 80:
        raise ValueError("milestone must be a positive multiple of 80")
    schedule = _schedule(root)
    complete = _completed_packet_ids(root)
    first = max(0, milestone - 80)
    completed_range = schedule[first:min(milestone, len(schedule))]
    missing = [
        row["packet_id"]
        for row in completed_range
        if row["packet_id"] not in complete
    ]
    if missing:
        raise ValueError(
            f"milestone is not complete; {len(missing)} packets are missing"
        )
    remaining = [
        row for row in schedule if row["packet_id"] not in complete
    ]
    handoff_dir = root / "handoffs"
    handoff_dir.mkdir(exist_ok=True)
    path = handoff_dir / f"milestone_{milestone:04d}.md"
    if path.exists():
        return path
    completed_ids = [
        str(row["packet_id"]) for row in completed_range
    ]
    pointer_path = root / "blind_judge_workspace.json"
    blind_workspace: Path | None = None
    if pointer_path.exists():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        blind_workspace = Path(str(pointer["workspace_dir"])).absolute()
    visible_root = blind_workspace or root
    frozen_rubric = (
        visible_root / "rubric" / "frozen_judge_protocol.json"
        if blind_workspace is not None
        else root / "frozen_judge_protocol.json"
    )
    verification = (
        "python dataset/scripts/run_single_codex_judge_v2.py "
        f'blind-status --workspace-dir "{visible_root}"'
        if blind_workspace is not None
        else (
            "python dataset/scripts/run_single_codex_judge_v2.py status "
            f'--benchmark-dir "{root}"'
        )
    )
    text = "\n".join(
        [
            "# Single-Codex GenUI Judge v2 handoff",
            "",
            "Objective: continue blinded, two-pass Codex judging without "
            "changing the frozen rubric or model identifier.",
            "",
            f"Blind workspace: `{visible_root}`",
            f"Frozen rubric: `{frozen_rubric}`",
            f"Completed milestone: {milestone}",
            f"Completed packet IDs: `{','.join(completed_ids)}`",
            "Remaining range: "
            + (
                f"{remaining[0]['schedule_position']}.."
                f"{remaining[-1]['schedule_position']}"
                if remaining
                else "none"
            ),
            "",
            "Verification command:",
            "",
            "```powershell",
            verification,
            "```",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8", newline="\n")
    if blind_workspace is not None:
        blind_path = (
            blind_workspace
            / "handoffs"
            / f"milestone_{milestone:04d}.md"
        )
        blind_path.parent.mkdir(parents=True, exist_ok=True)
        if blind_path.exists():
            if blind_path.read_text(encoding="utf-8") != text:
                raise ValueError(f"blind handoff drift: {blind_path}")
        else:
            blind_path.write_text(text, encoding="utf-8", newline="\n")
        return blind_path
    return path


def _prepare_adjudication_batch(root: Path) -> dict[str, Any]:
    pending = _read_jsonl(root / "pending_adjudications.jsonl")
    completed = _completed_packet_ids(root)
    packets = [
        {
            "packet_id": str(row["packet_id"]),
            "screenshot_only_packet": str(
                root
                / "packets"
                / "screenshot_only"
                / f"{row['packet_id']}.json"
            ),
            "source_conditioned_packet": str(
                root
                / "packets"
                / "source_conditioned"
                / f"{row['packet_id']}.json"
            ),
        }
        for row in pending
        if str(row["packet_id"]) not in completed
    ]
    value = {
        "schema_version": "genui_single_codex_judge_batch.v2",
        "protocol_fingerprint": protocol_fingerprint(),
        "packet_count": len(packets),
        "packets": packets,
    }
    path = root / "judge_batches" / "batch_extra.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise ValueError("adjudication batch changed after creation")
        return existing
    _write_json(path, value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    select = commands.add_parser("select")
    select.add_argument("--source-dir", required=True)
    select.add_argument("--output-dir", required=True)
    select.add_argument("--metric-scores")
    select.add_argument("--prior-v1")
    select.add_argument(
        "--seed", default="a2ui-single-codex-judge-960-v2"
    )

    precompute = commands.add_parser("precompute-selection-v5-4")
    precompute.add_argument("--source-dir", required=True)
    precompute.add_argument("--output-dir", required=True)
    precompute.add_argument("--workers", type=int, default=8)
    precompute.add_argument("--batch-size", type=int, default=256)

    audit_selection = commands.add_parser("audit-selection")
    audit_selection.add_argument("--source-dir", required=True)
    audit_selection.add_argument("--benchmark-dir", required=True)

    packets = commands.add_parser("build-packets")
    packets.add_argument("--benchmark-dir", required=True)
    packets.add_argument(
        "--allow-incomplete-capture", action="store_true"
    )

    blind_init = commands.add_parser("init-blind-workspace")
    blind_init.add_argument("--benchmark-dir", required=True)
    blind_init.add_argument("--workspace-dir", required=True)

    blind_batch = commands.add_parser("prepare-blind-batch")
    blind_batch.add_argument("--benchmark-dir", required=True)
    blind_batch.add_argument("--start", type=int, required=True)
    blind_batch.add_argument("--count", type=int, default=20)
    blind_batch.add_argument(
        "--pass-type",
        choices=("screenshot_only", "source_conditioned"),
        required=True,
    )

    blind_import = commands.add_parser("import-blind-batch")
    blind_import.add_argument("--benchmark-dir", required=True)
    blind_import.add_argument("--start", type=int, required=True)
    blind_import.add_argument("--count", type=int, default=20)
    blind_import.add_argument(
        "--pass-type",
        choices=("screenshot_only", "source_conditioned"),
        required=True,
    )
    blind_import.add_argument("--expected-task-id")

    blind_adjudication = commands.add_parser(
        "prepare-blind-adjudication-batch"
    )
    blind_adjudication.add_argument("--benchmark-dir", required=True)
    blind_adjudication.add_argument("--start", type=int, required=True)
    blind_adjudication.add_argument("--count", type=int, default=20)
    blind_adjudication.add_argument(
        "--pass-type",
        choices=("screenshot_only", "source_conditioned"),
        required=True,
    )

    blind_adjudication_import = commands.add_parser(
        "import-blind-adjudication-batch"
    )
    blind_adjudication_import.add_argument(
        "--benchmark-dir", required=True
    )
    blind_adjudication_import.add_argument(
        "--start", type=int, required=True
    )
    blind_adjudication_import.add_argument(
        "--count", type=int, default=20
    )
    blind_adjudication_import.add_argument(
        "--pass-type",
        choices=("screenshot_only", "source_conditioned"),
        required=True,
    )
    blind_adjudication_import.add_argument("--expected-task-id")

    blind_status = commands.add_parser("blind-status")
    blind_status.add_argument("--workspace-dir", required=True)

    seal = commands.add_parser("seal-implementation")
    seal.add_argument("--benchmark-dir", required=True)

    record = commands.add_parser("record-pass")
    record.add_argument("--benchmark-dir", required=True)
    record.add_argument("--judgment-json", required=True)

    record_batch = commands.add_parser("record-batch")
    record_batch.add_argument("--benchmark-dir", required=True)
    record_batch.add_argument("--judgments-jsonl", required=True)

    status = commands.add_parser("status")
    status.add_argument("--benchmark-dir", required=True)

    freeze = commands.add_parser("freeze-protocol")
    freeze.add_argument("--benchmark-dir", required=True)

    batch = commands.add_parser("prepare-batch")
    batch.add_argument("--benchmark-dir", required=True)
    batch.add_argument("--start", type=int, required=True)
    batch.add_argument("--count", type=int, default=20)

    repeat = commands.add_parser("repeat-analysis")
    repeat.add_argument("--benchmark-dir", required=True)

    adjudication = commands.add_parser("prepare-adjudication-batch")
    adjudication.add_argument("--benchmark-dir", required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument("--benchmark-dir", required=True)
    finalize.add_argument("--allow-incomplete", action="store_true")

    score = commands.add_parser("rescore-v5-4")
    score.add_argument("--benchmark-dir", required=True)
    score.add_argument(
        "--allow-incomplete-native-evidence", action="store_true"
    )

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--benchmark-dir", required=True)
    analyze.add_argument("--bootstrap-iterations", type=int, default=2000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260729)

    handoff = commands.add_parser("write-handoff")
    handoff.add_argument("--benchmark-dir", required=True)
    handoff.add_argument("--milestone", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "select":
        result = build_benchmark_selection(
            args.source_dir,
            args.output_dir,
            metric_scores_path=args.metric_scores,
            prior_v1_path=args.prior_v1,
            seed=args.seed,
        )
    elif args.command == "precompute-selection-v5-4":
        result = precompute_selection_v5_4(
            args.source_dir,
            args.output_dir,
            workers=args.workers,
            batch_size=args.batch_size,
        )
    elif args.command == "audit-selection":
        result = write_selection_coverage_audit(
            args.source_dir,
            args.benchmark_dir,
        )
    elif args.command == "blind-status":
        result = blind_workspace_status(args.workspace_dir)
    else:
        root = Path(args.benchmark_dir).resolve()
        if args.command == "build-packets":
            result = build_blinded_packets(
                root,
                allow_incomplete_capture=args.allow_incomplete_capture,
            )
        elif args.command == "seal-implementation":
            result = seal_implementation_provenance(root)
        elif args.command == "init-blind-workspace":
            result = initialize_blind_workspace(
                root,
                args.workspace_dir,
            )
        elif args.command == "prepare-blind-batch":
            result = prepare_blind_batch(
                root,
                start=args.start,
                count=args.count,
                pass_type=args.pass_type,
            )
        elif args.command == "import-blind-batch":
            result = import_blind_batch(
                root,
                start=args.start,
                count=args.count,
                pass_type=args.pass_type,
                expected_task_id=args.expected_task_id,
            )
        elif args.command == "prepare-blind-adjudication-batch":
            result = prepare_blind_adjudication_batch(
                root,
                start=args.start,
                count=args.count,
                pass_type=args.pass_type,
            )
        elif args.command == "import-blind-adjudication-batch":
            result = import_blind_adjudication_batch(
                root,
                start=args.start,
                count=args.count,
                pass_type=args.pass_type,
                expected_task_id=args.expected_task_id,
            )
        elif args.command == "record-pass":
            judgment = json.loads(
                Path(args.judgment_json).read_text(encoding="utf-8")
            )
            result = append_judgment_pass(root, judgment)
        elif args.command == "record-batch":
            judgments = _read_jsonl(
                Path(args.judgments_jsonl).resolve()
            )
            saved = [
                append_judgment_pass(root, judgment)
                for judgment in judgments
            ]
            result = {
                "saved_pass_count": len(saved),
                "completed_packet_count": (
                    saved[-1]["completed_packet_count"] if saved else 0
                ),
                "last_packet": (
                    saved[-1]["saved_pass"]["packet_id"]
                    if saved
                    else None
                ),
            }
        elif args.command == "status":
            result = _status(root)
        elif args.command == "freeze-protocol":
            result = _freeze_protocol(root)
        elif args.command == "prepare-batch":
            result = _prepare_batch(
                root, start=args.start, count=args.count
            )
        elif args.command == "repeat-analysis":
            repeats, pending = build_repeat_analysis(root)
            result = {
                "repeat_pairs_complete": len(repeats),
                "pending_adjudications": len(pending),
                "adjudication_packets": build_adjudication_packets(root),
            }
        elif args.command == "prepare-adjudication-batch":
            result = _prepare_adjudication_batch(root)
        elif args.command == "finalize":
            rows = finalize_groundtruth(
                root, require_complete=not args.allow_incomplete
            )
            result = {"finalized_count": len(rows)}
        elif args.command == "rescore-v5-4":
            result = rescore_selected_v5_4(
                root,
                require_complete_native_evidence=(
                    not args.allow_incomplete_native_evidence
                ),
            )
        elif args.command == "analyze":
            result = analyze_benchmark(
                root,
                bootstrap_iterations=args.bootstrap_iterations,
                bootstrap_seed=args.bootstrap_seed,
            )
        elif args.command == "write-handoff":
            result = {"handoff": str(_write_handoff(root, args.milestone))}
        else:
            raise AssertionError(args.command)
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
