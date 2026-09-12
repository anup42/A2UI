"""Checked, opt-in train-to-dual-Golden workflow; no cloud data generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable

from ir_training.common.config import repo_root
from ir_training.common.jsonl import read_jsonl, write_jsonl

GOLDENS = {
    "golden32": ("training/data/eval/golden32_archive_repeat_v1/golden32.jsonl", 32,
                 "8c7357103e6ea52d99d66430dd4b93242e1c4a4f0bf02f2fcfdf2a2e9c48de4c"),
    "golden35": ("training/data/eval/golden35_v1/golden35.jsonl", 35,
                 "8fec7fde8c31634f66e4c77dd0ca7a0e3b37d36e453398f167fd30b9604c2d20"),
}
GOLDEN_MANIFEST_SHA256 = {
    "golden32": "2b62797db7fd267c3f75e8ab1a7cb1a450d601df3f7785f5b78a62fe2657caa8",
    "golden35": "1d468e05015744b8165c03e92ed009e03aac845513471b8f0b99cbe9d93939fb",
}


@dataclass(frozen=True)
class GoldenTrainingOptions:
    model_dir: Path
    output_dir: Path
    profile: str = "e2b"
    source_run_dir: Path | None = None
    input_dir: Path | None = None
    devices: str = "auto"
    epochs: float = 1.0
    steps: int | None = None
    eval_steps: int = 500
    golden_every_steps: int = 1000
    max_seq_length: int = 4096
    max_input_tokens: int = 4096
    max_new_tokens: int = 2048
    tensorboard_root: str = "/tensorboard"
    microbatch: int | None = None
    effective_batch: int | None = None
    dataloader_workers: int | None = None
    qat: bool = False
    seed: int = 42


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _options(options: GoldenTrainingOptions) -> dict[str, Any]:
    result = asdict(options)
    for name in ("model_dir", "output_dir", "source_run_dir", "input_dir"):
        if result[name] is not None:
            result[name] = str(Path(result[name]).expanduser().resolve())
    if not result["input_dir"] and not result["source_run_dir"]:
        result["source_run_dir"] = str(repo_root() / "dataset/data/runs/dataset_v1")
    return result


def build_plan(options: GoldenTrainingOptions) -> dict[str, Any]:
    """Read-only plan, requiring no CUDA, model loading or tokenizer download."""
    values = _options(options)
    if options.profile not in {"e2b", "270m"} or (options.qat and options.profile != "270m"):
        raise ValueError("Profiles are e2b dense LoRA and 270m full SFT; --qat is only for 270m. Official E2B retained-scale QAT uses its separate launcher.")
    if options.input_dir and options.source_run_dir:
        raise ValueError("Choose --input-dir or --source-run-dir, not both")
    if options.epochs <= 0 or (options.steps is not None and options.steps <= 0):
        raise ValueError("Epochs/steps must be positive")
    if options.eval_steps <= 0 or options.golden_every_steps <= 0 or options.golden_every_steps % options.eval_steps:
        raise ValueError("Golden cadence must be a positive multiple of --eval-steps")
    if min(options.max_seq_length, options.max_input_tokens, options.max_new_tokens) <= 0:
        raise ValueError("Token budgets must be positive")
    # Review recipes intentionally bind one context limit through preparation,
    # preflight and standalone inference. Do not silently rewrite just one side.
    if options.max_input_tokens != options.max_seq_length:
        raise ValueError("This launcher requires --max-input-tokens == --max-seq-length")
    if options.max_input_tokens + options.max_new_tokens > (8192 if options.profile == "e2b" else 32768):
        raise ValueError("Prompt plus generation budget exceeds the selected recipe context")
    model = Path(values["model_dir"])
    if not (model / "config.json").is_file() or not (model / "tokenizer_config.json").is_file() or not any(model.glob("*.safetensors")):
        raise ValueError("--model-dir must contain local dense HF safetensors, config.json and tokenizer_config.json; no weights are downloaded")
    source = Path(values["input_dir"] or values["source_run_dir"])
    source_files = [source / name for name in (("train.jsonl", "val.jsonl") if values["input_dir"] else ("genui.jsonl", "responses.jsonl"))]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required source is missing: {path}")
    output = Path(values["output_dir"])
    if output == model or output in model.parents or output == source or output in source.parents:
        raise ValueError("Output directory must not contain the model or source inputs")
    from ir_training.data.shared_prompt import create_shared_prompt_contract
    prompt = create_shared_prompt_contract(ordering="root-first")
    return {
        "schema_version": 1, "workflow": "shared_prompt_dual_golden_training_v1",
        "options": values, "source_files": [str(path) for path in source_files],
        "shared_prompt": prompt,
        "goldens": {name: {"path": str(repo_root() / path), "rows": count, "sha256": digest,
                           "benchmark_manifest_path": str((repo_root() / path).parent / "benchmark_manifest.json"),
                           "benchmark_manifest_sha256": GOLDEN_MANIFEST_SHA256[name]}
                    for name, (path, count, digest) in GOLDENS.items()},
        "stages": ["prepare", "configure", "preflight", "training", "best_golden32", "best_golden35", "final_golden32", "final_golden35", "scorecard"],
        "model_training": "270m W8 QAT" if options.qat else ("dense E2B LoRA SFT" if options.profile == "e2b" else "270m full SFT"),
        "exports_performed": False, "automatic_model_downloads": False,
        "golden35_role": "final evaluation only; never checkpoint selection",
        "note": "A short run can finish before the default 500/1000 cadence; final Golden32 and both final-evaluation cohorts still run.",
    }


def _group_split(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    """Keep transitive query/response aliases together, even with different IDs."""
    from ir_training.data.audit_filter import _identity_keys, _source_hashes
    from ir_training.data.golden_replacement import source_identity_record
    from ir_training.data.splits import stratified_split
    parents = list(range(len(rows)))
    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        keys = {"id:" + key for key in _identity_keys(source_identity_record(row))} | {"text:" + key for key in _source_hashes(row)}
        for key in keys:
            if key in seen:
                a, b = find(index), find(seen[key])
                parents[max(a, b)] = min(a, b)
            else:
                seen[key] = index
    grouped = [{"source_id": str(find(index)), "intent_bucket": row.get("intent_bucket") or (row.get("metadata") or {}).get("intent_bucket"), "row": row} for index, row in enumerate(rows)]
    split = stratified_split(grouped, .9, .1, 0, "intent_bucket", seed)
    return {name: [item["row"] for item in values] for name, values in split.items() if name in {"train", "val"}}


def _load_tokenizer(model_dir: Path, profile: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    loader = PreTrainedTokenizerFast if profile == "e2b" else AutoTokenizer
    return loader.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)


def prepare_data(plan: dict[str, Any], *, tokenizer_loader: Callable = _load_tokenizer) -> dict[str, Any]:
    from ir_training.data.audit_filter import audit_and_filter_rows, load_reserved_cohorts
    from ir_training.data.build_pairs import prepare_dataset
    from ir_training.data.express_preparation import prepare_splits
    from ir_training.eval.golden_set import load_fixed_golden_rows
    options = plan["options"]
    output = Path(options["output_dir"])
    golden_paths = {name: Path(item["path"]) for name, item in plan["goldens"].items()}
    for name, path in golden_paths.items():
        if sha256(path) != plan["goldens"][name]["sha256"]:
            raise ValueError(f"Frozen {name} differs from the independently pinned revision")
        if sha256(Path(plan["goldens"][name]["benchmark_manifest_path"])) != plan["goldens"][name]["benchmark_manifest_sha256"]:
            raise ValueError(f"Frozen {name} benchmark manifest differs from the independently pinned revision")
        load_fixed_golden_rows(path, required_rows=plan["goldens"][name]["rows"])
    reserved = load_reserved_cohorts(golden_paths.values())
    source_pins = {path: sha256(Path(path)) for path in plan["source_files"]}
    source_manifest = None
    if options["input_dir"]:
        originals = {name: list(read_jsonl(Path(options["input_dir"]) / f"{name}.jsonl")) for name in ("train", "val")}
    else:
        raw = output / "materialized"
        if raw.exists():
            raise FileExistsError(raw)
        source_manifest = prepare_dataset({
            "run": {"id": "golden_training_source", "source_run_dir": options["source_run_dir"], "output_dir": str(raw),
                    "target_format": "a2ui_express_v1", "system_prompt": "", "seed": options["seed"]},
            "filters": {"require_strict_express": True, "deduplicate": True, "max_input_chars": 60000, "max_output_chars": 60000},
            "url_preprocessing": {"enabled": True}, "split": {"train": 1.0, "val": 0.0, "test": 0.0},
        })
        originals = {"all": list(read_jsonl(raw / "all.jsonl"))}
    filtered, reports = {}, {}
    filtered_dir = output / "filtered"
    filtered_dir.mkdir(exist_ok=False)
    for name, rows in originals.items():
        accepted, quarantine, report = audit_and_filter_rows(rows, reserved=reserved, require_source_identities=True)
        filtered[name], reports[name] = accepted, report
        write_jsonl(filtered_dir / f"{name}_quarantine.jsonl", quarantine)
    if "all" in filtered:
        filtered = _group_split(filtered["all"], options["seed"])
    for name, rows in filtered.items():
        if not rows:
            raise ValueError(f"No {name} rows remain after strict/reserved-source filtering")
        write_jsonl(filtered_dir / f"{name}.jsonl", rows)
    tokenizer = tokenizer_loader(Path(options["model_dir"]), options["profile"])
    manifest = prepare_splits(
        {**{name: filtered_dir / f"{name}.jsonl" for name in ("train", "val")}, **golden_paths},
        output / "prepared", ordering="root-first", tokenizer=tokenizer,
        max_seq_length=options["max_seq_length"], max_input_tokens=options["max_input_tokens"],
        chat_template_kwargs={"enable_thinking": False}, shared_prompt=plan["shared_prompt"],
        evaluation_splits={"golden32", "golden35"},
    )
    for path, digest in source_pins.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Source changed during preparation: {path}")
    report = {"source_files": source_pins, "source_materialization": source_manifest, "filtering": reports,
              "prepared_counts": {name: {key: values.get(key, 0) for key in ("input_rows", "accepted_rows", "quarantined_rows", "quarantine_reasons")} for name, values in manifest["splits"].items()},
              "synthetic_targets_created": 0, "golden_membership_changed": False}
    _write(output / "data_audit.json", report)
    # Run the same pre-model guard now, including --prepare-only on a CPU host.
    sys.path.insert(0, str(repo_root() / "training/scripts"))
    from prepare_review_training import verify_prepared
    verify_prepared(output / "prepared", output / "prepared/golden32.jsonl", golden35=output / "prepared/golden35.jsonl",
                    max_sequence=options["max_seq_length"], max_prompt=options["max_input_tokens"])
    return report


def configure_command(plan: dict[str, Any]) -> list[str]:
    values = plan["options"]
    output = Path(values["output_dir"])
    command = [sys.executable, str(repo_root() / "training/scripts/prepare_review_training.py"),
               "--profile", values["profile"], "--model-dir", values["model_dir"],
               "--dataset-dir", str(output / "prepared"), "--golden-file", str(output / "prepared/golden32.jsonl"),
               "--golden35-file", str(output / "prepared/golden35.jsonl"), "--output-dir", str(output / "fit"),
               "--run-id", output.name,
               "--devices", values["devices"], "--epochs", str(values["epochs"]),
               "--eval-steps", str(values["eval_steps"]), "--golden-every-steps", str(values["golden_every_steps"]),
               "--max-seq-length", str(values["max_seq_length"]), "--max-new-tokens", str(values["max_new_tokens"])]
    for name in ("steps", "microbatch", "effective_batch", "dataloader_workers"):
        if values[name] is not None:
            command.extend(["--" + name.replace("_", "-"), str(values[name])])
    if values["qat"]:
        command.append("--qat")
    return command


def evaluation_command(plan: dict[str, Any], role: str, cohort: str, destination: Path) -> list[str]:
    values = plan["options"]
    output = Path(values["output_dir"])
    checkpoint_name = "best_golden_checkpoint" if role == "best" else ("final_adapter" if values["profile"] == "e2b" else "final_model")
    return [sys.executable, str(repo_root() / "training/scripts/evaluate_checkpoint_on_golden.py"),
            "--config", str(output / "fit/training_config.yaml"), "--checkpoint", str(output / "fit/training" / checkpoint_name),
            "--checkpoint-kind", "adapter" if values["profile"] == "e2b" else "merged",
            "--qat-mode", "on" if values["qat"] else "auto",
            "--split", str(output / "prepared" / f"{cohort}.jsonl"),
            "--max-rows", str(plan["goldens"][cohort]["rows"]), "--required-rows", str(plan["goldens"][cohort]["rows"]),
            "--max-input-tokens", str(values["max_input_tokens"]), "--max-new-tokens", str(values["max_new_tokens"]),
            "--run-id", output.name, "--evaluation-name", f"{role}_{cohort}", "--output-dir", str(destination),
            "--tensorboard-root", values["tensorboard_root"], "--metric-version", "v5_4", "--require-prepared-contract"]


def _run_command(command: list[str], log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running {log.stem}; log: {log}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=repo_root(), env=environment, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise RuntimeError(f"Stage failed with exit {completed.returncode}; inspect {log}")


def _bindings(paths: list[Path]) -> dict[str, str]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Stage did not publish required output: {path}")
    return {str(path): sha256(path) for path in paths}


def run_pipeline(options: GoldenTrainingOptions, *, execute: bool = False, prepare_only: bool = False,
                 continue_run: bool = False, tokenizer_loader: Callable = _load_tokenizer,
                 command_runner: Callable = _run_command) -> dict[str, Any]:
    plan = build_plan(options)
    if not execute and not prepare_only:
        return {**plan, "status": "plan_only", "training_executed": False}
    if execute and prepare_only:
        raise ValueError("Choose --execute or --prepare-only")
    output = Path(plan["options"]["output_dir"])
    record = output / "pipeline_manifest.json"
    if output.exists():
        if not continue_run or not record.is_file():
            raise FileExistsError(f"Choose a fresh output directory or explicitly --continue-run a verified workflow: {output}")
        state = json.loads(record.read_text(encoding="utf-8"))
        if state["plan"] != plan:
            raise ValueError("Workflow options or prompt contract changed; do not reuse the run")
        for stage in state["completed"].values():
            for path, digest in stage["files"].items():
                if not Path(path).is_file() or sha256(Path(path)) != digest:
                    raise ValueError(f"Completed-stage artifact changed: {path}")
        if state.get("active_stage") in {"prepare", "configure", "training"}:
            raise ValueError("An interrupted preparation/configuration/training stage needs explicit recovery; this launcher will not restart its optimizer silently. Inspect the stage log and saved training config.")
    else:
        if continue_run:
            raise FileNotFoundError("--continue-run requires an existing workflow")
        output.mkdir(parents=True, exist_ok=False)
        state = {"schema_version": 1, "plan": plan, "status": "running", "completed": {}, "attempts": {}, "active_stage": None}
    environment = dict(os.environ)
    environment.update(A2UI_TENSORBOARD_ROOT=plan["options"]["tensorboard_root"], PYTHONUNBUFFERED="1")
    def stage(name: str, work: Callable[[], list[Path]]) -> None:
        if name in state["completed"]:
            return
        state.update(status="running", active_stage=name)
        state["attempts"][name] = state["attempts"].get(name, 0) + 1
        _write(record, state)
        try:
            paths = work()
            state["completed"][name] = {"files": _bindings(paths), "finished_at": datetime.now(timezone.utc).isoformat()}
            state.update(active_stage=None)
            _write(record, state)
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write(record, state)
            raise
    def prepare() -> list[Path]:
        prepare_data(plan, tokenizer_loader=tokenizer_loader)
        return [*map(Path, plan["source_files"]), *sorted((output / "prepared").glob("*.json*")), output / "data_audit.json",
                *[Path(item[key]) for item in plan["goldens"].values() for key in ("path", "benchmark_manifest_path")]]
    stage("prepare", prepare)
    if prepare_only:
        state.update(status="prepared", active_stage=None)
        _write(record, state)
        return state
    def configure() -> list[Path]:
        command_runner(configure_command(plan), output / "logs/configure.log", environment)
        return [output / "fit/training_config.yaml", output / "fit/preparation_report.json"]
    stage("configure", configure)
    launch = [sys.executable, str(repo_root() / "training/scripts/launch_review_training.py"), "--config", str(output / "fit/training_config.yaml")]
    stage("preflight", lambda: (command_runner([*launch, "--preflight-only", "--execute"], output / "logs/preflight.log", environment) or []))
    def train() -> list[Path]:
        command_runner([*launch, "--execute"], output / "logs/training.log", environment)
        folders = [output / "fit/training/best_golden_checkpoint", output / "fit/training" / ("final_adapter" if options.profile == "e2b" else "final_model")]
        for folder in folders:
            if not folder.is_dir() or not any(folder.glob("*.safetensors")):
                raise ValueError(f"Training did not publish the required checkpoint: {folder}")
        return [file for folder in folders for file in sorted(folder.iterdir()) if file.is_file()]
    stage("training", train)
    for role in ("best", "final"):
        for cohort in GOLDENS:
            name = f"{role}_{cohort}"
            def evaluate(role=role, cohort=cohort, name=name) -> list[Path]:
                destination = output / "evaluations" / name / f"attempt_{state['attempts'][name]:03d}"
                if destination.exists():
                    raise FileExistsError(destination)
                from ir_training.common.config import load_yaml
                runtime = load_yaml(output / "fit/training_config.yaml")["runtime"]
                evaluation_environment = {**environment, "CUDA_VISIBLE_DEVICES": runtime["cuda_visible_devices"],
                                          "A2UI_SKIP_CUDA_DEVICE_NORMALIZE": "1", "TOKENIZERS_PARALLELISM": "false"}
                evaluation_environment.pop("A2UI_CUDA_VISIBLE_DEVICES", None)
                evaluation_environment.pop("A2UI_EXCLUDE_CUDA_DEVICES", None)
                command_runner(evaluation_command(plan, role, cohort, destination), output / f"logs/{name}_{state['attempts'][name]:03d}.log", evaluation_environment)
                result = destination / "evaluation_result.json"
                value = json.loads(result.read_text(encoding="utf-8"))
                if value.get("row_count") != plan["goldens"][cohort]["rows"]:
                    raise ValueError(f"Incomplete {name} evaluation")
                return [result, destination / "aggregate_metrics.json", destination / "predictions.jsonl", destination / "scored_predictions.jsonl"]
            stage(name, evaluate)
    def scorecard() -> list[Path]:
        comparisons = {}
        for role in ("best", "final"):
            for cohort in GOLDENS:
                name = f"{role}_{cohort}"
                results = [Path(path) for path in state["completed"][name]["files"] if path.endswith("/evaluation_result.json") or path.endswith("\\evaluation_result.json")]
                if len(results) != 1:
                    raise ValueError(f"Missing bound evaluation result: {name}")
                comparisons[name] = json.loads(results[0].read_text(encoding="utf-8"))
        result = {"schema_version": 1, "status": "complete", "profile": options.profile, "shared_prompt": plan["shared_prompt"],
                  "evaluations": comparisons, "golden32_unique_sources": 31, "golden35_unique_sources": 35,
                  "golden35_used_for_selection": False, "exports_performed": False,
                  "warning": "New shared-prompt scores are not directly comparable to historical short-prompt runs."}
        path = output / "evaluation_scorecard.json"
        _write(path, result)
        return [path]
    stage("scorecard", scorecard)
    state.update(status="complete", active_stage=None)
    state.pop("error", None)
    _write(record, state)
    return state
