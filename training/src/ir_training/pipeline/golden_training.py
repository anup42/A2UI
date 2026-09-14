"""Checked, opt-in train-to-dual-Golden workflow; no cloud data generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
import codecs
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
from typing import Any, Callable

from ir_training.common.config import repo_root
from ir_training.common.jsonl import write_jsonl
from ir_training.common.parallel import resolve_prepare_workers
from ir_training.common.progress import Progress, fingerprint_file, log

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
    prepare_workers: int = 0
    progress_seconds: float = 10
    preparation_cache: bool = True
    preparation_cache_dir: Path | None = None
    learning_rate: float | None = None
    weight_decay: float | None = None
    warmup_ratio: float | None = None
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    attn_implementation: str = "sdpa"
    augmentation: str = "none"
    augmentation_max_extra_fraction: float = 0.10
    augmentation_max_family_repeats: int = 2
    evaluate_golden35: bool = True
    token_cache: bool = True
    token_cache_dir: Path | None = None


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
    for name in ("model_dir", "output_dir", "source_run_dir", "input_dir", "preparation_cache_dir", "token_cache_dir"):
        if result[name] is not None:
            result[name] = str(Path(result[name]).expanduser().resolve())
    if not result["input_dir"] and not result["source_run_dir"]:
        result["source_run_dir"] = str(repo_root() / "dataset/data/runs/dataset_v1")
    result["preparation_cache_dir"] = str(Path(result["preparation_cache_dir"] or Path(result["output_dir"]).parent / ".golden-preparation-cache").resolve())
    result["token_cache_dir"] = str(Path(result["token_cache_dir"] or Path(result["preparation_cache_dir"]) / "tokens").resolve())
    return result


def build_plan(options: GoldenTrainingOptions) -> dict[str, Any]:
    """Read-only plan, requiring no CUDA, model loading or tokenizer download."""
    values = _options(options)
    from ir_training.train.hyperparameters import review_overrides
    review_overrides(learning_rate=options.learning_rate, weight_decay=options.weight_decay,
                     warmup_ratio=options.warmup_ratio, logging_steps=options.logging_steps, seed=options.seed)
    if options.augmentation not in {"none", "rare_components"}:
        raise ValueError("--augmentation must be none or rare_components")
    if not math.isfinite(options.augmentation_max_extra_fraction) or not 0 < options.augmentation_max_extra_fraction <= 0.5:
        raise ValueError("--augmentation-max-extra-fraction must be in (0, 0.5]")
    if type(options.augmentation_max_family_repeats) is not int or not 2 <= options.augmentation_max_family_repeats <= 5:
        raise ValueError("--augmentation-max-family-repeats must be an integer from 2 to 5")
    if options.attn_implementation not in {"sdpa", "eager"}:
        raise ValueError("--attn-implementation must be sdpa or eager")
    resolve_prepare_workers(options.prepare_workers)
    if not math.isfinite(options.progress_seconds) or options.progress_seconds <= 0:
        raise ValueError("--progress-seconds must be positive")
    if options.profile not in {"e2b", "270m"} or (options.qat and options.profile != "270m"):
        raise ValueError("Profiles are e2b dense LoRA and 270m full SFT; --qat is only for 270m. Official E2B retained-scale QAT uses its separate launcher.")
    if options.input_dir and options.source_run_dir:
        raise ValueError("Choose --input-dir or --source-run-dir, not both")
    if not math.isfinite(options.epochs) or options.epochs <= 0 or (options.steps is not None and options.steps <= 0):
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
    for key in ("preparation_cache_dir", "token_cache_dir"):
        cache = Path(values[key])
        if any(cache.is_relative_to(protected) or protected.is_relative_to(cache) for protected in (model, source, output)):
            raise ValueError(f"{key} must be outside and must not contain model, source or run output directories")
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
        "stages": ["prepare", *(["augment"] if options.augmentation != "none" else []), "configure", "preflight", "training",
                   "best_golden32", *(["best_golden35"] if options.evaluate_golden35 else []),
                   "final_golden32", *(["final_golden35"] if options.evaluate_golden35 else []), "scorecard"],
        "model_training": "270m W8 QAT" if options.qat else ("dense E2B LoRA SFT" if options.profile == "e2b" else "270m full SFT"),
        "exports_performed": False, "automatic_model_downloads": False,
        "golden35_role": "final evaluation only; never checkpoint selection" if options.evaluate_golden35 else "reserved, not evaluated in development trial",
        "note": "Final Golden32 always runs. Golden35 runs unless explicitly deferred for sequential tuning. Augmentation only repeats validated training examples; no new semantic coverage.",
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


def _strict_rows(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: malformed JSON; no rows may be silently skipped") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            yield row


def prepare_data(plan: dict[str, Any], *, tokenizer_loader: Callable | None = None) -> dict[str, Any]:
    from ir_training.data.audit_filter import load_reserved_cohorts
    from ir_training.eval.golden_set import load_fixed_golden_rows
    options = plan["options"]
    output = Path(options["output_dir"])
    interval = options["progress_seconds"]
    workers = resolve_prepare_workers(options["prepare_workers"])
    log(f"Preparation uses {workers} CPU workers; streaming input; GPUs are used only after preparation")
    golden_paths = {name: Path(item["path"]) for name, item in plan["goldens"].items()}
    for name, path in golden_paths.items():
        if sha256(path) != plan["goldens"][name]["sha256"]:
            raise ValueError(f"Frozen {name} differs from the independently pinned revision")
        if sha256(Path(plan["goldens"][name]["benchmark_manifest_path"])) != plan["goldens"][name]["benchmark_manifest_sha256"]:
            raise ValueError(f"Frozen {name} benchmark manifest differs from the independently pinned revision")
        load_fixed_golden_rows(path, required_rows=plan["goldens"][name]["rows"])
    reserved = load_reserved_cohorts(golden_paths.values())
    fingerprints = {path: fingerprint_file(Path(path), count_rows=True, interval=interval) for path in plan["source_files"]}
    source_pins = {path: value["sha256"] for path, value in fingerprints.items()}
    from ir_training.pipeline import preparation_cache
    cache = Path(options["preparation_cache_dir"] or output.parent / ".golden-preparation-cache")
    cache_binding = None
    # Custom/in-memory tokenizers have no trustworthy on-disk identity.
    if options["preparation_cache"] and tokenizer_loader is None:
        with Progress("Bind preparation inputs, tokenizer assets, code and schemas", unit="stage", interval=interval):
            cache_binding = preparation_cache.identity(plan, source_pins)
    else:
        log("Preparation CACHE DISABLED: explicit opt-out or custom in-memory tokenizer")
    # The build lock covers the whole miss path, not only publication: matching
    # concurrent launches wait and reuse one completed, independently verified copy.
    guard = preparation_cache.lock(cache, cache_binding, interval=interval) if cache_binding is not None else nullcontext()
    with guard:
        reused = cache_binding is not None and preparation_cache.restore(cache, output, cache_binding, interval=interval)
        if reused:
            report = json.loads((output / "data_audit.json").read_text(encoding="utf-8"))
        else:
            report = _prepare_uncached(plan, tokenizer_loader or _load_tokenizer, reserved, fingerprints, workers)
        for path, digest in source_pins.items():
            if fingerprint_file(Path(path), interval=interval)["sha256"] != digest:
                raise ValueError(f"Source changed during preparation: {path}")
        # Run the same pre-model guard on both fresh and reused preparations.
        sys.path.insert(0, str(repo_root() / "training/scripts"))
        from prepare_review_training import verify_prepared
        with Progress("Verify prepared splits and both Golden contracts", unit="stage", interval=interval):
            verify_prepared(output / "prepared", output / "prepared/golden32.jsonl", golden35=output / "prepared/golden35.jsonl",
                            max_sequence=options["max_seq_length"], max_prompt=options["max_input_tokens"])
        if cache_binding is not None:
            if preparation_cache.identity(plan, source_pins) != cache_binding:
                raise ValueError("Preparation implementation, schemas or tokenizer assets changed during preparation")
            if not reused:
                try:
                    preparation_cache.publish(cache, output, cache_binding, interval=interval)
                except OSError as exc:
                    # Do not discard an otherwise fully verified preparation
                    # when the optional persistent store cannot be published.
                    log(f"Preparation is verified, but cache publication was unavailable: {exc}")
        return report


def _prepare_uncached(plan, tokenizer_loader, reserved, fingerprints, workers):
    from ir_training.data.audit_filter import audit_and_filter_rows
    from ir_training.data.build_pairs import prepare_dataset
    from ir_training.data.express_preparation import prepare_splits
    options, output = plan["options"], Path(plan["options"]["output_dir"])
    interval = options["progress_seconds"]
    golden_paths = {name: Path(item["path"]) for name, item in plan["goldens"].items()}
    source_manifest = None
    if options["input_dir"]:
        originals = {name: _strict_rows(Path(options["input_dir"]) / f"{name}.jsonl") for name in ("train", "val")}
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
        originals = {"all": _strict_rows(raw / "all.jsonl")}
    filtered, reports = {}, {}
    filtered_dir = output / "filtered"
    filtered_dir.mkdir(exist_ok=False)
    for name, rows in originals.items():
        total = fingerprints.get(str(Path(options["input_dir"]) / f"{name}.jsonl"), {}).get("rows") if options["input_dir"] else None
        with (filtered_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as accepted_file, \
                (filtered_dir / f"{name}_quarantine.jsonl").open("w", encoding="utf-8", newline="\n") as rejected_file, \
                Progress(f"Strict filter {name} ({workers} CPU workers)", total=total, interval=interval) as progress:
            def write_row(stream, value):
                stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
            accepted, _, report = audit_and_filter_rows(
                rows, reserved=reserved, require_source_identities=True, workers=workers, progress=progress,
                accepted_sink=(lambda row: write_row(accepted_file, row)) if name != "all" else None,
                quarantine_sink=lambda row: write_row(rejected_file, row),
            )
        if not report["accepted_rows"]:
            raise ValueError(f"No {name} rows remain after strict/reserved-source filtering")
        reports[name] = report
        if name == "all":
            filtered[name] = accepted
    if "all" in filtered:
        filtered = _group_split(filtered["all"], options["seed"])
    for name, rows in filtered.items():
        if not rows:
            raise ValueError(f"No {name} rows remain after strict/reserved-source filtering")
        write_jsonl(filtered_dir / f"{name}.jsonl", rows)
    with Progress("Load local tokenizer (no model weights)", unit="stage", interval=interval):
        tokenizer = tokenizer_loader(Path(options["model_dir"]), options["profile"])
    manifest = prepare_splits(
        {**{name: filtered_dir / f"{name}.jsonl" for name in ("train", "val")}, **golden_paths},
        output / "prepared", ordering="root-first", tokenizer=tokenizer,
        max_seq_length=options["max_seq_length"], max_input_tokens=options["max_input_tokens"],
        chat_template_kwargs={"enable_thinking": False}, shared_prompt=plan["shared_prompt"],
        evaluation_splits={"golden32", "golden35"},
        workers=workers, progress_interval=interval, show_progress=True,
    )
    report = {"source_files": {path: value["sha256"] for path, value in fingerprints.items()}, "source_materialization": source_manifest, "filtering": reports,
              "prepared_counts": {name: {key: values.get(key, 0) for key in ("input_rows", "accepted_rows", "quarantined_rows", "quarantine_reasons")} for name, values in manifest["splits"].items()},
              "synthetic_targets_created": 0, "golden_membership_changed": False}
    _write(output / "data_audit.json", report)
    return report


def configure_command(plan: dict[str, Any]) -> list[str]:
    values = plan["options"]
    output = Path(values["output_dir"])
    command = [sys.executable, str(repo_root() / "training/scripts/prepare_review_training.py"),
               "--profile", values["profile"], "--model-dir", values["model_dir"],
               "--dataset-dir", str(output / ("augmented" if values["augmentation"] != "none" else "prepared")), "--golden-file", str(output / "prepared/golden32.jsonl"),
               "--golden35-file", str(output / "prepared/golden35.jsonl"), "--output-dir", str(output / "fit"),
               "--run-id", output.name,
               "--devices", values["devices"], "--epochs", str(values["epochs"]),
               "--eval-steps", str(values["eval_steps"]), "--golden-every-steps", str(values["golden_every_steps"]),
               "--max-seq-length", str(values["max_seq_length"]), "--max-new-tokens", str(values["max_new_tokens"])]
    for name in ("steps", "microbatch", "effective_batch", "dataloader_workers", "learning_rate", "weight_decay", "warmup_ratio", "logging_steps", "seed", "attn_implementation"):
        if values[name] is not None:
            command.extend(["--" + name.replace("_", "-"), str(values[name])])
    if values["qat"]:
        command.append("--qat")
    command.append("--gradient-checkpointing" if values["gradient_checkpointing"] else "--no-gradient-checkpointing")
    command.append("--token-cache" if values["token_cache"] else "--no-token-cache")
    command.extend(["--token-cache-dir", values["token_cache_dir"]])
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
    # Read chunks, not lines: tqdm and model loaders also emit carriage returns
    # without newlines. Keep one combined stream in the console AND on disk.
    env = {**environment, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with log.open("w", encoding="utf-8", newline="") as stream:
        process = subprocess.Popen(command, cwd=repo_root(), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, bufsize=0, start_new_session=os.name == "posix")
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                block = process.stdout.read(4096)
                text = decoder.decode(block, final=not block)
                if text:
                    stream.write(text)
                    stream.flush()
                    sys.stdout.write(text)
                    sys.stdout.flush()
                if not block:
                    break
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            raise
        finally:
            process.stdout.close()
    if returncode:
        raise RuntimeError(f"Stage failed with exit {returncode}; inspect {log}")


class _Tee:
    def __init__(self, console, stream):
        self.console, self.stream = console, stream

    def write(self, text):
        self.console.write(text)
        self.stream.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def _console_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        with redirect_stdout(_Tee(sys.stdout, stream)), redirect_stderr(_Tee(sys.stderr, stream)):
            yield


def _bindings(paths: list[Path]) -> dict[str, str]:
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Stage did not publish required output: {path}")
    return {str(path): sha256(path) for path in paths}


def run_pipeline(options: GoldenTrainingOptions, *, execute: bool = False, prepare_only: bool = False,
                  continue_run: bool = False, tokenizer_loader: Callable | None = None,
                 command_runner: Callable = _run_command) -> dict[str, Any]:
    if execute or prepare_only:
        log("Golden training startup: validating options and production prompt; no training has started yet")
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
        with Progress("Verify completed stages before continuation", unit="stage", interval=options.progress_seconds):
            for completed_stage in state["completed"].values():
                for path, digest in completed_stage["files"].items():
                    if not Path(path).is_file() or sha256(Path(path)) != digest:
                        raise ValueError(f"Completed-stage artifact changed: {path}")
        if state.get("active_stage") in {"prepare", "augment", "configure", "training"}:
            raise ValueError("An interrupted preparation/configuration/training stage needs explicit recovery; this launcher will not restart its optimizer silently. Inspect the stage log and saved training config.")
        receipt_path = output / "preparation_receipt.json"
        if "prepare" in state["completed"] and receipt_path.is_file():
            from ir_training.pipeline.preparation_cache import identity
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            source_pins = {path: state["completed"]["prepare"]["files"][path] for path in plan["source_files"]}
            with Progress("Verify preparation implementation and tokenizer before continuation", unit="stage", interval=options.progress_seconds):
                if saved["identity"] != identity(plan, source_pins):
                    raise ValueError("Preparation implementation, schemas or tokenizer changed; use a fresh run directory")
    else:
        if continue_run:
            raise FileNotFoundError("--continue-run requires an existing workflow")
        output.mkdir(parents=True, exist_ok=False)
        state = {"schema_version": 1, "plan": plan, "status": "running", "completed": {}, "attempts": {}, "active_stage": None}
    environment = dict(os.environ)
    environment.update(A2UI_TENSORBOARD_ROOT=plan["options"]["tensorboard_root"], PYTHONUNBUFFERED="1")
    def stage(name: str, work: Callable[[], list[Path]]) -> None:
        if name in state["completed"]:
            log(f"Stage {name}: reuse verified completed stage")
            return
        state.update(status="running", active_stage=name)
        state["attempts"][name] = state["attempts"].get(name, 0) + 1
        _write(record, state)
        try:
            with Progress(f"Stage {name}", unit="stage", interval=options.progress_seconds):
                paths = work()
                state["completed"][name] = {"files": _bindings(paths), "finished_at": datetime.now(timezone.utc).isoformat()}
            state.update(active_stage=None)
            _write(record, state)
        except BaseException as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write(record, state)
            raise
    def prepare() -> list[Path]:
        with _console_log(output / "logs/prepare.log"):
            prepare_data(plan, tokenizer_loader=tokenizer_loader)
        return [*map(Path, plan["source_files"]), *sorted((output / "prepared").glob("*.json*")), output / "data_audit.json",
                *[path for path in (output / "preparation_receipt.json", output / "cache_reuse.json") if path.exists()],
                *[Path(item[key]) for item in plan["goldens"].values() for key in ("path", "benchmark_manifest_path")]]
    stage("prepare", prepare)
    if options.augmentation != "none":
        def augment() -> list[Path]:
            from ir_training.data.augmentation import augment_prepared_training
            sys.path.insert(0, str(repo_root() / "training/scripts"))
            from prepare_review_training import verify_prepared
            with _console_log(output / "logs/augmentation.log"):
                augment_prepared_training(output / "prepared", output / "augmented", seed=options.seed,
                    max_extra_fraction=options.augmentation_max_extra_fraction,
                    max_family_copies=options.augmentation_max_family_repeats, progress_seconds=options.progress_seconds)
                verify_prepared(output / "augmented", output / "prepared/golden32.jsonl", golden35=output / "prepared/golden35.jsonl",
                                max_sequence=options.max_seq_length, max_prompt=options.max_input_tokens)
            return sorted((output / "augmented").glob("*.json*"))
        stage("augment", augment)
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
    evaluated_cohorts = list(GOLDENS) if options.evaluate_golden35 else ["golden32"]
    for role in ("best", "final"):
        for cohort in evaluated_cohorts:
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
            for cohort in evaluated_cohorts:
                name = f"{role}_{cohort}"
                results = [Path(path) for path in state["completed"][name]["files"] if path.endswith("/evaluation_result.json") or path.endswith("\\evaluation_result.json")]
                if len(results) != 1:
                    raise ValueError(f"Missing bound evaluation result: {name}")
                comparisons[name] = json.loads(results[0].read_text(encoding="utf-8"))
        result = {"schema_version": 1, "status": "complete", "profile": options.profile, "shared_prompt": plan["shared_prompt"],
                  "evaluations": comparisons, "golden32_unique_sources": 31, "golden35_unique_sources": 35,
                  "golden35_used_for_selection": False, "exports_performed": False,
                  "golden35_evaluated": options.evaluate_golden35, "augmentation": options.augmentation,
                  "warning": "New shared-prompt scores are not directly comparable to historical short-prompt runs."}
        path = output / "evaluation_scorecard.json"
        _write(path, result)
        return [path]
    stage("scorecard", scorecard)
    state.update(status="complete", active_stage=None)
    state.pop("error", None)
    _write(record, state)
    return state
