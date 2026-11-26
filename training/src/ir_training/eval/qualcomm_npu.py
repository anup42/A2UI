"""Run the frozen Bixby50 holdout with the SM8850 target-only Android driver."""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
import uuid
from pathlib import Path

from ir_training.common.jsonl import write_jsonl
from ir_training.data.bixby50 import APPROVED_SOURCE_IDS, validate_source_only_rows
from ir_training.eval.compare_to_baseline import evaluate_predictions
from ir_training.eval.generate import build_prediction_record
from ir_training.eval.golden_set import load_fixed_golden_rows
from ir_training.eval.prepared_contract import file_sha256
from ir_training.export.qualcomm_npu import (
    BIXBY,
    PACKAGE,
    REQUESTS,
    SOC,
    SPLIT,
    inspect_npu_package,
    read_json,
)

PROBE = "bixby50_npu"
LIBRARIES = (
    "libLiteRtDispatch_Qualcomm.so",
    "libQnnHtp.so",
    "libQnnSystem.so",
    "libQnnHtpPrepare.so",
    "libQnnHtpV81Stub.so",
    "libQnnHtpV81Skel.so",
)


def validate_export(folder: Path) -> tuple[dict, list[dict]]:
    receipt = read_json(folder / "export.json")
    if (
        receipt.get("format") != "a2ui-sm8850-npu-v1"
        or receipt.get("profile") not in {"270m", "e2b"}
        or receipt.get("soc") != SOC
        or receipt.get("mtp") is not False
        or file_sha256(folder / "manifest.json")
        != receipt.get("bixby50_manifest_sha256")
        or file_sha256(folder / SPLIT) != receipt.get("bixby50_split_sha256")
        or file_sha256(folder / PACKAGE) != receipt.get("model_sha256")
        or file_sha256(folder / REQUESTS) != receipt.get("requests_sha256")
    ):
        raise ValueError("NPU export identity, package, or prompt binding changed")
    audit = inspect_npu_package(folder / PACKAGE, receipt["profile"])
    if any(receipt.get(key) != value for key, value in audit.items()):
        raise ValueError("Compiled graph differs from export evidence")
    requests = [
        json.loads(line)
        for line in (folder / REQUESTS).read_text(encoding="utf-8").splitlines()
    ]
    if [row["id"] for row in requests] != list(APPROVED_SOURCE_IDS):
        raise ValueError("Expected all 50 ordered Bixby50 sources")
    cache_length = audit["compiled_cache_length"]
    if (
        receipt["cache_length"] != cache_length
        or not 0 < receipt["max_output_tokens"] < cache_length
    ):
        raise ValueError("Invalid NPU generation limits")
    for row in requests:
        ids = row["input_ids"]
        if (
            set(row) != {"id", "input_ids", "prompt"}
            or not isinstance(row["prompt"], str)
            or not row["prompt"]
            or not ids
            or len(ids) > receipt["max_input_tokens"]
            or not all(type(i) is int and i >= 0 for i in ids)
            or len(ids) + receipt["max_output_tokens"] > receipt["cache_length"]
        ):
            raise ValueError(f"Invalid source-only NPU request {row['id']}")
    return receipt, requests


def score_results(
    results: Path, output: Path, requests: list[dict], runtime: dict, split: Path
) -> None:
    if (results / "backend.txt").read_text(encoding="utf-8").splitlines() != [
        "qualcomm-htp",
        "mtp-off",
    ]:
        raise ValueError("Vendor dispatch/HTP evidence is missing")
    rows = load_fixed_golden_rows(split, required_rows=50)
    validate_source_only_rows(rows, read_json(BIXBY / "benchmark_manifest.json"))
    lines = (results / "index.tsv").read_text(encoding="utf-8").splitlines()
    if {path.stem for path in results.glob("*.out")} != set(APPROVED_SOURCE_IDS):
        raise ValueError("Incomplete native output set")
    predictions = []
    for row, request, line in zip(
        sorted(rows, key=lambda r: r["id"]), requests, lines, strict=True
    ):
        source_id, prompt_length, seconds = line.split("\t")
        elapsed = float(seconds)
        if (
            source_id != row["id"]
            or source_id != request["id"]
            or int(prompt_length) != len(request["input_ids"])
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError(f"Invalid native result for {row['id']}")
        raw = (results / f"{source_id}.out").read_bytes().decode("utf-8")
        predictions.append(
            build_prediction_record(
                row,
                raw,
                runtime={
                    **runtime,
                    "input_tokens": int(prompt_length),
                    "generation_seconds": elapsed,
                },
            )
        )
    path = output / "predictions.jsonl"
    write_jsonl(path, predictions)
    evaluate_predictions(path, output_dir=output, metric_version="v5_4")


def test_on_device(
    folder: Path, runtime_dir: Path, serial: str, output: Path, adb: str = "adb"
) -> Path:
    def device(*args, capture: bool = True):
        proc = subprocess.run(
            [adb, "-s", serial, *map(str, args)],
            check=True,
            stdout=subprocess.PIPE if capture else None,
            encoding="utf-8",
            errors="replace",
        )
        return proc.stdout.strip() if capture else ""

    soc = device("shell", "getprop", "ro.soc.model")
    if soc.upper() != SOC:
        raise ValueError(
            f"Device {serial} reports {soc!r}; this artifact requires {SOC}"
        )
    if device("shell", "getprop", "ro.product.cpu.abi") != "arm64-v8a":
        raise ValueError("Expected an arm64 Android device")
    phone = device("shell", "getprop", "ro.product.model")
    for name in (PROBE, *LIBRARIES):
        if not (runtime_dir / name).is_file():
            raise FileNotFoundError(runtime_dir / name)
    bound, requests = validate_export(folder)
    output = output.resolve()
    if output.is_relative_to(folder.resolve()) or output.is_relative_to(
        runtime_dir.resolve()
    ):
        raise ValueError(
            "Use a fresh report directory separate from the model and runtime"
        )
    output.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex
    root = "/data/local/tmp/a2ui-sm8850"
    remote = f"{root}/runs/{run_id}"
    model = f"{root}/models/{bound['model_sha256']}.litertlm"
    device("shell", f"mkdir -p {remote}/results {remote}/cache {root}/models")
    present = device("shell", f"if test -f {model}; then sha256sum {model}; fi")
    if present:
        if present.split()[0] != bound["model_sha256"]:
            raise ValueError(f"Cached device model has the wrong SHA-256: {model}")
    else:
        pending = f"{remote}/model.upload"
        device("push", folder / PACKAGE, pending, capture=False)
        if device("shell", "sha256sum", pending).split()[0] != bound["model_sha256"]:
            raise ValueError("Uploaded model has the wrong SHA-256")
        device("shell", "mv", pending, model)
    with tempfile.TemporaryDirectory() as tmp:
        inputs = Path(tmp) / "inputs"
        inputs.mkdir()
        for row in requests:
            (inputs / f"{row['id']}.prompt").write_bytes(row["prompt"].encode("utf-8"))
            (inputs / f"{row['id']}.ids").write_text(
                " ".join(map(str, row["input_ids"])), encoding="ascii"
            )
        device("push", inputs, remote + "/", capture=False)
    files = [runtime_dir / PROBE, *sorted(runtime_dir.glob("*.so"))]
    device("push", *files, remote + "/", capture=False)
    device("shell", "chmod", "700", f"{remote}/{PROBE}")
    runtime = {
        "backend": "npu",
        "soc": soc,
        "serial": serial,
        "device_model": phone,
        "mtp_enabled": False,
        "model_sha256": bound["model_sha256"],
        "requests_sha256": bound["requests_sha256"],
        "max_new_tokens": bound["max_output_tokens"],
        "device_run_dir": remote,
    }
    (output / "device.json").write_text(
        json.dumps(
            {
                **runtime,
                "runtime_files_sha256": {p.name: file_sha256(p) for p in files},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    command = (
        f"cd {remote} && LD_LIBRARY_PATH={remote} ADSP_LIBRARY_PATH={remote} "
        f"./{PROBE} {model} {remote} {bound['cache_length']} {bound['max_output_tokens']}"
    )
    # adb shell -T propagates the actual Android process status, without a PTY.
    with (output / "native.log").open("wb") as log:
        result = subprocess.run(
            [adb, "-s", serial, "shell", "-T", command],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    device("pull", f"{remote}/results", output, capture=False)
    if result.returncode:
        raise RuntimeError(
            f"Android NPU inference failed ({result.returncode}); see {output / 'native.log'}"
        )
    split = folder / SPLIT
    if file_sha256(split) != bound["bixby50_split_sha256"]:
        raise ValueError("Prepared Bixby50 scoring context changed during evaluation")
    score_results(output / "results", output, requests, runtime, split)
    return output / "aggregate_metrics.json"
