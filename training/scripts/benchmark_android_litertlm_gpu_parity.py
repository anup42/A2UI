#!/usr/bin/env python3
"""Compare official and candidate LiteRT-LM packages on an Android GPU.

The script never trains a model and never overwrites a catalog model. It stages
two temporary packages under ``/data/local/tmp/litert_parity``, invokes the
``LiteRtGpuInitParityProbeTest`` instrumentation test, parses LiteRT's own
delegation/MTP logs, and writes a compact JSON report. Temporary device models,
probe reports, and probe-only cache directories are removed unless explicitly
retained.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_ROOT = "/data/local/tmp/litert_parity"
DEFAULT_TEST_CLASS = "com.samsung.genuicraft.LiteRtGpuInitParityProbeTest"
DEFAULT_RUNNER = (
    "com.samsung.genuicraft.test/androidx.test.runner.AndroidJUnitRunner"
)
DEFAULT_APP_PACKAGE = "com.samsung.genuicraft"

DELEGATION_RE = re.compile(
    r"Replacing\s+(?P<delegated>\d+)\s+out of\s+(?P<total>\d+)\s+node\(s\)\s+"
    r"with delegate \(LITERT_CL\) node,\s+yielding\s+(?P<partitions>\d+)\s+"
    r"partitions for subgraph\s+(?P<subgraph>\d+)\s+\((?P<name>[^)]+)\)",
    re.IGNORECASE,
)
SIGNATURE_RE = re.compile(
    r"signature=(?P<name>[A-Za-z0-9_]+),\s+"
    r"subgraph_index=(?P<subgraph>\d+),\s+"
    r"num_tensors=(?P<tensors>\d+),\s+"
    r"num_inputs=(?P<inputs>\d+),\s+"
    r"num_outputs=(?P<outputs>\d+),\s+"
    r"num_ops=(?P<ops>\d+)",
    re.IGNORECASE,
)
MTP_SUCCESS_RE = re.compile(
    r"MTP\s+Drafter\s+-\s+Success\s+rate:\s*(?P<rate>[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
SHA256SUM_RE = re.compile(r"^\s*(?P<sha256>[0-9a-fA-F]{64})(?:\s+|$)")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash one host artifact without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256sum_output(text: str) -> str:
    """Extract a normalized SHA-256 from Android toybox/sha256sum output."""

    match = SHA256SUM_RE.match(text)
    if not match:
        raise ValueError(f"Could not parse SHA-256 output: {text.strip()[:200]!r}")
    return match.group("sha256").lower()


def parse_logcat_evidence(text: str) -> dict[str, Any]:
    """Extract delegation, signature, and MTP acceptance evidence."""

    delegations = [
        {
            "delegated_nodes": int(match.group("delegated")),
            "total_nodes": int(match.group("total")),
            "partitions": int(match.group("partitions")),
            "subgraph_index": int(match.group("subgraph")),
            "subgraph_name": match.group("name"),
        }
        for match in DELEGATION_RE.finditer(text)
    ]
    signatures = [
        {
            "name": match.group("name"),
            "subgraph_index": int(match.group("subgraph")),
            "tensor_count": int(match.group("tensors")),
            "input_count": int(match.group("inputs")),
            "output_count": int(match.group("outputs")),
            "operator_count": int(match.group("ops")),
        }
        for match in SIGNATURE_RE.finditer(text)
    ]
    mtp_success_rates = [
        float(match.group("rate")) for match in MTP_SUCCESS_RE.finditer(text)
    ]
    return {
        "gpu_delegations": delegations,
        "gpu_delegation_count": len(delegations),
        "all_gpu_subgraphs_fully_delegated": bool(delegations)
        and all(
            item["delegated_nodes"] == item["total_nodes"]
            and item["partitions"] == 1
            for item in delegations
        ),
        "signatures": signatures,
        "mtp_success_rates": mtp_success_rates,
        "last_mtp_success_rate": mtp_success_rates[-1]
        if mtp_success_rates
        else None,
    }


def _delegation_shape(evidence: dict[str, Any]) -> list[tuple[int, str, int]]:
    return sorted(
        (
            int(item["subgraph_index"]),
            str(item["subgraph_name"]),
            int(item["total_nodes"]),
        )
        for item in evidence.get("gpu_delegations", [])
    )


def _signature_shape(evidence: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (
            str(item["name"]),
            int(item["subgraph_index"]),
            int(item["tensor_count"]),
            int(item["input_count"]),
            int(item["output_count"]),
            int(item["operator_count"]),
        )
        for item in evidence.get("signatures", [])
    )


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _artifact_identity_checks(probe: dict[str, Any]) -> dict[str, Any]:
    identity = probe.get("artifact_identity")
    device_report = probe.get("device_report") or {}
    if not isinstance(identity, dict):
        identity = {}

    host_sha256 = str(identity.get("host_sha256") or "").lower()
    staged_sha256 = str(identity.get("staged_device_sha256") or "").lower()
    post_run_sha256 = str(identity.get("post_run_device_sha256") or "").lower()
    valid_sha = re.compile(r"[0-9a-f]{64}").fullmatch
    host_size = _nonnegative_int(identity.get("host_size_bytes"))
    report_size = _nonnegative_int(device_report.get("model_size_bytes"))
    staged_path = str(identity.get("device_path") or "")
    report_path = str(device_report.get("model_path") or "")

    checks = {
        "host_sha256_valid": bool(valid_sha(host_sha256)),
        "staged_device_sha256_valid": bool(valid_sha(staged_sha256)),
        "post_run_device_sha256_valid": bool(valid_sha(post_run_sha256)),
        "host_matches_staged_device": bool(host_sha256)
        and host_sha256 == staged_sha256,
        "host_matches_post_run_device": bool(host_sha256)
        and host_sha256 == post_run_sha256,
        "device_report_size_matches_host": host_size is not None
        and report_size == host_size,
        "device_report_path_matches_staged_path": bool(staged_path)
        and report_path == staged_path,
    }
    checks["verified"] = all(checks.values())
    return {
        "host_sha256": host_sha256 or None,
        "staged_device_sha256": staged_sha256 or None,
        "post_run_device_sha256": post_run_sha256 or None,
        "host_size_bytes": host_size,
        "device_report_size_bytes": report_size,
        "staged_device_path": staged_path or None,
        "device_report_model_path": report_path or None,
        "checks": checks,
    }


def compare_probe_results(
    official: dict[str, Any],
    candidate: dict[str, Any],
    *,
    mtp_enabled: bool,
    output_tokens: int,
    max_throughput_regression_percent: float,
    max_mtp_success_rate_drop: float,
    mtp_max_decode_overshoot: int = 4,
) -> dict[str, Any]:
    """Build explicit structural, speed, and MTP-acceptance gates."""

    official_evidence = official["logcat_evidence"]
    candidate_evidence = candidate["logcat_evidence"]
    official_report = official.get("device_report") or {}
    candidate_report = candidate.get("device_report") or {}
    official_identity = _artifact_identity_checks(official)
    candidate_identity = _artifact_identity_checks(candidate)

    official_rate = _finite_number(official_report.get("decode_tokens_per_second"))
    candidate_rate = _finite_number(candidate_report.get("decode_tokens_per_second"))
    official_decode_count = _nonnegative_int(official_report.get("decode_token_count"))
    candidate_decode_count = _nonnegative_int(candidate_report.get("decode_token_count"))
    allowed_decode_overshoot = mtp_max_decode_overshoot if mtp_enabled else 0
    decode_length_match: bool | None = None
    requested_decode_length_reached: bool | None = None
    official_decode_cap_reached: bool | None = None
    candidate_decode_cap_reached: bool | None = None
    throughput_sample_comparable: bool | None = None
    if output_tokens > 0:
        decode_length_match = (
            official_decode_count is not None
            and candidate_decode_count is not None
            and official_decode_count == candidate_decode_count
        )
        maximum_valid_decode_count = output_tokens + allowed_decode_overshoot
        official_decode_cap_reached = (
            official_decode_count is not None
            and output_tokens <= official_decode_count <= maximum_valid_decode_count
        )
        candidate_decode_cap_reached = (
            candidate_decode_count is not None
            and output_tokens <= candidate_decode_count <= maximum_valid_decode_count
        )
        requested_decode_length_reached = bool(
            official_decode_cap_reached and candidate_decode_cap_reached
        )
        throughput_sample_comparable = requested_decode_length_reached
    throughput_regression = None
    throughput_gate = False if output_tokens > 0 else None
    if (
        output_tokens > 0
        and throughput_sample_comparable
        and official_rate is not None
        and official_rate > 0
        and candidate_rate is not None
    ):
        throughput_regression = 100.0 * (official_rate - candidate_rate) / official_rate
        throughput_gate = throughput_regression <= max_throughput_regression_percent

    official_mtp_rate = _finite_number(official_evidence.get("last_mtp_success_rate"))
    candidate_mtp_rate = _finite_number(candidate_evidence.get("last_mtp_success_rate"))
    mtp_rate_drop = None
    mtp_acceptance_gate = None
    if mtp_enabled and output_tokens > 0:
        if official_mtp_rate is not None and candidate_mtp_rate is not None:
            mtp_rate_drop = official_mtp_rate - candidate_mtp_rate
            mtp_acceptance_gate = mtp_rate_drop <= max_mtp_success_rate_drop
        else:
            mtp_acceptance_gate = False

    official_size = official_report.get("model_size_bytes")
    candidate_size = candidate_report.get("model_size_bytes")
    size_match = (
        official_size is not None
        and candidate_size is not None
        and official_size == candidate_size
    )
    delegation_shape_match = _delegation_shape(official_evidence) == _delegation_shape(
        candidate_evidence
    )
    official_signatures = _signature_shape(official_evidence)
    candidate_signatures = _signature_shape(candidate_evidence)
    signature_evidence_available = bool(official_signatures or candidate_signatures)
    signature_shape_match: bool | None
    if signature_evidence_available:
        signature_shape_match = bool(official_signatures and candidate_signatures) and (
            official_signatures == candidate_signatures
        )
    else:
        signature_shape_match = None
    structural_pass = all(
        (
            official.get("instrumentation_passed"),
            candidate.get("instrumentation_passed"),
            official_evidence.get("all_gpu_subgraphs_fully_delegated"),
            candidate_evidence.get("all_gpu_subgraphs_fully_delegated"),
            delegation_shape_match,
            signature_shape_match is not False,
            size_match,
            official_identity["checks"]["verified"],
            candidate_identity["checks"]["verified"],
        )
    )
    performance_pass = throughput_gate is not False and mtp_acceptance_gate is not False
    return {
        "official_and_candidate_package_size_match": size_match,
        "official_artifact_identity": official_identity,
        "candidate_artifact_identity": candidate_identity,
        "official_artifact_identity_verified": official_identity["checks"][
            "verified"
        ],
        "candidate_artifact_identity_verified": candidate_identity["checks"][
            "verified"
        ],
        "official_full_gpu_delegation": bool(
            official_evidence.get("all_gpu_subgraphs_fully_delegated")
        ),
        "candidate_full_gpu_delegation": bool(
            candidate_evidence.get("all_gpu_subgraphs_fully_delegated")
        ),
        "delegation_shape_match": delegation_shape_match,
        "signature_evidence_available": signature_evidence_available,
        "signature_shape_match": signature_shape_match,
        "structural_gpu_parity_pass": structural_pass,
        "official_decode_token_count": official_decode_count,
        "candidate_decode_token_count": candidate_decode_count,
        "official_decode_token_delta_from_request": (
            official_decode_count - output_tokens
            if official_decode_count is not None and output_tokens > 0
            else None
        ),
        "candidate_decode_token_delta_from_request": (
            candidate_decode_count - output_tokens
            if candidate_decode_count is not None and output_tokens > 0
            else None
        ),
        "mtp_max_decode_overshoot": mtp_max_decode_overshoot if mtp_enabled else None,
        "allowed_decode_overshoot": allowed_decode_overshoot,
        "official_decode_cap_reached": official_decode_cap_reached,
        "candidate_decode_cap_reached": candidate_decode_cap_reached,
        "decode_length_match": decode_length_match,
        "requested_decode_length_reached": requested_decode_length_reached,
        "throughput_sample_comparable": throughput_sample_comparable,
        "official_decode_tokens_per_second": official_rate,
        "candidate_decode_tokens_per_second": candidate_rate,
        "throughput_regression_percent": throughput_regression,
        "max_throughput_regression_percent": max_throughput_regression_percent,
        "throughput_gate_pass": throughput_gate,
        "official_mtp_success_rate": official_mtp_rate,
        "candidate_mtp_success_rate": candidate_mtp_rate,
        "mtp_success_rate_drop": mtp_rate_drop,
        "max_mtp_success_rate_drop": max_mtp_success_rate_drop,
        "mtp_acceptance_gate_pass": mtp_acceptance_gate,
        "performance_gate_pass": performance_pass,
        "overall_pass": bool(structural_pass and performance_pass),
        "interpretation": (
            "Structural GPU parity proves the same signatures and delegated node "
            "counts and cryptographically binds each run to the host and staged device "
            "artifact before and after execution. Throughput passes only when both "
            "probes reach the requested decode "
            "cap under identical sampler settings. A non-MTP run must report the exact "
            "cap; an MTP run may include at most one configured verifier batch of "
            "overshoot because LiteRT-LM checks the cap after Decode() advances the "
            "sequence. Decode throughput and MTP acceptance remain weight-dependent, "
            "and a preserved official drafter does not by itself guarantee official "
            "MTP speed."
        ),
    }


class AndroidProbeRunner:
    def __init__(
        self,
        *,
        adb: str,
        serial: str,
        app_package: str,
        test_class: str,
        runner: str,
        timeout_seconds: int,
    ) -> None:
        self.adb = adb
        self.serial = serial
        self.app_package = app_package
        self.test_class = test_class
        self.runner = runner
        self.timeout_seconds = timeout_seconds

    def adb_command(
        self,
        *arguments: str,
        check: bool = True,
        timeout: int | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            [self.adb, "-s", self.serial, *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or self.timeout_seconds,
            check=False,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(
                f"adb command failed ({completed.returncode}): {' '.join(arguments)}\n"
                f"{completed.stdout}"
            )
        return CommandResult(completed.returncode, completed.stdout)

    def device_sha256(self, device_path: str) -> str:
        failures: list[str] = []
        for command in (
            ("shell", "sha256sum", device_path),
            ("shell", "toybox", "sha256sum", device_path),
        ):
            result = self.adb_command(*command, check=False)
            if result.returncode == 0:
                try:
                    return parse_sha256sum_output(result.stdout)
                except ValueError as exc:
                    failures.append(str(exc))
            else:
                failures.append(result.stdout.strip()[:200])
        raise RuntimeError(
            f"Could not hash staged device artifact {device_path}: {failures}"
        )

    def stage(self, host_path: Path, device_path: str) -> dict[str, Any]:
        host_sha256 = sha256_file(host_path)
        self.adb_command("push", str(host_path), device_path)
        self.adb_command("shell", "chmod", "644", device_path)
        staged_sha256 = self.device_sha256(device_path)
        if staged_sha256 != host_sha256:
            raise RuntimeError(
                "Staged artifact SHA-256 mismatch: "
                f"host={host_sha256}, device={staged_sha256}, path={device_path}"
            )
        return {
            "algorithm": "SHA-256",
            "host_size_bytes": host_path.stat().st_size,
            "host_sha256": host_sha256,
            "device_path": device_path,
            "staged_device_sha256": staged_sha256,
            "post_run_device_sha256": None,
        }

    def run_once(
        self,
        *,
        device_path: str,
        label: str,
        mtp_enabled: bool,
        max_num_tokens: int,
        output_tokens: int,
        top_k: int,
        top_p: float,
        temperature: float,
        seed: int,
        prompt: str,
        output_dir: Path,
        run_index: int,
    ) -> dict[str, Any]:
        self.adb_command("logcat", "-c")
        instrumentation_args = [
            "shell",
            "am",
            "instrument",
            "-w",
            "-r",
            "-e",
            "class",
            self.test_class,
            "-e",
            "modelPath",
            device_path,
            "-e",
            "label",
            label,
            "-e",
            "mtp",
            str(mtp_enabled).lower(),
            "-e",
            "maxNumTokens",
            str(max_num_tokens),
            "-e",
            "outputTokens",
            str(output_tokens),
            "-e",
            "topK",
            str(top_k),
            "-e",
            "topP",
            str(top_p),
            "-e",
            "temperature",
            str(temperature),
            "-e",
            "seed",
            str(seed),
            "-e",
            "promptBase64",
            base64.b64encode(prompt.encode("utf-8")).decode("ascii"),
        ]
        instrumentation_args.append(self.runner)
        instrumentation = self.adb_command(
            *instrumentation_args,
            check=False,
            timeout=self.timeout_seconds,
        )
        logcat = self.adb_command("logcat", "-d", "-v", "threadtime").stdout
        instrumentation_passed = (
            instrumentation.returncode == 0
            and "OK (1 test)" in instrumentation.stdout
            and "FAILURES!!!" not in instrumentation.stdout
        )
        prefix = output_dir / f"run_{run_index:02d}"
        prefix.with_suffix(".instrumentation.txt").write_text(
            instrumentation.stdout, encoding="utf-8"
        )
        prefix.with_suffix(".logcat.txt").write_text(logcat, encoding="utf-8")
        device_report: dict[str, Any] | None = None
        device_report_error: str | None = None
        report_result = self.adb_command(
            "exec-out",
            "run-as",
            self.app_package,
            "cat",
            f"files/litert_gpu_parity_reports/{label}.json",
            check=False,
        )
        if report_result.returncode == 0 and report_result.stdout.strip():
            try:
                device_report = json.loads(report_result.stdout)
            except json.JSONDecodeError as exc:
                device_report_error = f"{exc}: {report_result.stdout.strip()[:500]}"
        if device_report is not None:
            prefix.with_suffix(".device_report.json").write_text(
                json.dumps(device_report, indent=2) + "\n", encoding="utf-8"
            )
        return {
            "run_index": run_index,
            "instrumentation_passed": instrumentation_passed,
            "instrumentation_returncode": instrumentation.returncode,
            "device_report": device_report,
            "device_report_error": device_report_error,
            "logcat_evidence": parse_logcat_evidence(logcat),
        }

    def cleanup_probe_artifacts(self, *, device_paths: Iterable[str], labels: Iterable[str]) -> None:
        for device_path in device_paths:
            if not device_path.startswith(f"{STAGE_ROOT}/parity_"):
                raise ValueError(f"Refusing to remove unexpected device path: {device_path}")
            self.adb_command("shell", "rm", "-f", device_path, check=False)
        for label in labels:
            if not re.fullmatch(r"parity_[A-Za-z0-9_.-]+", label):
                raise ValueError(f"Refusing to remove unexpected probe label: {label}")
            self.adb_command(
                "shell",
                "run-as",
                self.app_package,
                "rm",
                "-rf",
                f"cache/litert_gpu_parity/{label}",
                check=False,
            )
            self.adb_command(
                "shell",
                "run-as",
                self.app_package,
                "rm",
                "-f",
                f"files/litert_gpu_parity_reports/{label}.json",
                check=False,
            )


def _resolve_adb(explicit: str | None) -> str:
    candidate = explicit or shutil.which("adb")
    if not candidate:
        raise RuntimeError("adb was not found. Pass --adb or add platform-tools to PATH.")
    return candidate


def _resolve_serial(adb: str, explicit: str | None) -> str:
    completed = subprocess.run(
        [adb, "devices"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    devices = [
        line.split()[0]
        for line in completed.stdout.splitlines()
        if len(line.split()) >= 2 and line.split()[1] == "device"
    ]
    if explicit:
        if explicit not in devices:
            raise RuntimeError(
                f"Requested device {explicit!r} is not connected; found {devices}."
            )
        return explicit
    if len(devices) != 1:
        raise RuntimeError(f"Expected one connected device; found {devices}. Pass --serial.")
    return devices[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark official and candidate LiteRT-LM packages on Android GPU."
    )
    parser.add_argument("--official", required=True, help="Official .litertlm package.")
    parser.add_argument("--candidate", required=True, help="Candidate .litertlm package.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adb")
    parser.add_argument("--serial")
    parser.add_argument("--app-package", default=DEFAULT_APP_PACKAGE)
    parser.add_argument("--test-class", default=DEFAULT_TEST_CLASS)
    parser.add_argument("--runner", default=DEFAULT_RUNNER)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--max-num-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        default="Write exactly one hundred numbered words.",
        help="Bounded-generation prompt. Choose one unlikely to stop before the token cap.",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=1,
        help="Additional runs after the cold run; the last run is compared.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-throughput-regression-percent", type=float, default=10.0)
    parser.add_argument("--max-mtp-success-rate-drop", type=float, default=0.10)
    parser.add_argument(
        "--mtp-max-decode-overshoot",
        type=int,
        default=4,
        help=(
            "Maximum extra decode tokens allowed above --output-tokens for an MTP "
            "sample. The released Gemma 4 E2B verifier drafts four tokens, and "
            "LiteRT-LM checks the cap after a verifier batch. Ignored without --mtp."
        ),
    )
    parser.add_argument("--keep-device-artifacts", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    official = Path(args.official).expanduser().resolve()
    candidate = Path(args.candidate).expanduser().resolve()
    for path in (official, candidate):
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"LiteRT-LM package is missing or empty: {path}")
    if args.max_num_tokens < 1024:
        raise ValueError("--max-num-tokens must be at least 1024.")
    if not 0 <= args.output_tokens <= 512:
        raise ValueError("--output-tokens must be between 0 and 512.")
    if args.warm_runs < 0:
        raise ValueError("--warm-runs cannot be negative.")
    if not 0 <= args.mtp_max_decode_overshoot <= 512:
        raise ValueError("--mtp-max-decode-overshoot must be between 0 and 512.")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")
    if not math.isfinite(args.top_p) or not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top-p must be finite and between 0 and 1.")
    if not math.isfinite(args.temperature) or args.temperature < 0.0:
        raise ValueError("--temperature must be finite and non-negative.")

    output_root = Path(args.output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    device_dir = f"{STAGE_ROOT}/parity_{run_id}"
    device_paths = {
        "official": f"{device_dir}/official.litertlm",
        "candidate": f"{device_dir}/candidate.litertlm",
    }
    labels = {
        "official": f"parity_{run_id}_official",
        "candidate": f"parity_{run_id}_candidate",
    }

    adb = _resolve_adb(args.adb)
    serial = _resolve_serial(adb, args.serial)
    runner = AndroidProbeRunner(
        adb=adb,
        serial=serial,
        app_package=args.app_package,
        test_class=args.test_class,
        runner=args.runner,
        timeout_seconds=args.timeout_seconds,
    )
    runner.adb_command("shell", "mkdir", "-p", device_dir)
    runner.adb_command("shell", "chmod", "755", device_dir)
    results: dict[str, Any] = {}
    try:
        staged_identities = {
            "official": runner.stage(official, device_paths["official"]),
            "candidate": runner.stage(candidate, device_paths["candidate"]),
        }
        for role in ("official", "candidate"):
            role_dir = output_root / role
            role_dir.mkdir(parents=True, exist_ok=True)
            role_runs = []
            for run_index in range(args.warm_runs + 1):
                role_runs.append(
                    runner.run_once(
                        device_path=device_paths[role],
                        label=labels[role],
                        mtp_enabled=args.mtp,
                        max_num_tokens=args.max_num_tokens,
                        output_tokens=args.output_tokens,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        seed=args.seed,
                        prompt=args.prompt,
                        output_dir=role_dir,
                        run_index=run_index,
                    )
                )
            identity = dict(staged_identities[role])
            identity["post_run_device_sha256"] = runner.device_sha256(
                device_paths[role]
            )
            selected_run = dict(role_runs[-1])
            selected_run["artifact_identity"] = identity
            results[role] = {
                "host_path": str(official if role == "official" else candidate),
                "host_size_bytes": (
                    official if role == "official" else candidate
                ).stat().st_size,
                "device_path": device_paths[role],
                "artifact_identity": identity,
                "runs": role_runs,
                "selected_run": selected_run,
            }

        comparison = compare_probe_results(
            results["official"]["selected_run"],
            results["candidate"]["selected_run"],
            mtp_enabled=args.mtp,
            output_tokens=args.output_tokens,
            max_throughput_regression_percent=args.max_throughput_regression_percent,
            max_mtp_success_rate_drop=args.max_mtp_success_rate_drop,
            mtp_max_decode_overshoot=args.mtp_max_decode_overshoot,
        )
        report = {
            "schema_version": 4,
            "run_id": run_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "device_serial": serial,
            "mtp_enabled": args.mtp,
            "max_num_tokens": args.max_num_tokens,
            "requested_output_tokens": args.output_tokens,
            "mtp_max_decode_overshoot": (
                args.mtp_max_decode_overshoot if args.mtp else None
            ),
            "sampler": {
                "top_k": args.top_k,
                "top_p": args.top_p,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "prompt": args.prompt,
            "cold_run_count": 1,
            "warm_run_count": args.warm_runs,
            "training_executed": False,
            "results": results,
            "comparison": comparison,
        }
        report_path = output_root / "android_litertlm_gpu_parity_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "overall_pass": comparison["overall_pass"],
                    "structural_gpu_parity_pass": comparison[
                        "structural_gpu_parity_pass"
                    ],
                    "throughput_regression_percent": comparison[
                        "throughput_regression_percent"
                    ],
                    "throughput_sample_comparable": comparison[
                        "throughput_sample_comparable"
                    ],
                    "mtp_success_rate_drop": comparison["mtp_success_rate_drop"],
                },
                indent=2,
            )
        )
        return 0 if comparison["overall_pass"] else 2
    finally:
        if not args.keep_device_artifacts:
            runner.cleanup_probe_artifacts(
                device_paths=device_paths.values(), labels=labels.values()
            )
            if device_dir.startswith(f"{STAGE_ROOT}/parity_"):
                runner.adb_command("shell", "rmdir", device_dir, check=False)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - CLI boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
