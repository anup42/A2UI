from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


DATASET_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = DATASET_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from llm.base import BaseLLMAdapter, LLMResult, ModelSpec  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.stage3_genui import run_stage3  # noqa: E402
from pipeline.storage import get_run_paths, iter_jsonl  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.logging import setup_logger  # noqa: E402
from utils.rate_limit import RateLimiter  # noqa: E402


class VertexBatchGeminiAdapter(BaseLLMAdapter):
    """True Vertex AI batchPredictionJobs adapter for Stage 3 generation.

    This intentionally does not use Vertex Express API-key auth. Vertex batch
    prediction requires OAuth bearer auth plus GCS/BigQuery input and output.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        project_id: str,
        location: str,
        gcs_bucket: str,
        gcs_prefix: str,
        local_dir: Path,
        poll_seconds: float,
        timeout_seconds: float,
        logger,
    ) -> None:
        super().__init__(spec)
        self.batch_only = True
        self.project_id = project_id
        self.location = location
        self.gcs_bucket = gcs_bucket.replace("gs://", "").strip("/")
        self.gcs_prefix = gcs_prefix.strip("/")
        self.local_dir = local_dir
        self.poll_seconds = max(10.0, poll_seconds)
        self.timeout_seconds = max(600.0, timeout_seconds)
        self.logger = logger
        self.local_dir.mkdir(parents=True, exist_ok=True)

    def _token(self) -> str:
        return subprocess.check_output(
            ["gcloud.cmd", "auth", "print-access-token"],
            text=True,
            timeout=60,
        ).strip()

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self._token()}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310 - Google API endpoint
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vertex batch POST failed http={exc.code}: {body}") from exc

    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self._token()}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # nosec B310 - Google API endpoint
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vertex batch GET failed http={exc.code}: {body}") from exc

    def _gcs_uri(self, *parts: str) -> str:
        joined = "/".join(str(p).strip("/") for p in parts if str(p).strip("/"))
        return f"gs://{self.gcs_bucket}/{joined}"

    def _upload(self, local_path: Path, gcs_uri: str) -> None:
        subprocess.check_call(["gsutil.cmd", "cp", str(local_path), gcs_uri], timeout=300)

    def _download_prefix(self, gcs_prefix: str, local_path: Path) -> list[Path]:
        local_path.mkdir(parents=True, exist_ok=True)
        listing = subprocess.check_output(
            ["gsutil.cmd", "ls", "-r", gcs_prefix],
            text=True,
            timeout=300,
        )
        uris = [line.strip() for line in listing.splitlines() if line.strip().endswith(".jsonl")]
        out: list[Path] = []
        for idx, uri in enumerate(uris):
            filename = Path(uri.rstrip("/")).name or f"predictions_{idx}.jsonl"
            if filename in {p.name for p in out}:
                filename = f"{idx:04d}_{filename}"
            dest = local_path / filename
            subprocess.check_call(["gsutil.cmd", "cp", uri, str(dest)], timeout=1800)
            out.append(dest)
        return out

    def _make_request(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool,
    ) -> dict[str, Any]:
        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        if seed is not None:
            generation_config["seed"] = seed
        request: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        if system:
            request["systemInstruction"] = {"parts": [{"text": system}]}
        return request

    def _extract_prompt(self, request: dict[str, Any]) -> str:
        try:
            contents = request.get("contents") or []
            parts = contents[0].get("parts") or []
            return str(parts[0].get("text") or "")
        except Exception:
            return ""

    def _extract_text(self, response: dict[str, Any]) -> str:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return ""
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts:
            return ""
        return str(parts[0].get("text") or "")

    def generate(
        self,
        prompt: str,
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seed: Optional[int],
        json_mode: bool = False,
    ) -> LLMResult:
        return self.generate_batch(
            [prompt],
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            seeds=[seed or 0],
            json_mode=json_mode,
            batch_name=f"stage3_single_{int(time.time())}",
        )[0]

    def generate_batch(
        self,
        prompts: list[str],
        system: Optional[str],
        temperature: float,
        max_tokens: int,
        seeds: Optional[list[int]],
        json_mode: bool = False,
        batch_name: Optional[str] = None,
    ) -> list[LLMResult]:
        if not prompts:
            return []
        started = time.time()
        safe_name = (batch_name or f"stage3_{int(started)}").replace(":", "_").replace("/", "_")
        batch_dir = self.local_dir / safe_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        input_path = batch_dir / "input.jsonl"
        output_local = batch_dir / "output"

        prompt_to_indices: dict[str, list[int]] = {}
        with input_path.open("w", encoding="utf-8") as handle:
            for idx, prompt in enumerate(prompts):
                prompt_to_indices.setdefault(prompt, []).append(idx)
                seed = seeds[idx] if seeds and idx < len(seeds) else None
                request = self._make_request(prompt, system, temperature, max_tokens, seed, json_mode)
                handle.write(json.dumps({"request": request}, ensure_ascii=False, separators=(",", ":")) + "\n")

        input_uri = self._gcs_uri(self.gcs_prefix, safe_name, "input.jsonl")
        output_prefix = self._gcs_uri(self.gcs_prefix, safe_name, "output")
        self.logger.info("Vertex batch upload name=%s prompts=%s input=%s", safe_name, len(prompts), input_uri)
        self._upload(input_path, input_uri)

        endpoint = (
            f"https://{self.location}-aiplatform.googleapis.com/v1/"
            f"projects/{self.project_id}/locations/{self.location}/batchPredictionJobs"
        )
        payload = {
            "displayName": safe_name,
            "model": f"publishers/google/models/{self.spec.model}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [input_uri]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": output_prefix},
            },
        }
        job = self._post_json(endpoint, payload)
        job_name = str(job.get("name") or "")
        if not job_name:
            raise RuntimeError(f"Vertex batch response missing job name: {job}")
        self.logger.info("Vertex batch submitted name=%s job=%s", safe_name, job_name)

        deadline = time.time() + self.timeout_seconds
        get_url = f"https://{self.location}-aiplatform.googleapis.com/v1/{job_name}"
        state = ""
        while time.time() < deadline:
            job = self._get_json(get_url)
            state = str(job.get("state") or "")
            self.logger.info("Vertex batch poll name=%s state=%s", safe_name, state)
            if state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}:
                break
            time.sleep(self.poll_seconds)
        if state != "JOB_STATE_SUCCEEDED":
            raise RuntimeError(f"Vertex batch job did not succeed state={state} job={job}")

        output_dir = (
            job.get("outputInfo", {}).get("gcsOutputDirectory")
            or job.get("outputConfig", {}).get("gcsDestination", {}).get("outputUriPrefix")
            or output_prefix
        )
        output_files = self._download_prefix(str(output_dir).rstrip("/") + "/", output_local)
        if not output_files:
            raise RuntimeError(f"Vertex batch output contains no jsonl files: {output_dir}")

        results: list[LLMResult | None] = [None] * len(prompts)
        usage_by_idx: dict[int, tuple[int, int]] = {}
        errors: list[str] = []
        for output_file in output_files:
            with output_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    request = row.get("request") if isinstance(row.get("request"), dict) else {}
                    prompt = self._extract_prompt(request)
                    indices = prompt_to_indices.get(prompt)
                    if not indices:
                        errors.append(f"unmatched output row from {output_file.name}")
                        continue
                    idx = indices.pop(0)
                    status = str(row.get("status") or "")
                    response = row.get("response") if isinstance(row.get("response"), dict) else {}
                    usage = response.get("usageMetadata") if isinstance(response, dict) else {}
                    if isinstance(usage, dict):
                        usage_by_idx[idx] = (
                            int(usage.get("promptTokenCount") or 0),
                            int(usage.get("candidatesTokenCount") or 0),
                        )
                    text = "" if status else self._extract_text(response)
                    error = status or (None if text else "empty batch response")
                    in_tokens, out_tokens = usage_by_idx.get(idx, (0, 0))
                    results[idx] = LLMResult(
                        text=text,
                        raw={
                            "batch_mode": "vertex_batch_prediction",
                            "batch_name": safe_name,
                            "job_name": job_name,
                            "output_dir": output_dir,
                            "task_index": idx,
                            "status": status,
                        },
                        latency_ms=(time.time() - started) * 1000.0,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        cost_usd=None,
                        model=self.spec.model,
                        provider=self.spec.provider,
                        error=error,
                    )

        missing = [idx for idx, result in enumerate(results) if result is None]
        if missing:
            raise RuntimeError(f"Vertex batch missing {len(missing)} outputs; first_missing={missing[:10]} errors={errors[:5]}")
        if errors:
            self.logger.warning("Vertex batch output mapping warnings: %s", errors[:5])
        return [result for result in results if result is not None]


def count_jsonl(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 using true Vertex AI Gemini batch prediction.")
    parser.add_argument("--run_id", default="dataset_v0")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--project_id", default=None)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--gcs_bucket", default="a2ui-stage3-batch-gen-lang-client-0335524277")
    parser.add_argument("--gcs_prefix", default=None)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--max_total", type=int, default=None)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--timeout_seconds", type=float, default=21600.0)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--prompt_max_tokens", type=int, default=None)
    parser.add_argument("--schema_file", default=None)
    parser.add_argument("--prompt_file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_cfg = load_yaml(DATASET_ROOT / "configs" / "run.yaml").get("run", {})
    eval_cfg = load_yaml(DATASET_ROOT / "configs" / "run.yaml").get("evaluation", {})

    project_id = args.project_id or subprocess.check_output(
        ["gcloud.cmd", "config", "get-value", "project"],
        text=True,
        timeout=60,
    ).strip()
    if not project_id:
        raise SystemExit("Missing project id. Pass --project_id or configure gcloud project.")

    output_dir = Path(run_cfg.get("output_dir", "data/runs"))
    if not output_dir.is_absolute():
        output_dir = DATASET_ROOT / output_dir
    run_paths = get_run_paths(output_dir, args.run_id, run_cfg.get("artifact_dir", "artifacts"))
    logger = setup_logger(run_paths.run_dir)

    prompt_path = DATASET_ROOT / (args.prompt_file or run_cfg.get("stage3_prompt_file", "prompts/genui_gen_mobile_a2ui_express_v1.md"))
    schema_path = DATASET_ROOT / (args.schema_file or run_cfg.get("stage3_schema_file", "schema/canonical_ui_graph_v1.schema.json"))
    if not prompt_path.exists():
        prompt_path = DATASET_ROOT / "prompts" / "genui_gen_mobile_a2ui_express_v1.md"
    if not schema_path.exists():
        schema_path = DATASET_ROOT / "schema" / "canonical_ui_graph_v1.schema.json"

    model_spec = ModelSpec(
        name="vertex_batch_gemini_2_5_flash",
        provider="gemini",
        model=args.model,
        supports_json_mode=True,
        limits={"max_output_tokens": 65536},
    )
    gcs_prefix = args.gcs_prefix or f"a2ui-stage3/{args.run_id}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    adapter = VertexBatchGeminiAdapter(
        model_spec,
        project_id=project_id,
        location=args.location,
        gcs_bucket=args.gcs_bucket,
        gcs_prefix=gcs_prefix,
        local_dir=run_paths.run_dir / "vertex_batch_stage3",
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        logger=logger,
    )

    before = count_jsonl(run_paths.genui_path)
    logger.info(
        "Starting true Vertex Stage3 batch run run_id=%s model=%s batch_size=%s before=%s responses=%s",
        args.run_id,
        args.model,
        args.batch_size,
        before,
        count_jsonl(run_paths.responses_path),
    )
    run_stage3(
        queries_path=run_paths.queries_path,
        responses_path=run_paths.responses_path,
        prompt_path=prompt_path,
        adapter=adapter,
        genui_path=run_paths.genui_path,
        schema_path=schema_path,
        artifacts_dir=run_paths.artifacts_dir,
        candidates_per_response=int(run_cfg.get("genui_candidates_per_response", 1)),
        max_repair_attempts=int(run_cfg.get("max_repair_attempts", 1)),
        max_tokens=int(args.max_tokens or run_cfg.get("genui_max_tokens", 8192)),
        prompt_max_tokens=int(args.prompt_max_tokens or run_cfg.get("genui_prompt_max_tokens", 60000)),
        batch_size=max(1, int(args.batch_size)),
        seed=int(run_cfg.get("seed", 42)),
        rate_limiter=RateLimiter(1000.0, 0.0),
        cache=PromptCache(run_paths.run_dir / ".prompt_cache_stage3_vertex_batch.jsonl"),
        logger=logger,
        max_total=args.max_total,
        max_attempts=1,
        aggregates_path=run_paths.aggregates_path,
        aggregate_weights=eval_cfg.get("weights", {}),
        metric_version=eval_cfg.get("metric_version", "v5_4"),
    )
    after = count_jsonl(run_paths.genui_path)
    logger.info("Finished true Vertex Stage3 batch run run_id=%s before=%s after=%s created=%s", args.run_id, before, after, after - before)


if __name__ == "__main__":
    main()
