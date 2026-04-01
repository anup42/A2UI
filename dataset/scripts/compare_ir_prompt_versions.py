import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.factory import build_adapter, load_model_specs  # noqa: E402
from main import _compute_aggregates_with_backfill, _load_env  # noqa: E402
from pipeline.cache import PromptCache  # noqa: E402
from pipeline.stage3_genui import run_stage3  # noqa: E402
from pipeline.stage4_render import run_stage4  # noqa: E402
from pipeline.storage import get_run_paths  # noqa: E402
from utils.config import load_yaml  # noqa: E402
from utils.logging import setup_logger  # noqa: E402
from utils.rate_limit import RateLimiter  # noqa: E402


def _resolve_run_dir(runs_dir: Path, run_id: str) -> Path:
    candidate = Path(run_id)
    if candidate.exists():
        return candidate.resolve()
    return (runs_dir / run_id).resolve()


def _coerce_viewport(raw: dict | None, fallback: dict[str, int]) -> dict[str, int]:
    if not isinstance(raw, dict):
        return dict(fallback)
    width = raw.get("width")
    height = raw.get("height")
    try:
        width_int = int(width)
        height_int = int(height)
    except Exception:
        return dict(fallback)
    if width_int <= 0 or height_int <= 0:
        return dict(fallback)
    return {"width": width_int, "height": height_int}


def _resolve_render_viewport(render_cfg: dict[str, Any]) -> tuple[dict[str, int], bool]:
    default_viewport = {"width": 1280, "height": 720}
    preset_name = str(render_cfg.get("default_viewport_preset") or "").strip()
    presets = render_cfg.get("viewport_presets")
    if preset_name and isinstance(presets, dict):
        preset_cfg = presets.get(preset_name)
        if isinstance(preset_cfg, dict):
            viewport = _coerce_viewport(preset_cfg, default_viewport)
            emulate_mobile = bool(preset_cfg.get("emulate_mobile", False))
            return viewport, emulate_mobile
    viewport = _coerce_viewport(render_cfg.get("viewport"), default_viewport)
    emulate_mobile = bool(render_cfg.get("emulate_mobile", False))
    return viewport, emulate_mobile


def _load_prompt_steps(path: Path) -> list[dict[str, Any]]:
    data = load_yaml(path)
    section = data.get("ir_prompt_versions") if isinstance(data, dict) else None
    if not isinstance(section, dict):
        raise SystemExit(f"Invalid prompt versions file: {path}")
    steps = section.get("steps")
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"No ir_prompt_versions.steps found in: {path}")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(steps, start=1):
        if not isinstance(item, dict):
            continue
        step_id = str(item.get("id") or "").strip()
        prompt_path = str(item.get("prompt_path") or "").strip()
        if not step_id or not prompt_path:
            continue
        normalized.append(
            {
                "index": index,
                "id": step_id,
                "label": str(item.get("label") or step_id),
                "prompt_path": prompt_path,
                "change_summary": str(item.get("change_summary") or ""),
            }
        )
    if not normalized:
        raise SystemExit(f"No valid steps with id/prompt_path in: {path}")
    return normalized


def _prompt_heading(prompt_path: Path) -> str:
    try:
        first_line = prompt_path.read_text(encoding="utf-8").splitlines()[0].lstrip("\ufeff").strip()
    except Exception:
        return prompt_path.stem
    match = re.match(r"^#\s*([A-Za-z0-9_.-]+)", first_line)
    if match:
        return match.group(1)
    return prompt_path.stem


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run-id",
        type=str,
        default="subset10_g3pro_iconcatalog_headings_20260216_091411",
        help="Run id or absolute path that provides source queries.jsonl/responses.jsonl",
    )
    parser.add_argument("--runs-dir", type=str, default=str(ROOT / "data" / "runs"))
    parser.add_argument("--model", type=str, default="gemini_2_5_flash")
    parser.add_argument(
        "--versions-file",
        type=str,
        default=str(ROOT / "versions" / "ir_prompt_versions.yaml"),
        help="YAML file containing ir_prompt_versions.steps",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(ROOT / "comparison.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--run-id-prefix",
        type=str,
        default=None,
        help="Optional prefix for generated step run_ids",
    )
    parser.add_argument("--force", action="store_true", help="Regenerate stage3 even if genui already exists")
    parser.add_argument("--skip-render", action="store_true", help="Skip stage4 rendering")
    parser.add_argument("--render-workers", type=int, default=None, help="Override stage4 render worker count")
    parser.add_argument("--rate-limit-qps", type=float, default=None)
    parser.add_argument("--max-total", type=int, default=None, help="Optional stage3 max_total override")
    parser.add_argument("--seed", type=int, default=None, help="Optional stage3 seed override")
    args = parser.parse_args()

    _load_env(ROOT)
    runs_dir = Path(args.runs_dir).resolve()
    source_run_dir = _resolve_run_dir(runs_dir, args.source_run_id)
    if not source_run_dir.exists():
        raise SystemExit(f"Source run not found: {source_run_dir}")
    source_queries = source_run_dir / "queries.jsonl"
    source_responses = source_run_dir / "responses.jsonl"
    if not source_queries.exists() or not source_responses.exists():
        raise SystemExit(f"Source run is missing queries/responses JSONL: {source_run_dir}")

    run_cfg_path = ROOT / "configs" / "run.yaml"
    models_cfg_path = ROOT / "configs" / "models.yaml"
    all_run_cfg = load_yaml(run_cfg_path)
    run_cfg = all_run_cfg.get("run", {}) if isinstance(all_run_cfg, dict) else {}
    eval_cfg = all_run_cfg.get("evaluation", {}) if isinstance(all_run_cfg, dict) else {}
    render_cfg = all_run_cfg.get("render", {}) if isinstance(all_run_cfg, dict) else {}
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    if not isinstance(eval_cfg, dict):
        eval_cfg = {}
    if not isinstance(render_cfg, dict):
        render_cfg = {}
    weights = eval_cfg.get("weights", {}) if isinstance(eval_cfg, dict) else {}
    if not isinstance(weights, dict):
        weights = {}

    specs = load_model_specs(load_yaml(models_cfg_path))
    if not specs:
        raise SystemExit("No model specs found in configs/models.yaml")
    model_map = {spec.name: spec for spec in specs}
    spec = model_map.get(args.model)
    if spec is None:
        available = ", ".join(sorted(model_map.keys()))
        raise SystemExit(f"Unknown model '{args.model}'. Available: {available}")
    adapter = build_adapter(spec)

    steps = _load_prompt_steps(Path(args.versions_file).resolve())
    artifact_dir_name = str(run_cfg.get("artifact_dir", "artifacts"))
    cache = PromptCache(ROOT / str(run_cfg.get("cache_dir", "data/cache")))
    rate_limiter = RateLimiter(
        float(args.rate_limit_qps) if args.rate_limit_qps is not None else float(run_cfg.get("rate_limit_qps", 1.0)),
        float(run_cfg.get("call_sleep_seconds", 0.0)),
    )
    schema_path = ROOT / "schema" / "genui.schema.json"
    stage3_batch_size = int(run_cfg.get("genui_batch_size", 1))
    stage3_seed = int(args.seed) if args.seed is not None else int(run_cfg.get("seed", 42))
    stage3_max_total = int(args.max_total) if args.max_total is not None else run_cfg.get("max_genui_total")
    stage3_max_tokens = int(run_cfg.get("genui_max_tokens", 1024))
    stage3_prompt_max_tokens = run_cfg.get("genui_prompt_max_tokens")
    stage3_prompt_max_tokens = int(stage3_prompt_max_tokens) if stage3_prompt_max_tokens else None
    stage3_candidates = int(run_cfg.get("genui_candidates_per_response", 1))
    stage3_max_repair_attempts = int(run_cfg.get("max_repair_attempts", 1))
    stage3_max_attempts = int(run_cfg.get("max_attempts", 3))

    render_output_dir_name = str(render_cfg.get("output_dir", "rendered"))
    render_assets_dir = ROOT / str(render_cfg.get("assets_dir", "renderer/lit"))
    viewport, emulate_mobile = _resolve_render_viewport(render_cfg)
    render_workers = (
        int(args.render_workers)
        if args.render_workers is not None
        else int(render_cfg.get("parallel_workers", 1))
    )

    run_prefix = args.run_id_prefix or f"{source_run_dir.name}_ir_step"
    output_rows: list[dict[str, Any]] = []
    previous_overall: float | None = None

    for step in steps:
        step_index = int(step["index"])
        step_id = str(step["id"])
        step_run_id = f"{run_prefix}{step_index:02d}_{step_id}"
        run_paths = get_run_paths(runs_dir, step_run_id, artifact_dir_name)
        run_paths.run_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger(run_paths.run_dir)

        prompt_path = (ROOT / step["prompt_path"]).resolve()
        if not prompt_path.exists():
            raise SystemExit(f"Prompt file for step '{step_id}' not found: {prompt_path}")

        if not run_paths.queries_path.exists():
            shutil.copy2(source_queries, run_paths.queries_path)
        if not run_paths.responses_path.exists():
            shutil.copy2(source_responses, run_paths.responses_path)

        if args.force:
            if run_paths.genui_path.exists():
                run_paths.genui_path.unlink()
            if run_paths.aggregates_path.exists():
                run_paths.aggregates_path.unlink()
            render_log = run_paths.run_dir / "render.jsonl"
            if render_log.exists():
                render_log.unlink()
            render_dir = run_paths.run_dir / render_output_dir_name
            if render_dir.exists():
                shutil.rmtree(render_dir, ignore_errors=True)

        if not run_paths.genui_path.exists():
            logger.info("Generating stage3 for step=%s prompt=%s", step_id, prompt_path)
            run_stage3(
                queries_path=run_paths.queries_path,
                responses_path=run_paths.responses_path,
                prompt_path=prompt_path,
                adapter=adapter,
                genui_path=run_paths.genui_path,
                schema_path=schema_path,
                artifacts_dir=run_paths.artifacts_dir,
                candidates_per_response=stage3_candidates,
                max_repair_attempts=stage3_max_repair_attempts,
                max_tokens=stage3_max_tokens,
                prompt_max_tokens=stage3_prompt_max_tokens,
                batch_size=stage3_batch_size,
                seed=stage3_seed,
                rate_limiter=rate_limiter,
                cache=cache,
                logger=logger,
                max_total=stage3_max_total,
                max_attempts=stage3_max_attempts,
                aggregates_path=run_paths.aggregates_path,
                aggregate_weights=weights,
            )
        else:
            logger.info("Reusing existing stage3 output for step=%s (%s)", step_id, run_paths.genui_path)

        if not args.skip_render:
            render_output_dir = run_paths.run_dir / render_output_dir_name
            logger.info(
                "Rendering step=%s viewport=%s emulate_mobile=%s workers=%s",
                step_id,
                viewport,
                emulate_mobile,
                render_workers,
            )
            run_stage4(
                genui_path=run_paths.genui_path,
                output_dir=render_output_dir,
                assets_dir=render_assets_dir,
                server_root=ROOT,
                logger=logger,
                max_total=render_cfg.get("max_total"),
                render_images=bool(render_cfg.get("render_images", True)),
                image_format=str(render_cfg.get("image_format", "png")),
                viewport=viewport,
                timeout_ms=int(render_cfg.get("timeout_ms", 15000)),
                wait_ms=int(render_cfg.get("wait_ms", 200)),
                use_http_server=bool(render_cfg.get("use_http_server", True)),
                parallel_workers=render_workers,
                emulate_mobile=emulate_mobile,
            )
            refreshed = _compute_aggregates_with_backfill(
                run_paths.genui_path,
                run_paths.responses_path,
                weights,
            )
            run_paths.aggregates_path.write_text(json.dumps(refreshed, indent=2), encoding="utf-8")

        if not run_paths.aggregates_path.exists():
            raise SystemExit(f"Missing aggregates for step '{step_id}': {run_paths.aggregates_path}")
        aggregates = json.loads(run_paths.aggregates_path.read_text(encoding="utf-8"))
        overall = _safe_float(aggregates.get("overall_score"))
        delta = None if (overall is None or previous_overall is None) else (overall - previous_overall)
        metric_achieved = (
            f"overall_score={overall:.4f}"
            if delta is None
            else f"overall_score={overall:.4f} (delta_vs_prev={delta:+.4f})"
        ) if overall is not None else ""

        output_rows.append(
            {
                "step": step_index,
                "step_id": step_id,
                "label": step["label"],
                "run_id": step_run_id,
                "prompt_file": str(prompt_path.relative_to(ROOT)),
                "prompt_heading": _prompt_heading(prompt_path),
                "changes_done": step["change_summary"],
                "metric_achieved": metric_achieved,
                "overall_score": _fmt(overall),
                "overall_delta_vs_prev": _fmt(delta),
                "schema_valid_strict_rate": _fmt(_safe_float(aggregates.get("schema_valid_strict_rate"))),
                "content_coverage_avg": _fmt(_safe_float(aggregates.get("content_coverage_avg"))),
                "ui_decomposition_score_avg": _fmt(_safe_float(aggregates.get("ui_decomposition_score_avg"))),
                "table_cell_coverage_avg": _fmt(_safe_float(aggregates.get("table_cell_coverage_avg"))),
                "section_heading_coverage_avg": _fmt(_safe_float(aggregates.get("section_heading_coverage_avg"))),
                "action_coverage_avg": _fmt(_safe_float(aggregates.get("action_coverage_avg"))),
                "media_score": _fmt(_safe_float(aggregates.get("media_score"))),
                "rendered_image_ok_rate": _fmt(_safe_float(aggregates.get("rendered_image_ok_rate"))),
            }
        )
        previous_overall = overall if overall is not None else previous_overall

    output_csv_path = Path(args.output_csv).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "step_id",
        "label",
        "run_id",
        "prompt_file",
        "prompt_heading",
        "changes_done",
        "metric_achieved",
        "overall_score",
        "overall_delta_vs_prev",
        "schema_valid_strict_rate",
        "content_coverage_avg",
        "ui_decomposition_score_avg",
        "table_cell_coverage_avg",
        "section_heading_coverage_avg",
        "action_coverage_avg",
        "media_score",
        "rendered_image_ok_rate",
    ]
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote comparison CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
