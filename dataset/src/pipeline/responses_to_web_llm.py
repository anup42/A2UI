from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm.factory import build_adapter, load_model_specs
from pipeline.common import extract_json, load_prompt, render_prompt
from utils.config import load_yaml
from utils.retry import with_retry


def _load_env(root: Path) -> None:
    force_override_keys = {
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_API_KEYS",
        "OPENROUTER_API_KEY",
        "PERPLEXITY_API_KEY",
        "GAUSS_OPENAPI_TOKEN",
        "GAUSS_CLIENT_KEY",
    }

    def _should_force_override(key: str) -> bool:
        upper = key.upper()
        if upper in force_override_keys:
            return True
        return upper.endswith("_API_KEY") or upper.endswith("_API_KEYS")

    for env_path in [root / ".env", root.parent / ".env"]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[7:].strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue
            if _should_force_override(key):
                os.environ[key] = value
                continue
            if not os.environ.get(key):
                os.environ[key] = value


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def build_page_html(title: str, body_html: str, css_name: str, js_name: str) -> str:
    safe_title = title.replace("<", "").replace(">", "").strip() or "Generated Page"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{safe_title}</title>
  <link rel=\"stylesheet\" href=\"{css_name}\" />
</head>
<body>
{body_html}
  <script src=\"{js_name}\" defer></script>
</body>
</html>
"""


def screenshot_html(html_path: Path, png_path: Path, width: int, height: int, timeout_ms: int) -> str | None:
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        return f"playwright_not_installed: {exc}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(html_path.as_uri(), wait_until="load", timeout=timeout_ms)
            page.wait_for_timeout(350)
            page.screenshot(path=str(png_path), full_page=True)
            browser.close()
        return None
    except Exception as exc:
        return f"screenshot_error: {exc}"


def coerce_codegen_payload(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object.")
    html_text = payload.get("html")
    css_text = payload.get("css")
    js_text = payload.get("js")
    if not all(isinstance(v, str) for v in (html_text, css_text, js_text)):
        raise ValueError("Model output must include string fields: html, css, js.")
    return {
        "html": html_text,
        "css": css_text,
        "js": js_text,
    }


def validate_codegen_payload(payload: dict[str, str]) -> list[str]:
    errors: list[str] = []
    html_text = payload.get("html", "")
    css_text = payload.get("css", "")
    js_text = payload.get("js", "")

    if not html_text.strip():
        errors.append("html is empty")
    if not css_text.strip():
        errors.append("css is empty")
    if not js_text.strip():
        errors.append("js is empty")

    # Quick sanity checks (lightweight, non-blocking syntax proxy).
    if "<" not in html_text or ">" not in html_text:
        errors.append("html does not look like markup")
    if "{" not in css_text or "}" not in css_text:
        errors.append("css does not look like stylesheet")
    if "<script" in js_text.lower():
        errors.append("js should not include <script> tag")
    if "<style" in css_text.lower():
        errors.append("css should not include <style> tag")

    return errors


def parse_and_validate_codegen(text: str) -> tuple[dict[str, str] | None, str | None]:
    try:
        parsed = extract_json(text)
        payload = coerce_codegen_payload(parsed)
    except Exception as exc:
        return None, f"json_parse_error: {exc}"

    validation_errors = validate_codegen_payload(payload)
    if validation_errors:
        return None, "validation_error: " + "; ".join(validation_errors)
    return payload, None


def build_repair_prompt(previous_output: str, reason: str) -> str:
    return (
        "Your previous output is invalid.\n"
        "Repair it and return ONLY a valid JSON object with exactly these string keys: html, css, js.\n"
        "No markdown, no comments, no extra keys.\n"
        f"Failure reason: {reason}\n\n"
        "Previous output:\n"
        f"{previous_output}"
    )


def write_error(path: Path, error: str, raw: Any = None) -> None:
    payload = {"error": error, "raw": raw}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate html/css/js from responses.jsonl using configured LLM model.")
    parser.add_argument("--responses", required=True, help="Path to responses.jsonl")
    parser.add_argument("--model", required=True, help="Model alias from configs/models.yaml")
    parser.add_argument("--models", default=str(ROOT / "configs" / "models.yaml"), help="Path to models.yaml")
    parser.add_argument("--prompt", default=str(ROOT / "prompts" / "web_codegen.md"), help="Prompt template path")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: <responses_dir>/llm_web)")
    parser.add_argument("--max_total", type=int, default=None, help="Max responses to process")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max output tokens")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max_attempts", type=int, default=3, help="API retry attempts")
    parser.add_argument("--repair_attempts", type=int, default=2, help="Repair attempts when JSON/validation fails")
    parser.add_argument("--viewport_width", type=int, default=1280, help="Screenshot viewport width")
    parser.add_argument("--viewport_height", type=int, default=720, help="Screenshot viewport height")
    parser.add_argument("--timeout_ms", type=int, default=20000, help="Render timeout for screenshot")
    args = parser.parse_args()

    _load_env(ROOT)

    responses_path = Path(args.responses).resolve()
    if not responses_path.exists():
        raise SystemExit(f"responses.jsonl not found: {responses_path}")

    models_path = Path(args.models).resolve()
    models_cfg = load_yaml(models_path)
    specs = load_model_specs(models_cfg)
    spec = next((s for s in specs if s.name == args.model), None)
    if spec is None:
        raise SystemExit(f"Unknown model alias: {args.model}")

    adapter = build_adapter(spec)
    prompt_template = load_prompt(Path(args.prompt).resolve())

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (responses_path.parent / "llm_web")
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_dir = out_dir / "errors"
    errors_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    created = 0
    failed = 0

    for row in iter_jsonl(responses_path):
        response_id = str(row.get("response_id") or "").strip()
        response_text = str(row.get("response_text") or "").strip()
        if not response_id or not response_text:
            continue
        if args.max_total is not None and processed >= args.max_total:
            break
        processed += 1

        prompt = render_prompt(prompt_template, response_text=response_text)

        def _call_model():
            return adapter.generate(
                prompt=prompt,
                system=None,
                temperature=float(args.temperature),
                max_tokens=int(args.max_tokens),
                seed=None,
                json_mode=True if spec.supports_json_mode else False,
            )

        try:
            result = with_retry(_call_model, max_attempts=max(1, int(args.max_attempts)))
        except Exception as exc:
            failed += 1
            write_error(errors_dir / f"{response_id}.json", f"model_call_exception: {exc}", None)
            continue

        if result.error:
            failed += 1
            write_error(errors_dir / f"{response_id}.json", result.error, result.raw)
            continue

        payload, parse_error = parse_and_validate_codegen(result.text)
        if payload is None:
            repair_error = parse_error or "unknown_parse_error"
            repaired = False
            last_repair_raw = result.text
            for repair_idx in range(max(0, int(args.repair_attempts))):
                repair_prompt = build_repair_prompt(last_repair_raw, repair_error)

                def _call_repair():
                    return adapter.generate(
                        prompt=repair_prompt,
                        system=None,
                        temperature=0.1,
                        max_tokens=int(args.max_tokens),
                        seed=None,
                        json_mode=True if spec.supports_json_mode else False,
                    )

                try:
                    repair_result = with_retry(_call_repair, max_attempts=max(1, int(args.max_attempts)))
                except Exception as exc:
                    repair_error = f"repair_call_exception[{repair_idx + 1}]: {exc}"
                    continue

                if repair_result.error:
                    repair_error = f"repair_error[{repair_idx + 1}]: {repair_result.error}"
                    last_repair_raw = str(repair_result.raw or "")
                    continue

                last_repair_raw = repair_result.text
                payload, parse_error = parse_and_validate_codegen(repair_result.text)
                if payload is not None:
                    repaired = True
                    break
                repair_error = f"repair_parse_error[{repair_idx + 1}]: {parse_error}"

            if payload is None:
                failed += 1
                write_error(
                    errors_dir / f"{response_id}.json",
                    f"{repair_error}",
                    {"initial_raw": result.text, "last_repair_raw": last_repair_raw},
                )
                continue
            if repaired:
                write_error(
                    errors_dir / f"{response_id}.json",
                    "repaired_output_used",
                    None,
                )

        title = response_text.splitlines()[0].strip() if response_text.splitlines() else response_id
        css_name = f"{response_id}.css"
        js_name = f"{response_id}.js"
        html_name = f"{response_id}.html"
        png_name = f"{response_id}.png"

        css_path = out_dir / css_name
        js_path = out_dir / js_name
        html_path = out_dir / html_name
        png_path = out_dir / png_name

        css_path.write_text(payload["css"], encoding="utf-8")
        js_path.write_text(payload["js"], encoding="utf-8")
        html_path.write_text(
            build_page_html(title=title, body_html=payload["html"], css_name=css_name, js_name=js_name),
            encoding="utf-8",
        )

        ss_err = screenshot_html(
            html_path=html_path,
            png_path=png_path,
            width=int(args.viewport_width),
            height=int(args.viewport_height),
            timeout_ms=int(args.timeout_ms),
        )
        if ss_err:
            failed += 1
            write_error(errors_dir / f"{response_id}.json", ss_err, {"html": html_name})
            continue

        created += 1
        time.sleep(0.05)

    print(
        f"Done. model={spec.name} processed={processed} created={created} failed={failed} output={out_dir}"
    )


if __name__ == "__main__":
    main()