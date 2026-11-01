#!/usr/bin/env python3
"""GenUI: ChatGPT sign-in, separate Stage 3 generation, and a local dashboard."""
from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "dataset/src"), str(ROOT / "training/src")]

from pipeline import chatgpt_stage3, openai_stage3_batch
from pipeline.genui_codex_client import CodexClient
from pipeline.genui_dashboard import Controller, make_server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Offline preparation; never starts generation")
    prepare.add_argument("--input-dir", type=Path, required=True)
    prepare.add_argument("--samples", type=int, default=100)
    prepare.add_argument("--seed", type=int, default=123)
    prepare.add_argument("--model", default="gpt-6.1-sol", help="Exact model ID advertised by the GenUI account command; no silent fallback")
    prepare.add_argument("--selection-profile", choices=("random", "failure_mix_v1"), default="failure_mix_v1")
    prepare.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"), default="medium")
    prepare.add_argument("--reasoning-summary", choices=("none", "auto"), default="auto")
    prepare.add_argument("--generation-timeout", type=int, default=600, help="Per-sample wait limit in seconds; stored in the new run")
    run = commands.add_parser("run", help="Explicit generation using ChatGPT/Codex allowance; no paid/API retries")
    run.add_argument("--timeout", type=int, help="Override the prepared per-sample wait limit (default 600 seconds)")
    run.add_argument("--skip-uncertain", action="store_true", help="Keep uncertain attempts as skipped, never regenerate them")
    collect = commands.add_parser("collect", help="Offline canonical Stage 3 and strict quality checks")
    dashboard = commands.add_parser("dashboard", help="Local GenUI dashboard; no automatic sign-in or generation")
    dashboard.add_argument("--port", type=int, default=8767)
    dashboard.add_argument("--rescore-dir", type=Path, help="Optional separately sealed offline reassessment of this pilot")
    dashboard.add_argument("--open", action="store_true", help="Open the local dashboard in your browser")
    login = commands.add_parser("login", help="Sign in with ChatGPT using Codex-managed OAuth")
    login.add_argument("--device", action="store_true", help="Use the official device-code flow for a remote host")
    login.add_argument("--no-browser", action="store_true")
    status = commands.add_parser("account", help="Check GenUI sign-in without generation")
    for command in (prepare, run, collect, dashboard):
        command.add_argument("--output-dir", type=Path, required=True)
    for command in (run, dashboard, login, status):
        command.add_argument("--auth-home", type=Path, help="Dedicated auth directory outside the repository; default ~/.genui/codex")
        command.add_argument("--codex-executable", help="Pin a Codex binary; default is the newest codex available on PATH")
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "prepare":
            result = openai_stage3_batch.prepare(**args, transport="chatgpt", pilot_profile="fidelity_v2")
        elif command in {"run", "collect"}:
            result = getattr(chatgpt_stage3, command)(**args)
        elif command == "dashboard":
            controller = Controller(args["output_dir"], args["auth_home"], args["rescore_dir"], codex_executable=args["codex_executable"])
            server = make_server(controller, args["port"])
            url = f"http://127.0.0.1:{server.server_port}"
            print(f"GenUI dashboard: {url}\nGeneration starts only through an explicit Start action.", flush=True)
            if args["open"]:
                webbrowser.open(url)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.server_close()
                controller.close()
            return 0
        else:
            client = CodexClient(args.get("auth_home"), executable=args.get("codex_executable"))
            try:
                if command == "login":
                    flow = client.login(device=args["device"])
                    url = flow.get("authUrl") or flow.get("verificationUrl")
                    print("GenUI: complete sign-in on the official OpenAI page.")
                    print(url)
                    if flow.get("userCode"):
                        print("Device code:", flow["userCode"])
                    if not args["no_browser"]:
                        webbrowser.open(url)
                    deadline = time.monotonic() + 600
                    while not client.account()["signed_in"] and time.monotonic() < deadline:
                        time.sleep(2)
                    if not client.account()["signed_in"]:
                        raise ValueError("Sign-in timed out; no generation started")
                result = client.account()
                result["codex_version"] = client.version
                if result["signed_in"]:
                    result["limits"] = client.limits()
                    result["models"] = client.models()
            finally:
                client.close()
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"GenUI failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
