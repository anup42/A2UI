"""Prompt development only. Final acceptance uses the Android AAR benchmark."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="BXP-001,BXP-003,BXP-010,BXP-021,BXP-038")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "dataset" / "src"))
    from pipeline.ir_formats import express, a2ui_wire
    from pipeline.ir_formats.canonical import validate_canonical_graph
    rows = [json.loads(line) for line in (root / "android/app/src/main/assets/genuicraft_bixby50.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = set(args.cases.split(","))
    rows = [row for row in rows if not selected or row["id"] in selected]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    prompt = (root / "GenUICraft/genuicraft/src/main/assets/genuicraft/prompts/gauss.txt").read_text(encoding="utf-8")

    def generate(row):
        start = time.monotonic()
        result = {"id": row["id"], "source": "host_prompt_probe_not_AAR"}
        try:
            payload = {"model": "gaussa-30b-v0.5-128k", "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({"text": row["text"], "sources": []}, ensure_ascii=False)}], "temperature": 0.0, "max_tokens": 8192, "stream": False, "chat_template_kwargs": {"reasoning_strength": "low"}}
            headers = {"Content-Type": "application/json", "User-Agent": "GenUICraft/0.1"}
            if os.environ.get("GAUSS_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["GAUSS_API_KEY"]
            request = urllib.request.Request("https://gaussa.post-train.win/v1/chat/completions", json.dumps(payload).encode(), headers)
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.load(response)
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
            (output / f'{row["id"]}.express').write_text(text, encoding="utf-8")
            result["finish_reason"] = choice["finish_reason"]
            result["usage"] = body.get("usage")
            graph = express.decode(text)
            validation = validate_canonical_graph(graph)
            if not validation.is_valid:
                raise ValueError(str(validation))
            wire = a2ui_wire.encode(graph)
            (output / f'{row["id"]}.json').write_text(json.dumps(wire, ensure_ascii=False, indent=2), encoding="utf-8")
            result["status"] = "valid"
            result["components"] = len(graph["elements"])
        except Exception as error:
            result["status"] = "error"
            result["error"] = str(error)
        result["elapsed_s"] = round(time.monotonic() - start, 2)
        (output / f'{row["id"]}.result.json').write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(generate, rows))
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
