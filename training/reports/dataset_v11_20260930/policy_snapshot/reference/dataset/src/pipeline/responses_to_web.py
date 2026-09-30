from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def escape_and_linkify(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(
        r"(https?://[^\s<]+)",
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        escaped,
    )


def render_paragraph(lines: list[str]) -> str:
    text = " ".join(s.strip() for s in lines if s.strip())
    if not text:
        return ""
    return f"<p>{escape_and_linkify(text)}</p>"


def parse_table_block(lines: list[str]) -> str:
    rows = []
    for line in lines:
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if parts:
            rows.append(parts)
    if len(rows) < 2:
        return ""
    if all(set(cell) <= {"-"} for cell in rows[1]):
        rows.pop(1)
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    th = "".join(f"<th>{escape_and_linkify(c)}</th>" for c in header)
    tr = []
    for row in body:
        td = "".join(f"<td>{escape_and_linkify(c)}</td>" for c in row)
        tr.append(f"<tr>{td}</tr>")
    return (
        '<div class="table-wrap"><table>'
        f"<thead><tr>{th}</tr></thead>"
        f"<tbody>{''.join(tr)}</tbody>"
        "</table></div>"
    )


def response_text_to_html(text: str) -> str:
    lines = text.splitlines()
    blocks: list[str] = []
    para_buf: list[str] = []
    list_buf: list[str] = []
    table_buf: list[str] = []

    def flush_para():
        nonlocal para_buf
        out = render_paragraph(para_buf)
        if out:
            blocks.append(out)
        para_buf = []

    def flush_list():
        nonlocal list_buf
        if list_buf:
            lis = "".join(f"<li>{escape_and_linkify(item)}</li>" for item in list_buf)
            blocks.append(f"<ul>{lis}</ul>")
        list_buf = []

    def flush_table():
        nonlocal table_buf
        if table_buf:
            t = parse_table_block(table_buf)
            if t:
                blocks.append(t)
            else:
                para = render_paragraph(table_buf)
                if para:
                    blocks.append(para)
        table_buf = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush_para()
            flush_list()
            flush_table()
            continue

        if "|" in stripped and stripped.count("|") >= 2:
            flush_para()
            flush_list()
            table_buf.append(stripped)
            continue
        flush_table()

        if re.match(r"^#{1,6}\s+", stripped):
            flush_para()
            flush_list()
            level = min(6, len(stripped.split(" ", 1)[0]))
            title = stripped[level + 1 :].strip()
            blocks.append(f"<h{level}>{escape_and_linkify(title)}</h{level}>")
            continue

        if re.match(r"^(\d+\.\s+|-\s+|\*\s+)", stripped):
            flush_para()
            item = re.sub(r"^(\d+\.\s+|-\s+|\*\s+)", "", stripped).strip()
            list_buf.append(item)
            continue

        if stripped == stripped.title() and len(stripped.split()) <= 8:
            flush_para()
            flush_list()
            blocks.append(f"<h2>{escape_and_linkify(stripped)}</h2>")
            continue

        flush_list()
        para_buf.append(stripped)

    flush_para()
    flush_list()
    flush_table()
    return "\n".join(blocks)


def build_css() -> str:
    return """*{box-sizing:border-box}body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#f6f7fb;color:#121722}main{max-width:980px;margin:24px auto;padding:0 16px}.card{background:#fff;border:1px solid #dce2ef;border-radius:14px;padding:20px;box-shadow:0 8px 24px rgba(12,31,66,.06)}h1{margin:0 0 8px;font-size:1.5rem}h2{margin:20px 0 8px;font-size:1.15rem}p{line-height:1.55;margin:10px 0}ul{margin:10px 0 10px 20px;line-height:1.5}.meta{font-size:.86rem;color:#50607d;margin-bottom:12px}.table-wrap{overflow:auto;margin:12px 0}table{width:100%;border-collapse:collapse;min-width:640px}th,td{border:1px solid #d6dfef;padding:8px 10px;text-align:left;vertical-align:top}thead th{background:#eef3ff}a{color:#0a63d8;text-decoration:none}a:hover{text-decoration:underline}"""


def build_js(content_html: str, title: str, meta_line: str) -> str:
    payload = json.dumps(
        {
            "title": title,
            "meta": meta_line,
            "contentHtml": content_html,
        },
        ensure_ascii=False,
    )
    return f"""const DATA = {payload};
document.getElementById("title").textContent = DATA.title;
document.getElementById("meta").textContent = DATA.meta;
document.getElementById("content").innerHTML = DATA.contentHtml;
"""


def build_html(page_title: str, css_name: str, js_name: str) -> str:
    safe_title = html.escape(page_title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{safe_title}</title>
  <link rel="stylesheet" href="{css_name}" />
</head>
<body>
  <main>
    <article class="card">
      <h1 id="title"></h1>
      <div id="meta" class="meta"></div>
      <div id="content"></div>
    </article>
  </main>
  <script src="{js_name}" defer></script>
</body>
</html>
"""


def write_web_files(row: dict[str, Any], out_dir: Path) -> None:
    response_id = str(row.get("response_id") or "").strip()
    query_id = str(row.get("query_id") or "").strip()
    if not response_id:
        return
    text = str(row.get("response_text") or "").strip()
    if not text:
        return

    first_line = text.splitlines()[0].strip() if text.splitlines() else response_id
    content_html = response_text_to_html(text)
    meta = f"response_id={response_id} | query_id={query_id}"

    html_name = f"{response_id}.html"
    css_name = f"{response_id}.css"
    js_name = f"{response_id}.js"

    (out_dir / css_name).write_text(build_css(), encoding="utf-8")
    (out_dir / js_name).write_text(
        build_js(content_html=content_html, title=first_line, meta_line=meta),
        encoding="utf-8",
    )
    (out_dir / html_name).write_text(
        build_html(page_title=first_line, css_name=css_name, js_name=js_name),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate standalone html/css/js files from responses.jsonl"
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="Path to responses.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        required=False,
        default=None,
        help="Output directory (default: <responses_dir>/direct_web)",
    )
    args = parser.parse_args()

    responses_path = Path(args.responses).resolve()
    if not responses_path.exists():
        raise SystemExit(f"responses.jsonl not found: {responses_path}")

    out_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else responses_path.parent / "direct_web"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for row in iter_jsonl(responses_path):
        write_web_files(row, out_dir)
        count += 1

    print(f"Generated web files for {count} responses at: {out_dir}")


if __name__ == "__main__":
    main()
