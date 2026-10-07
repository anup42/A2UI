"""Render an annotated square demo from an actual 1080 x 2520 phone recording.

Usage:
    python render_demo.py --video raw.mp4 --plan plan.json --out demo.mp4

All plan times are absolute seconds in the RAW recording, including annotation
times. The default keeps the complete recording at normal speed. Example:

    {
      "trim": [4.0, 65.0],
      "speed": [17.0, 41.3, 3.0],
      "spinner": [194, 542, 56, 56],
      "cuts": [[41.9, 45.3]],
      "annotations": [
        {"start": 4, "end": 17, "stage": "LIVE BIXBY REQUEST",
         "title": "Ask for flights", "body": "The request runs on the phone.",
         "accent": "blue", "arrow_y_source": 2220},
        {"start": 17, "end": 41.3, "stage": "LIVE A2UI EXPRESS",
         "title": "The answer builds live", "body": "Watch content arrive.",
         "accent": "green", "arrow_y_source": 1090}
      ]
    }

Omit trim/speed/spinner/cuts to preserve the full recording at 1x. "speeds" can
contain several [start, end, multiplier] segments; "speed_segment" is an alias
for "speed". Acceleration requires a spinner rectangle in raw SOURCE pixels.
The spinner is copied from the SAME recording, starting at each speed segment's
source start and playing at 1x for that segment's shortened output duration.
No spinner frame is looped or held. Use a tight rectangle covering the spinner
on a stable background; the script cannot identify the spinner automatically.
Optional "cuts" are ordered [source_start, source_end] intervals to omit, such
as a verified empty wait when switching views. Cuts must not intersect any
accelerated segment. Inspect the raw footage before choosing cuts; preserve
the request, live progress/streaming and final UI. No still holds replace cuts.

The default crop is [0, 108, 1080, 2292] (source Y=108..2400). An optional
"crop": [x, y, width, height] can adjust it while remaining inside those source
vertical bounds. Captions automatically wrap; explicit newlines are supported.
"accent" accepts blue, green, red, purple, or #RRGGBB. Annotation intervals
must be ordered and must not overlap. Blank gaps are allowed.

Writes the MP4, <stem>.metadata.json with hashes, cuts and source/output time
mapping, and <stem>.contact_sheet.jpg containing actual exported keyframes.
No timer, source-time text or speed label is added to the video. Audio is
omitted. Requires imageio_ffmpeg and Pillow; no ffprobe installation is needed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


CANVAS = 1080
SOURCE_SIZE = (1080, 2520)
DEFAULT_CROP = (0, 108, 1080, 2292)
PHONE_X, PHONE_Y, PHONE_HEIGHT = 40, 60, 960
TEXT_X, TEXT_WIDTH = 613, 425
FPS = 30
BACKGROUND = "#F7F8FA"
INK, MUTED = "#182333", "#556270"
ACCENTS = {"blue": "#1967C8", "green": "#198E44", "red": "#D93025", "purple": "#7044B7"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def run(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f"FFmpeg exited with code {result.returncode}:\n{result.stderr[-5000:]}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def rectangle(value: object, name: str) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        value = [value.get(key) for key in ("x", "y", "w", "h")]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be [x, y, width, height] or {{x, y, w, h}}")
    parts = [number(part, name) for part in value]
    if any(part != int(part) for part in parts):
        raise ValueError(f"{name} coordinates must be integer source pixels")
    x, y, width, height = map(int, parts)
    if min(x, y) < 0 or min(width, height) <= 0 or x + width > SOURCE_SIZE[0] or y + height > SOURCE_SIZE[1]:
        raise ValueError(f"{name} must be inside the {SOURCE_SIZE[0]} x {SOURCE_SIZE[1]} source")
    return x, y, width, height


def probe(ffmpeg: str, path: Path) -> dict:
    result = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
    details = result.stderr
    video = next((line for line in details.splitlines() if "Video:" in line), "")
    size = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", video)
    duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", details)
    rate = re.search(r"(\d+(?:\.\d+)?) fps", video)
    if not size or not duration:
        raise ValueError(f"Cannot read video dimensions and duration: {path}\n{details[-2000:]}")
    if re.search(r"rotation of (?!-?0(?:\.0+)? degrees)", details):
        raise ValueError("Rotated source videos are unsupported; provide the portrait raw recording")
    seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    if seconds <= 0:
        raise ValueError("Video duration must be positive")
    return {"width": int(size[1]), "height": int(size[2]), "duration_seconds": seconds,
            "nominal_fps": float(rate[1]) if rate else None}


def strict_decode(ffmpeg: str, path: Path) -> dict:
    result = run([ffmpeg, "-hide_banner", "-loglevel", "error", "-xerror", "-err_detect", "explode",
                  "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough",
                  "-enc_time_base", "demux", "-progress", "pipe:1",
                  "-nostats", "-f", "null", "-"])
    progress = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    frames = int(progress.get("frame", "0"))
    if progress.get("progress") != "end" or frames <= 0:
        raise RuntimeError(f"Strict decode did not finish with video frames: {path}")
    if result.stderr.strip():
        raise RuntimeError(f"Strict decode reported an error: {result.stderr[-3000:]}")
    return {"frames": frames, "decoded_time_seconds": int(progress.get("out_time_us", "0")) / 1e6,
            "strict_decode": "passed", "decode_errors": 0}


def load_plan(path: Path, duration: float) -> dict:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError("Plan must be an object with an annotations array")
    known = {"annotations", "trim", "speed", "speed_segment", "speeds", "spinner", "crop", "cuts"}
    if set(data) - known:
        raise ValueError(f"Unknown plan field(s): {', '.join(sorted(set(data) - known))}")
    trim = data.get("trim", [0, duration])
    if not isinstance(trim, list) or len(trim) != 2:
        raise ValueError("trim must be [source_start, source_end]")
    start, end = [number(part, "trim") for part in trim]
    if start < 0 or end > duration + 0.02 or end - start < 0.5:
        raise ValueError(f"trim must retain at least 0.5 s inside source 0..{duration:.3f}")
    end = min(end, duration)
    crop = rectangle(data.get("crop", DEFAULT_CROP), "crop")
    if crop[1] < 108 or crop[1] + crop[3] > 2400:
        raise ValueError("crop must exclude the status bar (Y<108) and bottom area (Y>2400)")
    phone_width = round(crop[2] * PHONE_HEIGHT / crop[3])
    if phone_width > 480 or phone_width < 250:
        raise ValueError("crop must preserve a portrait aspect ratio for the left phone column")
    keys = [key for key in ("speed", "speed_segment", "speeds") if key in data]
    if len(keys) > 1:
        raise ValueError("Use one of speed, speed_segment or speeds")
    speed_specs = data.get("speeds", []) if keys == ["speeds"] else [data[keys[0]]] if keys else []
    if not isinstance(speed_specs, list):
        raise ValueError("speeds must contain [start, end, multiplier] segments")
    speeds = []
    last_end = start
    for index, spec in enumerate(speed_specs, 1):
        if not isinstance(spec, list) or len(spec) != 3:
            raise ValueError(f"Speed segment {index} must be [start, end, multiplier]")
        a, b, multiplier = [number(part, f"speed {index}") for part in spec]
        if a < last_end - 1e-6 or b > end + 1e-6 or b <= a or multiplier < 1:
            raise ValueError("Speed segments must be ordered, non-overlapping, inside trim, with multiplier >= 1")
        if (b - a) / multiplier < 0.25:
            raise ValueError("Each speed segment must retain at least 0.25 s of output")
        if multiplier > 1:
            speeds.append({"source_start": a, "source_end": b, "multiplier": multiplier})
        last_end = b
    spinner = rectangle(data["spinner"], "spinner") if "spinner" in data else None
    if speeds and spinner is None:
        raise ValueError("Acceleration requires spinner=[x,y,w,h] to preserve its actual animation at 1x")
    if spinner:
        x, y, width, height = spinner
        if x < crop[0] or y < crop[1] or x + width > crop[0] + crop[2] or y + height > crop[1] + crop[3]:
            raise ValueError("spinner must be completely inside the visible crop")
    cut_specs = data.get("cuts", [])
    if not isinstance(cut_specs, list):
        raise ValueError("cuts must contain ordered [source_start, source_end] intervals")
    cuts = []
    last_end = start
    for index, spec in enumerate(cut_specs, 1):
        if not isinstance(spec, list) or len(spec) != 2:
            raise ValueError(f"Cut {index} must be [source_start, source_end]")
        a, b = [number(part, f"cut {index}") for part in spec]
        if a < last_end - 1e-6 or b > end + 1e-6 or b <= a:
            raise ValueError("Cuts must be ordered, non-overlapping, and inside trim")
        if any(a < speed["source_end"] and b > speed["source_start"] for speed in speeds):
            raise ValueError("Cuts cannot intersect accelerated streaming segments")
        cuts.append({"source_start": a, "source_end": b, "position": "internal"})
        last_end = b
    if end - start - sum(cut["source_end"] - cut["source_start"] for cut in cuts) < 0.5:
        raise ValueError("Cuts must retain at least 0.5 s of the recording")
    annotations = []
    previous_end = 0.0
    for index, item in enumerate(data["annotations"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"Annotation {index} must be an object")
        unknown = set(item) - {"start", "end", "stage", "title", "body", "accent", "arrow_y_source"}
        if unknown:
            raise ValueError(f"Unknown field(s) in annotation {index}: {', '.join(sorted(unknown))}")
        a, b = number(item.get("start"), "annotation.start"), number(item.get("end"), "annotation.end")
        if a < previous_end - 1e-6 or a < 0 or b <= a or b > duration + 0.02:
            raise ValueError(f"Annotation {index} must be ordered and non-overlapping inside the source")
        text = {}
        for field in ("stage", "title", "body"):
            value = item.get(field, "")
            if not isinstance(value, str) or (field != "body" and not value.strip()):
                raise ValueError(f"Annotation {index}.{field} must be a {'nonempty ' if field != 'body' else ''}string")
            text[field] = value.replace("\r\n", "\n").replace("\\n", "\n").strip()
        colour = item.get("accent", "blue")
        if not isinstance(colour, str):
            raise ValueError("accent must be a named colour or #RRGGBB")
        colour = ACCENTS.get(colour.lower(), colour)
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", colour):
            raise ValueError("accent must be blue, green, red, purple or #RRGGBB")
        arrow = item.get("arrow_y_source")
        if arrow is not None:
            arrow = number(arrow, "arrow_y_source")
            if not crop[1] <= arrow <= crop[1] + crop[3]:
                raise ValueError("arrow_y_source must be inside the visible source crop")
        annotations.append({"source_start": a, "source_end": min(b, duration), **text,
                            "accent": colour.upper(), "arrow_y_source": arrow})
        previous_end = b
    return {"source_start": start, "source_end": end, "crop": crop, "phone_width": phone_width,
            "speeds": speeds, "spinner": spinner, "annotations": annotations, "cuts": cuts}


def make_timeline(plan: dict) -> list[dict]:
    segments = []
    boundaries = {plan["source_start"], plan["source_end"]}
    for region in plan["speeds"] + plan["cuts"]:
        boundaries.update((region["source_start"], region["source_end"]))
    boundaries = sorted(boundaries)
    for start, end in zip(boundaries, boundaries[1:]):
        middle = (start + end) / 2
        if any(cut["source_start"] <= middle < cut["source_end"] for cut in plan["cuts"]):
            continue
        multiplier = next((speed["multiplier"] for speed in plan["speeds"]
                           if speed["source_start"] <= middle < speed["source_end"]), 1.0)
        if segments and segments[-1]["source_end"] == start and segments[-1]["multiplier"] == multiplier:
            segments[-1]["source_end"] = end
        else:
            segments.append({"source_start": start, "source_end": end, "multiplier": multiplier})
    output_cursor = 0.0
    for segment in segments:
        segment["output_start"] = output_cursor
        output_cursor += (segment["source_end"] - segment["source_start"]) / segment["multiplier"]
        segment["output_end"] = output_cursor
    return segments


def source_to_output(seconds: float, segments: list[dict]) -> float:
    seconds = min(max(seconds, segments[0]["source_start"]), segments[-1]["source_end"])
    for segment in segments:
        if seconds < segment["source_start"]:
            return segment["output_start"]
        if seconds <= segment["source_end"] + 1e-7:
            return segment["output_start"] + (seconds - segment["source_start"]) / segment["multiplier"]
    return segments[-1]["output_end"]


def output_to_source(seconds: float, segments: list[dict]) -> float:
    for index, segment in enumerate(segments):
        if seconds < segment["output_end"] - 1e-7 or index == len(segments) - 1:
            return segment["source_start"] + max(0, seconds - segment["output_start"]) * segment["multiplier"]
    return segments[-1]["source_end"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / ("segoeuib.ttf" if bold else "segoeui.ttf"),
                  Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def wrapped(text: str, face: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines = []
    for paragraph in text.splitlines():
        if not paragraph.strip():
            raise ValueError("Caption text cannot contain blank lines")
        line = ""
        for word in paragraph.split():
            if face.getlength(word) > width:
                raise ValueError(f"Caption word is too wide: {word}")
            candidate = f"{line} {word}".strip()
            if face.getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
    return lines


def fit(text: str, largest: int, smallest: int, max_lines: int, bold: bool) -> tuple:
    if not text:
        return [], font(largest, bold), round(largest * 1.28)
    for size in range(largest, smallest - 1, -1):
        face = font(size, bold)
        try:
            lines = wrapped(text, face, TEXT_WIDTH)
        except ValueError:
            continue
        if len(lines) <= max_lines:
            return lines, face, round(size * 1.28)
    raise ValueError(f"Caption is too long; shorten it or add a concise line break: {text}")


def draw_lines(draw: ImageDraw.ImageDraw, lines: list[str], face: ImageFont.FreeTypeFont,
               x: int, y: int, step: int, colour: str) -> None:
    for line in lines:
        draw.text((x, y), line, font=face, fill=colour, anchor="lt")
        y += step


def caption_image(item: dict, plan: dict, path: Path) -> None:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    stage = fit(item["stage"].upper(), 26, 20, 2, True)
    title = fit(item["title"], 54, 34, 3, True)
    body = fit(item["body"], 31, 25, 4, False)
    stage_height, title_height, body_height = [len(part[0]) * part[2] for part in (stage, title, body)]
    total = stage_height + 18 + title_height + (22 + body_height if body_height else 0)
    arrow = item["arrow_y_source"]
    arrow_y = None if arrow is None else round(PHONE_Y + (arrow - plan["crop"][1]) * PHONE_HEIGHT / plan["crop"][3])
    top = round((CANVAS - total) / 2) if arrow_y is None else round(arrow_y - stage_height - 18 - title_height / 2)
    top = max(155, min(top, CANVAS - 60 - total))
    if total > CANVAS - 215:
        raise ValueError("Caption does not fit vertically; shorten its text")
    draw_lines(draw, *stage[:2], TEXT_X, top, stage[2], item["accent"])
    title_y = top + stage_height + 18
    draw_lines(draw, *title[:2], TEXT_X, title_y, title[2], INK)
    draw_lines(draw, *body[:2], TEXT_X, title_y + title_height + 22, body[2], MUTED)
    if arrow_y is not None:
        tip, tail = PHONE_X + plan["phone_width"] + 12, TEXT_X - 17
        draw.line((tip + 16, arrow_y, tail, arrow_y), fill=item["accent"], width=5)
        draw.polygon([(tip, arrow_y), (tip + 19, arrow_y - 11), (tip + 19, arrow_y + 11)], fill=item["accent"])
    image.save(path)


def common_image(plan: dict, path: Path) -> None:
    image = Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rectangle((PHONE_X - 3, PHONE_Y - 3, PHONE_X + plan["phone_width"] + 2,
                    PHONE_Y + PHONE_HEIGHT + 2), outline="#708090", width=3)
    image.save(path)


def filter_graph(plan: dict, segments: list[dict], annotations: list[dict]) -> tuple[str, list[dict]]:
    fast = [segment for segment in segments if segment["multiplier"] > 1]
    labels = [f"v{index}" for index in range(len(segments))]
    spinner_labels = [f"s{index}" for index in range(len(fast))]
    all_labels = labels + spinner_labels
    # Normalize the recording's VFR clock before trimming. A retained interval
    # can start between recorded frames (e.g. within a natural empty wait); its
    # existing recorded state must cover that time, rather than silently moving
    # the cut to the next newly recorded frame. This resamples actual frames.
    graph = [f"[0:v:0]setpts=PTS-STARTPTS,fps={FPS},split={len(all_labels)}" + "".join(f"[{label}]" for label in all_labels)]
    for index, segment in enumerate(segments):
        duration = segment["output_end"] - segment["output_start"]
        # Android VFR frames can carry a long duration across an empty wait.
        # Bound the normalized frames too, so that duration cannot leak back
        # into a deliberately cut interval at a concat boundary.
        graph.append(f"[{labels[index]}]trim=start={segment['source_start']:.9f}:end={segment['source_end']:.9f},"
                     f"setpts=(PTS-STARTPTS)/{segment['multiplier']:.9f},fps={FPS},"
                     f"trim=duration={duration:.9f},setpts=PTS-STARTPTS[part{index}]")
    if len(segments) == 1:
        graph.append("[part0]null[timeline]")
    else:
        graph.append("".join(f"[part{index}]" for index in range(len(segments))) + f"concat=n={len(segments)}:v=1:a=0[timeline]")
    patches = []
    base = "timeline"
    for index, segment in enumerate(fast):
        x, y, width, height = plan["spinner"]
        length = segment["output_end"] - segment["output_start"]
        graph.append(f"[{spinner_labels[index]}]trim=start={segment['source_start']:.9f}:end={segment['source_start'] + length:.9f},"
                     f"setpts=PTS-STARTPTS+{segment['output_start']:.9f}/TB,format=rgb24,"
                     f"crop={width}:{height}:{x}:{y},fps={FPS}[spinner{index}]")
        next_base = f"patched{index}"
        graph.append(f"[{base}][spinner{index}]overlay=x={x}:y={y}:format=rgb:eof_action=pass:repeatlast=0:"
                     f"enable='gte(t,{segment['output_start']:.9f})*lt(t,{segment['output_end']:.9f})'[{next_base}]")
        patches.append({"roi_xywh": list(plan["spinner"]), "multiplier": 1.0,
                        "source_start": segment["source_start"], "source_end": segment["source_start"] + length,
                        "output_start": segment["output_start"], "output_end": segment["output_end"],
                        "same_source_recording": True, "looped": False, "held": False})
        base = next_base
    x, y, width, height = plan["crop"]
    graph.append(f"[{base}]crop={width}:{height}:{x}:{y},scale={plan['phone_width']}:{PHONE_HEIGHT}:flags=lanczos,"
                 f"setsar=1,pad={CANVAS}:{CANVAS}:{PHONE_X}:{PHONE_Y}:color={BACKGROUND}[canvas]")
    graph.append("[canvas][1:v]overlay=eof_action=repeat:repeatlast=1[common]")
    base = "common"
    for index, item in enumerate(annotations):
        next_base = f"caption{index}"
        graph.append(f"[{base}][{index + 2}:v]overlay=eof_action=repeat:repeatlast=1:"
                     f"enable='gte(t,{item['output_start']:.9f})*lt(t,{item['output_end']:.9f})'[{next_base}]")
        base = next_base
    graph.append(f"[{base}]format=yuv420p[out]")
    return ";\n".join(graph) + "\n", patches


def contact_sheet(ffmpeg: str, video: Path, path: Path, work: Path, duration: float,
                  segments: list[dict], annotations: list[dict]) -> list[dict]:
    times = {0.0, max(0, duration - 0.15)}
    for item in annotations:
        times.add((item["output_start"] + item["output_end"]) / 2)
    for segment in segments:
        if segment["multiplier"] > 1:
            times.add(min(duration - 0.05, segment["output_start"] + 0.25))
            times.add(max(0, segment["output_end"] - 0.25))
    times = sorted({round(min(max(t, 0), max(0, duration - 0.05)), 3) for t in times})
    if len(times) > 12:
        times = [times[round(i * (len(times) - 1) / 11)] for i in range(12)]
    cell, margin, label_height = 540, 14, 43
    sheet = Image.new("RGB", (cell * 2 + margin * 3, (cell + label_height + margin) * math.ceil(len(times) / 2) + margin), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    face = font(18)
    samples = []
    for index, seconds in enumerate(times):
        frame = work / f"keyframe_{index}.png"
        run([ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{seconds:.6f}", "-i", str(video),
             "-map", "0:v:0", "-frames:v", "1", "-update", "1", str(frame)])
        with Image.open(frame) as pixels:
            thumbnail = pixels.convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        x, y = margin + (index % 2) * (cell + margin), margin + (index // 2) * (cell + label_height + margin)
        sheet.paste(thumbnail, (x, y))
        source_seconds = output_to_source(seconds, segments)
        draw.text((x + 4, y + cell + 10), f"Output {seconds:.2f}s   |   Source {source_seconds:.2f}s", font=face, fill=MUTED)
        samples.append({"output_seconds": seconds, "source_seconds": round(source_seconds, 6)})
    sheet.save(path, quality=94, subsampling=0)
    return samples


def render(source: Path, plan_path: Path, output: Path) -> dict:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    source_info = probe(ffmpeg, source)
    if (source_info["width"], source_info["height"]) != SOURCE_SIZE:
        raise ValueError(f"Source must be a raw {SOURCE_SIZE[0]} x {SOURCE_SIZE[1]} portrait recording")
    plan = load_plan(plan_path, source_info["duration_seconds"])
    segments = make_timeline(plan)
    expected_duration = segments[-1]["output_end"]
    annotations = []
    for item in plan["annotations"]:
        if item["source_end"] <= plan["source_start"] or item["source_start"] >= plan["source_end"]:
            continue
        output_start, output_end = source_to_output(item["source_start"], segments), source_to_output(item["source_end"], segments)
        if output_end > output_start:
            annotations.append({**item, "output_start": output_start, "output_end": output_end})
    print("Validating the raw recording...", file=sys.stderr, flush=True)
    source_info.update(strict_decode(ffmpeg, source))
    source_hash, plan_hash = sha256(source), sha256(plan_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path, sheet_path = output.with_suffix(".metadata.json"), output.with_suffix(".contact_sheet.jpg")
    if source in (metadata_path, sheet_path) or plan_path in (metadata_path, sheet_path):
        raise ValueError("Output metadata/contact-sheet paths must differ from the inputs")
    with tempfile.TemporaryDirectory(prefix=".bixby_render_", dir=output.parent) as temp:
        work = Path(temp)
        common_image(plan, work / "common.png")
        for index, item in enumerate(annotations):
            caption_image(item, plan, work / f"caption_{index}.png")
        graph, patches = filter_graph(plan, segments, annotations)
        (work / "filter.ffscript").write_text(graph, encoding="utf-8")
        pending = work / "render.mp4"
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-xerror", "-i", str(source), "-i", str(work / "common.png")]
        for index in range(len(annotations)):
            command.extend(["-i", str(work / f"caption_{index}.png")])
        command.extend(["-filter_complex_threads", "2", "-filter_complex_script", str(work / "filter.ffscript"),
                        "-map", "[out]", "-an", "-sn", "-dn", "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                        "-threads", "4", "-pix_fmt", "yuv420p", "-fps_mode", "cfr", "-r", str(FPS),
                        "-movflags", "+faststart", str(pending)])
        print(f"Rendering {expected_duration:.2f} seconds with {len(annotations)} captions...", file=sys.stderr, flush=True)
        run(command, cwd=work)
        print("Strictly decoding the export and extracting actual keyframes...", file=sys.stderr, flush=True)
        output_info = probe(ffmpeg, pending)
        output_info.update(strict_decode(ffmpeg, pending))
        output_info["duration_seconds"] = output_info["frames"] / FPS
        if (output_info["width"], output_info["height"]) != (CANVAS, CANVAS):
            raise RuntimeError("Export dimensions do not match the square canvas")
        if abs(output_info["duration_seconds"] - expected_duration) > 0.10 + len(segments) / FPS:
            raise RuntimeError(f"Unexpected export duration: {output_info['duration_seconds']:.6f} s; expected {expected_duration:.6f} s")
        samples = contact_sheet(ffmpeg, pending, work / "contact_sheet.jpg", work,
                                output_info["duration_seconds"], segments, annotations)
        # Reject an input that changed while FFmpeg was reading it.
        if sha256(source) != source_hash or sha256(plan_path) != plan_hash:
            raise RuntimeError("Source or plan changed during rendering; retry with stable inputs")
        cuts = [dict(cut) for cut in plan["cuts"]]
        if plan["source_start"] > 0:
            cuts.append({"source_start": 0.0, "source_end": plan["source_start"], "position": "head"})
        if plan["source_end"] < source_info["duration_seconds"]:
            cuts.append({"source_start": plan["source_end"], "source_end": source_info["duration_seconds"], "position": "tail"})
        cuts.sort(key=lambda cut: cut["source_start"])
        metadata = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                    "source": {"path": str(source), "bytes": source.stat().st_size, "sha256": source_hash, **source_info},
                    "plan": {"path": str(plan_path), "sha256": plan_hash, "time_basis": "absolute raw source seconds"},
                    "output": {"path": str(output), "bytes": pending.stat().st_size, "sha256": sha256(pending), **output_info,
                               "fps": FPS, "codec": "H.264", "pixel_format": "yuv420p", "audio": False},
                    "cuts": cuts, "source_speed_sections": segments, "spinner_patches": patches,
                    "crop_xywh": list(plan["crop"]), "phone_canvas_xywh": [PHONE_X, PHONE_Y, plan["phone_width"], PHONE_HEIGHT],
                    "annotations": annotations, "contact_sheet": {"path": str(sheet_path), "samples": samples},
                    "continuous_source": len(cuts) == 0, "one_source_recording": True,
                    "retained_source_contiguous": len(plan["cuts"]) == 0,
                    "static_phone_holds": False, "mock_phone_ui": False,
                    "timer_visible": False, "source_time_text_visible": False, "speed_label_in_video": False,
                    "frame_sampling": "30 fps resampling of raw capture timing; existing source frames cover natural VFR intervals",
                    "time_mapping_tolerance_seconds": round(0.1 + len(segments) / FPS, 6),
                    "ffmpeg_version": run([ffmpeg, "-version"]).stdout.splitlines()[0], "filter_graph": graph}
        (work / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(pending, output)
        os.replace(work / "contact_sheet.jpg", sheet_path)
        os.replace(work / "metadata.json", metadata_path)
    summary = {"output": str(output), "metadata": str(metadata_path), "contact_sheet": str(sheet_path),
               "duration_seconds": output_info["duration_seconds"], "frames": output_info["frames"],
               "sha256": metadata["output"]["sha256"], "strict_decode": "passed", "spinner_patches": len(patches)}
    print(json.dumps(summary, indent=2))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True, help="Actual raw 1080 x 2520 phone MP4")
    parser.add_argument("--plan", type=Path, required=True, help="JSON captions, optional source trim and speed segments")
    parser.add_argument("--out", type=Path, required=True, help="Square H.264 MP4 output")
    args = parser.parse_args()
    source, plan_path, output = args.video.resolve(), args.plan.resolve(), args.out.resolve()
    if not source.is_file() or not plan_path.is_file():
        parser.error("--video and --plan must name existing files")
    if output.suffix.lower() != ".mp4" or output in (source, plan_path):
        parser.error("--out must be a distinct MP4 path")
    try:
        render(source, plan_path, output)
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Render failed: {exc}\n")


if __name__ == "__main__":
    main()
