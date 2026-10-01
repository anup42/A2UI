"""Annotate one continuous, actual-device Bixby recording without cuts or holds.

The entire 1080 x 2520 source is processed as one video stream. Every source
frame passes once through crop, scale, pad and text overlays. Source timestamps
set the output playback speed; this script never trims, loops, pads time, joins
clips, or creates still-image holds. The source should visibly show the query,
submission, A2UI Express streaming, and final result in one recording.

Example plan (annotation times are seconds since the source recording starts):

{
  "annotations": [
    {
      "start": 0.0,
      "end": 7.5,
      "stage": "LIVE BIXBY REQUEST",
      "title": "Morning flights\\nto Varanasi",
      "body": "The query is entered\\non the device.",
      "accent": "blue",
      "arrow_y_source": 2220
    },
    {
      "start": 7.5,
      "end": 24.0,
      "stage": "A2UI EXPRESS STREAMING",
      "title": "The answer builds live",
      "body": "Watch new content arrive\\nin Bixby.",
      "accent": "green"
    }
  ]
}

Use explicit ``\\n`` in title/body for at most three lines. Omit
``arrow_y_source`` when no arrow is needed. It is a Y coordinate in the raw
1080 x 2520 screen recording. Blank gaps between annotations are allowed; the
LIVE DEVICE CAPTURE label and mm:ss elapsed timer remain visible throughout.
The recording must be no longer than 180 seconds.

Usage:
    python render_live_demo.py --video live_raw.mp4 --plan live_plan.json --out live_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unicodedata


CANVAS = 1080
SOURCE_WIDTH = 1080
SOURCE_HEIGHT = 2520
CROP_X = 0
CROP_Y = 108
CROP_WIDTH = 1056
CROP_HEIGHT = 2292  # Through source Y=2400: keeps the query field, excludes watermark.
PHONE_X = 40
PHONE_Y = 60
PHONE_WIDTH = 442
PHONE_HEIGHT = 960
BORDER_X = 37
BORDER_Y = 57
BORDER_WIDTH = 448
BORDER_HEIGHT = 966
TEXT_X = 613
TEXT_WIDTH = 425
ARROW_TIP_X = 492  # Seven pixels beyond the border's right edge.
ARROW_END_X = 596
BACKGROUND = "F4F5F7"
BORDER = "708090"
ACCENTS = {"blue": "1967C8", "green": "198E44", "red": "D93025"}


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def text_lines(value: object, name: str, *, maximum: int, required: bool) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not required and not value.strip():
        return []
    result = [line.strip() for line in value.replace("\r\n", "\n").split("\n")]
    if len(result) > maximum or any(not line for line in result):
        raise ValueError(f"{name} needs 1 to {maximum} nonempty lines")
    return result


def accent_hex(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("accent must be blue, green, red, or #RRGGBB")
    if value.lower() in ACCENTS:
        return ACCENTS[value.lower()]
    if re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        return value[1:].upper()
    raise ValueError("accent must be blue, green, red, or #RRGGBB")


def approximate_width(text: str, font_size: int, *, bold: bool) -> float:
    units = 0.0
    for char in text:
        if unicodedata.east_asian_width(char) in ("W", "F"):
            units += 1.02
        elif char in "MW@#%&":
            units += 0.91
        elif char in "ilI.,:;!|'`":
            units += 0.30
        elif char == " ":
            units += 0.33
        else:
            units += 0.61 if bold else 0.58
    return units * font_size


def fitted_size(text: list[str], maximum: int, minimum: int, name: str, *, bold: bool) -> int:
    for size in range(maximum, minimum - 1, -1):
        if all(approximate_width(line, size, bold=bold) <= TEXT_WIDTH for line in text):
            return size
    raise ValueError(f"{name} is too wide for the callout column; insert line breaks")


def load_plan(path: Path, duration: float) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError("plan must be an object with an annotations array")
    result = []
    previous_end = 0.0
    for index, item in enumerate(data["annotations"], 1):
        prefix = f"annotation {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")
        start = number(item.get("start"), f"{prefix}.start")
        end = number(item.get("end"), f"{prefix}.end")
        if start < 0 or end <= start or end - start < 0.5:
            raise ValueError(f"{prefix} needs end at least 0.5 s after start")
        if start < previous_end - 0.005:
            raise ValueError(f"{prefix} overlaps or precedes the previous annotation")
        if end > duration + 0.08:
            raise ValueError(f"{prefix} ends at {end:.2f}s, after the {duration:.2f}s source")
        stage = [s.upper() for s in text_lines(item.get("stage"), f"{prefix}.stage", maximum=1, required=True)]
        title = text_lines(item.get("title"), f"{prefix}.title", maximum=3, required=True)
        body = text_lines(item.get("body", ""), f"{prefix}.body", maximum=3, required=False)
        arrow_y = item.get("arrow_y_source")
        if arrow_y is not None:
            arrow_y = number(arrow_y, f"{prefix}.arrow_y_source")
            if not CROP_Y <= arrow_y <= CROP_Y + CROP_HEIGHT:
                raise ValueError(f"{prefix}.arrow_y_source must be within source Y={CROP_Y}..{CROP_Y+CROP_HEIGHT}")
        result.append({
            "start": start,
            "end": min(end, duration),
            "stage": stage,
            "title": title,
            "body": body,
            "accent": accent_hex(item.get("accent", "blue")),
            "arrow_y": arrow_y,
            "stage_size": fitted_size(stage, 28, 19, f"{prefix}.stage", bold=True),
            "title_size": fitted_size(title, 58, 31, f"{prefix}.title", bold=True),
            "body_size": fitted_size(body, 33, 23, f"{prefix}.body", bold=False) if body else 33,
        })
        previous_end = end
    return result


def ass_colour(rgb: str) -> str:
    return "&H00" + rgb[4:6] + rgb[2:4] + rgb[0:2] + "&"


def ass_text(text: list[str]) -> str:
    safe = [line.replace("\\", "／").replace("{", "(").replace("}", ")") for line in text]
    return r"\N".join(safe)


def ass_time(seconds: float) -> str:
    centiseconds = round(seconds * 100)
    return f"{centiseconds // 360000}:{centiseconds // 6000 % 60:02}:{centiseconds // 100 % 60:02}.{centiseconds % 100:02}"


def event(start: float, end: float, style: str, content: str, layer: int = 1) -> str:
    return f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},{style},,0,0,0,,{content}"


def callout_positions(item: dict, arrow_y: int | None) -> tuple[int, int, int]:
    stage_height = round(item["stage_size"] * 1.25)
    title_height = round(item["title_size"] * 1.18) * len(item["title"])
    body_height = round(item["body_size"] * 1.27) * len(item["body"])
    total = stage_height + 18 + title_height + (21 + body_height if body_height else 0)
    if total > 860:
        raise ValueError("callout is too tall; shorten its lines")
    if arrow_y is None:
        top = round((CANVAS - total) / 2)
    else:
        top = round(arrow_y - stage_height - 18 - title_height / 2)
        top = max(155, min(top, CANVAS - 58 - total))
    return top, top + stage_height + 18, top + stage_height + 18 + title_height + 21


def make_ass(duration: float, annotations: list[dict]) -> str:
    header = """[Script Info]
Title: Continuous live device capture with flight-answer annotations
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1080
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Live,Segoe UI,20,&H00C86719,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0.4,0,1,0,0,7,0,0,0,1
Style: Timer,Segoe UI,28,&H00332318,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Label,Segoe UI,28,&H00C86719,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0.6,0,1,0,0,7,0,0,0,1
Style: Main,Segoe UI,58,&H00332318,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Body,Segoe UI,33,&H00706255,&H00FFFFFF,&H00F7F5F4,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Vector,Arial,1,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = [event(0, duration, "Live", r"{\pos(613,72)}LIVE DEVICE CAPTURE", 3)]
    for second in range(math.ceil(duration)):
        end = min(float(second + 1), duration)
        if end <= second:
            continue
        elapsed = f"{second // 60:02}:{second % 60:02}"
        events.append(event(float(second), end, "Timer", rf"{{\pos(951,68)}}{elapsed}", 3))
    for item in annotations:
        raw_y = item["arrow_y"]
        canvas_y = None if raw_y is None else round(PHONE_Y + (raw_y - CROP_Y) * PHONE_HEIGHT / CROP_HEIGHT)
        stage_y, title_y, body_y = callout_positions(item, canvas_y)
        fade = r"\fad(90,90)"
        accent = ass_colour(item["accent"])
        start, end = item["start"], item["end"]
        events.append(event(start, end, "Label", f"{{{fade}\\pos({TEXT_X},{stage_y})\\fs{item['stage_size']}\\c{accent}}}{ass_text(item['stage'])}"))
        events.append(event(start, end, "Main", f"{{{fade}\\pos({TEXT_X},{title_y})\\fs{item['title_size']}}}{ass_text(item['title'])}"))
        if item["body"]:
            events.append(event(start, end, "Body", f"{{{fade}\\pos({TEXT_X},{body_y})\\fs{item['body_size']}}}{ass_text(item['body'])}"))
        if canvas_y is not None:
            y = canvas_y
            arrow = f"m {ARROW_TIP_X} {y} l {ARROW_TIP_X+21} {y-11} {ARROW_TIP_X+21} {y-3} {ARROW_END_X} {y-3} {ARROW_END_X} {y+3} {ARROW_TIP_X+21} {y+3} {ARROW_TIP_X+21} {y+11} {ARROW_TIP_X} {y}"
            events.append(event(start, end, "Vector", f"{{{fade}\\an7\\pos(0,0)\\c{accent}\\p1}}{arrow}{{\\p0}}", 2))
    return header + "\n".join(events) + "\n"


def source_duration(ffmpeg: str, source: Path) -> float:
    probe = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", str(source)],
        capture_output=True, text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    details = probe.stderr + probe.stdout
    video_lines = [line for line in details.splitlines() if "Video:" in line]
    if not video_lines or not any(re.search(r"(?<!\d)1080x2520(?!\d)", line) for line in video_lines):
        raise ValueError(f"source must contain a {SOURCE_WIDTH} x {SOURCE_HEIGHT} video stream")
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", details)
    if match is None:
        raise ValueError("cannot read source duration from FFmpeg")
    duration = int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
    if duration <= 0 or duration > 180.0:
        raise ValueError(f"source duration must be in (0, 180] seconds; found {duration:.2f}")
    return duration


def render(source: Path, plan: Path, output: Path) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio_ffmpeg is required for the bundled FFmpeg executable") from exc
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = source_duration(ffmpeg, source)
    annotations = load_plan(plan, duration)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bixby_live_") as tmp:
        work = Path(tmp)
        (work / "annotations.ass").write_text(make_ass(duration, annotations), encoding="utf-8")
        # One input, one chain, one output frame per input frame. No trim, fps,
        # tpad, loop, overlay sync, concat, -r, -t, or -frames:v operations.
        filter_graph = (
            f"[0:v]setpts=PTS-STARTPTS,crop={CROP_WIDTH}:{CROP_HEIGHT}:{CROP_X}:{CROP_Y},"
            f"scale={PHONE_WIDTH}:{PHONE_HEIGHT}:flags=lanczos,setsar=1,"
            f"pad={CANVAS}:{CANVAS}:{PHONE_X}:{PHONE_Y}:color=0x{BACKGROUND},"
            f"drawbox=x={BORDER_X}:y={BORDER_Y}:w={BORDER_WIDTH}:h={BORDER_HEIGHT}:"
            f"color=0x{BORDER}:t=3,subtitles=annotations.ass,format=yuv420p[outv]\n"
        )
        (work / "filter.ffscript").write_text(filter_graph, encoding="utf-8")
        with tempfile.NamedTemporaryFile(prefix=".bixby_live_", suffix=".mp4", dir=output.parent, delete=False) as file:
            pending = Path(file.name)
        try:
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-filter_complex_script", "filter.ffscript", "-map", "[outv]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                "-movflags", "+faststart", "-an", str(pending),
            ]
            subprocess.run(command, cwd=work, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if pending.stat().st_size == 0:
                raise RuntimeError("FFmpeg produced an empty output")
            os.replace(pending, output)
        finally:
            pending.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "source_duration_seconds": duration, "annotations": len(annotations), "continuous_source": True, "audio": False}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path, help="One continuous 1080 x 2520 live device recording, up to 180 s")
    parser.add_argument("--plan", required=True, type=Path, help="JSON annotation plan; schema is in --help")
    parser.add_argument("--out", required=True, type=Path, help="Output square H.264 MP4 path")
    args = parser.parse_args()
    source, plan, output = args.video.resolve(), args.plan.resolve(), args.out.resolve()
    if not source.is_file() or not plan.is_file():
        parser.error("--video and --plan must name existing files")
    if output in (source, plan):
        parser.error("--out must differ from --video and --plan")
    try:
        render(source, plan, output)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"render failed: {exc}\n")


if __name__ == "__main__":
    main()
