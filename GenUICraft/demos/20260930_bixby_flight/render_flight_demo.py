"""Render an annotated, actual-device flight-answer demo in the CARO v7 style.

Example plan (times are seconds in the unmodified source recording):

{
  "segments": [
    {
      "source_start": 1.20,
      "source_end": 4.50,
      "stage": "THE SAME FLIGHT QUESTION",
      "title": "Two ways to answer",
      "body": "Captured on the device",
      "accent": "blue"
    },
    {
      "source_start": 7.00,
      "source_end": 12.40,
      "stage": "BIXBY ANSWER",
      "title": "Flight details in text",
      "body": "The original answer is shown here.",
      "accent": "blue",
      "arrow_y_source": 1210
    },
    {
      "source_start": 19.25,
      "source_end": 24.00,
      "stage": "GENUICRAFT ANSWER",
      "title": "Flights as cards",
      "body": "The route and times are easier to scan.",
      "accent": "green",
      "arrow_y_source": 1450
    }
  ]
}

Use explicit ``\\n`` in title/body for two or three lines. ``arrow_y_source`` is
the target row's Y coordinate in the original 1080 x 2520 device recording.
Omit it for an intro/outro without an arrow. The arrow stops outside the phone
border, so no annotation covers source pixels. Segments play at source speed;
only selected ranges and one-frame (1/30 s) timing quantization are used.

Usage:
    python render_flight_demo.py --video raw.mp4 --plan scenes.json --out demo.mp4
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


FPS = 30
CANVAS = 1080
SOURCE_WIDTH = 1080
SOURCE_HEIGHT = 2520
CROP_X = 0
CROP_Y = 108
CROP_WIDTH = 1056
CROP_HEIGHT = 2026
PHONE_X = 20
PHONE_Y = 60
PHONE_WIDTH = 500
PHONE_HEIGHT = 960
BORDER_X = 17
BORDER_Y = 57
BORDER_WIDTH = 506
BORDER_HEIGHT = 966
TEXT_X = 613
TEXT_WIDTH = 425
ARROW_TIP_X = 530
ARROW_END_X = 596

BACKGROUND = "F4F5F7"
BORDER = "708090"
TITLE = "182333"
BODY = "556270"
ACCENTS = {"blue": "1967C8", "green": "198E44", "red": "D93025"}


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def lines(value: object, name: str, *, maximum: int, required: bool) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not required and not value.strip():
        return []
    result = [line.strip() for line in value.replace("\r\n", "\n").split("\n")]
    if len(result) > maximum or any(not line for line in result):
        raise ValueError(f"{name} needs 1 to {maximum} nonempty lines")
    if required and not value.strip():
        raise ValueError(f"{name} cannot be empty")
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
    """Conservative English/Unicode estimate; long lines require manual breaks."""
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


def fitted_size(text_lines: list[str], maximum: int, minimum: int, name: str, *, bold: bool) -> int:
    for size in range(maximum, minimum - 1, -1):
        if all(approximate_width(line, size, bold=bold) <= TEXT_WIDTH for line in text_lines):
            return size
    raise ValueError(f"{name} is too wide for the callout column; insert line breaks")


def load_plan(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("plan must be an object with a segments array")
    if not data["segments"]:
        raise ValueError("plan needs at least one segment")

    result = []
    for index, item in enumerate(data["segments"], 1):
        prefix = f"segment {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{prefix} must be an object")
        start = number(item.get("source_start"), f"{prefix}.source_start")
        end = number(item.get("source_end"), f"{prefix}.source_end")
        if start < 0 or end <= start or end - start < 0.5:
            raise ValueError(f"{prefix} needs source_end at least 0.5 s after source_start")
        stage = lines(item.get("stage"), f"{prefix}.stage", maximum=1, required=True)
        title = lines(item.get("title"), f"{prefix}.title", maximum=3, required=True)
        body = lines(item.get("body", ""), f"{prefix}.body", maximum=3, required=False)
        arrow_y = item.get("arrow_y_source")
        if arrow_y is not None:
            arrow_y = number(arrow_y, f"{prefix}.arrow_y_source")
            if not CROP_Y <= arrow_y <= CROP_Y + CROP_HEIGHT:
                raise ValueError(f"{prefix}.arrow_y_source must be inside the cropped device view")
        frames = round((end - start) * FPS)
        if frames < 15:
            raise ValueError(f"{prefix} is too short after frame quantization")
        stage = [stage[0].upper()]
        stage_size = fitted_size(stage, 28, 19, f"{prefix}.stage", bold=True)
        title_size = fitted_size(title, 58, 31, f"{prefix}.title", bold=True)
        body_size = fitted_size(body, 33, 23, f"{prefix}.body", bold=False) if body else 33
        result.append({
            "start": start,
            "end": end,
            "frames": frames,
            "stage": stage,
            "title": title,
            "body": body,
            "accent": accent_hex(item.get("accent", "blue")),
            "arrow_y": arrow_y,
            "stage_size": stage_size,
            "title_size": title_size,
            "body_size": body_size,
        })
    return result


def ass_colour(rgb: str) -> str:
    return "&H00" + rgb[4:6] + rgb[2:4] + rgb[0:2] + "&"


def ass_text(text_lines: list[str]) -> str:
    # Prevent plan text from becoming an ASS override sequence.
    safe = [line.replace("\\", "／").replace("{", "(").replace("}", ")") for line in text_lines]
    return r"\N".join(safe)


def ass_time(frame: int) -> str:
    centiseconds = round(frame * 100 / FPS)
    return f"{centiseconds // 360000}:{centiseconds // 6000 % 60:02}:{centiseconds // 100 % 60:02}.{centiseconds % 100:02}"


def ass_event(start_frame: int, end_frame: int, style: str, text: str, layer: int = 1) -> str:
    return f"Dialogue: {layer},{ass_time(start_frame)},{ass_time(end_frame)},{style},,0,0,0,,{text}"


def callout_top(segment: dict, arrow_canvas_y: int | None) -> tuple[int, int, int]:
    stage_height = round(segment["stage_size"] * 1.25)
    title_height = round(segment["title_size"] * 1.18) * len(segment["title"])
    body_height = round(segment["body_size"] * 1.27) * len(segment["body"])
    total = stage_height + 18 + title_height + (21 + body_height if body_height else 0)
    if total > 960:
        raise ValueError("callout is too tall; shorten its lines")
    if arrow_canvas_y is None:
        top = round((CANVAS - total) / 2)
    else:
        # Like CARO v7, the arrow aligns around the main title, with the body below.
        top = round(arrow_canvas_y - stage_height - 18 - title_height / 2)
        top = max(58, min(top, CANVAS - 58 - total))
    return top, top + stage_height + 18, top + stage_height + 18 + title_height + 21


def make_ass(segments: list[dict]) -> str:
    header = """[Script Info]
Title: Actual-device Bixby flight answer comparison
ScriptType: v4.00+
WrapStyle: 2
ScaledBorderAndShadow: yes
PlayResX: 1080
PlayResY: 1080
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Label,Segoe UI,28,&H00C86719,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0.6,0,1,0,0,7,0,0,0,1
Style: Main,Segoe UI,58,&H00332318,&H00FFFFFF,&H00F7F5F4,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Body,Segoe UI,33,&H00706255,&H00FFFFFF,&H00F7F5F4,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1
Style: Vector,Arial,1,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    first_frame = 0
    for segment in segments:
        last_frame = first_frame + segment["frames"]
        arrow_y = segment["arrow_y"]
        canvas_y = None if arrow_y is None else round(PHONE_Y + (arrow_y - CROP_Y) * PHONE_HEIGHT / CROP_HEIGHT)
        stage_y, title_y, body_y = callout_top(segment, canvas_y)
        fade = r"\fad(90,90)"
        accent = ass_colour(segment["accent"])
        events.append(ass_event(first_frame, last_frame, "Label", f"{{{fade}\\pos({TEXT_X},{stage_y})\\fs{segment['stage_size']}\\c{accent}}}{ass_text(segment['stage'])}"))
        events.append(ass_event(first_frame, last_frame, "Main", f"{{{fade}\\pos({TEXT_X},{title_y})\\fs{segment['title_size']}}}{ass_text(segment['title'])}"))
        if segment["body"]:
            events.append(ass_event(first_frame, last_frame, "Body", f"{{{fade}\\pos({TEXT_X},{body_y})\\fs{segment['body_size']}}}{ass_text(segment['body'])}"))
        if canvas_y is not None:
            y = canvas_y
            arrow = f"m {ARROW_TIP_X} {y} l {ARROW_TIP_X+21} {y-11} {ARROW_TIP_X+21} {y-3} {ARROW_END_X} {y-3} {ARROW_END_X} {y+3} {ARROW_TIP_X+21} {y+3} {ARROW_TIP_X+21} {y+11} {ARROW_TIP_X} {y}"
            events.append(ass_event(first_frame, last_frame, "Vector", f"{{{fade}\\an7\\pos(0,0)\\c{accent}\\p1}}{arrow}{{\\p0}}", 2))
        first_frame = last_frame
    return header + "\n".join(events) + "\n"


def ff_seconds(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def make_filter(segments: list[dict]) -> str:
    split_labels = "".join(f"[s{i}]" for i in range(len(segments)))
    filters = [f"[0:v]split={len(segments)}{split_labels}"]
    for index, segment in enumerate(segments):
        # A three-frame pad handles fractional boundary rounding, then trim fixes
        # the segment length without altering the speed of any captured action.
        filters.append(
            f"[s{index}]trim=start={ff_seconds(segment['start'])}:end={ff_seconds(segment['end'])},"
            f"setpts=PTS-STARTPTS,fps={FPS},tpad=stop_mode=clone:stop_duration=0.1,"
            f"trim=end_frame={segment['frames']},setpts=PTS-STARTPTS[p{index}]"
        )
    if len(segments) == 1:
        sequence = "[p0]"
    else:
        joined = "".join(f"[p{i}]" for i in range(len(segments)))
        filters.append(f"{joined}concat=n={len(segments)}:v=1:a=0[sequence]")
        sequence = "[sequence]"
    filters.append(
        f"{sequence}crop={CROP_WIDTH}:{CROP_HEIGHT}:{CROP_X}:{CROP_Y},"
        f"scale={PHONE_WIDTH}:{PHONE_HEIGHT}:flags=lanczos,setsar=1[phone]"
    )
    filters.append(
        f"[1:v][phone]overlay=x={PHONE_X}:y={PHONE_Y}:shortest=1,"
        f"drawbox=x={BORDER_X}:y={BORDER_Y}:w={BORDER_WIDTH}:h={BORDER_HEIGHT}:"
        f"color=0x{BORDER}:t=3,subtitles=annotations.ass,format=yuv420p[outv]"
    )
    return ";\n".join(filters) + "\n"


def source_duration_and_size(ffmpeg: str, source: Path) -> float:
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(source)], capture_output=True, text=True, **kwargs)
    details = probe.stderr + probe.stdout
    video_lines = [line for line in details.splitlines() if "Video:" in line]
    if not video_lines or not any(re.search(r"(?<!\d)1080x2520(?!\d)", line) for line in video_lines):
        raise ValueError(f"source must contain a {SOURCE_WIDTH} x {SOURCE_HEIGHT} video stream")
    duration = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", details)
    if duration is None:
        raise ValueError("cannot read source duration from FFmpeg")
    return int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])


def render(source: Path, plan: Path, output: Path) -> None:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("imageio_ffmpeg is required for the bundled FFmpeg executable") from exc
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    segments = load_plan(plan)
    source_duration = source_duration_and_size(ffmpeg, source)
    latest_end = max(segment["end"] for segment in segments)
    if latest_end > source_duration + 0.08:
        raise ValueError(f"plan ends at {latest_end:.2f}s, after the source duration of {source_duration:.2f}s")
    total_frames = sum(segment["frames"] for segment in segments)
    total_duration = total_frames / FPS
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flight_video_") as tmp:
        work = Path(tmp)
        (work / "annotations.ass").write_text(make_ass(segments), encoding="utf-8")
        (work / "filter.ffscript").write_text(make_filter(segments), encoding="utf-8")
        with tempfile.NamedTemporaryFile(prefix=".flight_render_", suffix=".mp4", dir=output.parent, delete=False) as temp_file:
            temp_output = Path(temp_file.name)
        try:
            command = [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(source),
                "-f", "lavfi", "-i", f"color=c=0x{BACKGROUND}:s={CANVAS}x{CANVAS}:r={FPS}:d={ff_seconds(total_duration)}",
                "-filter_complex_script", "filter.ffscript", "-map", "[outv]",
                "-c:v", "libx264", "-preset", "medium", "-crf", "17",
                "-pix_fmt", "yuv420p", "-r", str(FPS), "-frames:v", str(total_frames),
                "-movflags", "+faststart", "-an", str(temp_output),
            ]
            subprocess.run(command, cwd=work, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if temp_output.stat().st_size == 0:
                raise RuntimeError("FFmpeg produced an empty output")
            os.replace(temp_output, output)
        finally:
            temp_output.unlink(missing_ok=True)
    print(json.dumps({"output": str(output), "width": CANVAS, "height": CANVAS, "fps": FPS, "frames": total_frames, "duration": total_duration}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, type=Path, help="Raw 1080 x 2520 device recording")
    parser.add_argument("--plan", required=True, type=Path, help="JSON scene plan; schema example is in this script's docstring")
    parser.add_argument("--out", required=True, type=Path, help="Output H.264 MP4 path")
    args = parser.parse_args()
    source = args.video.resolve()
    plan = args.plan.resolve()
    output = args.out.resolve()
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
