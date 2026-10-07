"""Capture and inspect an actual Bixby conversation; no generated/mock UI inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET

ADB = r"C:\Users\anupk\AppData\Local\Android\Sdk\platform-tools\adb.exe"
PACKAGE = "com.samsung.android.bixby.agent"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Device:
    def __init__(self, serial: str, directory: Path):
        self.prefix = [ADB, "-s", serial]
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def run(self, *args: str, timeout: float = 20) -> bytes:
        return subprocess.run(self.prefix + list(args), check=True, capture_output=True,
                              timeout=timeout, creationflags=CREATE_NO_WINDOW).stdout

    def shell(self, command: str, timeout: float = 20) -> bytes:
        return self.run("shell", command, timeout=timeout)

    def tree(self, name: str = "ui") -> ET.Element:
        remote = "/data/local/tmp/genuicraft_five_ui.xml"
        self.run("shell", "uiautomator", "dump", remote, timeout=8)
        data = self.run("shell", "cat", remote)
        (self.directory / f"{name}.xml").write_bytes(data)
        return ET.fromstring(data)

    def snapshot(self, name: str) -> ET.Element:
        tree = self.tree(name)
        self.screenshot(name)
        return tree

    def screenshot(self, name: str) -> None:
        remote = "/data/local/tmp/genuicraft_five_screen.png"
        self.run("shell", "screencap", "-p", remote)
        self.run("pull", remote, str(self.directory / f"{name}.png"))

    def tap_node(self, node: ET.Element) -> list[int]:
        coords = list(map(int, re.findall(r"\d+", node.get("bounds", ""))))
        if len(coords) != 4 or coords[2] <= coords[0] or coords[3] <= coords[1]:
            raise ValueError("Cannot tap node without positive visible bounds")
        self.run("shell", "input", "tap", str((coords[0]+coords[2])//2), str((coords[1]+coords[3])//2))
        return coords


def find(tree: ET.Element, *, text: str | None = None, description: str | None = None,
         cls: str | None = None) -> ET.Element | None:
    return next((n for n in tree.iter("node") if
                 (text is None or n.get("text", "").strip() == text) and
                 (description is None or n.get("content-desc", "") == description) and
                 (cls is None or n.get("class", "") == cls)), None)


def print_nodes(tree: ET.Element) -> None:
    visible = [n for n in tree.iter("node") if n.get("bounds") != "[0,0][0,0]" and (n.get("text") or n.get("content-desc"))]
    for n in visible[:24]:
        if n.get("text") or n.get("content-desc"):
            print(json.dumps({k: n.get(k, "")[:230] for k in ("text", "content-desc", "bounds", "clickable", "checked")}, ensure_ascii=True))


def prepare(device: Device, query: str) -> None:
    device.run("shell", "am", "start", "-W", "-n", PACKAGE + "/.mainui.main.fullscreen.FullScreenActivity")
    tree = device.tree("before_prepare")
    new = find(tree, description="New conversation")
    if new is not None:
        device.tap_node(new)
        time.sleep(1)
        tree = device.tree("new_conversation")
    edit = find(tree, cls="android.widget.EditText")
    if edit is None:
        edit = find(tree, text="Ask Bixby")
    if edit is None:
        raise RuntimeError("Bixby text input is not visible")
    device.tap_node(edit)
    # One remote shell command, with a quoted input argument. No local interpolation.
    device.shell("input text " + shlex.quote(query.replace(" ", "%s")))
    time.sleep(0.7)
    tree = device.snapshot("prepared")
    if find(tree, description="Run command") is None:
        raise RuntimeError("No Bixby Run command button after typing")
    (device.directory / "query.txt").write_text(query, encoding="utf-8")
    print_nodes(tree)


def capture(device: Device, limit: int) -> None:
    query = (device.directory / "query.txt").read_text(encoding="utf-8")
    remote = "/sdcard/Download/genuicraft_live_" + device.directory.name + ".mp4"
    events: list[dict] = []
    record_start = time.monotonic()

    def event(name: str, **fields) -> None:
        row = {"event": name, "elapsed_seconds": round(time.monotonic()-record_start, 3), **fields}
        events.append(row)
        (device.directory / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)

    log_file = (device.directory / "live_logcat.txt").open("wb")
    package_pid = device.run("shell", "pidof", PACKAGE).decode().strip().split()[0]
    logger = subprocess.Popen(device.prefix + ["logcat", "--pid=" + package_pid, "-v", "threadtime", "-T", "1"],
                              stdout=log_file, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW)
    recorder = subprocess.Popen(device.prefix + ["shell", "screenrecord", "--bit-rate", "14000000",
                                                  "--time-limit", str(limit), remote],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=CREATE_NO_WINDOW)
    event("recording_started", query=query)
    time.sleep(1.2)
    if recorder.poll() is not None:
        logger.terminate()
        logger.wait(timeout=5)
        log_file.close()
        _, error = recorder.communicate()
        raise RuntimeError("Device screenrecord did not start: " + error.decode(errors="replace"))
    pid_lines = device.run("shell", "ps", "-A", "-o", "PID,ARGS").decode(errors="replace").splitlines()
    recorder_pids = [line.split()[0] for line in pid_lines if "screenrecord" in line and remote in line]
    selected = False
    complete = False
    index = 0
    try:
        tree = device.tree("ready_to_send")
        button = find(tree, description="Run command")
        if button is None:
            raise RuntimeError("Run command control disappeared")
        event("submitted_query", bounds=device.tap_node(button))
        while time.monotonic()-record_start < limit-24:
            logs = (device.directory / "live_logcat.txt").read_text(encoding="utf-8", errors="replace")
            if "GenUICraft conversion succeeded:" in logs:
                complete = True
                event("conversion_succeeded")
                break
            if "GenUICraft conversion failed:" in logs:
                event("conversion_failed")
                break
            if selected:
                # Streaming updates prevent UIAutomator from becoming idle. The video and
                # PNG capture work without that idle wait; poll native completion instead.
                if index % 3 == 0:
                    device.screenshot(f"stream_{index:03d}")
                index += 1
                time.sleep(1.3)
                continue
            try:
                tree = device.tree(f"poll_{index:03d}")
            except (subprocess.SubprocessError, ET.ParseError) as exc:
                event("poll_retry", reason=type(exc).__name__)
                time.sleep(1)
                continue
            genui = find(tree, text="GenUICraft")
            if not selected and genui is not None:
                event("selected_genuicraft", bounds=device.tap_node(genui))
                selected = True
                time.sleep(0.5)
                device.screenshot("generation_selected")
            if selected and index % 3 == 0:
                device.screenshot(f"stream_{index:03d}")
            index += 1
            time.sleep(1.3)
        event("waiting_finished", complete=complete, selected=selected)
        time.sleep(2)
        tree = device.snapshot("genuicraft_final")
        event("rendered_view_visible")
        time.sleep(5)
        # Same-answer comparison, followed by a return to the generated cards.
        bixby = find(tree, text="Bixby")
        if complete and bixby is not None:
            event("selected_bixby", bounds=device.tap_node(bixby))
            time.sleep(1.2)
            tree = device.snapshot("bixby_original")
            time.sleep(4)
            genui = find(tree, text="GenUICraft")
            if genui is not None:
                event("returned_genuicraft", bounds=device.tap_node(genui))
                time.sleep(1)
                device.snapshot("genuicraft_returned")
                time.sleep(4)
            event("scroll_for_detail")
            device.run("shell", "input", "swipe", "560", "1950", "560", "1080", "650")
            time.sleep(1)
            tree = device.snapshot("genuicraft_detail")
            time.sleep(4)
        print_nodes(tree)
    finally:
        event("stopping_recording")
        for pid in recorder_pids:
            device.run("shell", "kill", "-2", pid)
        try:
            _, err = recorder.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            recorder.terminate()
            _, err = recorder.communicate(timeout=5)
        (device.directory / "screenrecord_stderr.txt").write_bytes(err)
        logger.terminate()
        logger.wait(timeout=5)
        log_file.close()
        for attempt in range(3):
            try:
                device.run("pull", remote, str(device.directory / "raw.mp4"), timeout=60)
                break
            except subprocess.CalledProcessError:
                if attempt == 2:
                    raise
                time.sleep(2)
        event("recording_pulled", complete=complete)
        lines = (device.directory / "live_logcat.txt").read_text(encoding="utf-8", errors="replace").splitlines()
        kept = [line for line in lines if re.search(r"GenUICraft|GenUiIntegration|GenUiHost|GenUiTrained|A2UI|FATAL EXCEPTION|AndroidRuntime", line)]
        (device.directory / "runtime_evidence.txt").write_text("\n".join(kept), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "capture", "snapshot", "tap", "scroll"))
    parser.add_argument("--serial", default="R3GL203AKSF")
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--query")
    parser.add_argument("--name", default="snapshot")
    parser.add_argument("--text")
    parser.add_argument("--description")
    parser.add_argument("--limit", type=int, default=180)
    args = parser.parse_args()
    device = Device(args.serial, args.dir.resolve())
    if args.mode == "prepare":
        if not args.query:
            parser.error("prepare requires --query")
        prepare(device, args.query)
    elif args.mode == "capture":
        capture(device, args.limit)
    elif args.mode == "snapshot":
        print_nodes(device.snapshot(args.name))
    elif args.mode == "tap":
        tree = device.tree()
        node = find(tree, text=args.text, description=args.description)
        if node is None:
            raise RuntimeError("Requested control is not visible")
        print(device.tap_node(node))
    else:
        device.run("shell", "input", "swipe", "560", "1950", "560", "1080", "650")
        print_nodes(device.snapshot(args.name))


if __name__ == "__main__":
    main()
