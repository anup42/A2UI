from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BOOTSTRAP_VERSION = "1.11.3"
BOOTSTRAP_BASE_URL = (
    f"https://cdn.jsdelivr.net/npm/bootstrap-icons@{BOOTSTRAP_VERSION}/icons"
)


@dataclass(frozen=True)
class IconDef:
    icon_id: str
    keywords: tuple[str, ...]
    intents: tuple[str, ...]


ICON_DEFS: tuple[IconDef, ...] = (
    IconDef("airplane", ("flight", "airport", "plane", "air"), ("travel", "booking")),
    IconDef("alarm", ("alarm", "timer", "wake", "schedule"), ("event_schedule",)),
    IconDef("bag-check", ("bag", "shopping", "purchase"), ("product_lookup",)),
    IconDef("bank", ("bank", "finance", "account"), ("information_retrieval",)),
    IconDef("upc-scan", ("barcode", "scan", "code"), ("qr_scanner",)),
    IconDef("bell", ("alert", "notification", "status"), ("status_check",)),
    IconDef("book", ("book", "read", "study"), ("education", "documentation")),
    IconDef("bookmark-star", ("favorite", "save", "bookmark"), ("productivity",)),
    IconDef("briefcase", ("work", "business", "office"), ("productivity", "planning")),
    IconDef("calendar-event", ("calendar", "date", "event", "schedule", "time"), ("event_schedule", "planning")),
    IconDef("camera", ("camera", "photo", "image"), ("media_playback", "entertainment")),
    IconDef("camera-video", ("video", "meeting", "call"), ("technical_support",)),
    IconDef("calculator", ("calculator", "calc", "math", "compute"), ("calculation",)),
    IconDef("car-front", ("car", "vehicle", "rental", "drive"), ("travel", "booking", "product_lookup")),
    IconDef("cart", ("cart", "order", "shop"), ("product_lookup", "booking")),
    IconDef("cash-coin", ("cash", "money", "price", "cost", "budget"), ("calculation", "comparison")),
    IconDef("bar-chart-line", ("chart", "trend", "analytics"), ("data_visualization", "comparison")),
    IconDef("check-circle", ("ok", "done", "success"), ("status_check",)),
    IconDef("clipboard-check", ("task", "checklist", "plan"), ("planning", "productivity")),
    IconDef("cloud-sun", ("weather", "forecast", "temperature"), ("weather", "localization")),
    IconDef("code-slash", ("code", "developer", "api"), ("technical_support", "documentation")),
    IconDef("compass", ("navigate", "direction", "route"), ("navigation", "travel")),
    IconDef("controller", ("game", "gaming", "play"), ("entertainment",)),
    IconDef("cup-hot", ("coffee", "drink", "cafe"), ("localization", "travel")),
    IconDef("emoji-smile", ("fun", "mood", "happy"), ("entertainment",)),
    IconDef("exclamation-triangle", ("warning", "error", "risk"), ("technical_support", "status_check")),
    IconDef("file-earmark-text", ("file", "document", "report"), ("documentation",)),
    IconDef("film", ("movie", "stream", "cinema"), ("media_playback", "entertainment")),
    IconDef("geo-alt", ("location", "place", "city"), ("travel", "localization", "navigation")),
    IconDef("globe", ("global", "world", "international"), ("localization", "travel")),
    IconDef("graph-up", ("growth", "stats", "metric"), ("comparison", "research_analysis")),
    IconDef("headphones", ("audio", "listen", "sound"), ("media_playback",)),
    IconDef("heart-pulse", ("health", "medical", "fitness"), ("status_check",)),
    IconDef("house", ("home", "housing", "stay"), ("booking", "travel")),
    IconDef("image", ("image", "picture", "gallery"), ("entertainment",)),
    IconDef("info-circle", ("info", "help", "details"), ("information_retrieval",)),
    IconDef("joystick", ("arcade", "console", "gaming"), ("entertainment",)),
    IconDef("key", ("access", "secure", "login"), ("technical_support", "status_check")),
    IconDef("laptop", ("device", "computer", "pc"), ("technical_support", "productivity")),
    IconDef("lightning-charge", ("power", "energy", "fast"), ("status_check",)),
    IconDef("link-45deg", ("link", "source", "reference"), ("information_retrieval", "research_analysis")),
    IconDef("list-check", ("list", "items", "todo"), ("planning", "productivity")),
    IconDef("map", ("map", "route", "area"), ("navigation", "travel")),
    IconDef("megaphone", ("announcement", "marketing", "campaign"), ("event_schedule", "product_lookup")),
    IconDef("mic", ("voice", "speak", "audio"), ("media_playback",)),
    IconDef("moon-stars", ("night", "evening", "dark"), ("weather", "travel")),
    IconDef("music-note-beamed", ("music", "song", "playlist"), ("media_playback", "entertainment")),
    IconDef("newspaper", ("news", "headline", "article"), ("information_retrieval",)),
    IconDef("palette", ("design", "style", "color"), ("creative_writing", "entertainment")),
    IconDef("patch-question", ("unknown", "faq", "question"), ("technical_support", "education")),
    IconDef("people", ("group", "team", "community"), ("planning", "productivity")),
    IconDef("person-check", ("profile", "verified", "account"), ("status_check",)),
    IconDef("phone", ("call", "contact", "support"), ("technical_support", "booking")),
    IconDef("play-circle", ("play", "start", "watch"), ("media_playback", "entertainment")),
    IconDef("question-circle", ("question", "help", "support"), ("education", "technical_support")),
    IconDef("qr-code-scan", ("qr", "scan", "barcode"), ("qr_scanner",)),
    IconDef("receipt", ("bill", "invoice", "payment"), ("calculation", "comparison")),
    IconDef("search", ("search", "find", "lookup"), ("information_retrieval", "product_lookup")),
    IconDef("shield-check", ("secure", "privacy", "safe"), ("technical_support", "status_check")),
    IconDef("stopwatch", ("time", "duration", "timer"), ("planning", "calculation")),
    IconDef("suitcase", ("trip", "luggage", "journey"), ("travel", "booking")),
    IconDef("table", ("table", "rows", "columns"), ("comparison", "data_visualization")),
    IconDef("ticket-perforated", ("ticket", "booking", "event"), ("booking", "event_schedule")),
    IconDef("tools", ("tool", "fix", "repair"), ("technical_support",)),
    IconDef("translate", ("language", "translation", "locale"), ("localization",)),
    IconDef("tree", ("outdoor", "nature", "park"), ("travel", "weather")),
    IconDef("trophy", ("winner", "rank", "award"), ("comparison", "entertainment")),
    IconDef("truck", ("delivery", "shipping", "logistics"), ("status_check", "product_lookup")),
    IconDef("ui-checks-grid", ("ui", "form", "options"), ("productivity", "documentation")),
    IconDef("wallet2", ("wallet", "budget", "expense"), ("calculation", "comparison")),
    IconDef("wifi", ("internet", "network", "signal"), ("technical_support", "status_check")),
    IconDef("wrench-adjustable", ("settings", "configure", "adjust"), ("technical_support",)),
)


DEFAULT_INTENT_DEFAULTS = {
    "information_retrieval": ["info-circle", "search", "link-45deg"],
    "entertainment": ["play-circle", "emoji-smile", "controller"],
    "product_lookup": ["cart", "bag-check", "search"],
    "booking": ["ticket-perforated", "calendar-event", "house"],
    "weather": ["cloud-sun", "geo-alt", "moon-stars"],
    "data_visualization": ["bar-chart-line", "graph-up", "table"],
    "planning": ["calendar-event", "list-check", "stopwatch"],
    "productivity": ["briefcase", "clipboard-check", "ui-checks-grid"],
    "recipe": ["cup-hot", "list-check", "emoji-smile"],
    "localization": ["translate", "geo-alt", "globe"],
    "technical_support": ["tools", "wrench-adjustable", "question-circle"],
    "creative_writing": ["palette", "book", "emoji-smile"],
    "event_schedule": ["calendar-event", "ticket-perforated", "megaphone"],
    "research_analysis": ["graph-up", "newspaper", "link-45deg"],
    "comparison": ["table", "graph-up", "trophy"],
    "calculation": ["calculator", "cash-coin", "receipt"],
    "travel": ["airplane", "car-front", "map"],
    "navigation": ["compass", "map", "geo-alt"],
    "education": ["book", "question-circle", "patch-question"],
    "documentation": ["file-earmark-text", "book", "ui-checks-grid"],
    "media_playback": ["play-circle", "music-note-beamed", "headphones"],
    "qr_scanner": ["qr-code-scan", "barcode", "camera"],
    "status_check": ["check-circle", "bell", "exclamation-triangle"],
}


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local Bootstrap icon catalog.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Dataset repo root (default: auto-detected).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    root = args.root.resolve()

    catalog_dir = root / "assets" / "icon_catalog" / "bootstrap-icons"
    icons_dir = catalog_dir / "icons"
    catalog_path = catalog_dir / "catalog.json"
    mapping_path = root / "configs" / "icon_map.json"
    license_path = catalog_dir / "SOURCES_AND_LICENSES.md"

    icons_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[dict] = []
    failures: list[str] = []

    for icon_def in ICON_DEFS:
        source_url = f"{BOOTSTRAP_BASE_URL}/{icon_def.icon_id}.svg"
        local_file = icons_dir / f"{icon_def.icon_id}.svg"
        try:
            data = _download(source_url)
            if b"<svg" not in data[:512]:
                raise RuntimeError("not an svg payload")
            local_file.write_bytes(data)
            downloaded.append(
                {
                    "id": icon_def.icon_id,
                    "source_url": source_url,
                    "local_path": str(local_file.relative_to(root)).replace("\\", "/"),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                    "keywords": list(icon_def.keywords),
                    "intents": list(icon_def.intents),
                    "license": "MIT",
                    "provider": "Bootstrap Icons",
                }
            )
        except Exception as exc:
            failures.append(f"{icon_def.icon_id}: {exc}")

    downloaded.sort(key=lambda item: item["id"])

    catalog_payload = {
        "catalog_version": "v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Bootstrap Icons",
        "provider_version": BOOTSTRAP_VERSION,
        "provider_homepage": "https://icons.getbootstrap.com/",
        "license": {
            "name": "MIT",
            "url": "https://github.com/twbs/icons/blob/main/LICENSE.md",
        },
        "icons": downloaded,
    }
    _write_json(catalog_path, catalog_payload)

    keyword_map: dict[str, str] = {}
    for icon_def in ICON_DEFS:
        for keyword in icon_def.keywords:
            keyword_map.setdefault(keyword, icon_def.icon_id)

    mapping_payload = {
        "catalog": str(catalog_path.relative_to(root)).replace("\\", "/"),
        "min_icons_per_response": 2,
        "max_icons_per_response": 3,
        "fallback_icons": ["info-circle", "link-45deg", "question-circle"],
        "keyword_map": keyword_map,
        "intent_defaults": DEFAULT_INTENT_DEFAULTS,
    }
    _write_json(mapping_path, mapping_payload)

    lines = [
        "# Icon Sources and Licenses",
        "",
        "This project uses icons from Bootstrap Icons.",
        "",
        "## Source",
        f"- Project: Bootstrap Icons {BOOTSTRAP_VERSION}",
        "- Homepage: https://icons.getbootstrap.com/",
        f"- CDN base: {BOOTSTRAP_BASE_URL}/",
        "",
        "## License",
        "- License: MIT",
        "- License text: https://github.com/twbs/icons/blob/main/LICENSE.md",
        "",
        "## Catalog",
        f"- Catalog file: `{catalog_path.relative_to(root).as_posix()}`",
        f"- Mapping file: `{mapping_path.relative_to(root).as_posix()}`",
        f"- Total downloaded icons: {len(downloaded)}",
    ]
    if failures:
        lines.append(f"- Failed downloads: {len(failures)}")
    lines.extend(["", "## Per-icon Provenance", ""])
    for item in downloaded:
        lines.append(
            f"- `{item['id']}`: `{item['local_path']}` | source: {item['source_url']} | license: MIT"
        )
    if failures:
        lines.extend(["", "## Failed Downloads", ""])
        for failure in failures:
            lines.append(f"- {failure}")
    license_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Prepared icon catalog: {catalog_path}")
    print(f"Prepared icon mapping: {mapping_path}")
    print(f"Prepared license manifest: {license_path}")
    if failures:
        print(f"Some icons failed to download ({len(failures)}).")


if __name__ == "__main__":
    main()

