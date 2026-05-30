from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "prompts" / "stage3_modules"
LEGACY_PROMPT = ROOT / "prompts" / "genui_gen_mobile_flatspec_v11.md"

REQUIRED_TERMS = {
    "flat_spec_contract": ["root", "state", "elements", "json object"],
    "markdown_cleanup": ["markdown", "CodeBlock", "ConsoleLog"],
    "compact_tables": ["statePath", "preferredPresentation", "primaryColumn", "highlightColumns"],
    "media_policy": ["loremflickr.com", "picsum.photos", "trailing galleries", "Icon only"],
    "weather": ["Weather domain", "humidity", "precipitation"],
    "flight": ["Flight domain", "layover", "actionLabel"],
    "booking": ["Booking/place domain", "bookingUrl", "detached Quick Actions"],
    "itinerary": ["Travel itinerary domain", "imageAlt", "matching day/place row"],
    "formula": ["Formula/calculation domain", "variables Table", "LaTeX"],
    "playlist": ["Playlist/music domain", "trackNumber", "genre"],
    "recipe": ["Recipe domain", "Ingredients", "Instructions", "unresolved placeholder"],
    "email": ["Email/message domain", "EmailPreview", "signature"],
    "chart": ["Chart/data domain", "Chart", "numeric values"],
    "status": ["Status/support domain", "diagnostic", "service health"],
    "comparison": ["Comparison domain", "Feature/Metric", "entityMedia"],
}


def main() -> int:
    if not LEGACY_PROMPT.exists():
        print(f"Missing legacy prompt: {LEGACY_PROMPT}", file=sys.stderr)
        return 2
    if not MODULE_DIR.exists():
        print(f"Missing module dir: {MODULE_DIR}", file=sys.stderr)
        return 2

    corpus = "\n".join(path.read_text(encoding="utf-8") for path in MODULE_DIR.glob("*.md"))
    lowered = corpus.lower()
    failures: list[str] = []
    for group, terms in REQUIRED_TERMS.items():
        missing = [term for term in terms if term.lower() not in lowered]
        if missing:
            failures.append(f"{group}: missing {missing}")
    if failures:
        print("Stage3 prompt module coverage FAILED:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"Stage3 prompt module coverage OK: {len(REQUIRED_TERMS)} rule groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
