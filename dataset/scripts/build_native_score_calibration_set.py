from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline.metrics import (  # noqa: E402
    aggregate_metrics,
    compute_intent_metrics,
    compute_media_score,
    compute_overall_score,
    compute_ui_metrics,
    content_coverage,
    count_characters,
    dup_rate,
    lint_score,
)
from pipeline.genui_quality import (  # noqa: E402
    RewardConfig,
    breakdown_to_mapping,
    extract_expected_ui_contract,
    load_default_reward_config,
    score_genui_completion,
)
from pipeline.toon_convert import encode_toon, roundtrip_ok  # noqa: E402
from pipeline.storage import iter_jsonl  # noqa: E402
from utils.config import load_yaml  # noqa: E402


TARGETS = [40, 50, 60, 70, 80, 90, 100]


def _asset_ref(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("assets/"):
        return "../" + normalized
    return normalized


def _load_base_assets(base_run: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    responses = base_run / "responses.jsonl"
    if not responses.exists():
        return out
    for row in iter_jsonl(responses):
        rid = str(row.get("response_id") or "")
        assets = row.get("assets") if isinstance(row.get("assets"), list) else []
        if rid:
            out[rid] = assets
    return out


def _first_asset(
    assets_by_response: dict[str, list[dict[str, Any]]],
    response_id: str,
    suffixes: tuple[str, ...],
) -> str | None:
    for asset in assets_by_response.get(response_id, []):
        path = str(asset.get("path") or "")
        if path.lower().endswith(suffixes):
            return _asset_ref(path)
    return None


def _asset_by_token(
    assets_by_response: dict[str, list[dict[str, Any]]],
    response_id: str,
    token: str,
    suffixes: tuple[str, ...],
) -> str | None:
    token_l = token.lower()
    for asset in assets_by_response.get(response_id, []):
        path = str(asset.get("path") or "")
        if token_l in path.lower() and path.lower().endswith(suffixes):
            return _asset_ref(path)
    return None


def _el(
    typ: str,
    props: dict[str, Any] | None = None,
    children: list[str] | None = None,
    on: dict[str, Any] | None = None,
) -> dict[str, Any]:
    node: dict[str, Any] = {
        "type": typ,
        "props": props or {},
        "children": children if children is not None else [],
    }
    if on:
        node["on"] = on
    return node


def _button(label: str, url: str) -> dict[str, Any]:
    return _el(
        "Button",
        {"label": label, "variant": "primary"},
        [],
        {"press": {"action": "openUrl", "params": {"url": url}}},
    )


def _table(columns: list[dict[str, str]], rows: list[dict[str, Any]], domain: str) -> dict[str, Any]:
    return _el(
        "Table",
        {
            "columns": columns,
            "rows": rows,
            "domain": domain,
            "preferredPresentation": "cards" if domain in {"weather", "flight", "booking", "playlist"} else "table",
        },
        [],
    )


def _mk_response(s: dict[str, Any]) -> str:
    lines = [
        f"# {s['title']}",
        s["overview"],
        "",
        f"## {s['table_title']}",
        "| " + " | ".join(c["label"] for c in s["columns"]) + " |",
        "| " + " | ".join("---" for _ in s["columns"]) + " |",
    ]
    for row in s["rows"]:
        lines.append("| " + " | ".join(str(row.get(c["key"], "")) for c in s["columns"]) + " |")
    for section in s.get("sections", []):
        lines.extend(["", f"## {section['title']}", section["body"]])
    if s.get("actions"):
        lines.extend(["", "## Quick Actions"])
        for action in s["actions"]:
            lines.append(f"Action: [Button: {action['label']}] {action['url']}")
    if s.get("sources"):
        lines.extend(["", "## Sources"])
        for source in s["sources"]:
            lines.append(f"- {source['label']}: {source['url']}")
    return "\n".join(lines)


def _scenarios(assets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    img = (".jpg", ".jpeg", ".png", ".webp")
    svg = (".svg",)
    return [
        {
            "slot": "01",
            "query_id": "q_self_001",
            "response_id": "r_self_001_01",
            "intent": "Weather",
            "tags": ["weather", "forecast", "bengaluru"],
            "title": "Bengaluru 5-Day Weather Outlook",
            "overview": "Current conditions are warm and partly cloudy, with rain risk increasing early in the week. Use this compact forecast to plan clothing, commute timing, and outdoor activities.",
            "table_title": "Detailed 5-Day Forecast",
            "domain": "weather",
            "image": None,
            "icon": _first_asset(assets, "r_cal_001_01", svg),
            "columns": [
                {"key": "day", "label": "Day"},
                {"key": "condition", "label": "Condition"},
                {"key": "temp", "label": "Temp"},
                {"key": "rain", "label": "Rain Chance"},
                {"key": "wind", "label": "Wind"},
                {"key": "wear", "label": "What to wear"},
            ],
            "rows": [
                {"day": "Today", "condition": "Partly cloudy", "temp": "28C/20C", "rain": "20%", "wind": "10 km/h NW", "wear": "Light clothing"},
                {"day": "Sunday", "condition": "Scattered showers", "temp": "27C/20C", "rain": "40%", "wind": "12 km/h NW", "wear": "Umbrella, light layers"},
                {"day": "Monday", "condition": "Moderate rain", "temp": "26C/19C", "rain": "60%", "wind": "15 km/h W", "wear": "Raincoat, closed shoes"},
                {"day": "Tuesday", "condition": "Cloudy", "temp": "27C/20C", "rain": "30%", "wind": "10 km/h W", "wear": "Light jacket"},
                {"day": "Wednesday", "condition": "Sunny intervals", "temp": "29C/21C", "rain": "10%", "wind": "8 km/h SW", "wear": "Comfortable attire"},
            ],
            "sections": [
                {"title": "Travel Comfort", "body": "Morning outdoor plans are best before the higher-rain days. Carry compact rain protection from Sunday onward."}
            ],
            "actions": [
                {"label": "Explore Bengaluru activities", "url": "https://www.tripadvisor.in/Attractions-g297628-Activities-Bengaluru_Bangalore_Karnataka.html"},
                {"label": "View extended forecast", "url": "https://www.accuweather.com/en/in/bengaluru/204108/weather-forecast/204108"},
            ],
            "sources": [{"label": "AccuWeather", "url": "https://www.accuweather.com/"}],
        },
        {
            "slot": "02",
            "query_id": "q_self_002",
            "response_id": "r_self_002_01",
            "intent": "Travel",
            "tags": ["flight", "booking", "comparison"],
            "title": "BLR to LKO Flight Options for June 15",
            "overview": "The most useful choices are non-stop morning and evening flights. Compare fares, timing, and booking actions before selecting a route.",
            "table_title": "Available Flight Options",
            "domain": "flight",
            "image": None,
            "icon": _first_asset(assets, "r_cal_002_01", svg),
            "columns": [
                {"key": "airline", "label": "Airline"},
                {"key": "departure", "label": "Departure"},
                {"key": "arrival", "label": "Arrival"},
                {"key": "duration", "label": "Duration"},
                {"key": "stops", "label": "Stops"},
                {"key": "fare", "label": "Fare"},
            ],
            "rows": [
                {"airline": "IndiGo", "departure": "06:00 AM", "arrival": "08:30 AM", "duration": "2h 30m", "stops": "Non-stop", "fare": "INR 6,850", "bookingUrl": "https://www.makemytrip.com/flights/", "actionLabel": "Check Fare"},
                {"airline": "Air India", "departure": "09:15 AM", "arrival": "11:50 AM", "duration": "2h 35m", "stops": "Non-stop", "fare": "INR 7,200", "bookingUrl": "https://www.airindia.com/", "actionLabel": "Open Airline"},
                {"airline": "Vistara", "departure": "11:30 AM", "arrival": "02:05 PM", "duration": "2h 35m", "stops": "Non-stop", "fare": "INR 7,550", "bookingUrl": "https://www.skyscanner.co.in/", "actionLabel": "Compare"},
                {"airline": "IndiGo", "departure": "03:45 PM", "arrival": "06:20 PM", "duration": "2h 35m", "stops": "Non-stop", "fare": "INR 7,100", "bookingUrl": "https://www.goindigo.in/", "actionLabel": "Book"},
            ],
            "sections": [{"title": "Best Pick", "body": "Choose the earliest IndiGo option if you want the lowest fare and a full day after arrival."}],
            "actions": [
                {"label": "Search on MakeMyTrip", "url": "https://www.makemytrip.com/flights/"},
                {"label": "Compare on Skyscanner", "url": "https://www.skyscanner.co.in/"},
            ],
            "sources": [{"label": "Skyscanner", "url": "https://www.skyscanner.co.in/"}],
        },
        {
            "slot": "03",
            "query_id": "q_self_003",
            "response_id": "r_self_003_01",
            "intent": "Booking",
            "tags": ["hotel", "booking", "kyoto"],
            "title": "Kyoto Family Hotel Shortlist",
            "overview": "These hotels balance family convenience, fitness access, Wi-Fi, and station-area logistics under the target budget.",
            "table_title": "Top Family-Friendly Hotels",
            "domain": "booking",
            "image": _first_asset(assets, "r_cal_003_01", img),
            "icon": _first_asset(assets, "r_cal_003_01", svg),
            "columns": [
                {"key": "hotel", "label": "Hotel"},
                {"key": "area", "label": "Area"},
                {"key": "price", "label": "Price"},
                {"key": "family", "label": "Family Fit"},
                {"key": "gym", "label": "Gym"},
                {"key": "wifi", "label": "Wi-Fi"},
            ],
            "rows": [
                {"hotel": "The Thousand Kyoto", "area": "Kyoto Station", "price": "~$175/night", "family": "Spacious rooms", "gym": "Yes", "wifi": "Free", "bookingUrl": "https://www.booking.com/hotel/jp/the-thousand-kyoto.html", "actionLabel": "Check Availability"},
                {"hotel": "Hotel Granvia Kyoto", "area": "Kyoto Station", "price": "~$170/night", "family": "Transit-friendly", "gym": "Yes", "wifi": "Free", "bookingUrl": "https://www.booking.com/hotel/jp/granvia-kyoto.html", "actionLabel": "Check Availability"},
                {"hotel": "Kyoto Century Hotel", "area": "Station east", "price": "~$165/night", "family": "Classic rooms", "gym": "Nearby", "wifi": "Free", "bookingUrl": "https://www.booking.com/hotel/jp/kyoto-century.html", "actionLabel": "Check Availability"},
            ],
            "sections": [{"title": "Recommendation", "body": "Hotel Granvia is easiest for families because it minimizes transfers and keeps meals, taxis, and rail access simple."}],
            "actions": [{"label": "Search All Kyoto Hotels", "url": "https://www.booking.com/searchresults.en-us.html?city=-235489"}],
            "sources": [{"label": "Booking.com", "url": "https://www.booking.com"}],
        },
        {
            "slot": "04",
            "query_id": "q_self_004",
            "response_id": "r_self_004_01",
            "intent": "Shopping",
            "tags": ["comparison", "coffee", "shopping"],
            "title": "Coffee Maker Comparison for Small Apartments",
            "overview": "The right coffee maker depends on space, cleanup tolerance, coffee style, and recurring cost.",
            "table_title": "Feature Comparison Table",
            "domain": "comparison",
            "image": _asset_by_token(assets, "r_cal_004_01", "french", img) or _first_asset(assets, "r_cal_004_01", img),
            "icon": _first_asset(assets, "r_cal_004_01", svg),
            "columns": [
                {"key": "feature", "label": "Feature"},
                {"key": "drip", "label": "Drip Coffee Maker"},
                {"key": "press", "label": "French Press"},
                {"key": "pod", "label": "Pod Coffee Machine"},
            ],
            "rows": [
                {"feature": "Space", "drip": "Medium to large", "press": "Small", "pod": "Small to medium"},
                {"feature": "Setup", "drip": "Easy", "press": "Manual", "pod": "Very easy"},
                {"feature": "Clean-up", "drip": "Moderate", "press": "Moderate", "pod": "Minimal"},
                {"feature": "Coffee Quality", "drip": "Consistent", "press": "Rich and full-bodied", "pod": "Convenient but varies"},
                {"feature": "Waste", "drip": "Filter and grounds", "press": "Grounds", "pod": "Pods and packaging"},
                {"feature": "Best For", "drip": "Guests", "press": "Craft coffee", "pod": "Fast single cups"},
            ],
            "sections": [{"title": "Apartment Pick", "body": "Choose French Press for the best space-to-quality ratio; choose pods only if speed matters most."}],
            "actions": [
                {"label": "Shop French Presses", "url": "https://www.amazon.com/s?k=french+press"},
                {"label": "Shop Pod Machines", "url": "https://www.amazon.com/s?k=pod+coffee+machine"},
            ],
            "sources": [{"label": "Coffee Detective", "url": "https://www.coffeedetective.com/types-of-coffee-makers.html"}],
        },
        {
            "slot": "05",
            "query_id": "q_self_005",
            "response_id": "r_self_005_01",
            "intent": "Entertainment",
            "tags": ["playlist", "music", "entertainment"],
            "title": "Cyberpunk Rain Night Soundtrack",
            "overview": "A dark, neon playlist for late-night focus, rain ambience, and cinematic city energy.",
            "table_title": "Tracklist for a Neo-Noir Atmosphere",
            "domain": "playlist",
            "image": None,
            "icon": _first_asset(assets, "r_cal_005_01", svg),
            "columns": [
                {"key": "order", "label": "Order"},
                {"key": "artist", "label": "Artist"},
                {"key": "track", "label": "Track Title"},
                {"key": "mood", "label": "Mood"},
            ],
            "rows": [
                {"order": "1", "artist": "Perturbator", "track": "Neo Tokyo", "mood": "Driving, cyberpunk"},
                {"order": "2", "artist": "Carpenter Brut", "track": "Roller Mobster", "mood": "Intense synthwave"},
                {"order": "3", "artist": "Lorn", "track": "Acid Rain", "mood": "Dark, brooding"},
                {"order": "4", "artist": "Kavinsky", "track": "Nightcall", "mood": "Retro, melancholic"},
                {"order": "5", "artist": "Com Truise", "track": "Cyanide Sisters", "mood": "Dreamy electronic"},
                {"order": "6", "artist": "Vangelis", "track": "Blade Runner Blues", "mood": "Somber classic"},
            ],
            "sections": [{"title": "Listening Flow", "body": "Start with momentum, drop into brooding rain texture, then close with a classic cinematic cool-down."}],
            "actions": [{"label": "Open Synthwave Playlist", "url": "https://open.spotify.com/search/synthwave%20cyberpunk"}],
            "sources": [{"label": "Spotify Search", "url": "https://open.spotify.com/"}],
        },
        {
            "slot": "06",
            "query_id": "q_self_006",
            "response_id": "r_self_006_01",
            "intent": "Travel",
            "tags": ["itinerary", "bengaluru", "travel"],
            "title": "Bengaluru 4-Day Vacation Itinerary",
            "overview": "This route mixes heritage, gardens, food, technology, shopping, and relaxed evening neighborhoods.",
            "table_title": "Day-by-Day Plan",
            "domain": "schedule",
            "image": _asset_by_token(assets, "r_cal_006_01", "iskcon", img) or _first_asset(assets, "r_cal_006_01", img),
            "icon": _first_asset(assets, "r_cal_006_01", svg),
            "columns": [
                {"key": "day", "label": "Day"},
                {"key": "morning", "label": "Morning"},
                {"key": "afternoon", "label": "Afternoon"},
                {"key": "food", "label": "Food Stop"},
                {"key": "action", "label": "Action"},
            ],
            "rows": [
                {"day": "Day 1", "morning": "Bangalore Palace", "afternoon": "Cubbon Park", "food": "Karnataka thali", "action": "Palace info"},
                {"day": "Day 2", "morning": "Lalbagh Garden", "afternoon": "Commercial Street", "food": "Dosa and idli", "action": "Garden details"},
                {"day": "Day 3", "morning": "Technology Museum", "afternoon": "ISKCON Temple", "food": "Indiranagar cafe", "action": "Temple map"},
                {"day": "Day 4", "morning": "Tipu Palace", "afternoon": "UB City", "food": "Modern Karnataka dinner", "action": "Route plan"},
            ],
            "sections": [
                {"title": "Day 1: Heritage Core", "body": "Keep the first day compact and central to avoid traffic fatigue."},
                {"title": "Day 2: Garden and Market", "body": "Start early at Lalbagh, then move to Commercial Street for shopping and snacks."},
                {"title": "Day 3: Museum and Temple", "body": "Use metro or app cab transfers for predictable timing."},
            ],
            "actions": [
                {"label": "Open Bengaluru Map", "url": "https://www.google.com/maps/search/Bengaluru+tourist+places"},
                {"label": "Bangalore Palace Info", "url": "https://en.wikipedia.org/wiki/Bangalore_Palace"},
            ],
            "sources": [{"label": "Bengaluru Tourism", "url": "https://www.karnatakatourism.org/"}],
        },
        {
            "slot": "07",
            "query_id": "q_self_007",
            "response_id": "r_self_007_01",
            "intent": "Calculation",
            "tags": ["formula", "finance", "mortgage"],
            "title": "Mortgage Payment Estimate",
            "overview": "For a 450,000 USD fixed-rate mortgage at 6.5 percent over 30 years, the estimated monthly payment is 2,842.25 USD before taxes and insurance.",
            "table_title": "Payment Variables and Totals",
            "domain": "calculation",
            "image": None,
            "icon": _first_asset(assets, "r_cal_007_01", svg),
            "columns": [
                {"key": "item", "label": "Item"},
                {"key": "description", "label": "Description"},
                {"key": "value", "label": "Value"},
            ],
            "rows": [
                {"item": "P", "description": "Principal loan amount", "value": "$450,000"},
                {"item": "i", "description": "Monthly interest rate", "value": "0.00541667"},
                {"item": "n", "description": "Total payments", "value": "360"},
                {"item": "M", "description": "Monthly payment", "value": "$2,842.25"},
                {"item": "Total Interest", "description": "Interest paid over 30 years", "value": "$573,210.00"},
            ],
            "sections": [{"title": "Formula", "body": "M = P * i(1+i)^n / ((1+i)^n - 1). This excludes tax, insurance, HOA fees, and closing costs."}],
            "actions": [{"label": "Explore Mortgage Calculator", "url": "https://www.bankrate.com/mortgages/mortgage-calculator/"}],
            "sources": [{"label": "Bankrate", "url": "https://www.bankrate.com/"}],
        },
        {
            "slot": "08",
            "query_id": "q_self_008",
            "response_id": "r_self_008_01",
            "intent": "Writing",
            "tags": ["email", "business", "message"],
            "title": "Vendor Timeline Revision Email",
            "overview": "Use a direct but collaborative tone: acknowledge the delay, request a revised timeline, and ask for mitigation steps.",
            "table_title": "Email Elements",
            "domain": "email",
            "image": None,
            "icon": _first_asset(assets, "r_cal_008_01", svg),
            "columns": [
                {"key": "part", "label": "Part"},
                {"key": "purpose", "label": "Purpose"},
                {"key": "text", "label": "Draft Text"},
            ],
            "rows": [
                {"part": "Subject", "purpose": "Set context", "text": "Project delivery timeline revision request"},
                {"part": "Opening", "purpose": "Acknowledge delay", "text": "I am following up on the revised delivery timeline."},
                {"part": "Request", "purpose": "Ask clearly", "text": "Please share an updated schedule and mitigation steps by Friday."},
                {"part": "Close", "purpose": "Keep tone constructive", "text": "Thank you for helping us adjust our downstream plan."},
            ],
            "sections": [{"title": "Ready-to-Send Draft", "body": "Dear Vendor, please share a revised project delivery schedule, key delay drivers, and mitigation steps by Friday so we can adjust dependent milestones."}],
            "actions": [{"label": "Copy Email Guidance", "url": "https://www.grammarly.com/blog/email-writing/"}],
            "sources": [{"label": "Grammarly", "url": "https://www.grammarly.com/blog/email-writing/"}],
        },
        {
            "slot": "09",
            "query_id": "q_self_009",
            "response_id": "r_self_009_01",
            "intent": "Tech Support",
            "tags": ["code_console", "support", "npm"],
            "title": "Fix npm EACCES Permission Errors",
            "overview": "EACCES means npm cannot write to the target directory. Prefer a user-level Node manager before changing ownership of system folders.",
            "table_title": "Safe Troubleshooting Steps",
            "domain": "status",
            "image": None,
            "icon": _first_asset(assets, "r_cal_009_01", svg),
            "columns": [
                {"key": "step", "label": "Step"},
                {"key": "command", "label": "Command"},
                {"key": "expected", "label": "Expected Result"},
            ],
            "rows": [
                {"step": "1", "command": "npm config get prefix", "expected": "Shows global npm path"},
                {"step": "2", "command": "whoami", "expected": "Confirms current user"},
                {"step": "3", "command": "nvm install node", "expected": "Installs Node in user space"},
                {"step": "4", "command": "npm cache clean --force", "expected": "Clears locked cache entries"},
                {"step": "5", "command": "npm install", "expected": "Package installs without EACCES"},
            ],
            "sections": [{"title": "Recommended Fix", "body": "Use NVM or another version manager. Avoid sudo npm install because it creates more ownership conflicts later."}],
            "actions": [{"label": "Open NVM Guide", "url": "https://github.com/nvm-sh/nvm#installing-and-updating"}],
            "sources": [{"label": "npm Docs", "url": "https://docs.npmjs.com/resolving-eacces-permissions-errors-when-installing-packages-globally"}],
        },
        {
            "slot": "10",
            "query_id": "q_self_010",
            "response_id": "r_self_010_01",
            "intent": "Planning",
            "tags": ["planning", "ecommerce", "business"],
            "title": "3-Month Handmade Candle Launch Plan",
            "overview": "This launch plan turns product testing, brand setup, store build, and launch marketing into a phased roadmap.",
            "table_title": "Launch Plan Breakdown",
            "domain": "schedule",
            "image": _first_asset(assets, "r_cal_010_01", img),
            "icon": _first_asset(assets, "r_cal_010_01", svg),
            "columns": [
                {"key": "month", "label": "Month"},
                {"key": "phase", "label": "Phase"},
                {"key": "goal", "label": "Goal"},
                {"key": "tasks", "label": "Key Tasks"},
                {"key": "deliverable", "label": "Deliverable"},
            ],
            "rows": [
                {"month": "Month 1", "phase": "Foundation", "goal": "Finalize product and brand", "tasks": "Testing, packaging, registration", "deliverable": "Launch-ready samples"},
                {"month": "Month 2", "phase": "E-commerce", "goal": "Build online store", "tasks": "Photos, listings, payments, shipping", "deliverable": "Functional store"},
                {"month": "Month 3", "phase": "Launch", "goal": "Drive first sales", "tasks": "Email, social, launch promo", "deliverable": "Public launch"},
            ],
            "sections": [
                {"title": "Month 1: Foundation", "body": "Complete burn tests, packaging labels, and business registration before public promotion."},
                {"title": "Month 2: Store Build", "body": "Create product pages, checkout, shipping rules, and first content calendar."},
                {"title": "Month 3: Launch", "body": "Run launch promotion, monitor orders, and collect first customer feedback."},
            ],
            "actions": [
                {"label": "Shopify Platform", "url": "https://www.shopify.com/"},
                {"label": "SBA Registration Guide", "url": "https://www.sba.gov/business-guide/launch-your-business/register-your-business"},
            ],
            "sources": [{"label": "SBA", "url": "https://www.sba.gov/"}],
        },
    ]


def _summary_text(s: dict[str, Any], target: int) -> str:
    if target <= 40:
        return s["overview"].split(".")[0] + "."
    if target <= 50:
        return s["overview"]
    if target <= 60:
        return s["overview"] + " " + (s.get("sections") or [{"body": ""}])[0]["body"]
    return s["overview"]


def _build_spec(s: dict[str, Any], target: int) -> dict[str, Any]:
    elements: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}

    def add(eid: str, node: dict[str, Any]) -> str:
        elements[eid] = node
        return eid

    root_children: list[str] = []
    add("root", _el("Stack", {"direction": "vertical", "gap": "md", "padding": "md"}, root_children))
    root_children.append(add("title", _el("Text", {"text": s["title"], "variant": "h1"}, [])))

    if target <= 40:
        if s.get("icon"):
            root_children.append(add("minimal_icon", _el("Icon", {"source": s["icon"], "size": 24}, [])))
        root_children.append(add("brief", _el("Text", {"text": _summary_text(s, target)}, [])))
        card_children: list[str] = []
        root_children.append(add("one_detail_card", _el("Card", {"padding": "md"}, card_children)))
        card_children.append(add("one_detail_title", _el("Text", {"text": s["table_title"], "variant": "h3"}, [])))
        first_row = s["rows"][0]
        first_bits = [str(first_row.get(c["key"], "")) for c in s["columns"][:3]]
        card_children.append(add("one_detail_body", _el("Text", {"text": " | ".join(first_bits)}, [])))
        second_card_children: list[str] = []
        root_children.append(add("second_detail_card", _el("Card", {"padding": "sm"}, second_card_children)))
        second_card_children.append(add("second_detail_title", _el("Text", {"text": str(first_row.get(s["columns"][0]["key"], "")), "variant": "label"}, [])))
        second_card_children.append(add("second_detail_body", _el("Text", {"text": str(first_row.get(s["columns"][1]["key"], ""))}, [])))
        return {"root": "root", "state": state, "elements": elements}

    hero_children: list[str] = []
    hero_id = add("hero", _el("Card", {"padding": "md"}, hero_children))
    root_children.append(hero_id)
    if s.get("icon"):
        hero_children.append(add("hero_icon", _el("Icon", {"source": s["icon"], "size": 28}, [])))
    if target >= 80 and s.get("image"):
        hero_children.append(add("hero_image", _el("Image", {"source": s["image"], "fit": "cover", "aspectRatio": 1.7}, [])))
    hero_children.append(add("overview", _el("Text", {"text": _summary_text(s, target)}, [])))

    if target <= 50:
        first_row = s["rows"][0]
        card_children = []
        root_children.append(add("quick_fact_card", _el("Card", {"padding": "md"}, card_children)))
        card_children.append(add("quick_fact_title", _el("Text", {"text": s["table_title"], "variant": "h3"}, [])))
        for idx, col in enumerate(s["columns"][:3], start=1):
            card_children.append(
                add(
                    f"quick_fact_{idx}",
                    _el("Text", {"text": f"{col['label']}: {first_row.get(col['key'], '')}"}, []),
                )
            )
        state["rows"] = s["rows"][:2]
        root_children.append(add("tiny_table_heading", _el("Text", {"text": s["table_title"], "variant": "h2"}, [])))
        root_children.append(add("tiny_table", _table(s["columns"][: min(3, len(s["columns"]))], s["rows"][:2], s["domain"])))
        return {"root": "root", "state": state, "elements": elements}

    table_rows = s["rows"]
    table_columns = s["columns"]
    if target <= 60:
        table_rows = s["rows"][:2]
        table_columns = s["columns"][: min(3, len(s["columns"]))]
    elif target <= 70:
        table_rows = s["rows"][: max(3, min(len(s["rows"]), 4))]
        table_columns = s["columns"]

    root_children.append(add("table_heading", _el("Text", {"text": s["table_title"], "variant": "h2"}, [])))
    state["rows"] = table_rows
    root_children.append(add("main_table", _table(table_columns, table_rows, s["domain"])))

    row_card_plan = 0
    if target >= 60:
        row_card_plan = min(4, len(s["rows"]))
    if target >= 70:
        row_card_plan = min(3, len(s["rows"]))
    if target >= 80:
        row_card_plan = min(4, len(s["rows"]))
    if target >= 90:
        row_card_plan = min(6, len(s["rows"]))
    if target >= 100:
        row_card_plan = len(s["rows"])

    if row_card_plan:
        root_children.append(add("highlights_heading", _el("Text", {"text": "Key Details", "variant": "h2"}, [])))
        highlights_children: list[str] = []
        root_children.append(add("highlights", _el("Stack", {"direction": "vertical", "gap": "sm"}, highlights_children)))
        for ridx, row in enumerate(s["rows"][:row_card_plan], start=1):
            card_children: list[str] = []
            highlights_children.append(add(f"detail_card_{ridx}", _el("Card", {"padding": "md"}, card_children)))
            title_col = s["columns"][0]
            subtitle_col = s["columns"][1] if len(s["columns"]) > 1 else s["columns"][0]
            card_children.append(
                add(
                    f"detail_title_{ridx}",
                    _el("Text", {"text": str(row.get(title_col["key"], "")), "variant": "h3"}, []),
                )
            )
            card_children.append(
                add(
                    f"detail_subtitle_{ridx}",
                    _el("Text", {"text": f"{subtitle_col['label']}: {row.get(subtitle_col['key'], '')}"}, []),
                )
            )
            if target >= 70:
                for cidx, col in enumerate(s["columns"][2:4], start=1):
                    card_children.append(
                        add(
                            f"detail_{ridx}_{cidx}",
                            _el("Text", {"text": f"{col['label']}: {row.get(col['key'], '')}", "variant": "caption"}, []),
                        )
                    )
            if target >= 90 and row.get("bookingUrl"):
                card_children.append(
                    add(
                        f"detail_action_{ridx}",
                        _button(str(row.get("actionLabel") or "Open"), str(row.get("bookingUrl"))),
                    )
                )

    if target >= 80:
        for idx, section in enumerate(s.get("sections", [])[: 1 if target == 80 else 3], start=1):
            card_children: list[str] = []
            cid = add(f"section_card_{idx}", _el("Card", {"padding": "md"}, card_children))
            card_children.append(add(f"section_title_{idx}", _el("Text", {"text": section["title"], "variant": "h3"}, [])))
            card_children.append(add(f"section_body_{idx}", _el("Text", {"text": section["body"]}, [])))
            root_children.append(cid)

    if (target >= 80 or target == 70) and s.get("actions"):
        root_children.append(add("actions_heading", _el("Text", {"text": "Quick Actions", "variant": "h2"}, [])))
        action_stack_children: list[str] = []
        root_children.append(add("actions", _el("Stack", {"direction": "vertical", "gap": "sm"}, action_stack_children)))
        action_limit = 1 if target < 90 else 1 if target == 90 else 3
        for idx, action in enumerate(s["actions"][:action_limit], start=1):
            action_stack_children.append(add(f"action_{idx}", _button(action["label"], action["url"])))

    if target >= 100 and s.get("sources"):
        root_children.append(add("sources_heading", _el("Text", {"text": "Sources", "variant": "h2"}, [])))
        for idx, source in enumerate(s["sources"][:2], start=1):
            root_children.append(add(f"source_{idx}", _button(source["label"], source["url"])))

    if target >= 90:
        # Add a compact checklist-style summary. This intentionally uses normal
        # renderer primitives rather than changing the IR contract.
        checklist_children: list[str] = []
        root_children.append(add("checklist_card", _el("Card", {"padding": "md"}, checklist_children)))
        checklist_children.append(add("checklist_title", _el("Text", {"text": "Decision Checklist", "variant": "h3"}, [])))
        for idx, col in enumerate(s["columns"][: min(4, len(s["columns"]))], start=1):
            checklist_children.append(
                add(
                    f"checklist_item_{idx}",
                    _el("Text", {"text": f"{col['label']}: verify before acting"}, []),
                )
            )

    if target >= 100:
        extra_children: list[str] = []
        root_children.append(add("compact_summary_grid", _el("Stack", {"direction": "vertical", "gap": "sm"}, extra_children)))
        for idx, section in enumerate((s.get("sections") or [])[:3], start=1):
            mini_children: list[str] = []
            extra_children.append(add(f"mini_card_{idx}", _el("Card", {"padding": "sm"}, mini_children)))
            mini_children.append(add(f"mini_title_{idx}", _el("Text", {"text": section["title"], "variant": "h3"}, [])))
            mini_children.append(add(f"mini_body_{idx}", _el("Text", {"text": section["body"]}, [])))

    return {"root": "root", "state": state, "elements": elements}


def _record_metrics(
    response_text: str,
    genui_json: dict[str, Any],
    intent: str,
    tags: list[str],
    assets: list[dict[str, Any]],
    expected_ui_contract: dict[str, Any],
    reward_config: RewardConfig,
    weights: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    toon = encode_toon(genui_json)
    json_text = json.dumps(genui_json, ensure_ascii=False)
    metrics: dict[str, Any] = {
        "content_coverage": content_coverage(response_text, genui_json),
        "dup_rate": dup_rate(genui_json),
        "lint_score": lint_score(genui_json),
        "output_tokens_toon": count_characters(toon),
        "output_tokens_json": count_characters(json_text),
        "output_chars_toon": count_characters(toon),
        "output_chars_json": count_characters(json_text),
    }
    metrics.update(compute_ui_metrics(response_text, genui_json))
    intent_metrics = compute_intent_metrics(intent, tags, response_text, metrics)
    intent_metrics.pop("intent_bucket", None)
    metrics.update(intent_metrics)
    result = score_genui_completion(
        genui_json,
        response_text,
        intent=intent,
        assets=assets,
        expected_ui_contract=expected_ui_contract,
        config=reward_config,
    )
    result_mapping = breakdown_to_mapping(result)
    sample = {
        "response_text": response_text,
        "genui_json": genui_json,
        "expected_ui_contract": expected_ui_contract,
        "genui_quality_v4": result_mapping,
        "metrics": metrics,
    }
    legacy_score = compute_overall_score(aggregate_metrics([sample], metric_version="legacy"), weights)
    metrics["legacy_structural_richness_score"] = legacy_score
    metrics["overall_score"] = legacy_score
    metrics["genui_quality_v4"] = result.quality_0_100
    metrics["genui_quality_v4_dimensions"] = result.dimensions
    metrics["genui_quality_v4_active_caps"] = result.active_caps
    metrics["genui_metric_version"] = result.metric_version
    return metrics, result_mapping


def _copy_assets(base_run: Path, out_run: Path) -> None:
    src = base_run / "assets"
    dst = out_run / "assets"
    if dst.exists():
        shutil.rmtree(dst)
    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


def build(run_id: str, base_run_id: str) -> Path:
    out_run = ROOT / "data" / "runs" / run_id
    base_run = ROOT / "data" / "runs" / base_run_id
    out_run.mkdir(parents=True, exist_ok=True)
    _copy_assets(base_run, out_run)
    assets_by_response = _load_base_assets(base_run)
    scenarios = _scenarios(assets_by_response)
    weights = load_yaml(ROOT / "configs" / "run.yaml").get("evaluation", {}).get("weights", {})
    reward_config = load_default_reward_config()

    query_rows = []
    response_rows = []
    genui_rows = []
    manifest_rows = []
    now = datetime.utcnow().isoformat() + "Z"

    assets_for_record: dict[str, list[dict[str, Any]]] = {}
    for scenario in scenarios:
        query_rows.append(
            {
                "query_id": scenario["query_id"],
                "query_text": scenario["overview"],
                "intent": scenario["intent"],
                "tags": scenario["tags"],
                "created_at": now,
            }
        )
        response_text = _mk_response(scenario)
        # Reuse all base assets as a simple run-local asset map; Stage 3/native
        # renderer only loads assets referenced by the IR.
        all_assets = []
        for items in assets_by_response.values():
            all_assets.extend(items)
        assets_for_record[scenario["response_id"]] = all_assets
        expected_ui_contract = extract_expected_ui_contract(
            response_text,
            intent=scenario["intent"],
            assets=all_assets,
        )
        response_rows.append(
            {
                "response_id": scenario["response_id"],
                "query_id": scenario["query_id"],
                "n_idx": 1,
                "response_text": response_text,
                "expected_ui_contract": expected_ui_contract,
                "expected_ui_contract_source": "deterministic fallback",
                "created_at": now,
                "assets": all_assets,
                "asset_stats": {
                    "declared_asset_urls": len(all_assets),
                    "downloaded_assets": len(all_assets),
                    "asset_quality_ok": True,
                    "generator": "codex_local_score_calibration",
                },
                "gen": {
                    "provider": "codex_local",
                    "model": "self_generated_from_stage2_prompt_rules",
                    "prompt_version": "response_gen_v9_heading_specific",
                },
            }
        )

        for target in TARGETS:
            genui_json = _build_spec(scenario, target)
            toon = encode_toon(genui_json)
            metrics, quality_v4 = _record_metrics(
                response_text,
                genui_json,
                scenario["intent"],
                scenario["tags"],
                assets_for_record[scenario["response_id"]],
                expected_ui_contract,
                reward_config,
                weights,
            )
            ui_id = f"s{target:03d}_{scenario['slot']}_u_self_{scenario['slot']}_01"
            record = {
                "ui_id": ui_id,
                "response_id": scenario["response_id"],
                "query_id": scenario["query_id"],
                "intent": scenario["intent"],
                "tags": scenario["tags"],
                "intent_bucket": metrics.get("intent_bucket", scenario["intent"].lower()),
                "target_score_bucket": target,
                "response_text": response_text,
                "expected_ui_contract": expected_ui_contract,
                "expected_ui_contract_source": "deterministic fallback",
                "genui_json": genui_json,
                "genui_quality_v4": quality_v4,
                "evaluation_metric_mode": "dual",
                "renderer_check_result": {
                    "adapter": "android_native",
                    "attempted": False,
                    "ok": None,
                    "source": "calibration_generation_not_rendered",
                },
                "assets": assets_for_record[scenario["response_id"]],
                "toon": toon,
                "validation": {
                    "json_parse_ok": True,
                    "schema_valid_strict": True,
                    "schema_valid_lenient": True,
                    "toon_roundtrip_ok": roundtrip_ok(genui_json, toon),
                    "converted_from_legacy": False,
                    "errors": [],
                    "warnings": ["self_generated_score_calibration_variant"],
                    "repair_attempts": 0,
                    "repair_needed": False,
                },
                "metrics": metrics,
                "gen": {
                    "provider": "codex_local",
                    "model": "self_generated_from_stage3_prompt_rules",
                    "prompt_version": "genui_gen_v10_flatspec_hardcut_style_rules",
                    "latency_ms": 0,
                    "input_tokens": 0,
                    "output_tokens": metrics["output_tokens_json"],
                    "cost_usd": 0,
                    "error": None,
                },
                "created_at": now,
            }
            genui_rows.append(record)
            manifest_rows.append(
                {
                    "target_score": target,
                    "actual_score": round(float(metrics["overall_score"]), 4),
                    "ui_id": ui_id,
                    "response_id": scenario["response_id"],
                    "query_id": scenario["query_id"],
                    "intent": scenario["intent"],
                    "title": scenario["title"],
                }
            )

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_jsonl(out_run / "queries.jsonl", query_rows)
    write_jsonl(out_run / "responses.jsonl", response_rows)
    write_jsonl(out_run / "genui.jsonl", genui_rows)

    by_bucket: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        rows = [row for row in genui_rows if row["target_score_bucket"] == target]
        agg = aggregate_metrics(rows)
        legacy_score = compute_overall_score(agg, weights)
        agg["legacy_structural_richness_score"] = legacy_score
        agg["overall_score"] = legacy_score
        agg["media_score"] = compute_media_score(agg)
        by_bucket[str(target)] = agg
    aggregate_all = aggregate_metrics(genui_rows)
    legacy_score = compute_overall_score(aggregate_all, weights)
    aggregate_all["legacy_structural_richness_score"] = legacy_score
    aggregate_all["overall_score"] = legacy_score
    aggregate_all["media_score"] = compute_media_score(aggregate_all)
    aggregate_all["bucket_scores"] = {
        target: by_bucket[str(target)].get("overall_score")
        for target in map(str, TARGETS)
    }
    (out_run / "aggregates.json").write_text(json.dumps(aggregate_all, indent=2), encoding="utf-8")
    (out_run / "aggregates_by_bucket.json").write_text(json.dumps(by_bucket, indent=2), encoding="utf-8")
    (out_run / "calibration_manifest.json").write_text(
        json.dumps({"run_id": run_id, "base_run_id": base_run_id, "items": manifest_rows}, indent=2),
        encoding="utf-8",
    )
    with (out_run / "calibration_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (out_run / "rating_sheet.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["target_score", "actual_score", "ui_id", "response_id", "intent", "screenshot", "user_rating_1_to_5", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest_rows:
            writer.writerow(
                {
                    "target_score": row["target_score"],
                    "actual_score": row["actual_score"],
                    "ui_id": row["ui_id"],
                    "response_id": row["response_id"],
                    "intent": row["intent"],
                    "screenshot": f"android_device_rendered/{row['ui_id']}.png",
                    "user_rating_1_to_5": "",
                    "notes": "",
                }
            )
    readme = [
        "# Native Score Calibration Set",
        "",
        "This run is generated locally for structural-score calibration. It uses 10 self-authored response texts that follow the Stage 2 response prompt shape, then creates seven strict flat-spec IR variants per response following the Stage 3 contract rules.",
        "",
        "No external LLM is used for the IR variants in this run. The purpose is controlled human calibration: compare native Android screenshots across structural metric score bands.",
        "",
        "Target buckets: 40, 50, 60, 70, 80, 90, 100.",
        "",
        "Important: `target_score` is the intended structural richness band; `actual_score` and `aggregates_by_bucket.json` are the authoritative computed scores from the current metric formula.",
    ]
    (out_run / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    return out_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="score_calibration_native_self_20260701")
    parser.add_argument("--base-run-id", default="score_calibration_native_generated_20260701_base")
    args = parser.parse_args()
    out_run = build(args.run_id, args.base_run_id)
    print(out_run)


if __name__ == "__main__":
    main()
