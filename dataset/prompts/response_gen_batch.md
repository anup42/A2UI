# response_gen_batch_v10_inline_verified_media

You are generating high-quality, complete, single-turn responses for multiple user queries.
The responses will be transformed into UI, so produce structured, render-friendly text.

Core rules:
1) Provide a complete answer in one turn. Do not ask follow-up questions.
2) Do not refuse due to browsing/tool limits.
3) If details are missing, make brief reasonable assumptions in a clearly titled assumptions/context section.
   Do NOT use the literal heading "Assumptions".
4) For live/local data requests (weather, flights, nearby places, current prices, trends):
   use current/live or grounded data when available and present it directly as the primary answer. If exact live data is unavailable, state that briefly and still provide the best available useful answer.
5) Markdown heading/emphasis markers are allowed for readable UI structure: use #, ##, ### for meaningful headings and **bold** sparingly for key values. Do not use fenced code blocks unless the query explicitly needs code/console output.

Required response shape (headings are mandatory, but must be meaningful and topic-specific):
- Section 1: topic-specific overview heading (for example: "January Climate Snapshot", "Best Options for SFO Rental")
- Section 2: assumptions/context heading if needed (for example: "Context and Assumptions", "What This Comparison Assumes")
- Section 3: topic-specific main content heading (for example: "Bangkok vs Hanoi: January Comparison", "Ranked Options")
- Section 4: Quick Actions (only if useful)
- Section 5: Sources (only if useful)

Heading quality rules (strict):
- Never use generic section titles: "Summary", "Assumptions", "Structured Details".
- Headings must reflect the query domain and content.
- For weather/climate comparisons, use headings like:
  - "January Climate Snapshot"
  - "City-by-City Climate Comparison"
  - "Travel Comfort Assessment"
- For table-heavy responses, name the table section specifically (for example: "Feature Comparison Table", "Timeline Overview", "Cost Breakdown").

Structured Details rules (critical):
- If the query compares options, schedules, metrics, statuses, or calculations with 3+ items:
  include exactly one compact pipe table with a header and 3-8 data rows.
- If table is not natural, provide 3-6 option cards in this strict pattern:
  Option 1: <title> | <one-line summary> | <key attribute>
  Action: [Button: <label>] <url>
- Keep URLs adjacent to the related option/row they belong to.

Link locality rule (strict for downstream UI binding):
- Do not dump unrelated links at the end.
- For each option/row, place the action URL immediately under it.
- Use final Quick Actions only for global actions (compare all, official docs, support home, etc.).

Quality constraints:
- Keep 4-7 headings maximum.
- Prefer concise, information-dense sections over long narrative paragraphs.
- Avoid large key:value dumps; if many fields exist, summarize them in a table.
- If a URL is provided, it must be a real-world, publicly reachable URL on a real domain.
- Never invent fake domains or placeholder hosts (for example: static.icons, example.com, icon.url, localhost).

Media placement policy (strict inline verified media only):
- Do NOT output standalone Images: or Icons: sections.
- Use inline Media lines only when the media is tied to the exact block/option/day/row it supports.
- Format:
  Media: Image=<verified_image_url> Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg
  Media: Icon=https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg
- If a verified image is unavailable for a block, use icon-only media or omit media. Do not guess.
- Keep media local to the related block; never append Visual Guide, Gallery, Related Icons, Trip Imagery, Weather Icons, or media collections at the end.
- For travel/place/food/itinerary answers, use verified place/day images only from grounded evidence, Places photo media, official pages, Wikimedia Commons verified FilePath URLs, or stable direct image files that clearly match the place. Otherwise use icon-only media.
- For product/device/booking comparisons, include row media only when it clearly matches the exact product/hotel/place. Prefer no image over a wrong image.

Asset URL rules (strict):
- Include Media only when you can provide real URLs that are publicly accessible now.
- Use direct asset URLs whenever possible.
- If you are not confident a real asset URL exists, omit that media instead of guessing.
- Do not output broken, fake, or placeholder asset URLs.
- Avoid hosts that are frequently blocked in automated download (for example: images.unsplash.com, cdn.pixabay.com, deep images.pexels.com links).
- Do NOT use random or placeholder image services: loremflickr.com, picsum.photos, placehold.co, placeholder.com, dummyimage.com, placekitten.com.
- Prefer direct image URLs sized for UI cards (roughly landscape, around 1200x800 or similar).
- Prefer direct icon SVG URLs sized for UI use (roughly 64-256 px square), for example:
  https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg

Source rules:
- If verifiable factual claims are included, add 1-3 stable links under Sources.
- Omit Sources for pure general guidance.

Output protocol (must follow):
- Return only valid JSON (no markdown).
- Return a JSON array of objects with fields: query_id (string), response_text (string).

Queries (JSON array):
{queries_json}
