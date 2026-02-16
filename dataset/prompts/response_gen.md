# response_gen_v9_heading_specific

You are generating a high-quality, complete, single-turn response to the user query below.
The response will be transformed into UI, so produce structured, render-friendly text.

Intent:
{intent}

Tags:
{tags}

Query:
{query_text}

Core rules:
1) Provide a complete answer in one turn. Do not ask follow-up questions.
2) Do not refuse due to browsing/tool limits.
3) If details are missing, make brief reasonable assumptions in a clearly titled assumptions/context section.
   Do NOT use the literal heading "Assumptions".
4) For live/local data requests (weather, flights, nearby places, current prices, trends):
   include both (A) an offline decision framework + verification steps and (B) clearly labeled EXAMPLE options.
5) Output plain text only. No markdown emphasis.

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

Media enrichment policy:
- If intent/tags imply visual content (travel, booking, product_lookup, recipe, entertainment, event_schedule, weather, localization, qr_scanner, status_check), include both:
  1) Images section with 2-5 relevant representative image URLs.
  2) Icons section with 1-3 relevant icon URLs.
- For non-visual intents, Images/Icons are optional.

Media output format:
Images:
- <title>: <image_url>
Icons:
- <name>: <icon_url>

Asset URL rules (strict):
- Include Images/Icons only when you can provide real sample URLs that are publicly accessible now.
- Use direct asset URLs whenever possible (image file URLs for image/icon entries).
- If you are not confident a real asset URL exists, omit that specific image/icon entry instead of guessing.
- Do not output broken, fake, or placeholder asset URLs.
- Avoid hosts that are frequently blocked in automated download (for example: upload.wikimedia.org, images.unsplash.com, cdn.pixabay.com, deep images.pexels.com links).
- Prefer direct image URLs sized for UI cards (roughly landscape, around 1200x800 or similar).
- When uncertain, use keyword-based real photos via: https://loremflickr.com/1200/800/<keyword>
- Prefer direct icon SVG URLs sized for UI use (roughly 64-256 px square), for example:
  https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/icons/<icon-name>.svg

Source rules:
- If verifiable factual claims are included, add 1-3 stable links under Sources.
- Omit Sources for pure general guidance.

Output protocol (must follow):
- If given a JSON array of {query_id, query_text}, return a JSON array of {query_id, response_text}.
- If asked for multiple responses to one query, return a JSON array of strings.
- Otherwise, return plain text only.

Now generate the response.
