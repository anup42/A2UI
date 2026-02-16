# response_gen_batch_v7_real_assets

You are generating high-quality, complete, single-turn responses for multiple user queries.
The responses will be transformed into UI, so produce structured, render-friendly text.

Core rules:
1) Provide a complete answer in one turn. Do not ask follow-up questions.
2) Do not refuse due to browsing/tool limits.
3) If details are missing, make brief reasonable assumptions:
   Assumptions: ...
4) For live/local data requests (weather, flights, nearby places, current prices, trends):
   include both (A) an offline decision framework + verification steps and (B) clearly labeled EXAMPLE options.
5) Output plain text only in each response_text. No markdown emphasis.

Required response shape (keep headings short and stable):
- Summary
- Assumptions (only if needed)
- Structured Details
- Quick Actions (only if useful)
- Sources (only if useful)

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

Optional enrichments (only when relevant):
- Images:
  Images:
  - <title>: <image_url>
- Icons:
  Icons:
  - <name>: <icon_url>

Asset URL rules (strict):
- Include Images/Icons only when you can provide real sample URLs that are publicly accessible now.
- Use direct asset URLs whenever possible (image file URLs for image/icon entries).
- If you are not confident a real asset URL exists, omit that specific image/icon entry instead of guessing.
- Do not output broken, fake, or placeholder asset URLs.

Source rules:
- If verifiable factual claims are included, add 1-3 stable links under Sources.
- Omit Sources for pure general guidance.

Output protocol (must follow):
- Return only valid JSON (no markdown).
- Return a JSON array of objects with fields: query_id (string), response_text (string).

Queries (JSON array):
{queries_json}