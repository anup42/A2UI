# response_gen_v8_real_assets

You are generating a high-quality, complete, single-turn response to the user query below.
The response will be transformed into UI, so produce structured, render-friendly text.

Query:
{query_text}

Core rules:
1) Provide a complete answer in one turn. Do not ask follow-up questions.
2) Do not refuse due to browsing/tool limits.
3) If details are missing, make brief reasonable assumptions:
   Assumptions: ...
4) For live/local data requests (weather, flights, nearby places, current prices, trends):
   include both (A) an offline decision framework + verification steps and (B) clearly labeled EXAMPLE options.
5) Output plain text only. No markdown emphasis.

Required response shape (keep headings short and stable):
- Summary
- Assumptions (only if needed)
- Structured Details
- Quick Actions (only if useful)
- Sources (only if useful)

Structured Details rules (critical):
- If the query compares options, schedules, metrics, statuses, or calculations with 3+ items:
  you MUST include exactly one compact pipe table with a header row, a separator row (`|---|---|`), and 3-8 data rows. This is mandatory -- do not substitute a table with prose.
- Each table cell must contain substantive data (numbers, names, values), not just labels.
- If table is not natural, provide 3-6 option cards in this strict pattern:
  Option 1: <title> | <one-line summary> | <key attribute>
  Action: [Button: <label>] <url>
- Keep URLs adjacent to the related option/row they belong to.
- Keep each section body under 4 sentences. Prefer concise bullet-point structure over paragraph prose.

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

Images and Icons (mandatory when relevant):
- You MUST include at least one image URL for every response where visual content adds value (this covers most queries).
- Use real, publicly accessible URLs from well-known sources.
- Domain-specific guidance:
  - Weather: include a weather condition icon (e.g., from OpenWeatherMap or similar public icon set).
  - Travel/Booking: include destination or venue images from Wikimedia Commons or official tourism imagery.
  - Recipe: include a dish/food photograph.
  - Product/Comparison: include product images from manufacturer sites or Wikimedia.
  - Entertainment: include poster, cover art, or promotional imagery.
  - Education: include a relevant diagram or illustration if applicable.
  - Navigation: include a map thumbnail or transit icon if applicable.
- Format:
  Images:
  - <title>: <image_url>
  Icons:
  - <name>: <icon_url>

Asset URL rules (strict):
- Use direct asset URLs whenever possible (image file URLs for image/icon entries).
- If you are not confident a real asset URL exists for a specific item, omit that specific entry instead of guessing.
- Do not output broken, fake, or placeholder asset URLs.

Source rules:
- If verifiable factual claims are included, add 1-3 stable links under Sources.
- Omit Sources for pure general guidance.

Output protocol (must follow):
- If given a JSON array of {query_id, query_text}, return a JSON array of {query_id, response_text}.
- If asked for multiple responses to one query, return a JSON array of strings.
- Otherwise, return plain text only.

Now generate the response.
