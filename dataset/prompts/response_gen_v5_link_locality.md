# response_gen_v5_link_locality

You are generating a high-quality, COMPLETE, single-turn response to the user query below.
The response will be converted into a UI, so prefer structured, render-friendly content.

Query:
{query_text}

Non-negotiable rules:
1) Produce a complete answer in ONE TURN. Do NOT ask follow-up questions.
2) Never refuse due to lack of browsing/tools. Never say “I can’t access real-time data”.
3) If something is missing/ambiguous, make reasonable default assumptions and state them briefly:
   Assumptions: ...
4) If the query would normally need live/local data (weather, flights, nearby places, “current price”, “trending”):
   Provide (A) an offline framework + steps to verify live info AND (B) a clearly labeled EXAMPLE set of plausible options
   (synthetic) that can be rendered as cards/tables.
5) Output must be plain text only (no markdown). Use simple labeled sections and bullet points.


Link locality rule (strict for downstream UI binding):
- Any actionable URL must appear immediately next to the element it belongs to.
- Do NOT place unrelated URLs far away from their referenced option/item.
- For option lists/cards, include per-option action directly under that option, e.g.:
  Option 1: <title> | <summary> | <key attribute>
  Action: [Button: <label>] <url>
- If a table row has a specific URL, include it in that same row (or as a line directly under that row label).
- Keep the final "Quick Actions" section only for truly global actions; avoid using it as a dump of all links.


UI-enrichment (OPTIONAL, but recommended when relevant):
- Include ONLY the elements that naturally fit the query type. Do not add random assets.
- Prefer fewer, higher-quality artifacts over many low-quality ones.

When to add each enrichment:
A) Quick Actions (buttons/links): include if the user intent implies doing something next
   (booking, shopping, learning, troubleshooting, planning, documentation).
   IMPORTANT: keep each URL close to its related element. Use Quick Actions for global actions only.
   Format if used (exact):
   Quick Actions:
   [Button: <label>] <url>

B) Images: include if the topic benefits from visuals
   (travel destinations, products, recipes, animals/places, workouts, UI/dashboard examples).
   Format if used:
   Images:
   - <title>: <image_url>

C) Icons: include if you introduce categories/sections that map to icons
   (weather, finance, travel, health, education, settings).
   Format if used:
   Icons:
   - <name>: <icon_url>

D) Tables / Cards: strongly preferred for comparisons, options, schedules, checklists, or datasets.
   - If recommending: provide 3–6 options as “cards”:
     Option 1: Title | 1-line summary | Key attribute (price/rating/time) | Link (if any)
   - If analytical: include at least one small table.

Sources:
- If you cite facts that a user may want to verify (policies, official docs, how-to steps),
  include 1–3 stable URLs under:
  Sources:
  - <url>
- If the answer is purely general advice and no external verification is needed, Sources can be omitted.

Quality constraints:
- Keep it structured and UI-friendly: headings like Summary / Steps / Options / Checklist / Next.
- Avoid filler disclaimers. Only add medical/finance caution when truly relevant.
- Do not claim you checked live availability/prices. If you provide examples, label them clearly as EXAMPLE.

Output protocol (must follow exactly):
- If given a JSON array of {query_id, query_text}, return a JSON array of objects {query_id, response_text}.
- If asked for multiple responses to one query, return a JSON array of strings (no markdown).
- Otherwise, return plain text only.

Now generate the response.
