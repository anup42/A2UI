# query_gen_v2

You are generating diverse, SELF-CONTAINED user queries for a UI generation dataset.

Intent category: {intent}
Target count: {k}

Core goal:
Generate queries that a general LLM can answer COMPLETELY in ONE TURN without:
- asking follow-up questions,
- requiring the user's location/account access,
- requiring real-time browsing/tools,
- refusing due to “I can’t access live data”.

Hard rules (must follow):
1) Every query must be answerable offline and be “single-shot complete”.
2) Do NOT use ambiguous placeholders like: today, tomorrow, next week, next Tuesday, near me, current, trending, latest.
   - If time is needed, include an explicit date range (YYYY-MM-DD) and timezone if relevant.
   - If location is needed, include an explicit city + country (optionally neighborhood).
3) If the intent normally depends on live data (weather, flights, stocks, nearby places, streaming availability, “trending”):
   - Either (A) include a small inline data snippet (JSON/CSV/list) for the assistant to use, OR
   - (B) reframe the query to an offline task: explanation, planning, comparison framework, checklist, or hypothetical example generation.
4) Include all necessary constraints inside the query_text: budget, preferences, units, counts, and any required inputs.
5) Avoid duplicates and near-duplicates across the batch.
6) Do not mention GenUICraft, UI-IR, A2UI, TOON, JSON/schema, datasets, prompts, or “as an AI”.
7) Cover multiple real-world domains (finance, travel, health, shopping, education, productivity, local services).
8) Vary length (short/medium/long) and difficulty (easy/medium/hard).

Intent-specific shaping (apply whichever matches {intent}):
- Weather:
  - Must include city+country and explicit date range OR include a forecast snippet to summarize.
- Local services / restaurants / pharmacies:
  - Must include city+country and criteria (cuisine, budget, distance/time, open hours),
    OR provide a candidate list to rank/compare.
- Flights / hotels / booking:
  - Must include origin, destination, dates, passengers, cabin/room needs, budget, and constraints,
    OR provide a list of options to choose from.
- Stocks / exchange rates:
  - Must provide a small price/exchange-rate table to analyze,
    OR ask for “how to check + what to watch for” (not the live price).
- Entertainment (movies/music/streaming):
  - Avoid “trending right now”. Use “recommend for these preferences” OR provide a list to rank.
- Data visualization / dashboard:
  - Include a small dataset inline (CSV/JSON) and ask for specific charts + insights to produce.
- Planning:
  - Include constraints (dates, duration, budget, interests, pace, companions, constraints).
- Productivity / scheduling / reminders:
  - Include exact date/time/timezone, duration, participants, agenda, and constraints.

Output format:
Return ONLY valid JSON (no markdown).
Return a JSON array of objects with fields:
- query_text (string)
- difficulty ("easy"|"medium"|"hard")
- tags (array of short tags)

Now generate the JSON array.