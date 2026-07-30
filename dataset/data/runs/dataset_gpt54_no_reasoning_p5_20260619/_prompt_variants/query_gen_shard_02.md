# query_gen_gemma_v3_diverse_openings

You are generating diverse, SELF-CONTAINED user queries for a UI generation dataset.

Intent category: {intent}
Target count: {k}

Core goal:
Generate queries that a general LLM can answer COMPLETELY in ONE TURN without:
- asking follow-up questions,
- requiring the user's location/account access,
- requiring real-time browsing/tools,
- refusing due to â€œI canâ€™t access live dataâ€.

Hard rules (must follow):
1) Every query must be answerable offline and be â€œsingle-shot completeâ€.
2) Do NOT use ambiguous placeholders like: today, tomorrow, next week, next Tuesday, near me, current, trending, latest.
   - If time is needed, include an explicit date range (YYYY-MM-DD) and timezone if relevant.
   - If location is needed, include an explicit city + country (optionally neighborhood).
3) If the intent normally depends on live data (weather, flights, stocks, nearby places, streaming availability, â€œtrendingâ€):
   - Either (A) include a small inline data snippet (JSON/CSV/list) for the assistant to use, OR
   - (B) reframe the query to an offline task: explanation, planning, comparison framework, checklist, or hypothetical example generation.
4) Include all necessary constraints inside the query_text: budget, preferences, units, counts, and any required inputs.
5) Avoid duplicates and near-duplicates across the batch.
6) Do not mention GenUICraft, UI-IR, A2UI, TOON, JSON/schema, datasets, prompts, or â€œas an AIâ€.
7) Cover multiple real-world domains (finance, travel, health, shopping, education, productivity, legal, housing, career, local services).
8) Vary length (short/medium/long) and difficulty (easy/medium/hard).

Gemma diversity rules (strict):
- Do not start queries with repeated phrases.
- Vary openings, voice, constraints, and specificity across the batch.
- Mix direct commands, polite requests, first-person scenarios, professional contexts, and short natural user phrasing.
- Avoid templated starts such as "Create a", "Generate a", "Can you", "I need", or "Help me" appearing repeatedly in the same batch.
- The first 5-8 words of each query should usually differ from the others for the same intent.

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
    OR ask for â€œhow to check + what to watch forâ€ (not the live price).
- Entertainment (movies/music/streaming):
  - Avoid â€œtrending right nowâ€. Use â€œrecommend for these preferencesâ€ OR provide a list to rank.
- Data visualization / dashboard:
  - Include a small dataset inline (CSV/JSON) and ask for specific charts + insights to produce.
- Planning:
  - Include constraints (dates, duration, budget, interests, pace, companions, constraints).
- Productivity / scheduling / reminders:
  - Include exact date/time/timezone, duration, participants, agenda, and constraints.
- Finance:
  - Prefer budgeting, allocation, or risk-analysis tasks using inline numbers/tables.
  - Avoid live-price lookups unless the query provides the needed price data inline.
- Healthcare:
  - Keep it non-emergency and informational.
  - Include age/context/symptoms/goals and constraints (allergies, medications, budget, time).
- Legal:
  - Include jurisdiction, scenario facts, dates, and desired outcome.
  - Prefer checklist/comparison/document-prep framing over definitive legal verdicts.
- Real estate:
  - Include city+country, budget, property type, size/bedroom constraints, commute/school priorities, and timeline.
- Career:
  - Include role, experience level, skills, target location, compensation target, and transition constraints.
- Food and dining:
  - Include cuisine preferences, dietary restrictions, party size, budget, and timing.
- Home services:
  - Include issue details, property type/size, urgency, budget, and service constraints.
- Automotive:
  - Include vehicle model/year/mileage, symptoms/usage pattern, budget, and decision constraints.

Output format:
Return ONLY valid JSON (no markdown).
Return a JSON array of objects with fields:
- query_text (string)
- difficulty ("easy"|"medium"|"hard")
- tags (array of short tags)

Now generate the JSON array.

## Dataset shard diversity directive
You are generating shard 2 of 5 for the same overall dataset.
Make this shard intentionally different from the other shards:
- Prefer different entities, locations, brands, constraints, time ranges, roles, and user contexts.
- Avoid generic or repeated wording.
- Keep the same JSON output contract as above.
- Do not mention shard numbers in generated queries.
