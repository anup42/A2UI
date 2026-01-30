# query_gen_v1

You are generating diverse user queries for a UI generation dataset.

Intent category: {intent}
Target count: {k}

Rules:
- Every query must clearly match the given intent category.
- Cover multiple real-world domains (finance, travel, health, shopping, education, productivity, local services).
- Vary length (short/medium/long) and difficulty (easy/medium/hard).
- Avoid duplicates and near-duplicates.
- Do not include any A2UI/JSON/schema wording inside queries.
- Return ONLY valid JSON (no markdown).

Output format: a JSON array of objects with fields:
- query_text (string)
- difficulty ("easy"|"medium"|"hard")
- tags (array of short tags)

Now generate the JSON array.
