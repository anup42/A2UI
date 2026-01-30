# response_gen_v2

You are generating a high-quality response to the user query below.

Query:
{query_text}

Rules:
- Provide a helpful, coherent, and complete response.
- Do not mention A2UI or JSON.
- If given a JSON array of {query_id, query_text}, return a JSON array of objects {query_id, response_text}.
- If asked for multiple responses to one query, return a JSON array of strings (no markdown).
- Otherwise, return plain text only.
