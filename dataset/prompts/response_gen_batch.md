# response_gen_batch_v1

You are generating responses for multiple user queries.

Return ONLY valid JSON (no markdown). Output a JSON array of objects with fields:
- query_id (string)
- response_text (string)
- If the response references assets (images, files, documents), include direct, publicly accessible URLs in response_text.

Queries (JSON array):
{queries_json}
