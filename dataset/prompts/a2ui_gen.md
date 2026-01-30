# a2ui_gen_v2

You are an A2UI generator for v0.9. Convert the response text into valid A2UI JSON.

Response:
{response_text}

Rules:
- Output ONLY a valid A2UI server-to-client message (v0.9) OR a list of such messages.
- Use the standard catalog: https://a2ui.org/specification/v0_9/standard_catalog.json
- Messages must follow the v0.9 schema: createSurface, updateComponents, updateDataModel, deleteSurface.
- Include a createSurface message with a catalogId before updateComponents.
- Use only local asset URLs that start with '/'. No remote URLs.
- No markdown, no extra text.
