# stage25_domain_classifier_v1

Classify this response for Stage 3 prompt routing.

Return ONLY one JSON object:
{
  "primary_domain": "one allowed domain",
  "secondary_domains": ["up to two allowed domains"],
  "confidence": 0.0,
  "evidence": ["short phrases that justify the decision"],
  "required_components": ["Table", "Button"],
  "presentation_hints": {
    "tableDomain": "weather|flight|booking|playlist|schedule|status|formula|comparison|generic",
    "preferredPresentation": "cards|table"
  }
}

Allowed domains:
{allowed_domains}

Classification rules:
- Pick the UI pattern needed for rendering, not just the business topic.
- Use open_domain only when none of the predefined domains fit.
- Use generic for ordinary informational content with no specialized UI needs.
- Prefer the most specific primary domain. Example: a restaurant list is restaurant, not generic; a vacation day plan is travel_itinerary, not schedule_planning.
- Secondary domains should be used only when they change Stage 3 rendering guidance.

Query: {query_text}
Intent: {intent}
Tags: {tags}

Response:
{response_text}
