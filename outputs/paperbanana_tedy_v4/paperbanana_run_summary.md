# PaperBanana TEDY Figure Run Summary

Generated with PaperBanana full pipeline (`demo_full`) using Retriever -> Planner -> Stylist -> Visualizer -> Critic.

## Images
- `tedy_architecture_paperbanana.png`
- `tedy_spec_adaptive_paperbanana.png`

## Run configuration
- Main VLM model: `gemini-3.1-pro-preview`
- Image generation model: `gemini-3-pro-image-preview`
- Auth route: Vertex AI Express
- Candidate count: 1 per figure
- Retrieval setting: none (few-shot retrieval skipped to avoid upstream retrieval/cache instability)
- Requested critic rounds: 4

## Observed critiques
- Architecture figure: observed critic rounds 0 and 1, then stopped early because critic requested no further changes.
- Spec/adaptive figure: observed critic rounds 0, 1, 2, and 3.

## Notes
- Managed PaperBanana cache was patched locally to route `gemini-*` models to Gemini/Vertex before OpenRouter when both clients are configured.
- Some transient Vertex 429 retries occurred, but both final image generations completed successfully.
