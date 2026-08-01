# Blinded GenUI Anchored Criterion Judge task procedure v3

Work only from the neutral batch manifest and files explicitly listed by it.
Do not inspect the repository, host benchmark, sealed maps, other batches,
conversation history, FlatSpec JSON, metric values, previous judgments, repeat
identity, or generator identity.

For every block:

1. Read the frozen GACJ v3 judge instructions and packet rubric.
2. Record the exact isolated task ID and model identifier.
3. Inspect all screenshot-only packets and their original native images.
4. Return criterion levels and defect severities for dimensions 3-10 only.
   Validate, count-check, and durably seal the screenshot result before any
   source packet becomes available.
5. After the host releases the source-conditioned manifest, inspect the source,
   expected contract, and same images.
6. Return criterion levels and defect severities for dimensions 1-2 only.
7. Stop. Do not calculate dimension scores, R, U, J, policy caps,
   correlations, calibration, or pairwise preferences.

Paired repeat occurrences must be judged in different fresh tasks. A model
identifier change requires a new protocol version. Screenshot timestamps must
precede source-conditioned timestamps for each packet.
