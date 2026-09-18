# Literal text in the GenUICraft profile

Ordinary source strings remain unchanged. A string that could be interpreted as a state path or
template, such as `${HOME}`, `$/price`, `$state.foo`, or `{$index}`, is prefixed with the reserved
sequence `U+001E` followed by `GenUICraftLiteral:v1:`. The source value follows that prefix without
rewriting, base64 encoding, or evaluation. JSON and Express encode the control character as
`\u001e` when serializing it.

This is **GenUICraft catalog/profile escaping**, not a new A2UI v0.9 object type. The wire value
remains a plain JSON string. Another renderer must understand this profile's escape to display
these exceptional literal values. The legacy `{literalString: ...}` object is deliberately not
emitted: the pinned v0.9 `dynamicString` schema rejects it.

The renderer removes exactly one prefix and retains an opaque runtime text value until display.
Repeated property/cell resolution cannot evaluate its contents. If the source itself starts with
the reserved prefix, conversion adds another prefix; one decoding step recovers the original
source exactly. Existing strings without this prefix retain their dynamic renderer semantics.
Source bindings and the conversion-only `LiteralTextCodec.protectVisibleContent` helper apply
the same escape. The latter visits content properties only, leaving layout, component references,
explicit dynamic objects, and action URLs unchanged. Call it once on raw model output, before
compilation; do not reapply it to a previously protected document.

The Express lexer also accepts JSON Unicode escapes and document delimiters inside quoted strings.
Actual nested/duplicate document delimiters outside strings and comments remain invalid.

Validation includes prefix collisions, repeated resolution, text/list/table rendering data paths,
source-bound code and plain text, literal document delimiters, Unicode controls, invalid escapes,
preserved ordinary dynamic expressions, and Express/wire round trips. The 50-case test corpus is
unchanged. A representative escaped wire document was validated with the repository's pinned JSON
schema; the equivalent legacy `literalString` object was rejected. Schema SHA256:
`90137067429316d2ecba4997c43198bccf7241a9bbeeb34ef8c4ae52f6e14881`.
