# Development observations

The initial screen uses v10 with the existing JSON user-prompt serialization and frozen runtime settings. These observations are hypotheses from actual raw outputs, not changes to production acceptance.

- Fresh baseline reproduces all three historical first-attempt failures: shortened closing tag (030), skipped/shifted element after a two-binding table (032), malformed opening/root (037).
- The no-example compact prompt failed all four cases. Raw outputs copied JSON root escape sequences and introduced XML-like or per-block document syntax.
- The example-first prompt passed 003/030 and failed 032/037. The latter failures include JSON syntax inside the Express document and a changed source-binding token. Both successful tables exactly preserve source cells and select suitable card/table presentations.
- The grammar candidate's first case emitted the rule notation as code and copied escaped root punctuation. This suggests abstract rule descriptions are insufficient for this model in the current input format.
- The input currently uses Gson HTML escaping, so an equals sign inside the root JSON string is represented as \u003d. The mapping and targeted-baseline candidates explicitly explain decoding. No serialization, validator, model, thinking, GPU, or MTP change has been made for this screen.
- All of these results are from a deliberately difficult development subset. They do not estimate whole-corpus accuracy, and thermal drift prevents treating the timings as controlled prompt-only speed measurements.
