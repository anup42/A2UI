# Packaged-AAR synthetic review

`ps_synthetic_scaffold_v11`: 3/3 first-attempt successes, zero repairs/fallbacks. Each recorded call used GPU+MTP with thinking enabled and exactly one scaffold supplied by the AAR. The independent semantic graph audit preserves all three scaffolds except for valid table selections.

Manual screenshot review:

- `SYN-001/screen.png`: heading, both sensor rows with signed temperatures, Kotlin code and language title, divider, and the final example-data caveat/citation are visible and legible.
- `SYN-003/screen_scrolled.png`: comparison columns, units/numbers, both list items, Python code with literal `${HOME}`, divider, limitations heading, citation, and final follow-up offer are visible and intact.

SYN-002's source/literal preservation is established by conversion/content validation and graph auditing; its pixels were not separately reviewed here. These synthetic cases exercise code/divider/literal shapes missing from Bixby50. They are not additional Bixby50 samples or evidence of held-out performance.
