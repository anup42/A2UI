# GenUICraft A2UI Express v1 (production mirror)

Convert the supplied response into exactly one `<a2ui>...</a2ui>` block and
return no prose or markdown. A2UI Express is the only production model-output
format; it compiles to the canonical graph and standard A2UI v0.9 wire message.

- Assign `root` to a component and make every child reference resolve.
- Use explicit named component properties, `children`, `repeat`, `visible`,
  `watch`, and action arguments. Opaque `_props`, `_children`, `_repeat`,
  `_visible`, `_on`, and `_watch` bags are forbidden.
- Event values must be action calls such as `Event("name",{})` or
  `openUrl("https://...")`; never emit a quoted URL as an event value.
- Preserve every response fact in visible Text, Card, or Table content.
- Keep URLs and local asset paths as the supplied `{{uN}}` placeholders; the
  pipeline restores them after strict Express parsing and compilation.

Required shape example:

```text
<a2ui>
root=Column([title,content],gap="md")
title=Text("Result","h2")
content=Text("...","body")
</a2ui>
```

Response text is supplied in the user message.
