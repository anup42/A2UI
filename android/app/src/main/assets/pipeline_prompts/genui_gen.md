# genui_gen_v10_flatspec_hardcut

You are a GenUICraft generator. Convert the response text into a flat-spec JSON object.

Response:
{response_text}

## Output contract
- Return ONLY one JSON object. No prose, no markdown, no comments.
- Top-level shape:
  `{"root":"<id>","state":{...},"elements":{...}}`
- `root` must reference an existing key in `elements`.
- Every element object must contain:
  - `type` (component name from CATALOG)
  - `props` (object, can be empty)
  - `children` (array of element ids, can be empty)
- Every id in `children` must exist in `elements`.
- Use only component types listed in CATALOG.
- Supported actions: `openUrl`, `setState`, `pushState`, `removeState`, `validateForm`.
- Interaction bindings must use element-level `on` objects.
- Do NOT use legacy `props.action` or `functionCall`.

## Hard rule: no legacy message array
- Correct output is a single flat object with `root/state/elements`.

## Positive and negative format examples

Valid (flat-spec):
```json
{
  "root": "main",
  "state": {},
  "elements": {
    "main": { "type": "Column", "props": {}, "children": ["title"] },
    "title": { "type": "Text", "props": { "text": "Example", "variant": "h2" }, "children": [] }
  }
}
```

Invalid (legacy, DO NOT OUTPUT):
```json
[
  { "version": "v0.9", "createSurface": { "surfaceId": "surface_live", "catalogId": "..." } },
  { "version": "v0.9", "updateComponents": { "surfaceId": "surface_live", "components": [] } }
]
```

## Asset URL policy
- If no Assets mapping is provided, preserve media URLs exactly as given.
- Never invent placeholder paths like `/image.jpg` or `/asset/foo.png`.

## Dynamic fields
Allowed dynamic value expressions in props:
- `{ "$item": "fieldName" }`
- `{ "$state": "/path/to/value" }`
- `{ "$bindItem": "fieldName" }`
- `{ "$index": true }`
- `{ "$cond": <condition>, "$then": <value>, "$else": <value> }`
- `{ "$template": "Hello ${/user/name}!" }`
- `{ "$computed": "<name>", "args": { ... } }`

## Repeat and visibility
- Use `repeat` (top-level field on an element) for list templating.
- Strict semantics: repeat renders the parent element once, and repeats the parent element's `children` for each item.
- Use `repeat` with:
  - `"repeat": { "statePath": "/hotels", "key": "id" }`
- Use `visible` (top-level field on an element) for conditional rendering:
  - `{"$state":"/tab","eq":"hotels"}`
  - `{"$and":[...]}`
  - `{"$or":[...]}`

## Watch bindings
- You may use element-level `watch` to react to state changes:
  - `"watch": { "/form/submit": { "action": "validateForm", "params": { "statePath": "/formValidation" } } }`

## Layout rules
- Build structured app-like UI, not one giant text block.
- Use headings and sections for medium/long responses.
- Keep title, media, body, and CTA together inside each card.
- Convert `Media: Image=<url>` into `Image` elements.
- Convert `Media: Icon=<url>` into `Icon` elements.
- If response has any media URL, output must include at least one `Image` element (or `$item` image binding).
- Convert links/CTAs to `Button` with `openUrl`.
- Keep `Tags: A | B | C` lines: emit chips via `Text` with `variant: "chip"`.
- For hotel/restaurant/place result sets, prefer one card template with `repeat` over a state array.
- Preserve all numbers, dates, times, units, and currency exactly.

## CATALOG

Layout:
- `Column`
- `Row` with optional `justify`
- `List`
- `Card`

Content:
- `Text` props: `text` (required), `variant` optional (`h1|h2|h3|body|caption|chip|label`)
- `Image` props: `url` (required), `fit` optional (`cover|contain`)
- `Icon` props: `name` (required, icon URL)
- `Video` props: `url` (required)
- `AudioPlayer` props: `url` (required), `description` optional
- `Divider`

Interactive:
- `Button` props: `label`, `variant` optional (`primary|borderless`)
- Use element-level `on.press` for button actions.
- `Tabs` props: `tabs` (required list of `{ "title": "...", "child": "<id>" }`)
- `Modal` props: `trigger` optional, `content` optional

Form:
- `TextField` props: `label` (required), `value` optional
- `CheckBox` props: `label` (required), `value` (required boolean)
- `ChoicePicker` props: `label` (required), `options` (required), `value` (required array)
- `Slider` props: `min`, `max`, `value` (required numbers), `label` optional
- `DateTimeInput` props: `value` (required), `label` optional, `enableDate` optional, `enableTime` optional

## Canonical domain example (flight cards with repeat)
```json
{
  "root": "main",
  "state": {
    "flights": [
      { "id": "f1", "airline": "IndiGo", "time": "05:50 - 08:20", "fare": "INR 5,488", "url": "https://example.com/f1" },
      { "id": "f2", "airline": "Akasa Air", "time": "07:00 - 09:25", "fare": "INR 5,799", "url": "https://example.com/f2" }
    ]
  },
  "elements": {
    "main": { "type": "Column", "props": {}, "children": ["title", "flight_list"] },
    "title": { "type": "Text", "props": { "text": "Flights BLR to LKO", "variant": "h2" }, "children": [] },
    "flight_list": {
      "type": "Column",
      "props": {},
      "repeat": { "statePath": "/flights", "key": "id" },
      "children": ["flight_card"]
    },
    "flight_card": { "type": "Card", "props": {}, "children": ["airline", "time", "fare", "cta"] },
    "airline": { "type": "Text", "props": { "text": { "$item": "airline" }, "variant": "h3" }, "children": [] },
    "time": { "type": "Text", "props": { "text": { "$item": "time" } }, "children": [] },
    "fare": { "type": "Text", "props": { "text": { "$item": "fare" } }, "children": [] },
    "cta": {
      "type": "Button",
      "props": {
        "label": "Open"
      },
      "on": {
        "press": { "action": "openUrl", "params": { "url": { "$item": "url" } } }
      },
      "children": []
    }
  }
}
```
