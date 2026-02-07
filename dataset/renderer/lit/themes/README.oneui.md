# OneUI Theme Overlay for Stage4 Renderer

This folder contains OneUI-inspired renderer overlays kept separate from upstream
vendor code so updates are easy to maintain.

- `oneui.tokens.css`: shared color/radius/shadow tokens.
- `oneui.overrides.css`: page/surface-level look-and-feel overrides.

Template wiring is in `dataset/renderer/lit/template.html` and uses:

- `__ASSET_BASE__themes/oneui.tokens.css`
- `__ASSET_BASE__themes/oneui.overrides.css`

The template also applies theme-level `additionalStyles` so component internals
(inside shadow DOM) pick up OneUI-like spacing, typography, and button/card styles.
