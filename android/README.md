# GenUI Craft Renderer

This Android app renders GenUI payloads (`genui.json` and `genui.jsonl`) into HTML and visualizes them in a `WebView`.

## Features

- First screen contains only source selection actions: `Select GenUI File` and `Open Sample Dataset`.
- Loads JSON/JSONL from Android file picker (no path input field required).
- Loads bundled JSONL sample: `app/src/main/assets/sample_genui.jsonl`.
- Second screen shows all parsed JSON/JSONL rows in an improved selectable list.
- Third screen renders the selected item in either native Compose or `WebView` mode (toggle on first screen).
- Bundles sample run assets/icons in `app/src/main/assets/r_*` so `/assets/...` paths render inside WebView.
- Converts GenUI component trees (`v0.9` and basic `v0.8`) to HTML/CSS
- Renders common components: `Column`, `Row`, `List`, `Text`, `Image`, `Icon`, `Button`, `Divider`, `Card`, `Tabs`
- Settings supports two inference backends for the GenUI assistant flow:
  - `Gemini API` (default, with selectable Gemini model)
  - `Local Server` (configure server URL + Hugging Face model path)
- Applies device configuration adaptively:
  - Dynamic color + dark/light mode from system theme
  - Font scale for accessibility (`WebView` `textZoom` + Compose large-text layout handling)
  - Screen width/height classes for responsive paddings and table sizing
  - Orientation and locale/layout direction (`LTR`/`RTL`) for rendered HTML

## Build / Test

```powershell
$env:ANDROID_HOME='C:\Users\anupk\AppData\Local\Android\Sdk'
$env:ANDROID_SDK_ROOT='C:\Users\anupk\AppData\Local\Android\Sdk'
.\gradlew.bat testDebugUnitTest assembleDebug
```

## Output

- APK: `app/build/outputs/apk/debug/app-debug.apk`
- Unit test report: `app/build/reports/tests/testDebugUnitTest/index.html`
