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

## Gemini Key Setup (No Keys In APK)

- This app does not embed Gemini keys in `BuildConfig` or APK.
- Runtime key file location on device:
  - `/sdcard/Android/data/com.samsung.genuicraft/files/genuicraft_keys.env`
- Required entries:
  - Stage 2: `GEMINI_STAGE2_API_KEY` (or `GEMINI_RESPONSE_API_KEY` / `GEMINI_API_KEY`)
  - Stage 3: `GEMINI_STAGE3_API_KEY` (or `GEMINI_IR_API_KEY` / `GEMINI_API_KEY_2`)

Push keys from `../dataset/.env`:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\push_gemini_keys.ps1
```

## UI Automation Method (Agent Reuse)

- Canonical testing method is documented in:
  - `android/tools/TESTING_METHODS.md`
- Reusable 10-scenario audit script:
  - `android/tools/run_quickactions_audit.ps1`

Run from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\android\tools\run_quickactions_audit.ps1
```

## Output

- APK: `app/build/outputs/apk/debug/app-debug.apk`
- Unit test report: `app/build/reports/tests/testDebugUnitTest/index.html`
