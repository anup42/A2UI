# GenUI Craft Renderer

This Android app renders GenUI flat-spec IR payloads (`genui.json` and `genui.jsonl`) into native Jetpack Compose UI.
A legacy HTML/`WebView` path still exists for pre-rendered HTML sample datasets, but it does not support the flat-spec
IR and is not the primary renderer.

## Features

- First screen contains only source selection actions: `Select GenUI File` and `Open Sample Dataset`.
- Loads JSON/JSONL from Android file picker (no path input field required).
- Loads bundled JSONL sample: `app/src/main/assets/sample_genui.jsonl`.
- Second screen shows all parsed JSON/JSONL rows in an improved selectable list.
- Third screen renders the selected item in either native Compose or `WebView` mode (toggle on first screen).
- Bundles sample run assets/icons in `app/src/main/assets/r_*` so `/assets/...` paths render inside WebView.
- Renders the 25 flat-spec element types natively in Compose: `Stack` (plus `Row`/`Column` aliases), `List`, `Card`,
  `Table`, `Chart`, `Formula`, `CodeBlock`, `ConsoleLog`, `EmailPreview`, `Text`, `Image`, `Icon`, `Video`,
  `AudioPlayer`, `Divider`, `Button`, `Tabs`, `Modal`, `TextField`, `CheckBox`, `ChoicePicker`, `Slider`,
  `DateTimeInput`. The authoritative list is `FlatSpecContract.allowedTypes`; dispatch lives in
  `FlatSpecRenderer.RenderByType`.
- Legacy `v0.9`/basic `v0.8` component trees are converted to HTML/CSS only on the `WebView` path
- Settings supports two inference backends for the GenUI assistant flow:
  - `Gemini API` (default, with selectable Gemini model)
  - `Local Server` (configure server URL + Hugging Face model path)
- Applies device configuration adaptively:
  - Dynamic color + dark/light mode from system theme
  - Font scale for accessibility (`WebView` `textZoom` + Compose large-text layout handling)
  - Screen width/height classes for responsive paddings and table sizing
  - Orientation and locale/layout direction (`LTR`/`RTL`) for rendered HTML

## Renderer API

Two entry points, both `FlatSpecContent`:

- `FlatSpecContent(spec, resolveAssetUrl, computedFunctions, ...)` — legacy, unchanged behaviour.
- `FlatSpecContent(spec, host, stateHolder, computedFunctions, onEvent, ...)` — host-agnostic. `FlatRendererHost`
  supplies `openUrl`, `resolveAssetUrl`, `imageLoader` and `onDiagnostic`, so the renderer no longer launches
  intents or builds image loaders itself. `FlatSpecStateHolder` owns render state, so an incremental surface
  update can preserve user input; it also exposes `applyDataModelUpdate` for upsert/delete. `onEvent` reports
  actions, state changes, navigation and diagnostics.

Element dispatch is a registry (`FlatRenderRegistry.kt`): `FLAT_ELEMENT_RENDERERS` keyed by canonical type,
with alias spellings held as data in `FLAT_TYPE_ALIASES`.

### Known gaps

- `FlatSpecRenderer.kt` is still large; domain card renderers have been extracted to
  `renderer/flat/domain/`, but table layout, source sections and several card families remain in it.
- Table presentation is still partly inferred from English column headers, so non-English column labels can
  route to a plain table instead of a domain card layout.
- Inline video/audio playback is intentionally absent — it would make render-capture screenshots
  non-deterministic. Media elements show a poster and open externally.
- The renderer is not yet a separate Gradle module, so it cannot be consumed without this app module.
- Ingest paths do not run `FlatSpecContract` canonicalization; doing so changes table routing on ~56% of the
  golden corpus and needs a deliberate metric re-baseline.

## Build / Test

```powershell
$env:ANDROID_HOME='C:\Users\anupk\AppData\Local\Android\Sdk'
$env:ANDROID_SDK_ROOT='C:\Users\anupk\AppData\Local\Android\Sdk'
.\gradlew.bat testDebugUnitTest assembleDebug
```

CI runs the same unit tests plus lint and a no-secrets assemble on every PR touching `android/`
(`.github/workflows/android_build_and_test.yml`).

### If Gradle fails with "Unable to establish loopback connection"

That error is not about networking. JDK 17+ on Windows backs the NIO selector wakeup pipe with an AF_UNIX socket, and
on some machines `Selector.open()` fails with `SocketException: Invalid argument: connect`. Android Studio is
unaffected; only shell builds fail. Point `JAVA_HOME` at a JDK whose selector works — locally that is
`corretto-18.0.2`:

```powershell
$env:JAVA_HOME="$env:USERPROFILE\.jdks\corretto-18.0.2"
```

To test a candidate JDK, run `Selector.open()` in a scratch file; if it throws, that JDK cannot run Gradle here.

## Gemini / Vertex Express key setup

- Release and `judgeCapture` APKs keep credential-backed `BuildConfig` fields empty.
- A local debug/demo APK can embed credentials resolved from ignored `local.properties`,
  `.env`, Gradle properties, or environment variables. Do not distribute that APK.
- Runtime key file location on device:
  - `/sdcard/Android/data/com.samsung.genuicraft/files/genuicraft_keys.env`
- Required entries:
  - Vertex Express: `GEMINI_VERTEX_EXPRESS_API_KEY` or `VERTEX_EXPRESS_API_KEY`
  - Direct Gemini Stage 2: `GEMINI_STAGE2_API_KEY` (or `GEMINI_RESPONSE_API_KEY` / `GEMINI_API_KEY`)
  - Direct Gemini Stage 3: `GEMINI_STAGE3_API_KEY` (or `GEMINI_IR_API_KEY` / `GEMINI_API_KEY_2`)

Push the ignored Android `local.properties` values to a specific connected device:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\push_gemini_keys.ps1 -Serial <adb-serial>
```

The app imports the external file into private storage and deletes the staged copy.
Runtime values take precedence over debug `BuildConfig` defaults.

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
