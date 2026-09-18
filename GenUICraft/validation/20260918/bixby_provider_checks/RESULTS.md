# Isolated Bixby provider-holder validation

Passed on 2026-09-18 at approximately 03:49 IST: **2 tests, 0 failures, 0 errors, 0 skipped**. Gradle completed successfully in 1 minute 7 seconds; the JUnit suite ran for 1.858 seconds.

This harness compiles the actual Bixby `GenUiProviderHolder.kt` and SDK `Api.kt` and runs the actual Bixby `GenUiProviderHolderTest.kt`, all copied verbatim. There are no replacement source declarations or stubs. MockK is used by the original tests to create provider instances.

This validates only the provider cleanup/cancellation helper. It does **not** validate the full Bixby Android integration, Android rendering, or device lifecycle behavior. No Bixby build, APK build, installation, or device command was performed for this check.

## Results

| Unmodified test | Result | Duration |
| --- | --- | --- |
| `canceled caller cannot initialize a provider()` | Passed | 0.284 s |
| `double close preserves cleanup barrier before creating next provider()` | Passed | 1.572 s |

The compiler reported three nonfatal `ExperimentalCoroutinesApi` opt-in warnings for the original test's `runCurrent()` calls. JDK 21 reported the usual MockK/Byte Buddy dynamic-agent warning. Neither caused a test failure.

## Runtime and dependencies

- SDK Gradle wrapper: 8.10.2; the Bixby wrapper/build configuration was not invoked.
- Java: Android Studio OpenJDK 21.0.7, compiled with Java/Kotlin JVM target 17.
- Kotlin JVM plugin: 2.2.21.
- Coroutines core and test: 1.11.0.
- MockK: 1.14.11.
- JUnit Jupiter/BOM: 5.13.4, with the corresponding platform launcher.
- Dependency repositories: public Maven Central and Gradle Plugin Portal only.

Kotlin, coroutines, and MockK versions match the Bixby version catalog. This isolated harness uses the requested JUnit 5; Bixby currently declares JUnit 6.1.3, so this is not a reproduction of its complete test runtime.

## Source provenance

| File | SHA-256 |
| --- | --- |
| Bixby `GenUiProviderHolder.kt` | `9BDA871C7CFA9CCF4DB4CBB83CDD7E99046E39C1C5CEDF2077A03E365040592B` |
| SDK `Api.kt` | `049D8EB630E9F56EB89B830A78BED56F7D3D6FFF40242E20177B3EF97E830190` |
| Bixby `GenUiProviderHolderTest.kt` | `EAB0307C840B73A1E5957F90A0C95B196BB50DC96BF142F592BDBCCEB07DBBDB` |

`source-hashes.json` records absolute original/copy paths and verified copy hashes. `post-run-source-hashes.json` verifies that all three originals still matched their tested snapshots after execution. No production source was edited by this harness.

## Reproduction and evidence

From this directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run-checks.ps1
```

The script recopies and hashes the current source, then invokes the SDK wrapper from the isolated project directory:

```text
GenUICraft\gradlew.bat --no-daemon --max-workers=2 --console=plain test --tests com.samsung.android.bixby.agent.renderer.internal.genuicraft.GenUiProviderHolderTest
```

Evidence files:

- `gradle-test.log`
- `source-hashes.json`
- `post-run-source-hashes.json`
- `build/test-results/test/TEST-com.samsung.android.bixby.agent.renderer.internal.genuicraft.GenUiProviderHolderTest.xml`
- `build/reports/tests/test/index.html`

This entire harness is a temporary validation artifact under `A2UI/tmp`; it is not part of SDK or Bixby production source.
