# Bixby Common `GenUiFeatureStore` isolated validation

- Validation date: 2026-09-22 (Asia/Calcutta)
- Production source compiled directly: `C:\Users\anupk\Downloads\Bixby_18Sep\Common\src\main\java\com\samsung\android\bixby\agent\common\genuicraft\GenUiFeatureStore.kt`
- Source SHA-256: `22fbd0fd6d8284afbcad1ae17e86daa1aa67df455ba5291407658589ff29b47d`
- Source size: 9,601 bytes
- Result: 4 tests, 0 failures, 0 errors, 0 skipped (`BUILD SUCCESSFUL`)
- JUnit result: `bixby_feature_store_host_tests.xml`

## Scope

The ignored harness at `A2UI/tmp/genuicraft_feature_store_harness` adds the exact external Bixby source directory to an isolated Android library source set. It uses public dependencies only: Android Gradle Plugin/Kotlin from the GenUICraft wrapper, Robolectric 4.14.1, AndroidX Test Core 1.6.1, JUnit 4.13.2, and kotlinx-coroutines 1.9.0. It does not copy the production source into A2UI and does not run the full Bixby build or ADB.

The focused test run covers:

1. Safe preference defaults (`enabled=false`, `MTP=true`) and persistence.
2. Missing and wrong-size model status, including an empty verified-model cache.
3. Rejection of a 21-byte import while preserving the previously installed destination and removing the `.importing` file.
4. Listener duplicate suppression and teardown through the returned `AutoCloseable`.

The final source revision also compiled with the shared `modelIoMutex` guarding both `verifyInstalledModel` and `importModel` through `withLock`.

No test allocates or writes the expected 2,588,147,712-byte model. Consequently, this run does not exercise the valid full-size SHA-256 success path or real storage throughput; it verifies the safe early-rejection paths and state preservation without a large model artifact.

## Command

```powershell
$env:ANDROID_HOME='C:\Users\anupk\AppData\Local\Android\Sdk'
$env:ANDROID_SDK_ROOT=$env:ANDROID_HOME
.\gradlew.bat -p C:\Users\anupk\Documents\git\A2UI\tmp\genuicraft_feature_store_harness :harness:testDebugUnitTest --tests com.samsung.android.bixby.agent.common.genuicraft.GenUiFeatureStoreHarnessTest --no-daemon --console=plain
```

Final run: `BUILD SUCCESSFUL in 32s`; 31 actionable tasks (5 executed, 26 up-to-date). JUnit suite time: 4.314 seconds.
