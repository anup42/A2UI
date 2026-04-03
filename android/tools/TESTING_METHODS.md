# Android UI Automation Testing Methods

This file documents the stable device-testing flow for GenUI assistant rendering checks.

## Critical Send-Flow Method

When testing by injecting text and pressing send:

1. Focus input and inject text with `adb shell input text`.
2. Wait `1s` after text injection.
3. Dismiss keyboard (`adb shell input keyevent 4`).
4. Wait `1s` after keyboard dismiss.
5. Dump UI again (`uiautomator dump`) and re-read current bounds of `content-desc="Send prompt"`.
6. Tap send using the updated bounds center.

Do not use a fixed send-button coordinate immediately after typing. Keyboard transitions can shift layout.

## Reusable Script

Use:

```powershell
powershell -ExecutionPolicy Bypass -File .\android\tools\run_quickactions_audit.ps1
```

What it does:

1. Opens `GenUI Demo`.
2. Runs 10 queries from `android/app/src/main/assets/ir_demo_subset10_queries.jsonl`.
3. Applies the critical send-flow method above for every query.
4. Captures top/mid/low UI dumps and one screenshot per scenario.
5. Flags raw action-markup leaks (`[Button: ...]`, `Action: [...]`) in rendered text.
6. Writes summary CSV at `android/tmp/selftest/quickactions_audit/summary.csv`.

## Notes

- Device expected: package `com.samsung.genuicraft` installed and unlocked.
- Default adb path in script:
  `C:\Users\anups\AppData\Local\Android\Sdk\platform-tools\adb.exe`
- You can override `-AdbPath`, `-OutDir`, and `-QueriesPath` script params if needed.
