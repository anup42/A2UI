param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [Parameter(Mandatory = $false)]
    [string]$Adb = "adb",
    [Parameter(Mandatory = $false)]
    [string]$Serial = "",
    [Parameter(Mandatory = $false)]
    [switch]$Mtp,
    [Parameter(Mandatory = $false)]
    [switch]$SkipBuildInstall
)

$ErrorActionPreference = "Stop"
$model = (Resolve-Path -LiteralPath $ModelPath).Path
$modelInfo = Get-Item -LiteralPath $model
if ($modelInfo.Extension -ne ".litertlm") {
    throw "Expected a .litertlm package: $model"
}
if ($modelInfo.Length -lt 2500000000L) {
    throw "The trained Express package is incomplete: $($modelInfo.Length) bytes"
}

$androidRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$validationPackage = "com.samsung.genuicraft.judgecapture"
$runner = "$validationPackage.test/androidx.test.runner.AndroidJUnitRunner"
$deviceDir = "/sdcard/Android/data/$validationPackage/files/on_device_models"
$deviceModel = "$deviceDir/gemma-4-e2b-trained-express-int4.litertlm"
$mode = if ($Mtp) { "gpu_mtp" } else { "gpu_target_only" }
$mtpText = if ($Mtp) { "true" } else { "false" }
$adbPrefix = @()
if (-not [string]::IsNullOrWhiteSpace($Serial)) {
    $adbPrefix = @("-s", $Serial)
}

function Invoke-CheckedAdb {
    param([string[]]$CommandArgs)
    $output = & $Adb @adbPrefix @CommandArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "adb command failed: $Adb $($adbPrefix -join ' ') $($CommandArgs -join ' ')`n$($output -join "`n")"
    }
    return @($output)
}

if (-not $SkipBuildInstall) {
    $priorAndroidSerial = $env:ANDROID_SERIAL
    try {
        if (-not [string]::IsNullOrWhiteSpace($Serial)) {
            $env:ANDROID_SERIAL = $Serial
        }
        Push-Location $androidRoot
        try {
            & .\gradlew.bat -PandroidTestBuildType=judgeCapture `
                :app:installJudgeCapture :app:installJudgeCaptureAndroidTest
            if ($LASTEXITCODE -ne 0) {
                throw "The isolated judgeCapture app/test installation failed."
            }
        } finally {
            Pop-Location
        }
    } finally {
        $env:ANDROID_SERIAL = $priorAndroidSerial
    }
}

Invoke-CheckedAdb -CommandArgs @("get-state") | Out-Null
Invoke-CheckedAdb -CommandArgs @("shell", "mkdir", "-p", $deviceDir) | Out-Null
Invoke-CheckedAdb -CommandArgs @("push", $model, $deviceModel) | Out-Host

$hostHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $model).Hash.ToLowerInvariant()
$deviceHashText = Invoke-CheckedAdb -CommandArgs @("shell", "sha256sum", $deviceModel)
$deviceHash = (($deviceHashText -join " ") -split "\s+")[0].ToLowerInvariant()
if ($hostHash -ne $deviceHash) {
    throw "Host/device model SHA-256 mismatch: host=$hostHash device=$deviceHash"
}

$instrumentation = Invoke-CheckedAdb -CommandArgs @(
    "shell", "am", "instrument", "-w", "-r",
    "-e", "mtp", $mtpText,
    "-e", "class", "com.samsung.genuicraft.Gemma4E2bTrainedExpressRawProbeTest",
    $runner
)
$instrumentation | Out-Host
if (($instrumentation -join "`n") -notmatch "OK \(1 test\)") {
    throw "$mode trained-Express instrumentation did not report OK (1 test)."
}

$reportRelative = "files/result_gemma4_e2b_trained_express_raw_probe/$mode-report.json"
$reportText = Invoke-CheckedAdb -CommandArgs @(
    "shell", "run-as", $validationPackage, "cat", $reportRelative
)
$reportJson = $reportText -join "`n"
$report = $reportJson | ConvertFrom-Json
$requiredTrue = @(
    "starts_with_a2ui",
    "strict_decode_succeeded",
    "strict_validation_succeeded",
    "response_fact_coverage_succeeded",
    "native_render_succeeded"
)
$failed = @($requiredTrue | Where-Object { $report.$_ -ne $true })
if ($failed.Count -gt 0) {
    throw "$mode strict-Express semantic gate failed ($($failed -join ', ')):`n$reportJson"
}

Write-Output "PASS: $mode strict A2UI Express generation, fact coverage, and native rendering"
Write-Output "Host/device SHA-256: $hostHash"
Write-Output "Isolated validation package: $validationPackage"
Write-Output "Device report: $reportRelative"
if (-not $Mtp) {
    Write-Output "MTP remained disabled. Run the target-only GPU parity benchmark before any -Mtp validation."
}
