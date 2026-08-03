param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [Parameter(Mandatory = $false)]
    [string]$ExpectedSha256 = "171c6507ea320e955b2dd8fc91b78e44085e03290f11d26ea576f10dde643fa1",
    [Parameter(Mandatory = $false)]
    [string]$Adb = "adb",
    [Parameter(Mandatory = $false)]
    [string]$Serial = ""
)

$ErrorActionPreference = "Stop"
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
$model = Get-Item -LiteralPath $resolvedModel
if ($model.Extension -ne ".litertlm") {
    throw "Expected a .litertlm model: $resolvedModel"
}
if ($model.Length -lt 2500000000) {
    throw "The v6 model is unexpectedly small ($($model.Length) bytes): $resolvedModel"
}
$actualSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $resolvedModel).Hash.ToLowerInvariant()
if ($actualSha256 -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "SHA-256 mismatch for $resolvedModel. Expected $ExpectedSha256, got $actualSha256."
}

$adbArgs = @()
if (-not [string]::IsNullOrWhiteSpace($Serial)) {
    $adbArgs += @("-s", $Serial)
}
$packageName = "com.samsung.genuicraft"
$deviceDir = "/sdcard/Android/data/$packageName/files/on_device_models"
$deviceModel = "$deviceDir/gemma-4-e2b-a2ui-express-v6.litertlm"

& $Adb @adbArgs get-state | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "No Android device is connected."
}
& $Adb @adbArgs shell mkdir -p $deviceDir
if ($LASTEXITCODE -ne 0) {
    throw "Could not create $deviceDir on the device."
}
& $Adb @adbArgs push $resolvedModel $deviceModel
if ($LASTEXITCODE -ne 0) {
    throw "Model push failed."
}
$remoteSize = (& $Adb @adbArgs shell stat -c %s $deviceModel).Trim()
if ($LASTEXITCODE -ne 0 -or $remoteSize -ne $model.Length.ToString()) {
    throw "Remote model size verification failed. Expected $($model.Length), got $remoteSize."
}

Write-Output "Installed and verified Gemma 4 E2B A2UI Express v6 at $deviceModel"
Write-Output "SHA-256: $actualSha256"
Write-Output "Open Settings > IR generation backend > On-device LiteRT IR, select the v6 model, and choose GPU only."
