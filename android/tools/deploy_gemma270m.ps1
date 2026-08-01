param(
    [Parameter(Mandatory = $false)]
    [string]$ModelPath = "..\..\training\outputs\export\gemma270m_ir_int8\litertlm\gemma-3-270m-ir-int8.litertlm",
    [Parameter(Mandatory = $false)]
    [string]$Adb = "adb"
)

$ErrorActionPreference = "Stop"
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
if ([IO.Path]::GetExtension($resolvedModel) -ne ".litertlm") {
    throw "Expected a .litertlm model: $resolvedModel"
}
if ((Get-Item -LiteralPath $resolvedModel).Length -le 0) {
    throw "Model file is empty: $resolvedModel"
}

$packageName = "com.samsung.genuicraft"
$deviceDir = "/sdcard/Android/data/$packageName/files/on_device_models"
$deviceModel = "$deviceDir/gemma-3-270m-ir-int8.litertlm"

& $Adb get-state | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No Android device is connected." }
& $Adb shell mkdir -p $deviceDir
if ($LASTEXITCODE -ne 0) { throw "Could not create $deviceDir on the device." }
& $Adb push $resolvedModel $deviceModel
if ($LASTEXITCODE -ne 0) { throw "Model push failed." }
& $Adb shell ls -l $deviceModel
if ($LASTEXITCODE -ne 0) { throw "The pushed model could not be verified." }

Write-Output "Installed Gemma 270M INT8 at $deviceModel"
Write-Output "Open GenUICraft Settings > IR generation backend > On-device LiteRT IR, then select Gemma 3 270M IR."
