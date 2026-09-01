param(
    [Parameter(Mandatory = $false)]
    [string]$ModelPath = "..\..\training\outputs\export\gemma3_270m_a2ui_express_int8\litertlm\gemma-3-270m-a2ui-express-int8.litertlm",
    [Parameter(Mandatory = $false)]
    [string]$Adb = "adb"
)

$ErrorActionPreference = "Stop"
$resolvedModel = (Resolve-Path -LiteralPath $ModelPath).Path
if ([IO.Path]::GetExtension($resolvedModel) -ne ".litertlm") {
    throw "Expected a .litertlm model: $resolvedModel"
}
if ((Get-Item -LiteralPath $resolvedModel).Length -le 0) {
    throw "Model is empty: $resolvedModel"
}

$packageName = "com.samsung.genuicraft"
$deviceDir = "/sdcard/Android/data/$packageName/files/on_device_models"
$deviceModel = "$deviceDir/gemma-3-270m-a2ui-express-int8.litertlm"

& $Adb get-state | Out-Null
if ($LASTEXITCODE -ne 0) { throw "No Android device is connected." }
& $Adb shell mkdir -p $deviceDir
if ($LASTEXITCODE -ne 0) { throw "Could not create $deviceDir on the device." }
& $Adb push $resolvedModel $deviceModel
if ($LASTEXITCODE -ne 0) { throw "Model push failed." }
& $Adb shell chmod 644 $deviceModel
& $Adb shell ls -l $deviceModel
if ($LASTEXITCODE -ne 0) { throw "The pushed model could not be verified." }

# Samsung Android builds may expose shell-owned files through adb but return
# File.isFile()==false to the app process. Keep an app-owned private copy so
# OnDeviceModelCatalog can use its internal fallback reliably in tests.
$privateModel = "files/on_device_models/gemma-3-270m-a2ui-express-int8.litertlm"
& $Adb shell run-as $packageName mkdir -p files/on_device_models
if ($LASTEXITCODE -ne 0) { throw "Could not create the app-private model directory." }
& $Adb shell run-as $packageName cp $deviceModel $privateModel
if ($LASTEXITCODE -ne 0) { throw "Could not copy the model into app-private storage." }
& $Adb shell run-as $packageName ls -l $privateModel
if ($LASTEXITCODE -ne 0) { throw "The app-private model copy could not be verified." }

Write-Output "Installed Gemma 270M A2UI Express at $deviceModel"
Write-Output "App-private fallback copy: $privateModel"
Write-Output "Select Gemma 3 270M A2UI Express under Settings > IR generation backend > On-device LiteRT IR."
