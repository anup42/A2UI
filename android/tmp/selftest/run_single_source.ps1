param(
    [string]$OutDir = "tmp/selftest/single_source"
)

$ErrorActionPreference = 'Stop'

function Ensure-Dir([string]$path) {
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

function Get-CenterFromBounds([string]$bounds) {
    if ($bounds -match "\[(\d+),(\d+)\]\[(\d+),(\d+)\]") {
        $x1 = [int]$Matches[1]
        $y1 = [int]$Matches[2]
        $x2 = [int]$Matches[3]
        $y2 = [int]$Matches[4]
        return @([int](($x1 + $x2) / 2), [int](($y1 + $y2) / 2))
    }
    throw "Invalid bounds: $bounds"
}

function Dump-Ui([string]$path) {
    adb shell uiautomator dump /sdcard/window_dump.xml | Out-Null
    adb pull /sdcard/window_dump.xml $path | Out-Null
}

function Read-Raw([string]$path) { Get-Content $path -Raw }

function Ensure-AppForeground() {
    adb shell input keyevent 224 | Out-Null
    Start-Sleep -Milliseconds 300
    adb shell input keyevent 82 | Out-Null
    Start-Sleep -Milliseconds 300
    adb shell wm dismiss-keyguard | Out-Null
    Start-Sleep -Milliseconds 300
    adb shell am start -n com.samsung.genuicraft/.MainActivity | Out-Null
    Start-Sleep -Seconds 2
}

function Tap-ByTextContains([string]$xmlPath, [string]$textPart) {
    [xml]$xml = Get-Content $xmlPath
    $nodes = $xml.SelectNodes("//node[contains(@text, '$textPart')]")
    if ($null -eq $nodes -or $nodes.Count -eq 0) { throw "Node containing text '$textPart' not found" }
    $node = $nodes.Item(0)
    $center = Get-CenterFromBounds $node.bounds
    adb shell input tap $center[0] $center[1] | Out-Null
}

function Tap-ByClass([string]$xmlPath, [string]$klass) {
    [xml]$xml = Get-Content $xmlPath
    $node = $xml.SelectSingleNode("//node[@class='$klass']")
    if ($null -eq $node) { throw "Node with class '$klass' not found" }
    $center = Get-CenterFromBounds $node.bounds
    adb shell input tap $center[0] $center[1] | Out-Null
}

function Tap-ByContentDesc([string]$xmlPath, [string]$desc) {
    [xml]$xml = Get-Content $xmlPath
    $node = $xml.SelectSingleNode("//node[@content-desc='$desc']")
    if ($null -eq $node) { throw "Node with content-desc '$desc' not found" }
    $center = Get-CenterFromBounds $node.bounds
    adb shell input tap $center[0] $center[1] | Out-Null
}

function Clear-EditField() {
    adb shell input keyevent 123 | Out-Null
    1..120 | ForEach-Object { adb shell input keyevent 67 | Out-Null }
}

function Enter-Text([string]$text) {
    $encoded = $text.Trim()
    $encoded = $encoded -replace " ", "%s"
    $encoded = $encoded -replace "&", "\\&"
    $encoded = $encoded -replace "\|", ""
    $encoded = $encoded -replace "\(", ""
    $encoded = $encoded -replace "\)", ""
    adb shell input text "$encoded" | Out-Null
}

function Wait-For-Result([int]$timeoutSec = 180) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $pollPath = Join-Path $OutDir ("poll_{0}.xml" -f $elapsed)
        Dump-Ui $pollPath
        $raw = Read-Raw $pollPath
        if ($raw -match "Completed\. Rendered output is ready\.") {
            return @{ status = "completed"; elapsed = $elapsed; xml = $pollPath }
        }
        if ($raw -match "Failed:" -or $raw -match "Pipeline failed") {
            return @{ status = "failed"; elapsed = $elapsed; xml = $pollPath }
        }
    }
    return @{ status = "timeout"; elapsed = $elapsed; xml = "" }
}

Ensure-Dir $OutDir
adb shell am force-stop com.samsung.genuicraft | Out-Null
Ensure-AppForeground

$mainXml = Join-Path $OutDir "main.xml"
Dump-Ui $mainXml
Tap-ByTextContains $mainXml "GenUI Demo"
Start-Sleep -Seconds 2

$assistantXml = Join-Path $OutDir "assistant.xml"
Dump-Ui $assistantXml

$beforeXml = Join-Path $OutDir "before.xml"
Dump-Ui $beforeXml
Tap-ByClass $beforeXml "android.widget.EditText"
Start-Sleep -Milliseconds 400
Clear-EditField
Enter-Text "compare samsung s24 and iphone 16 camera battery and price with source links"
Start-Sleep -Milliseconds 1000

$typedXml = Join-Path $OutDir "typed.xml"
Dump-Ui $typedXml
Tap-ByContentDesc $typedXml "Send prompt"

$state = Wait-For-Result

$doneXml = Join-Path $OutDir "done.xml"
Dump-Ui $doneXml
$remotePng = "/sdcard/Download/single_source.png"
adb shell screencap -p $remotePng | Out-Null
adb pull $remotePng (Join-Path $OutDir "screen.png") | Out-Null
$logPath = Join-Path $OutDir "logcat.txt"
adb logcat -d -v time > $logPath

Write-Output "status=$($state.status) elapsed=$($state.elapsed)"
