param(
    [string]$OutDir = "tmp/selftest/5scenarios"
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
    adb shell input keyevent 224 | Out-Null # wake
    Start-Sleep -Milliseconds 350
    adb shell input keyevent 82 | Out-Null  # menu/unlock
    Start-Sleep -Milliseconds 350
    adb shell wm dismiss-keyguard | Out-Null
    Start-Sleep -Milliseconds 350
    adb shell am start -n com.samsung.genuicraft/.MainActivity | Out-Null
    Start-Sleep -Seconds 2

    $checkPath = Join-Path $OutDir "_foreground_check.xml"
    Dump-Ui $checkPath
    $raw = Read-Raw $checkPath
    if ($raw -notmatch 'package="com.samsung.genuicraft"') {
        throw "App not in foreground after unlock/start"
    }
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
    adb shell input keyevent 123 | Out-Null # move end
    1..90 | ForEach-Object { adb shell input keyevent 67 | Out-Null } # DEL
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

function Wait-For-Result([string]$scenarioName, [int]$timeoutSec = 210) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $pollPath = Join-Path $OutDir ("{0}_poll_{1}.xml" -f $scenarioName, $elapsed)
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

$scenarios = @(
    @{ name = "s1_weather"; query = "weather in bengaluru next 5 days with current conditions" },
    @{ name = "s2_flights"; query = "best flights from BLR to LKO on march 15 with fares and stops" },
    @{ name = "s3_itinerary"; query = "3 day kyoto family itinerary with activities and quick actions" },
    @{ name = "s4_compare"; query = "compare samsung s24 and iphone 16 camera battery and price in table" },
    @{ name = "s5_bullets"; query = "best high protein vegetarian breakfasts with bullet tips and sources" }
)

adb shell am force-stop com.samsung.genuicraft | Out-Null
Ensure-AppForeground

$mainXml = Join-Path $OutDir "main.xml"
Dump-Ui $mainXml
Tap-ByTextContains $mainXml "GenUI Demo"
Start-Sleep -Seconds 2

$assistantXml = Join-Path $OutDir "assistant_open.xml"
Dump-Ui $assistantXml

$resultSummary = @()

foreach ($s in $scenarios) {
    $name = $s.name
    $query = $s.query

    adb logcat -c

    $beforeXml = Join-Path $OutDir ("{0}_before.xml" -f $name)
    Dump-Ui $beforeXml

    Tap-ByClass $beforeXml "android.widget.EditText"
    Start-Sleep -Milliseconds 400
    Clear-EditField
    Enter-Text $query
    Start-Sleep -Milliseconds 1200

    $afterTypeXml = Join-Path $OutDir ("{0}_typed.xml" -f $name)
    Dump-Ui $afterTypeXml

    Tap-ByContentDesc $afterTypeXml "Send prompt"

    $state = Wait-For-Result -scenarioName $name

    $doneXml = Join-Path $OutDir ("{0}_done.xml" -f $name)
    Dump-Ui $doneXml

    $png = Join-Path $OutDir ("{0}.png" -f $name)
    $remotePng = "/sdcard/Download/$name.png"; adb shell screencap -p $remotePng | Out-Null; adb pull $remotePng $png | Out-Null

    $logPath = Join-Path $OutDir ("{0}.logcat.txt" -f $name)
    adb logcat -d -v time > $logPath

    $rawDone = Read-Raw $doneXml
    $hasErrorText = [bool]($rawDone -match "Failed:")
    $hasCompleted = [bool]($rawDone -match "Completed\. Rendered output is ready\.")
    $hasMarkdownLeak = [bool]($rawDone -match "\*\*|###")

    $resultSummary += [pscustomobject]@{
        scenario = $name
        status = $state.status
        elapsed_sec = $state.elapsed
        completed = $hasCompleted
        has_error_text = $hasErrorText
        markdown_leak = $hasMarkdownLeak
        screenshot = $png
        xml = $doneXml
        log = $logPath
    }
}

$resultSummary | Export-Csv -NoTypeInformation -Path (Join-Path $OutDir "summary.csv")
$resultSummary | Format-Table -AutoSize
