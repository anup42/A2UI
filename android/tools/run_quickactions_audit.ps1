param(
    [string]$OutDir = "android/tmp/selftest/quickactions_audit",
    [string]$QueriesPath = "android/app/src/main/assets/ir_demo_subset10_queries.jsonl",
    [string]$AdbPath = "C:\Users\anups\AppData\Local\Android\Sdk\platform-tools\adb.exe"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (!(Test-Path $AdbPath)) { throw "adb not found at $AdbPath" }
if (!(Test-Path $QueriesPath)) { throw "queries file not found: $QueriesPath" }

function Invoke-Adb {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & $AdbPath @Args
}

function Ensure-Dir([string]$path) {
    if (!(Test-Path $path)) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
}

function Dump-Ui([string]$path) {
    Invoke-Adb shell uiautomator dump /sdcard/window_dump.xml | Out-Null
    Invoke-Adb pull /sdcard/window_dump.xml $path | Out-Null
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

function Get-SendPromptCenter([string]$xmlPath) {
    [xml]$xml = Get-Content $xmlPath
    $sendNode = $xml.SelectSingleNode("//node[@content-desc='Send prompt']")
    if ($null -eq $sendNode) {
        return $null
    }
    return Get-CenterFromBounds $sendNode.bounds
}

function Encode-Input([string]$text) {
    $encoded = $text.Trim()
    $encoded = $encoded -replace " ", "%s"
    $encoded = $encoded -replace "&", "\\&"
    $encoded = $encoded -replace "\|", ""
    $encoded = $encoded -replace "\(", ""
    $encoded = $encoded -replace "\)", ""
    $encoded = $encoded -replace "'", ""
    $encoded = $encoded -replace '"', ""
    return $encoded
}

function Open-GenUiScreen {
    Invoke-Adb shell am force-stop com.samsung.genuicraft | Out-Null
    Invoke-Adb shell input keyevent 224 | Out-Null
    Start-Sleep -Milliseconds 250
    Invoke-Adb shell input keyevent 82 | Out-Null
    Start-Sleep -Milliseconds 250
    Invoke-Adb shell wm dismiss-keyguard | Out-Null
    Start-Sleep -Milliseconds 250
    Invoke-Adb shell am start -n com.samsung.genuicraft/.MainActivity | Out-Null
    Start-Sleep -Seconds 2
    # GenUI Demo card center
    Invoke-Adb shell input tap 220 390 | Out-Null
    Start-Sleep -Seconds 2
}

function Tap-SendPromptAfterKeyboardDismiss([string]$scenarioDir) {
    Start-Sleep -Seconds 1
    # remove keyboard
    Invoke-Adb shell input keyevent 4 | Out-Null
    # IMPORTANT: keyboard hide animation takes time on device
    Start-Sleep -Seconds 1

    $typedReadyXml = Join-Path $scenarioDir 'typed_ready.xml'
    Dump-Ui $typedReadyXml
    $center = Get-SendPromptCenter $typedReadyXml

    if ($null -eq $center) {
        # fallback center if node not found
        $center = @(955, 2285)
    }
    Invoke-Adb shell input tap $center[0] $center[1] | Out-Null
}

function Wait-For-Completion([string]$scenarioDir, [int]$timeoutSec = 210) {
    $elapsed = 0
    while ($elapsed -lt $timeoutSec) {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $pollPath = Join-Path $scenarioDir ("poll_{0}.xml" -f $elapsed)
        Dump-Ui $pollPath
        $raw = Get-Content $pollPath -Raw
        if ($raw -match "Completed\. Rendered output is ready\." -or $raw -match "Failed:" -or $raw -match "Pipeline failed") {
            return $elapsed
        }
    }
    return $elapsed
}

function Scenario-HasRawActionLeak([string[]]$xmlPaths) {
    $combined = ""
    foreach ($p in $xmlPaths) {
        if (Test-Path $p) { $combined += (Get-Content $p -Raw) + "`n" }
    }
    if ([string]::IsNullOrWhiteSpace($combined)) { return $false }

    $patterns = @(
        '(?i)\[Button:\s*',
        '(?i)Action:\s*\[',
        '(?i)Action:\s*[^\n]*https?://'
    )

    foreach ($pat in $patterns) {
        if ([regex]::IsMatch($combined, $pat)) {
            return $true
        }
    }
    return $false
}

function Scenario-ButtonCount([string[]]$xmlPaths) {
    $count = 0
    foreach ($p in $xmlPaths) {
        if (Test-Path $p) {
            $raw = Get-Content $p -Raw
            $count += ([regex]::Matches($raw, 'class="android.widget.Button"')).Count
        }
    }
    return $count
}

Ensure-Dir $OutDir

$queries = Get-Content $QueriesPath |
    Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
    Select-Object -First 10 |
    ForEach-Object { ($_ | ConvertFrom-Json).query_text.ToString() }

if ($queries.Count -lt 10) { throw "expected 10 queries, found $($queries.Count)" }

$rows = @()

for ($i = 0; $i -lt $queries.Count; $i++) {
    $idx = $i + 1
    $name = "s{0:00}" -f $idx
    $query = $queries[$i]
    $scenarioDir = Join-Path $OutDir $name
    Ensure-Dir $scenarioDir

    Open-GenUiScreen

    Invoke-Adb shell input tap 180 2270 | Out-Null
    Start-Sleep -Milliseconds 350
    Invoke-Adb shell input keyevent 123 | Out-Null
    1..140 | ForEach-Object { Invoke-Adb shell input keyevent 67 | Out-Null }

    $encoded = Encode-Input $query
    Invoke-Adb shell input text "$encoded" | Out-Null

    Tap-SendPromptAfterKeyboardDismiss -scenarioDir $scenarioDir

    $elapsed = Wait-For-Completion -scenarioDir $scenarioDir

    $topXml = Join-Path $scenarioDir 'done_top.xml'
    Dump-Ui $topXml

    Invoke-Adb shell input swipe 540 1900 540 650 350 | Out-Null
    Start-Sleep -Milliseconds 700
    $midXml = Join-Path $scenarioDir 'done_mid.xml'
    Dump-Ui $midXml

    Invoke-Adb shell input swipe 540 1900 540 650 350 | Out-Null
    Start-Sleep -Milliseconds 700
    $lowXml = Join-Path $scenarioDir 'done_low.xml'
    Dump-Ui $lowXml

    $png = Join-Path $scenarioDir 'screen.png'
    $remotePng = '/sdcard/Download/qaudit_screen.png'
    Invoke-Adb shell screencap -p $remotePng | Out-Null
    Invoke-Adb pull $remotePng $png | Out-Null

    $xmls = @($topXml, $midXml, $lowXml)
    $rawLeak = Scenario-HasRawActionLeak -xmlPaths $xmls
    $buttonCount = Scenario-ButtonCount -xmlPaths $xmls

    $rows += [pscustomobject]@{
        scenario = $name
        elapsed_sec = $elapsed
        raw_action_leak = $rawLeak
        button_count = $buttonCount
        query = $query
        dir = $scenarioDir
    }
}

$summaryCsv = Join-Path $OutDir 'summary.csv'
$rows | Export-Csv -NoTypeInformation -Path $summaryCsv
$rows | Format-Table -AutoSize
$leaks = @($rows | Where-Object { $_.raw_action_leak }).Count
Write-Output "raw_action_leak_count=$leaks"
Write-Output "summary_csv=$summaryCsv"
