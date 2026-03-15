$ErrorActionPreference = 'Stop'
$out = 'tmp/selftest/source_repro'
New-Item -ItemType Directory -Force -Path $out | Out-Null

function DumpUi([string]$name) {
  adb shell uiautomator dump /sdcard/window_dump.xml | Out-Null
  adb pull /sdcard/window_dump.xml (Join-Path $out $name) | Out-Null
}

adb shell input keyevent 224 | Out-Null
adb shell wm dismiss-keyguard | Out-Null
adb shell am force-stop com.samsung.genuicraft | Out-Null
adb shell am start -n com.samsung.genuicraft/.MainActivity | Out-Null
Start-Sleep -Seconds 2
DumpUi 'main.xml'

# Open GenUI Demo tile (fixed location)
adb shell input tap 540 390 | Out-Null
Start-Sleep -Seconds 2
DumpUi 'assistant_open.xml'

# Focus input and clear
adb shell input tap 190 2270 | Out-Null
Start-Sleep -Milliseconds 400
adb shell input keyevent 123 | Out-Null
for ($k = 0; $k -lt 160; $k++) {
  adb shell input keyevent 67 | Out-Null
}

$query = 'weather in bengaluru today with current condition and sources'
$typed = $query -replace ' ', '%s'
adb shell input text $typed | Out-Null
Start-Sleep -Milliseconds 1200
DumpUi 'typed.xml'

# Send prompt
adb shell input tap 956 2288 | Out-Null

$done = $false
for ($i = 1; $i -le 90; $i++) {
  Start-Sleep -Seconds 3
  $pollName = "poll_$i.xml"
  DumpUi $pollName
  $raw = Get-Content (Join-Path $out $pollName) -Raw
  if ($raw -match 'Completed\. Rendered output is ready\.' -or $raw -match 'Failed:') {
    $done = $true
    break
  }
}

DumpUi 'done.xml'
$final = Get-Content (Join-Path $out 'done.xml') -Raw
$hasSources = [bool]($final -match 'Sources|Source')
$hasButtons = [bool]($final -match 'android\.widget\.Button')
$hasRawUrl = [bool]($final -match 'https?://|www\.')
$hasFailed = [bool]($final -match 'Failed:')
Write-Host "DONE=$done HAS_SOURCES=$hasSources HAS_BUTTONS=$hasButtons HAS_RAW_URL=$hasRawUrl FAILED=$hasFailed"

# Attempt source-button tap and detect browser package
adb shell input swipe 540 1900 540 1300 300 | Out-Null
Start-Sleep -Milliseconds 700
DumpUi 'after_scroll.xml'
adb shell input tap 240 1890 | Out-Null
Start-Sleep -Seconds 1
DumpUi 'after_source_tap.xml'
$afterTap = Get-Content (Join-Path $out 'after_source_tap.xml') -Raw
$openedBrowser = [bool]($afterTap -match 'package="com\.android\.chrome"|package="com\.sec\.android\.app\.sbrowser"|package="com\.google\.android\.googlequicksearchbox"')
Write-Host "BROWSER_OPENED=$openedBrowser"
