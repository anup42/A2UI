param(
    [string]$OutDir = "tmp/selftest/single_query",
    [string]$Query = "best flights from bangalore to langkawi on april 15 with fares and sources"
)
$ErrorActionPreference='Stop'
function Ensure-Dir([string]$path){ if(!(Test-Path $path)){ New-Item -ItemType Directory -Path $path -Force | Out-Null } }
function Get-CenterFromBounds([string]$bounds){ if($bounds -match "\[(\d+),(\d+)\]\[(\d+),(\d+)\]"){ return @([int](($Matches[1]+$Matches[3])/2),[int](($Matches[2]+$Matches[4])/2)) }; throw "Invalid bounds: $bounds" }
function Dump-Ui([string]$path){ adb shell uiautomator dump /sdcard/window_dump.xml | Out-Null; adb pull /sdcard/window_dump.xml $path | Out-Null }
function Tap-ByTextContains([string]$xmlPath,[string]$textPart){ [xml]$xml=Get-Content $xmlPath; $n=$xml.SelectNodes("//node[contains(@text, '$textPart')]"); if($null -eq $n -or $n.Count -eq 0){ throw "node not found: $textPart"}; $c=Get-CenterFromBounds $n.Item(0).bounds; adb shell input tap $c[0] $c[1] | Out-Null }
function Tap-ByClass([string]$xmlPath,[string]$klass){ [xml]$xml=Get-Content $xmlPath; $n=$xml.SelectSingleNode("//node[@class='$klass']"); if($null -eq $n){ throw "class not found: $klass"}; $c=Get-CenterFromBounds $n.bounds; adb shell input tap $c[0] $c[1] | Out-Null }
function Tap-ByContentDesc([string]$xmlPath,[string]$desc){ [xml]$xml=Get-Content $xmlPath; $n=$xml.SelectSingleNode("//node[@content-desc='$desc']"); if($null -eq $n){ throw "desc not found: $desc"}; $c=Get-CenterFromBounds $n.bounds; adb shell input tap $c[0] $c[1] | Out-Null }
function Enter-Text([string]$text){ $e=$text.Trim(); $e=$e -replace " ", "%s"; $e=$e -replace "&", "\\&"; $e=$e -replace "\|", ""; $e=$e -replace "\(", ""; $e=$e -replace "\)", ""; adb shell input text "$e" | Out-Null }
Ensure-Dir $OutDir
adb shell am force-stop com.samsung.genuicraft | Out-Null
adb shell input keyevent 224 | Out-Null
Start-Sleep -Milliseconds 300
adb shell input keyevent 82 | Out-Null
Start-Sleep -Milliseconds 300
adb shell wm dismiss-keyguard | Out-Null
Start-Sleep -Milliseconds 300
adb shell am start -n com.samsung.genuicraft/.MainActivity | Out-Null
Start-Sleep -Seconds 2
Dump-Ui (Join-Path $OutDir 'main.xml')
Tap-ByTextContains (Join-Path $OutDir 'main.xml') 'GenUI Demo'
Start-Sleep -Seconds 2
Dump-Ui (Join-Path $OutDir 'assistant.xml')
Tap-ByClass (Join-Path $OutDir 'assistant.xml') 'android.widget.EditText'
Start-Sleep -Milliseconds 400
adb shell input keyevent 123 | Out-Null
1..140 | % { adb shell input keyevent 67 | Out-Null }
Enter-Text $Query
Start-Sleep -Milliseconds 1000
Dump-Ui (Join-Path $OutDir 'typed.xml')
Tap-ByContentDesc (Join-Path $OutDir 'typed.xml') 'Send prompt'
$elapsed=0
while($elapsed -lt 180){ Start-Sleep -Seconds 5; $elapsed += 5; Dump-Ui (Join-Path $OutDir ("poll_$elapsed.xml")); $raw=Get-Content (Join-Path $OutDir ("poll_$elapsed.xml")) -Raw; if($raw -match "Completed\. Rendered output is ready\." -or $raw -match "Failed:"){ break } }
Dump-Ui (Join-Path $OutDir 'done.xml')
adb shell screencap -p /sdcard/single_query.png | Out-Null
adb pull /sdcard/single_query.png (Join-Path $OutDir 'screen.png') | Out-Null
adb logcat -d -v time > (Join-Path $OutDir 'logcat.txt')
Write-Output "elapsed=$elapsed"
