param([string]$Serial = 'R3GL203AKSF')
$ErrorActionPreference = 'Stop'
$modelUrl = 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it.litertlm'
$expectedHash = '181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c'
$expectedLength = 2588147712L
$modelDirectory = '/sdcard/Android/data/com.samsung.genuicraft/files/sdk_models'
$modelTarget = "$modelDirectory/gemma-4-E2B-it.litertlm"
& adb -s $Serial shell mkdir -p $modelDirectory
$httpClient = [Net.Http.HttpClient]::new()
$httpClient.Timeout = [TimeSpan]::FromMinutes(60)
$httpClient.DefaultRequestHeaders.UserAgent.ParseAdd('GenUICraft/0.1')
$modelResponse = $null
for ($connectAttempt = 1; $connectAttempt -le 4; $connectAttempt++) {
    try {
        $modelResponse = $httpClient.GetAsync($modelUrl, [Net.Http.HttpCompletionOption]::ResponseHeadersRead).GetAwaiter().GetResult()
        break
    } catch {
        if ($connectAttempt -eq 4) { throw }
        Start-Sleep -Seconds (2 * $connectAttempt)
    }
}
$modelResponse.EnsureSuccessStatusCode() | Out-Null
if ($modelResponse.Content.Headers.ContentLength -ne $expectedLength) { throw 'Unexpected model content length.' }
$modelStream = $modelResponse.Content.ReadAsStreamAsync().GetAwaiter().GetResult()
$adbInfo = [Diagnostics.ProcessStartInfo]::new((Get-Command adb).Source)
$adbInfo.UseShellExecute = $false
$adbInfo.CreateNoWindow = $true
$adbInfo.RedirectStandardInput = $true
foreach ($argument in @('-s', $Serial, 'exec-in', "cat > $modelTarget.partial")) { $adbInfo.ArgumentList.Add($argument) }
$adbSink = [Diagnostics.Process]::Start($adbInfo)
$modelHash = [Security.Cryptography.SHA256]::Create()
$modelBuffer = [byte[]]::new(4MB)
$modelBytes = 0L
$nextProgress = 64MB
try {
    while (($readBytes = $modelStream.Read($modelBuffer, 0, $modelBuffer.Length)) -gt 0) {
        $adbSink.StandardInput.BaseStream.Write($modelBuffer, 0, $readBytes)
        $modelHash.TransformBlock($modelBuffer, 0, $readBytes, $null, 0) | Out-Null
        $modelBytes += $readBytes
        if ($modelBytes -ge $nextProgress) {
            Write-Output "Model transfer: $modelBytes / $expectedLength bytes"
            $nextProgress += 64MB
        }
    }
    $modelHash.TransformFinalBlock([byte[]]::new(0), 0, 0) | Out-Null
} finally {
    $adbSink.StandardInput.Close()
    $modelStream.Dispose()
    $modelResponse.Dispose()
    $httpClient.Dispose()
}
$adbSink.WaitForExit()
if ($adbSink.ExitCode -ne 0) { throw 'ADB model write failed.' }
$actualHash = [Convert]::ToHexString($modelHash.Hash).ToLowerInvariant()
if ($modelBytes -ne $expectedLength -or $actualHash -ne $expectedHash) { throw "Integrity mismatch: $modelBytes bytes SHA=$actualHash" }
$deviceHash = ((& adb -s $Serial shell sha256sum "$modelTarget.partial") -split '\s+')[0]
if ($deviceHash -ne $expectedHash) { throw 'Device model SHA-256 differs from official artifact.' }
& adb -s $Serial shell mv "$modelTarget.partial" $modelTarget
$modelManifest = @{ url=$modelUrl; repository_commit='b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1'; sha256=$actualHash; bytes=$modelBytes; device=$Serial; device_path=$modelTarget }
$manifestDirectory = Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts'
New-Item -ItemType Directory -Path $manifestDirectory -Force | Out-Null
$modelManifest | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $manifestDirectory 'official_gemma_model.json')
$modelManifest | ConvertTo-Json
