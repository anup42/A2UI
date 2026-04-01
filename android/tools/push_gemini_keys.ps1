param(
    [string]$DatasetEnvPath = "..\dataset\.env",
    [string]$PackageName = "com.samsung.genuicraft",
    [string]$OutputName = "genuicraft_keys.env"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Parse-DotEnv([string]$path) {
    $result = @{}
    foreach ($raw in Get-Content -Path $path) {
        $line = $raw.Trim()
        if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
            continue
        }
        if ($line.StartsWith("export ")) {
            $line = $line.Substring(7).Trim()
        }
        $idx = $line.IndexOf("=")
        if ($idx -le 0) {
            continue
        }
        $key = $line.Substring(0, $idx).Trim()
        $value = $line.Substring($idx + 1).Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        if (-not [string]::IsNullOrWhiteSpace($key) -and -not [string]::IsNullOrWhiteSpace($value)) {
            $result[$key] = $value
        }
    }
    return $result
}

if (-not (Test-Path $DatasetEnvPath)) {
    throw "Missing .env file: $DatasetEnvPath"
}

$envMap = Parse-DotEnv -path $DatasetEnvPath
$vertexKey = if ($envMap.ContainsKey("GEMINI_VERTEX_EXPRESS_API_KEY")) {
    $envMap["GEMINI_VERTEX_EXPRESS_API_KEY"]
} elseif ($envMap.ContainsKey("VERTEX_EXPRESS_API_KEY")) {
    $envMap["VERTEX_EXPRESS_API_KEY"]
} else {
    $null
}

if ([string]::IsNullOrWhiteSpace($vertexKey)) {
    throw "GEMINI_VERTEX_EXPRESS_API_KEY or VERTEX_EXPRESS_API_KEY is missing in $DatasetEnvPath"
}

$tmpFile = Join-Path $env:TEMP $OutputName
@(
    "GEMINI_VERTEX_EXPRESS_API_KEY=$vertexKey"
    "VERTEX_EXPRESS_API_KEY=$vertexKey"
) | Set-Content -Path $tmpFile -NoNewline:$false -Encoding ascii

$devicePath = "/sdcard/Android/data/$PackageName/files/$OutputName"
adb push "$tmpFile" "$devicePath" | Out-Null
adb shell ls -l "$devicePath"

Remove-Item -Force "$tmpFile"
Write-Host "Pushed key file to $devicePath"
