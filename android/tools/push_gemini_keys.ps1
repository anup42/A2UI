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
$stage2Key = $envMap["GEMINI_API_KEY"]
$stage3Key = if ($envMap.ContainsKey("GEMINI_IR_API_KEY")) {
    $envMap["GEMINI_IR_API_KEY"]
} elseif ($envMap.ContainsKey("GEMINI_API_KEY_2")) {
    $envMap["GEMINI_API_KEY_2"]
} else {
    $null
}
$vertexExpressKey = if ($envMap.ContainsKey("VERTEX_EXPRESS_API_KEY")) {
    $envMap["VERTEX_EXPRESS_API_KEY"]
} elseif ($envMap.ContainsKey("GEMINI_VERTEX_EXPRESS_API_KEY")) {
    $envMap["GEMINI_VERTEX_EXPRESS_API_KEY"]
} else {
    $null
}
$vertexOauthToken = if ($envMap.ContainsKey("VERTEX_OAUTH_ACCESS_TOKEN")) {
    $envMap["VERTEX_OAUTH_ACCESS_TOKEN"]
} elseif ($envMap.ContainsKey("GOOGLE_OAUTH_ACCESS_TOKEN")) {
    $envMap["GOOGLE_OAUTH_ACCESS_TOKEN"]
} else {
    $null
}
$vertexProjectId = if ($envMap.ContainsKey("VERTEX_PROJECT_ID")) {
    $envMap["VERTEX_PROJECT_ID"]
} elseif ($envMap.ContainsKey("GOOGLE_CLOUD_PROJECT")) {
    $envMap["GOOGLE_CLOUD_PROJECT"]
} elseif ($envMap.ContainsKey("GCP_PROJECT_ID")) {
    $envMap["GCP_PROJECT_ID"]
} else {
    $null
}

if ([string]::IsNullOrWhiteSpace($stage2Key)) {
    throw "GEMINI_API_KEY is missing in $DatasetEnvPath"
}
if ([string]::IsNullOrWhiteSpace($stage3Key)) {
    throw "GEMINI_IR_API_KEY or GEMINI_API_KEY_2 is missing in $DatasetEnvPath"
}
if ([string]::IsNullOrWhiteSpace($vertexExpressKey)) {
    throw "VERTEX_EXPRESS_API_KEY (or GEMINI_VERTEX_EXPRESS_API_KEY) is missing in $DatasetEnvPath"
}

$tmpFile = Join-Path $env:TEMP $OutputName
$lines = @(
    "GEMINI_STAGE2_API_KEY=$stage2Key"
    "GEMINI_RESPONSE_API_KEY=$stage2Key"
    "GEMINI_API_KEY=$stage2Key"
    "GEMINI_STAGE3_API_KEY=$stage3Key"
    "GEMINI_IR_API_KEY=$stage3Key"
    "GEMINI_API_KEY_2=$stage3Key"
    "VERTEX_EXPRESS_API_KEY=$vertexExpressKey"
    "GEMINI_VERTEX_EXPRESS_API_KEY=$vertexExpressKey"
)
if (-not [string]::IsNullOrWhiteSpace($vertexOauthToken)) {
    $lines += "VERTEX_OAUTH_ACCESS_TOKEN=$vertexOauthToken"
    $lines += "GOOGLE_OAUTH_ACCESS_TOKEN=$vertexOauthToken"
}
if (-not [string]::IsNullOrWhiteSpace($vertexProjectId)) {
    $lines += "VERTEX_PROJECT_ID=$vertexProjectId"
    $lines += "GOOGLE_CLOUD_PROJECT=$vertexProjectId"
    $lines += "GCP_PROJECT_ID=$vertexProjectId"
}
$lines | Set-Content -Path $tmpFile -NoNewline:$false -Encoding ascii

$devicePath = "/sdcard/Android/data/$PackageName/files/$OutputName"
adb push "$tmpFile" "$devicePath" | Out-Null
adb shell ls -l "$devicePath"

Remove-Item -Force "$tmpFile"
Write-Host "Pushed key file to $devicePath"
