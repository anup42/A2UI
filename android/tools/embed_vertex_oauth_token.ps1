param(
    [string]$ProjectId = "genui-486215",
    [string]$DatasetEnvPath = "..\..\dataset\.env",
    [string]$Token = "",
    [switch]$Install
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-InputPath([string]$path, [string]$baseDir) {
    if ([System.IO.Path]::IsPathRooted($path)) {
        return $path
    }
    return Join-Path $baseDir $path
}

function Upsert-DotEnvValue(
    [string[]]$lines,
    [string]$key,
    [string]$value
) {
    $escapedKey = [Regex]::Escape($key)
    $replacement = "$key=`"$value`""
    $updated = $false
    for ($i = 0; $i -lt $lines.Length; $i++) {
        if ($lines[$i] -match "^\s*(export\s+)?$escapedKey\s*=") {
            $lines[$i] = $replacement
            $updated = $true
            break
        }
    }
    if (-not $updated) {
        $lines += $replacement
    }
    return ,$lines
}

function Get-VertexAccessToken([string]$projectId) {
    $gcloudCmd = Get-Command gcloud -ErrorAction SilentlyContinue
    if (-not $gcloudCmd) {
        throw "gcloud is not installed. Install Google Cloud SDK and run this script again."
    }

    $activeProject = (& gcloud config get-value project 2>$null).Trim()
    if ($activeProject -ne $projectId) {
        Write-Host "Setting gcloud project to '$projectId'..."
        & gcloud config set project $projectId | Out-Null
    }

    $token = (& gcloud auth application-default print-access-token 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw @"
Could not mint OAuth token from ADC.
Run:
  gcloud auth application-default login --scopes=https://www.googleapis.com/auth/cloud-platform
Then rerun:
  android/tools/embed_vertex_oauth_token.ps1
"@
    }
    return $token
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$androidDir = Resolve-Path (Join-Path $scriptDir "..")
$repoRoot = Resolve-Path (Join-Path $androidDir "..")
$envPath = Resolve-InputPath -path $DatasetEnvPath -baseDir $scriptDir

if (-not (Test-Path $envPath)) {
    throw "dataset .env not found: $envPath"
}

$oauthToken = $Token.Trim()
if ([string]::IsNullOrWhiteSpace($oauthToken)) {
    $oauthToken = Get-VertexAccessToken -projectId $ProjectId
}

$lines = Get-Content -Path $envPath
$lines = Upsert-DotEnvValue -lines $lines -key "VERTEX_OAUTH_ACCESS_TOKEN" -value $oauthToken
$lines = Upsert-DotEnvValue -lines $lines -key "GOOGLE_OAUTH_ACCESS_TOKEN" -value $oauthToken

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($envPath, $lines, $utf8NoBom)

Write-Host "Updated OAuth token in: $envPath"

Push-Location $androidDir
try {
    $gradleTask = if ($Install) { ":app:installDebug" } else { ":app:assembleDebug" }
    Write-Host "Running gradlew $gradleTask ..."
    & .\gradlew $gradleTask
    if ($LASTEXITCODE -ne 0) {
        throw "Gradle task failed: $gradleTask"
    }
} finally {
    Pop-Location
}

Write-Host "Done. Vertex OAuth token is embedded through BuildConfig in the built APK."
