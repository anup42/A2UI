param(
    [string]$RepoRoot = "C:\Users\anupk\Documents\git\A2UI",
    [string]$RunPath = "dataset/data/runs/dataset_v1"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$logPath = Join-Path $RepoRoot "dataset/.dataset_auto_commit.log"
function Write-Log([string]$message) {
    Add-Content -LiteralPath $logPath -Value "[$timestamp] $message"
}

try {
    if (!(Test-Path -LiteralPath $RunPath)) {
        Write-Log "Run path missing: $RunPath"
        exit 0
    }

    $branch = (& git rev-parse --abbrev-ref HEAD).Trim()
    & git add -- $RunPath

    & git diff --cached --quiet -- $RunPath
    if ($LASTEXITCODE -eq 0) {
        Write-Log "No generated data changes to commit."
        exit 0
    }

    $queryCount = if (Test-Path "$RunPath/queries.jsonl") { (Get-Content "$RunPath/queries.jsonl" | Measure-Object -Line).Lines } else { 0 }
    $responseCount = if (Test-Path "$RunPath/responses.jsonl") { (Get-Content "$RunPath/responses.jsonl" | Measure-Object -Line).Lines } else { 0 }
    $irCount = if (Test-Path "$RunPath/genui.jsonl") { (Get-Content "$RunPath/genui.jsonl" | Measure-Object -Line).Lines } else { 0 }

    $message = "Auto snapshot dataset_v1 q=$queryCount r=$responseCount ir=$irCount"
    & git commit -m $message
    if ($LASTEXITCODE -ne 0) {
        Write-Log "git commit failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    & git push origin $branch
    if ($LASTEXITCODE -ne 0) {
        Write-Log "git push failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Log "Committed and pushed: $message on $branch"
} catch {
    Write-Log "Failed: $($_.Exception.Message)"
    exit 1
}
