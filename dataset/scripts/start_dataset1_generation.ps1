param(
    [int]$Stage3MinPending = 256,
    [int]$Target = 10000,
    [int]$PollSeconds = 300,
    [switch]$NoHourlyCommit
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$runDir = Join-Path $repoRoot "dataset\data\runs"

function Stop-ExistingDataset1Generation {
    $self = $PID
    $workers = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $self -and
        $_.Name -match "python" -and
        $_.CommandLine -match "dataset/scripts/(manage_gemini_dataset_workers|watch_stage1_queries|watch_stage2_for_queries|watch_stage3_for_responses)\.py"
    }
    foreach ($proc in $workers) {
        Stop-Process -Id $proc.ProcessId -Force
    }

    $committers = Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $self -and
        $_.Name -match "powershell|pwsh" -and
        $_.CommandLine -match "dataset/scripts/hourly_commit_generated_data\.ps1"
    }
    foreach ($proc in $committers) {
        Stop-Process -Id $proc.ProcessId -Force
    }
}

Set-Location $repoRoot
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

Stop-ExistingDataset1Generation
Start-Sleep -Seconds 2

$managerOut = Join-Path $runDir "gemini_worker_manager.out.log"
$managerErr = Join-Path $runDir "gemini_worker_manager.err.log"
$managerArgs = @(
    "dataset/scripts/manage_gemini_dataset_workers.py",
    "--runs", "dataset_v3,dataset_v1",
    "--target", "$Target",
    "--poll_seconds", "$PollSeconds",
    "--stage3_min_pending", "$Stage3MinPending"
)
$manager = Start-Process `
    -FilePath python `
    -ArgumentList $managerArgs `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden `
    -RedirectStandardOutput $managerOut `
    -RedirectStandardError $managerErr `
    -PassThru

Write-Output "dataset1 generation manager started pid=$($manager.Id)"

if (-not $NoHourlyCommit) {
    $committerScript = Join-Path $repoRoot "dataset\scripts\hourly_commit_generated_data.ps1"
    $committer = Start-Process `
        -FilePath powershell `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $committerScript, "-IntervalSeconds", "3600") `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden `
        -PassThru
    Write-Output "dataset1 generation hourly committer started pid=$($committer.Id)"
}
