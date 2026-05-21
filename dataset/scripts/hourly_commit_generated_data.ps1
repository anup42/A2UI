param(
    [int]$IntervalSeconds = 3600,
    [switch]$RunOnce
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$logPath = Join-Path $repoRoot "dataset\.hourly_generated_commit.log"
$lockPath = Join-Path $repoRoot "dataset\.hourly_generated_commit.lock"
$pidPath = Join-Path $repoRoot "dataset\.hourly_generated_commit.pid"

function Write-CommitLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $logPath -Value $line
}

function Invoke-Git {
    param([string[]]$GitArgs)
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $output = & git @GitArgs 2>&1
        $exit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exit -ne 0) {
        throw "git $($GitArgs -join ' ') failed ($exit): $($output -join [Environment]::NewLine)"
    }
    return $output
}

function Get-FailedIndexPath {
    param([string]$GitOutput)

    $patterns = @(
        "unable to index file '([^']+)'",
        "short read while indexing ([^\r\n]+)"
    )
    foreach ($pattern in $patterns) {
        $match = [regex]::Match($GitOutput, $pattern)
        if ($match.Success) {
            return $match.Groups[1].Value.Trim()
        }
    }
    return $null
}

function Add-GeneratedRunFiles {
    $excluded = New-Object System.Collections.Generic.List[string]

    for ($attempt = 1; $attempt -le 12; $attempt++) {
        $gitAddArgs = @(
            "add",
            "-A",
            "--",
            "dataset/data/runs",
            ":(exclude)dataset/data/runs/*/artifacts/error_*.json"
        )
        foreach ($path in $excluded) {
            $gitAddArgs += ":(exclude)$path"
        }

        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $output = & git @gitAddArgs 2>&1
            $exit = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }

        if ($exit -eq 0) {
            if ($excluded.Count -gt 0) {
                Write-CommitLog "git add skipped active writer files: $($excluded -join ', ')"
            }
            return
        }

        $joined = $output -join [Environment]::NewLine
        $badPath = Get-FailedIndexPath $joined
        if ([string]::IsNullOrWhiteSpace($badPath) -or $excluded.Contains($badPath)) {
            throw "git add failed: $joined"
        }

        Write-CommitLog "git add hit active writer file, skipping for this snapshot: $badPath"
        $excluded.Add($badPath) | Out-Null
        Start-Sleep -Seconds 2
    }

    throw "git add failed after excluding active writer files: $($excluded -join ', ')"
}

function Get-JsonlCount {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }

    $count = 0
    $reader = [System.IO.File]::OpenText($Path)
    try {
        while ($null -ne $reader.ReadLine()) {
            $count++
        }
    } finally {
        $reader.Dispose()
    }
    return $count
}

function Get-RunSummary {
    param([string]$RunName)
    $runDir = Join-Path $repoRoot ("dataset\data\runs\{0}" -f $RunName)
    if (-not (Test-Path -LiteralPath $runDir)) {
        return "{0}: missing" -f $RunName
    }

    $queries = Get-JsonlCount (Join-Path $runDir "queries.jsonl")
    $responses = Get-JsonlCount (Join-Path $runDir "responses.jsonl")
    $irs = Get-JsonlCount (Join-Path $runDir "genui.jsonl")
    return "{0}: q={1} r={2} ir={3}" -f $RunName, $queries, $responses, $irs
}

function Unstage-ForbiddenGeneratedFiles {
    $forbidden = @(
        "^dataset/data/cache/",
        "^presentation_work/",
        "/run\.log$",
        "\.prompt_cache",
        "/progress_stage[123]",
        "/\.stage[123]_worker",
        "/vertex_batch_stage3/",
        "/artifacts/error_.*\.json$",
        "^dataset/\.dataset_",
        "^dataset/\.hourly_generated_commit"
    )

    $staged = & git diff --cached --name-only
    foreach ($file in $staged) {
        foreach ($pattern in $forbidden) {
            if ($file -match $pattern) {
                & git restore --staged -- $file | Out-Null
                Write-CommitLog "unstaged ignored/scratch file: $file"
                break
            }
        }
    }
}

function Invoke-GeneratedDataSnapshot {
    Set-Location $repoRoot

    if (Test-Path -LiteralPath $lockPath) {
        $ageMinutes = ((Get-Date) - (Get-Item -LiteralPath $lockPath).LastWriteTime).TotalMinutes
        if ($ageMinutes -lt 90) {
            Write-CommitLog "snapshot skipped; lock file exists ($([math]::Round($ageMinutes, 1)) minutes old)"
            return
        }
        Write-CommitLog "removing stale lock file ($([math]::Round($ageMinutes, 1)) minutes old)"
        Remove-Item -LiteralPath $lockPath -Force
    }

    Set-Content -LiteralPath $lockPath -Value ("pid={0}`nstarted={1:o}" -f $PID, (Get-Date))
    try {
        Write-CommitLog "snapshot started"

        Add-GeneratedRunFiles

        Unstage-ForbiddenGeneratedFiles

        $staged = & git diff --cached --name-only
        if (-not $staged) {
            Write-CommitLog "nothing staged for generated data snapshot"
            return
        }

        $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $summaries = @()
        foreach ($runName in @("dataset_v0", "dataset_v1", "dataset_v2", "dataset_v3")) {
            $summaries += Get-RunSummary $runName
        }

        $message = "Hourly generated dataset snapshot $timestamp"
        $body = ($summaries -join [Environment]::NewLine)
        Invoke-Git -GitArgs @("commit", "-m", $message, "-m", $body) | Out-Null
        Write-CommitLog "committed: $message; $($summaries -join '; ')"

        Invoke-Git -GitArgs @("push", "origin", "HEAD") | Out-Null
        Write-CommitLog "pushed snapshot commit"
    } catch {
        Write-CommitLog "snapshot failed: $($_.Exception.Message)"
        throw
    } finally {
        if (Test-Path -LiteralPath $lockPath) {
            Remove-Item -LiteralPath $lockPath -Force
        }
    }
}

Set-Content -LiteralPath $pidPath -Value $PID
Write-CommitLog "hourly generated-data committer started; pid=$PID intervalSeconds=$IntervalSeconds runOnce=$RunOnce"

if ($RunOnce) {
    Invoke-GeneratedDataSnapshot
    exit 0
}

while ($true) {
    Invoke-GeneratedDataSnapshot
    Start-Sleep -Seconds $IntervalSeconds
}
