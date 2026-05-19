param(
    [string]$RepoRoot = "C:\Users\anupk\Documents\git\A2UI"
)

$ErrorActionPreference = "Stop"

function Count-JsonlLines {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path | Measure-Object -Line).Lines
}

function Write-WatchdogLog {
    param([string]$Message)
    $logPath = Join-Path $RepoRoot "dataset/.dataset_worker_watchdog.log"
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -LiteralPath $logPath -Value "$timestamp $Message"
}

function Test-ExpectedProcess {
    param(
        [int]$PidValue,
        [string]$ScriptName,
        [string]$RunId
    )
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $PidValue" -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        return $false
    }
    $commandLine = [string]$proc.CommandLine
    return ($commandLine -like "*$ScriptName*" -and $commandLine -like "*--run_id*" -and $commandLine -like "*$RunId*")
}

function Start-DatasetWorker {
    param([hashtable]$Worker)

    $python = (Get-Command python).Source
    $stdout = Join-Path $RepoRoot $Worker.Stdout
    $stderr = Join-Path $RepoRoot $Worker.Stderr
    $pidPath = Join-Path $RepoRoot $Worker.PidPath
    $args = $Worker.Args

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $args `
        -WorkingDirectory $RepoRoot `
        -PassThru `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr
    Set-Content -LiteralPath $pidPath -Value $process.Id
    Write-WatchdogLog "restarted name=$($Worker.Name) pid=$($process.Id)"
}

Set-Location -LiteralPath $RepoRoot

$workers = @(
    @{
        Name = "dataset_v1_stage2"
        RunId = "dataset_v1"
        Stage = "stage2"
        TargetPath = "dataset/data/runs/dataset_v1/responses.jsonl"
        Target = 20000
        PidPath = "dataset/.dataset_v1_stage2_pid.txt"
        ScriptName = "continue_stage2_dataset_v1.py"
        Stdout = "dataset/.dataset_v1_stage2_stdout.log"
        Stderr = "dataset/.dataset_v1_stage2_stderr.log"
        Args = @("dataset/scripts/continue_stage2_dataset_v1.py", "--run_id", "dataset_v1", "--target", "20000", "--model", "azure_gpt54_mini", "--pass_size", "32", "--query_chunk_size", "10000", "--stage1_per_intent_batch_size", "10")
    },
    @{
        Name = "dataset_v1_stage3_0"
        RunId = "dataset_v1"
        Stage = "stage3"
        TargetPath = "dataset/data/runs/dataset_v1/genui.jsonl"
        Target = 20000
        PidPath = "dataset/.dataset_v1_stage3_0_pid.txt"
        ScriptName = "watch_stage3_for_responses.py"
        Stdout = "dataset/.dataset_v1_stage3_0_stdout.log"
        Stderr = "dataset/.dataset_v1_stage3_0_stderr.log"
        Args = @("dataset/scripts/watch_stage3_for_responses.py", "--run_id", "dataset_v1", "--target", "20000", "--model", "azure_gpt54_mini", "--worker_index", "0", "--worker_count", "2", "--pass_size", "8", "--poll_seconds", "45", "--idle_checks", "5")
    },
    @{
        Name = "dataset_v1_stage3_1"
        RunId = "dataset_v1"
        Stage = "stage3"
        TargetPath = "dataset/data/runs/dataset_v1/genui.jsonl"
        Target = 20000
        PidPath = "dataset/.dataset_v1_stage3_1_pid.txt"
        ScriptName = "watch_stage3_for_responses.py"
        Stdout = "dataset/.dataset_v1_stage3_1_stdout.log"
        Stderr = "dataset/.dataset_v1_stage3_1_stderr.log"
        Args = @("dataset/scripts/watch_stage3_for_responses.py", "--run_id", "dataset_v1", "--target", "20000", "--model", "azure_gpt54_mini", "--worker_index", "1", "--worker_count", "2", "--pass_size", "8", "--poll_seconds", "45", "--idle_checks", "5")
    },
    @{
        Name = "dataset_v3_stage1"
        RunId = "dataset_v3"
        Stage = "stage1"
        TargetPath = "dataset/data/runs/dataset_v3/queries.jsonl"
        Target = 10000
        PidPath = "dataset/.dataset_v3_stage1_pid.txt"
        ScriptName = "watch_stage1_queries.py"
        Stdout = "dataset/.dataset_v3_stage1_stdout.log"
        Stderr = "dataset/.dataset_v3_stage1_stderr.log"
        Args = @("dataset/scripts/watch_stage1_queries.py", "--run_id", "dataset_v3", "--target", "10000", "--model", "azure_gpt54_mini", "--pass_size", "320", "--stage1_per_intent_batch_size", "10")
    },
    @{
        Name = "dataset_v3_stage2_0"
        RunId = "dataset_v3"
        Stage = "stage2"
        TargetPath = "dataset/data/runs/dataset_v3/responses.jsonl"
        Target = 10000
        PidPath = "dataset/.dataset_v3_stage2_0_pid.txt"
        ScriptName = "watch_stage2_for_queries.py"
        Stdout = "dataset/.dataset_v3_stage2_0_stdout.log"
        Stderr = "dataset/.dataset_v3_stage2_0_stderr.log"
        Args = @("dataset/scripts/watch_stage2_for_queries.py", "--run_id", "dataset_v3", "--target", "10000", "--model", "azure_gpt54_mini", "--worker_index", "0", "--worker_count", "2", "--pass_size", "8", "--poll_seconds", "30")
    },
    @{
        Name = "dataset_v3_stage2_1"
        RunId = "dataset_v3"
        Stage = "stage2"
        TargetPath = "dataset/data/runs/dataset_v3/responses.jsonl"
        Target = 10000
        PidPath = "dataset/.dataset_v3_stage2_1_pid.txt"
        ScriptName = "watch_stage2_for_queries.py"
        Stdout = "dataset/.dataset_v3_stage2_1_stdout.log"
        Stderr = "dataset/.dataset_v3_stage2_1_stderr.log"
        Args = @("dataset/scripts/watch_stage2_for_queries.py", "--run_id", "dataset_v3", "--target", "10000", "--model", "azure_gpt54_mini", "--worker_index", "1", "--worker_count", "2", "--pass_size", "8", "--poll_seconds", "30")
    },
    @{
        Name = "dataset_v3_stage3_0"
        RunId = "dataset_v3"
        Stage = "stage3"
        TargetPath = "dataset/data/runs/dataset_v3/genui.jsonl"
        Target = 10000
        PidPath = "dataset/.dataset_v3_stage3_0_pid.txt"
        ScriptName = "watch_stage3_for_responses.py"
        Stdout = "dataset/.dataset_v3_stage3_0_stdout.log"
        Stderr = "dataset/.dataset_v3_stage3_0_stderr.log"
        Args = @("dataset/scripts/watch_stage3_for_responses.py", "--run_id", "dataset_v3", "--target", "10000", "--model", "azure_gpt54_mini", "--worker_index", "0", "--worker_count", "2", "--pass_size", "4", "--poll_seconds", "30", "--idle_checks", "5")
    },
    @{
        Name = "dataset_v3_stage3_1"
        RunId = "dataset_v3"
        Stage = "stage3"
        TargetPath = "dataset/data/runs/dataset_v3/genui.jsonl"
        Target = 10000
        PidPath = "dataset/.dataset_v3_stage3_1_pid.txt"
        ScriptName = "watch_stage3_for_responses.py"
        Stdout = "dataset/.dataset_v3_stage3_1_stdout.log"
        Stderr = "dataset/.dataset_v3_stage3_1_stderr.log"
        Args = @("dataset/scripts/watch_stage3_for_responses.py", "--run_id", "dataset_v3", "--target", "10000", "--model", "azure_gpt54_mini", "--worker_index", "1", "--worker_count", "2", "--pass_size", "4", "--poll_seconds", "30", "--idle_checks", "5")
    }
)

foreach ($worker in $workers) {
    $targetPath = Join-Path $RepoRoot $worker.TargetPath
    $count = Count-JsonlLines $targetPath
    if ($count -ge [int]$worker.Target) {
        Write-WatchdogLog "skip-complete name=$($worker.Name) count=$count target=$($worker.Target)"
        continue
    }

    $pidPath = Join-Path $RepoRoot $worker.PidPath
    $running = $false
    if (Test-Path -LiteralPath $pidPath) {
        $pidText = (Get-Content -LiteralPath $pidPath -ErrorAction SilentlyContinue | Select-Object -First 1)
        $pidValue = 0
        if ([int]::TryParse([string]$pidText, [ref]$pidValue)) {
            $running = Test-ExpectedProcess -PidValue $pidValue -ScriptName $worker.ScriptName -RunId $worker.RunId
        }
    }

    if ($running) {
        Write-WatchdogLog "ok name=$($worker.Name) count=$count target=$($worker.Target)"
        continue
    }

    Write-WatchdogLog "missing name=$($worker.Name) count=$count target=$($worker.Target)"
    Start-DatasetWorker -Worker $worker
}
