param(
  [string]$RunGroup = ("azure_gpt54_reasoning32_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
  [int]$MaxParallel = 3,
  [string]$Genui001ApiKey = $env:AZURE_OPENAI_GENUI001_API_KEY,
  [string]$ProApiKey = $env:AZURE_OPENAI_PRO_API_KEY,
  [string]$Stage1Prompt = "prompts/query_gen_gemma_v3_diverse_openings.md",
  [string]$Stage3Prompt = "prompts/genui_gen_gemma_v12_structure_preserve.md"
)

$ErrorActionPreference = "Stop"

$datasetRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runsRoot = Join-Path $datasetRoot "data\runs"
$groupDir = Join-Path $runsRoot $RunGroup
New-Item -ItemType Directory -Force -Path $groupDir | Out-Null

if (-not $Genui001ApiKey) {
  throw "AZURE_OPENAI_GENUI001_API_KEY is required for gpt-5.4 and gpt-5.4-mini benchmark runs."
}
if (-not $ProApiKey) {
  throw "AZURE_OPENAI_PRO_API_KEY is required for gpt-5.4-pro benchmark runs."
}

$configs = @(
  @{ Name = "gpt54_mini_no_reasoning"; Model = "azure_gpt54_mini_benchmark"; ApiKey = $Genui001ApiKey; Reasoning = "" },
  @{ Name = "gpt54_mini_reasoning_medium"; Model = "azure_gpt54_mini_benchmark"; ApiKey = $Genui001ApiKey; Reasoning = "medium" },
  @{ Name = "gpt54_no_reasoning"; Model = "azure_gpt54_benchmark"; ApiKey = $Genui001ApiKey; Reasoning = "" },
  @{ Name = "gpt54_reasoning_medium"; Model = "azure_gpt54_benchmark"; ApiKey = $Genui001ApiKey; Reasoning = "medium" },
  @{ Name = "gpt54_pro_no_reasoning"; Model = "azure_gpt54_pro_benchmark"; ApiKey = $ProApiKey; Reasoning = "" },
  @{ Name = "gpt54_pro_reasoning_medium"; Model = "azure_gpt54_pro_benchmark"; ApiKey = $ProApiKey; Reasoning = "medium" }
)

$manifest = [ordered]@{
  run_group = $RunGroup
  created_at = (Get-Date).ToUniversalTime().ToString("o")
  stage1_prompt = $Stage1Prompt
  stage2_prompt = "prompts/response_gen.md"
  stage2_batch_prompt = "prompts/response_gen_batch.md"
  stage3_prompt = $Stage3Prompt
  target_per_stage = 32
  one_query_per_intent = $true
  max_parallel = $MaxParallel
  configs = @($configs | ForEach-Object {
    [ordered]@{
      name = $_.Name
      model = $_.Model
      reasoning_effort = $(if ($_.Reasoning) { $_.Reasoning } else { "none" })
      run_id = "$RunGroup/$($_.Name)"
    }
  })
}
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path (Join-Path $groupDir "benchmark_manifest.json")

function Start-BenchmarkJob($cfg) {
  $logPath = Join-Path $groupDir ($cfg.Name + ".log")
  $runId = "$RunGroup/$($cfg.Name)"
  Start-Job -Name $cfg.Name -ArgumentList @(
    $datasetRoot,
    $runId,
    $cfg.Model,
    $cfg.ApiKey,
    $cfg.Reasoning,
    $Stage1Prompt,
    $Stage3Prompt,
    $logPath
  ) -ScriptBlock {
    param($datasetRoot, $runId, $model, $apiKey, $reasoning, $stage1Prompt, $stage3Prompt, $logPath)
    $ErrorActionPreference = "Stop"
    Set-Location $datasetRoot

    $env:AZURE_OPENAI_API_KEY = $apiKey
    $env:AZURE_OPENAI_SUBSCRIPTION_KEY = $apiKey
    $env:AZURE_OPENAI_MAX_OUTPUT_TOKENS_CAP = "16384"
    $env:A2UI_AZURE_MAX_OUTPUT_TOKENS_CAP = "16384"
    $env:A2UI_QUERY_MAX_TOKENS = "4096"
    $env:A2UI_RESPONSE_MAX_TOKENS = "8192"
    $env:A2UI_GENUI_MAX_TOKENS = "16384"
    $env:A2UI_STAGE1_PROMPT_FILE = $stage1Prompt
    $env:A2UI_STAGE3_PROMPT_FILE = $stage3Prompt
    $env:STAGE3_PROMPT_MODE = "system_prefix"
    $env:A2UI_STAGE2_BATCH_SIZE = "16"
    $env:A2UI_STAGE2_RESPONSE_BATCH_SIZE = "1"
    $env:A2UI_STAGE3_BATCH_SIZE = "8"
    if ($reasoning) {
      $env:AZURE_OPENAI_REASONING_EFFORT = $reasoning
    } else {
      Remove-Item Env:\AZURE_OPENAI_REASONING_EFFORT -ErrorAction SilentlyContinue
    }

    function Run-Step([string]$label, [string[]]$argsList) {
      "[$(Get-Date -Format o)] START $label" | Add-Content -Encoding UTF8 -Path $logPath
      $quotedArgs = @("python") + ($argsList | ForEach-Object {
        '"' + ($_ -replace '"', '\"') + '"'
      })
      $cmdLine = ($quotedArgs -join " ") + " >> " + '"' + $logPath + '"' + " 2>&1"
      & cmd.exe /d /c $cmdLine
      if ($LASTEXITCODE -ne 0) {
        throw "$label failed with exit code $LASTEXITCODE"
      }
      "[$(Get-Date -Format o)] END $label" | Add-Content -Encoding UTF8 -Path $logPath
    }

    Run-Step "stage1" @(
      "src/main.py", "--stage", "1", "--model", $model, "--run_id", $runId,
      "--k_queries_per_intent", "1", "--max_queries_total", "32",
      "--stage1_batch_size", "1", "--rate_limit_qps", "0.5"
    )
    Run-Step "stage2" @(
      "src/main.py", "--stage", "2", "--model", $model, "--run_id", $runId,
      "--max_responses_total", "32", "--stage2_batch_size", "16",
      "--stage2_response_batch_size", "1", "--rate_limit_qps", "0.5"
    )
    Run-Step "stage3" @(
      "src/main.py", "--stage", "3", "--model", $model, "--run_id", $runId,
      "--max_genui_total", "32", "--genui_batch_size", "8", "--rate_limit_qps", "0.5"
    )
    Run-Step "recompute_aggregates" @("scripts/recompute_aggregates_stream.py", "--run-id", $runId)
  }
}

$jobs = @()
foreach ($cfg in $configs) {
  while ((Get-Job -State Running).Count -ge $MaxParallel) {
    Start-Sleep -Seconds 10
    Get-Job | Where-Object { $_.State -in @("Completed", "Failed", "Stopped") -and $jobs.Id -contains $_.Id } | ForEach-Object {
      Receive-Job $_ -Keep -ErrorAction SilentlyContinue | Out-Null
    }
  }
  $jobs += Start-BenchmarkJob $cfg
  Write-Host "Started $($cfg.Name)"
}

while ((Get-Job | Where-Object { $jobs.Id -contains $_.Id -and $_.State -eq "Running" }).Count -gt 0) {
  $running = (Get-Job | Where-Object { $jobs.Id -contains $_.Id -and $_.State -eq "Running" }).Name -join ", "
  Write-Host "Running: $running"
  Start-Sleep -Seconds 30
}

$failed = @()
foreach ($job in $jobs) {
  Receive-Job $job -Keep -ErrorAction SilentlyContinue | Out-Null
  if ($job.State -ne "Completed") {
    $failed += $job.Name
  }
}

$summaryRows = @()
foreach ($cfg in $configs) {
  $runDir = Join-Path $runsRoot "$RunGroup\$($cfg.Name)"
  $aggPath = Join-Path $runDir "aggregates.json"
  $queriesPath = Join-Path $runDir "queries.jsonl"
  $responsesPath = Join-Path $runDir "responses.jsonl"
  $genuiPath = Join-Path $runDir "genui.jsonl"
  $agg = $null
  if (Test-Path $aggPath) {
    $agg = Get-Content $aggPath -Raw | ConvertFrom-Json
  }
  $summaryRows += [pscustomobject]@{
    config = $cfg.Name
    model = $cfg.Model
    reasoning = $(if ($cfg.Reasoning) { $cfg.Reasoning } else { "none" })
    run_id = "$RunGroup/$($cfg.Name)"
    query_count = $(if (Test-Path $queriesPath) { (Get-Content $queriesPath).Count } else { 0 })
    response_count = $(if (Test-Path $responsesPath) { (Get-Content $responsesPath).Count } else { 0 })
    ir_count = $(if (Test-Path $genuiPath) { (Get-Content $genuiPath).Count } else { 0 })
    overall_score = $(if ($agg) { [math]::Round([double]$agg.overall_score, 4) } else { $null })
    schema_valid_strict_rate = $(if ($agg) { [math]::Round([double]$agg.schema_valid_strict_rate, 4) } else { $null })
    content_coverage_avg = $(if ($agg) { [math]::Round([double]$agg.content_coverage_avg, 4) } else { $null })
    intent_score_avg = $(if ($agg) { [math]::Round([double]$agg.intent_score_avg, 4) } else { $null })
  }
}

$summaryCsv = Join-Path $groupDir "summary.csv"
$summaryJson = Join-Path $groupDir "summary.json"
$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryCsv
$summaryRows | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -Path $summaryJson

Write-Host "Summary: $summaryCsv"
$summaryRows | Format-Table -AutoSize

if ($failed.Count -gt 0) {
  throw "Benchmark jobs failed: $($failed -join ', ')"
}
