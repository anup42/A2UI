param(
  [string]$RunId = "dataset_gpt54_no_reasoning_20k_20260619",
  [int]$Target = 20000,
  [int]$Stage1Chunk = 1000,
  [int]$Stage2Chunk = 1000,
  [int]$Stage3Chunk = 1000,
  [switch]$InterleaveResponseIr,
  [int]$ResponseIrChunk = 1,
  [string]$Model = "azure_gpt54_benchmark",
  [string]$Stage1Prompt = "prompts/query_gen_gemma_v3_diverse_openings.md",
  [string]$Stage3Prompt = "prompts/genui_gen_gemma_v12_structure_preserve.md",
  [double]$RateLimitQps = 0.5,
  [int]$Stage1BatchSize = 16,
  [int]$Stage2QueryBatchSize = 16,
  [int]$Stage2ResponseBatchSize = 1,
  [int]$Stage3BatchSize = 8
)

$ErrorActionPreference = "Stop"

$datasetRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $datasetRoot

$runDir = Join-Path $datasetRoot "data\runs\$RunId"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$cycleLog = Join-Path $runDir "cycle_generation.log"
$progressPath = Join-Path $runDir "cycle_progress.json"

function Write-Log([string]$Message) {
  $line = "[{0}] {1}" -f (Get-Date -Format o), $Message
  Write-Output $line
  Add-Content -Path $cycleLog -Value $line -Encoding UTF8
}

function Count-Jsonl([string]$Path) {
  if (-not (Test-Path $Path)) {
    return 0
  }
  return ([System.IO.File]::ReadLines($Path) | Measure-Object).Count
}

function Count-Intents() {
  $intentsPath = Join-Path $datasetRoot "intents.info"
  return ([System.IO.File]::ReadLines($intentsPath) | Where-Object { $_.Trim() }).Count
}

function Write-ProgressJson([int]$Cycle, [int]$Queries, [int]$Responses, [int]$Genui) {
  $payload = [ordered]@{
    run_id = $RunId
    model = $Model
    reasoning = "none"
    target = $Target
    cycle = $Cycle
    counts = [ordered]@{
      queries = $Queries
      responses = $Responses
      genui = $Genui
    }
    chunks = [ordered]@{
      stage1 = $Stage1Chunk
      stage2 = $Stage2Chunk
      stage3 = $Stage3Chunk
      response_ir = $ResponseIrChunk
    }
    mode = if ($InterleaveResponseIr) { "interleave_response_ir" } else { "stage_chunks" }
    updated_at = (Get-Date).ToUniversalTime().ToString("o")
  }
  $payload | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 -Path $progressPath
}

function Next-ChunkCreate([int]$Current, [int]$Chunk, [int]$Limit) {
  if ($Current -ge $Limit) {
    return 0
  }
  $chunkIndex = [Math]::Floor($Current / [Math]::Max(1, $Chunk)) + 1
  $nextBoundary = [Math]::Min($Limit, [int]($chunkIndex * $Chunk))
  return [Math]::Max(0, $nextBoundary - $Current)
}

function Invoke-PythonStep([string]$Label, [string[]]$ArgsList) {
  $stepLog = Join-Path $runDir ("{0}.output.log" -f $Label)
  Write-Log "START $Label output_log=$stepLog args=$($ArgsList -join ' ')"
  $quotedArgs = @("python") + ($ArgsList | ForEach-Object {
    '"' + ($_ -replace '"', '\"') + '"'
  })
  $cmdLine = ($quotedArgs -join " ") + " >> " + '"' + $stepLog + '"' + " 2>&1"
  & cmd.exe /d /c $cmdLine
  if ($LASTEXITCODE -ne 0) {
    throw "$Label failed with exit code $LASTEXITCODE. See $stepLog"
  }
  Write-Log "END $Label"
}

if (-not $env:AZURE_OPENAI_API_KEY -and -not $env:AZURE_OPENAI_SUBSCRIPTION_KEY) {
  throw "AZURE_OPENAI_API_KEY or AZURE_OPENAI_SUBSCRIPTION_KEY must be set before running."
}

$env:AZURE_OPENAI_SUBSCRIPTION_KEY = if ($env:AZURE_OPENAI_SUBSCRIPTION_KEY) { $env:AZURE_OPENAI_SUBSCRIPTION_KEY } else { $env:AZURE_OPENAI_API_KEY }
Remove-Item Env:\AZURE_OPENAI_REASONING_EFFORT -ErrorAction SilentlyContinue
Remove-Item Env:\OPENAI_REASONING_EFFORT -ErrorAction SilentlyContinue
$env:A2UI_ENABLE_PROMPT_CACHE = "0"
$env:A2UI_STAGE1_PROMPT_FILE = $Stage1Prompt
$env:A2UI_STAGE3_PROMPT_FILE = $Stage3Prompt
$env:STAGE3_PROMPT_MODE = "system_prefix"
$env:A2UI_QUERY_MAX_TOKENS = "4096"
$env:A2UI_RESPONSE_MAX_TOKENS = "8192"
$env:A2UI_GENUI_MAX_TOKENS = "16384"
$env:AZURE_OPENAI_MAX_OUTPUT_TOKENS_CAP = "16384"
$env:A2UI_AZURE_MAX_OUTPUT_TOKENS_CAP = "16384"
$env:A2UI_STAGE2_BATCH_SIZE = [string]$Stage2QueryBatchSize
$env:A2UI_STAGE2_RESPONSE_BATCH_SIZE = [string]$Stage2ResponseBatchSize
$env:A2UI_STAGE3_BATCH_SIZE = [string]$Stage3BatchSize

$intentCount = [Math]::Max(1, (Count-Intents))
$kPerIntent = [int][Math]::Ceiling($Target / $intentCount) + 2

Write-Log "Cycle generation started run_id=$RunId model=$Model target=$Target intent_count=$intentCount k_per_intent=$kPerIntent"
Write-Log "Prompts stage1=$Stage1Prompt stage2=prompts/response_gen.md stage3=$Stage3Prompt prompt_cache=disabled reasoning=none mode=$(if ($InterleaveResponseIr) { 'interleave_response_ir' } else { 'stage_chunks' }) response_ir_chunk=$ResponseIrChunk"

$cycle = 0
while ($true) {
  $queriesPath = Join-Path $runDir "queries.jsonl"
  $responsesPath = Join-Path $runDir "responses.jsonl"
  $genuiPath = Join-Path $runDir "genui.jsonl"
  $queries = Count-Jsonl $queriesPath
  $responses = Count-Jsonl $responsesPath
  $genui = Count-Jsonl $genuiPath
  Write-ProgressJson -Cycle $cycle -Queries $queries -Responses $responses -Genui $genui

  if ($queries -ge $Target -and $responses -ge $Target -and $genui -ge $Target) {
    Write-Log "Target complete queries=$queries responses=$responses genui=$genui"
    Invoke-PythonStep "recompute_aggregates" @("scripts/recompute_aggregates_stream.py", "--run-id", $RunId)
    break
  }

  $cycle += 1
  Write-Log "CYCLE $cycle counts before queries=$queries responses=$responses genui=$genui"

  $canCreateQueries = $queries -lt $Target
  if ($InterleaveResponseIr -and ($responses -lt $queries -or $genui -lt $responses)) {
    $canCreateQueries = $false
    Write-Log "Interleave mode holding Stage1 until catch-up queries=$queries responses=$responses genui=$genui"
  }
  if ($canCreateQueries) {
    $create = Next-ChunkCreate -Current $queries -Chunk $Stage1Chunk -Limit $Target
    Invoke-PythonStep "stage1_cycle_$cycle" @(
      "src/main.py", "--stage", "1", "--model", $Model, "--run_id", $RunId,
      "--k_queries_per_intent", [string]$kPerIntent,
      "--max_queries_total", [string]$create,
      "--stage1_batch_size", [string]$Stage1BatchSize,
      "--rate_limit_qps", [string]$RateLimitQps
    )
  }

  $queries = Count-Jsonl $queriesPath
  $responses = Count-Jsonl $responsesPath
  if ($InterleaveResponseIr) {
    while ($responses -lt $Target -and $responses -lt $queries) {
      $beforeResponses = $responses
      $beforeGenui = Count-Jsonl $genuiPath
      $createResponses = [Math]::Min([Math]::Max(1, $ResponseIrChunk), $Target - $responses)
      $createResponses = [Math]::Min($createResponses, $queries - $responses)
      if ($createResponses -le 0) {
        break
      }
      Invoke-PythonStep "stage2_cycle_${cycle}_from_${beforeResponses}" @(
        "src/main.py", "--stage", "2", "--model", $Model, "--run_id", $RunId,
        "--max_responses_total", [string]$createResponses,
        "--stage2_batch_size", [string]$Stage2QueryBatchSize,
        "--stage2_response_batch_size", [string]$Stage2ResponseBatchSize,
        "--rate_limit_qps", [string]$RateLimitQps
      )
      $responses = Count-Jsonl $responsesPath
      if ($responses -le $beforeResponses) {
        Write-Log "Interleave mode made no Stage2 progress responses=$responses; breaking to avoid spin"
        break
      }

      $genui = Count-Jsonl $genuiPath
      while ($genui -lt $Target -and $genui -lt $responses) {
        $createGenui = [Math]::Min([Math]::Max(1, $ResponseIrChunk), $Target - $genui)
        $createGenui = [Math]::Min($createGenui, $responses - $genui)
        if ($createGenui -le 0) {
          break
        }
        Invoke-PythonStep "stage3_cycle_${cycle}_from_${beforeGenui}" @(
          "src/main.py", "--stage", "3", "--model", $Model, "--run_id", $RunId,
          "--max_genui_total", [string]$createGenui,
          "--genui_batch_size", [string]$Stage3BatchSize,
          "--rate_limit_qps", [string]$RateLimitQps
        )
        $newGenui = Count-Jsonl $genuiPath
        if ($newGenui -le $genui) {
          Write-Log "Interleave mode made no Stage3 progress genui=$genui responses=$responses; breaking to avoid spin"
          break
        }
        $genui = $newGenui
        $beforeGenui = $genui
      }
    }
  } else {
    if ($responses -lt $Target -and $responses -lt $queries) {
      $create = Next-ChunkCreate -Current $responses -Chunk $Stage2Chunk -Limit $Target
      $create = [Math]::Min($create, $queries - $responses)
      if ($create -gt 0) {
        Invoke-PythonStep "stage2_cycle_$cycle" @(
          "src/main.py", "--stage", "2", "--model", $Model, "--run_id", $RunId,
          "--max_responses_total", [string]$create,
          "--stage2_batch_size", [string]$Stage2QueryBatchSize,
          "--stage2_response_batch_size", [string]$Stage2ResponseBatchSize,
          "--rate_limit_qps", [string]$RateLimitQps
        )
      }
    }

    $responses = Count-Jsonl $responsesPath
    $genui = Count-Jsonl $genuiPath
    if ($genui -lt $Target -and $genui -lt $responses) {
      $create = Next-ChunkCreate -Current $genui -Chunk $Stage3Chunk -Limit $Target
      $create = [Math]::Min($create, $responses - $genui)
      if ($create -gt 0) {
        Invoke-PythonStep "stage3_cycle_$cycle" @(
          "src/main.py", "--stage", "3", "--model", $Model, "--run_id", $RunId,
          "--max_genui_total", [string]$create,
          "--genui_batch_size", [string]$Stage3BatchSize,
          "--rate_limit_qps", [string]$RateLimitQps
        )
      }
    }
  }

  $queries = Count-Jsonl $queriesPath
  $responses = Count-Jsonl $responsesPath
  $genui = Count-Jsonl $genuiPath
  Write-ProgressJson -Cycle $cycle -Queries $queries -Responses $responses -Genui $genui
  Write-Log "CYCLE $cycle counts after queries=$queries responses=$responses genui=$genui"
}
