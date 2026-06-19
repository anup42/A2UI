param(
  [string]$RunGroup = ("dataset_gpt54_no_reasoning_p5_" + (Get-Date -Format "yyyyMMdd_HHmmss")),
  [int]$Shards = 5,
  [int]$PerShardTarget = 4000,
  [int]$Stage1Chunk = 1000,
  [int]$Stage2Chunk = 1000,
  [int]$Stage3Chunk = 2000,
  [string]$Model = "azure_gpt54_benchmark",
  [string]$BaseStage1Prompt = "prompts/query_gen_gemma_v3_diverse_openings.md",
  [string]$Stage3Prompt = "prompts/genui_gen_gemma_v12_structure_preserve.md"
)

$ErrorActionPreference = "Stop"

$datasetRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$runsRoot = Join-Path $datasetRoot "data\runs"
$groupDir = Join-Path $runsRoot $RunGroup
$promptDir = Join-Path $groupDir "_prompt_variants"
$cycleScript = Join-Path $datasetRoot "scripts\run_gpt54_no_reasoning_cycle20k.ps1"

New-Item -ItemType Directory -Force -Path $groupDir | Out-Null
New-Item -ItemType Directory -Force -Path $promptDir | Out-Null

if (-not $env:AZURE_OPENAI_API_KEY -and -not $env:AZURE_OPENAI_SUBSCRIPTION_KEY) {
  throw "AZURE_OPENAI_API_KEY or AZURE_OPENAI_SUBSCRIPTION_KEY must be set before running."
}

$env:AZURE_OPENAI_SUBSCRIPTION_KEY = if ($env:AZURE_OPENAI_SUBSCRIPTION_KEY) { $env:AZURE_OPENAI_SUBSCRIPTION_KEY } else { $env:AZURE_OPENAI_API_KEY }
Remove-Item Env:\AZURE_OPENAI_REASONING_EFFORT -ErrorAction SilentlyContinue
Remove-Item Env:\OPENAI_REASONING_EFFORT -ErrorAction SilentlyContinue
$env:A2UI_ENABLE_PROMPT_CACHE = "0"

$basePromptPath = [System.IO.Path]::GetFullPath((Join-Path $datasetRoot $BaseStage1Prompt))
$basePrompt = Get-Content -Raw -Encoding UTF8 -Path $basePromptPath
$started = @()

for ($i = 1; $i -le $Shards; $i++) {
  $shardLabel = "{0:D2}" -f $i
  $runId = "${RunGroup}_part${shardLabel}"
  $variantPath = Join-Path $promptDir ("query_gen_shard_{0}.md" -f $shardLabel)
  $variantDirective = @"

## Dataset shard diversity directive
You are generating shard $i of $Shards for the same overall dataset.
Make this shard intentionally different from the other shards:
- Prefer different entities, locations, brands, constraints, time ranges, roles, and user contexts.
- Avoid generic or repeated wording.
- Keep the same JSON output contract as above.
- Do not mention shard numbers in generated queries.
"@
  Set-Content -Path $variantPath -Value ($basePrompt + $variantDirective) -Encoding UTF8

  $argsList = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    $cycleScript,
    "-RunId",
    $runId,
    "-Target",
    [string]$PerShardTarget,
    "-Stage1Chunk",
    [string]$Stage1Chunk,
    "-Stage2Chunk",
    [string]$Stage2Chunk,
    "-Stage3Chunk",
    [string]$Stage3Chunk,
    "-Model",
    $Model,
    "-Stage1Prompt",
    $variantPath,
    "-Stage3Prompt",
    $Stage3Prompt
  )
  $proc = Start-Process -FilePath powershell.exe `
    -ArgumentList $argsList `
    -WindowStyle Hidden `
    -PassThru `
    -WorkingDirectory $datasetRoot

  $runDir = Join-Path $runsRoot $runId
  New-Item -ItemType Directory -Force -Path $runDir | Out-Null
  Set-Content -Path (Join-Path $runDir "cycle_generation.pid") -Value $proc.Id -Encoding ASCII

  $started += [ordered]@{
    shard = $i
    run_id = $runId
    pid = $proc.Id
    target = $PerShardTarget
    stage1_prompt = $variantPath
    started_at = (Get-Date).ToUniversalTime().ToString("o")
  }
}

$manifest = [ordered]@{
  run_group = $RunGroup
  created_at = (Get-Date).ToUniversalTime().ToString("o")
  model = $Model
  reasoning = "none"
  prompt_cache = "disabled"
  shards = $Shards
  per_shard_target = $PerShardTarget
  total_target = $Shards * $PerShardTarget
  chunks = [ordered]@{
    stage1 = $Stage1Chunk
    stage2 = $Stage2Chunk
    stage3 = $Stage3Chunk
  }
  base_stage1_prompt = $BaseStage1Prompt
  stage3_prompt = $Stage3Prompt
  runs = $started
}

$manifestPath = Join-Path $groupDir "parallel_manifest.json"
$manifest | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 -Path $manifestPath

Write-Host "Started $Shards GPT-5.4 no-reasoning generation shards."
Write-Host "Manifest: $manifestPath"
$started | ForEach-Object {
  Write-Host ("{0}: pid={1} target={2}" -f $_.run_id, $_.pid, $_.target)
}
