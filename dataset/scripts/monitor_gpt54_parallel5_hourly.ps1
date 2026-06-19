param(
  [string]$RunGroup = "dataset_gpt54_no_reasoning_p5_20260619",
  [int]$Shards = 5,
  [int]$PerShardTarget = 4000,
  [int]$IntervalSeconds = 3600,
  [switch]$Once,
  [switch]$StopWhenComplete
)

$ErrorActionPreference = "Stop"
$script = Join-Path $PSScriptRoot "monitor_gpt54_parallel5_hourly.py"
$argsList = @(
  $script,
  "--run-group", $RunGroup,
  "--shards", [string]$Shards,
  "--per-shard-target", [string]$PerShardTarget,
  "--interval-seconds", [string]$IntervalSeconds
)
if ($Once) {
  $argsList += "--once"
}
if ($StopWhenComplete) {
  $argsList += "--stop-when-complete"
}
python @argsList
