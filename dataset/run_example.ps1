$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (Test-Path .\\.env) {
  Get-Content .\\.env | ForEach-Object {
    if ($_ -match \"^\\s*#\") { return }
    if ($_ -match \"^\\s*$\") { return }
    $pair = $_.Split('=', 2)
    if ($pair.Length -eq 2) {
      $name = $pair[0].Trim()
      $value = $pair[1].Trim()
      if ($name) { [System.Environment]::SetEnvironmentVariable($name, $value, 'Process') }
    }
  }
}

if (-not $env:OPENAI_API_KEY -and -not $env:GEMINI_API_KEY) {
  Write-Host "Set OPENAI_API_KEY and/or GEMINI_API_KEY before running." -ForegroundColor Yellow
}

python .\src\main.py --stage 1 --model openai_gpt4o
python .\src\main.py --stage 2 --model openai_gpt4o
python .\src\main.py --stage 3 --model openai_gpt4o
