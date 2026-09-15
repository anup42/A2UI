param([switch]$Execute)

$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..'))
$DatasetRoot = (Resolve-Path -LiteralPath (Join-Path $RepoRoot 'training/outputs/datasets')).Path
$Names = @(
    'full_data_archive_recovered_v1.partial-357700',
    'full_data_archive_recovered_v1.partial-369348',
    'full_data_archive_recovered_v1.partial-381088',
    'full_data_archive_recovered_v1.partial-392360',
    'full_data_archive_recovered_v1.partial-397768',
    'full_data_archive_recovered_v1.partial-400064',
    'full_data_archive_recovered_v1.partial-400276',
    'full_data_archive_recovered_v1.partial-406400',
    'full_data_archive_recovered_v1.partial-414540',
    'full_data_archive_recovered_v1.partial-418348',
    'full_data_archive_recovered_v1.partial-419444',
    'full_data_archive_recovered_v9.partial-385612'
)
$AllowedFiles = @('train.jsonl', 'val.jsonl', 'decisions.csv', 'quarantine.csv')
$EvidencePath = Join-Path $RepoRoot 'training/outputs/audits/temp_cleanup_20260913/cleanup.json'
if ($Execute -and (Test-Path -LiteralPath $EvidencePath)) {
    throw "Cleanup evidence already exists; inspect before any retry: $EvidencePath"
}

function Get-ExclusiveDigest([string]$Path) {
    $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
    $Hasher = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($Hasher.ComputeHash($Stream)).Replace('-', '').ToLowerInvariant() }
    finally { $Hasher.Dispose(); $Stream.Dispose() }
}

function Get-Digest([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-FreeBytes {
    return [IO.DriveInfo]::new([IO.Path]::GetPathRoot($RepoRoot)).AvailableFreeSpace
}

# Pin the final copy, the original source files, and frozen benchmarks.
$FinalDir = Join-Path $DatasetRoot 'full_data_archive_recovered_v9'
$ManifestPath = Join-Path $FinalDir 'manifest.json'
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
$Verified = Get-Content -LiteralPath (Join-Path $RepoRoot 'training/reports/offline_recovery_20260913_v9/verification.json') -Raw | ConvertFrom-Json
if ($Manifest.status -ne 'candidate_export_complete' -or $Verified.status -ne 'verified') {
    throw 'The retained final dataset must be complete and verified'
}
$Protected = [ordered]@{}
$Protected[$ManifestPath] = $Verified.manifest_sha256
foreach ($Item in $Manifest.outputs.PSObject.Properties) { $Protected[(Join-Path $FinalDir $Item.Name)] = $Item.Value }
foreach ($Item in $Manifest.source_files.PSObject.Properties) { $Protected[$Item.Value.path] = $Item.Value.sha256 }
foreach ($Item in $Manifest.benchmark_file_sha256.PSObject.Properties) { $Protected[(Join-Path $RepoRoot $Item.Name)] = $Item.Value }
foreach ($Path in @($Protected.Keys)) {
    if ((Get-Digest $Path) -cne $Protected[$Path]) { throw "Protected file differs from its verified hash: $Path" }
}

$Targets = @()
foreach ($Name in $Names) {
    $Expected = [IO.Path]::GetFullPath((Join-Path $DatasetRoot $Name))
    $Resolved = (Resolve-Path -LiteralPath $Expected).Path
    if ($Resolved -cne $Expected -or (Split-Path -Parent $Resolved) -cne $DatasetRoot) {
        throw "Target is not the exact reviewed child: $Resolved"
    }
    $Directory = Get-Item -LiteralPath $Resolved -Force
    if (-not $Directory.PSIsContainer -or ($Directory.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing non-directory or reparse point: $Resolved"
    }
    $Relative = 'training/outputs/datasets/' + $Name
    $Tracked = @(& git -C $RepoRoot -c core.excludesFile=NUL ls-files -- $Relative)
    if ($LASTEXITCODE -ne 0 -or $Tracked.Count) { throw "Target is Git-tracked or Git check failed: $Resolved" }
    $Files = @()
    foreach ($File in @(Get-ChildItem -LiteralPath $Resolved -Force)) {
        if ($File.PSIsContainer -or $File.Name -notin $AllowedFiles -or ($File.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Unexpected content in failed export: $($File.FullName)"
        }
        if ($File.LastWriteTimeUtc -gt [DateTime]::UtcNow.AddMinutes(-30)) {
            throw "Refusing a recently written export: $($File.FullName)"
        }
        $Files += [pscustomobject][ordered]@{ path=$File.FullName; bytes=$File.Length; last_write_utc=$File.LastWriteTimeUtc.ToString('o'); sha256=(Get-ExclusiveDigest $File.FullName) }
    }
    $Targets += [pscustomobject][ordered]@{ path=$Resolved; reason='Failed incomplete export from this repair task; no completed manifest'; files=$Files; bytes=($Files | Measure-Object -Property bytes -Sum).Sum }
}
$TotalBytes = ($Targets | Measure-Object -Property bytes -Sum).Sum
$Evidence = [ordered]@{
    status='validated_plan'; created_at_utc=[DateTime]::UtcNow.ToString('o');
    logical_bytes=$TotalBytes; target_directories=$Targets.Count;
    file_count=($Targets.files | Measure-Object).Count; targets=$Targets;
    protected_sha256=$Protected; before_free_bytes=(Get-FreeBytes);
    deletion_mode='Permanent removal of incomplete derived copies only; originals and completed copies retained';
    deleted=@(); blocked=@()
}
if (-not $Execute) {
    [ordered]@{ status='plan_only'; directories=$Targets.Count; files=$Evidence.file_count; bytes=$TotalBytes; gib=[Math]::Round($TotalBytes/1GB,3); protected_files=$Protected.Count } | ConvertTo-Json
    exit 0
}

New-Item -ItemType Directory -Path (Split-Path -Parent $EvidencePath) -ErrorAction Stop | Out-Null
function Save-Evidence { [IO.File]::WriteAllText($EvidencePath, ($Evidence | ConvertTo-Json -Depth 10), [Text.UTF8Encoding]::new($false)) }
Save-Evidence
foreach ($Target in $Targets) {
    try {
        $Resolved = (Resolve-Path -LiteralPath $Target.path).Path
        if ($Resolved -cne $Target.path -or (Split-Path -Parent $Resolved) -cne $DatasetRoot) { throw 'Target path changed' }
        $Current = @(Get-ChildItem -LiteralPath $Resolved -Force)
        if ($Current.Count -ne $Target.files.Count) { throw 'Target membership changed' }
        foreach ($File in $Target.files) {
            $CurrentFile = Get-Item -LiteralPath $File.path -Force
            if ($CurrentFile.PSIsContainer -or ($CurrentFile.Attributes -band [IO.FileAttributes]::ReparsePoint) -or $CurrentFile.Length -ne $File.bytes -or (Get-ExclusiveDigest $File.path) -cne $File.sha256) {
                throw "Target content changed: $($File.path)"
            }
        }
        Remove-Item -LiteralPath $Resolved -Recurse -Force -ErrorAction Stop
        if (Test-Path -LiteralPath $Resolved) { throw 'Deletion did not complete' }
        $Evidence.deleted += $Resolved
        Save-Evidence
    }
    catch {
        $Evidence.blocked += [ordered]@{path=$Target.path; error=$_.Exception.Message}
        Save-Evidence
        break
    }
}
foreach ($Path in @($Protected.Keys)) {
    if ((Get-Digest $Path) -cne $Protected[$Path]) { throw "Protected file changed after cleanup: $Path" }
}
$Evidence.status = if ($Evidence.blocked.Count) { 'partially_blocked' } else { 'completed' }
$Evidence['protected_files_verified_after'] = $Protected.Count
$Evidence['after_free_bytes'] = Get-FreeBytes
$Evidence['completed_at_utc'] = [DateTime]::UtcNow.ToString('o')
Save-Evidence
[ordered]@{status=$Evidence.status; directories_deleted=$Evidence.deleted.Count; directories_blocked=$Evidence.blocked.Count; logical_bytes_planned=$TotalBytes; protected_files_verified_after=$Protected.Count; before_free_bytes=$Evidence.before_free_bytes; after_free_bytes=$Evidence.after_free_bytes; report=$EvidencePath} | ConvertTo-Json
if ($Evidence.blocked.Count) { exit 2 }
