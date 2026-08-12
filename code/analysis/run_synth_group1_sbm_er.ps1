param(
  [string]$ApiKey=$env:EOH_API_KEY,
  [string]$ApiEndpoint = $env:EOH_API_ENDPOINT,
  [string]$Model = "gpt-4.1-mini",
  [int]$NumInstance = 5,
  [int]$PopSize =6 ,
  [int]$NPop = 30,
  [int]$NProc = 1,
  [int]$Timeout = 1000,
  [int]$FailureStopPatience = 15,
  [switch]$NewRun,
  [switch]$ResumeFromLatest = $true
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PackageRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
$RunEoH = Join-Path $PackageRoot "code\eoh\runEoH.py"
$RunsRoot = Join-Path $PackageRoot "evolve_experiment\evolution\synthetic"
if (-not $ApiEndpoint) { throw "Set EOH_API_ENDPOINT or pass -ApiEndpoint." }
if (-not $ApiKey) { throw "Set EOH_API_KEY or pass -ApiKey." }
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

function Get-LatestRunDir {
  param(
    [string]$GroupName,
    [string]$Dataset
  )

  $base = Join-Path $RunsRoot (Join-Path $GroupName $Model)
  if (-not (Test-Path -LiteralPath $base)) { return $null }

  $prefix = ($Dataset + "_")
  $dirs = Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name.StartsWith($prefix) } |
    Sort-Object -Property Name -Descending

  if ($dirs.Count -ge 1) { return $dirs[0].FullName }
  return $null
}

function Get-LatestPopulationCheckpoint {
  param(
    [string]$OutDir,
    [string]$Dataset
  )

  $popsDir = Join-Path $OutDir (Join-Path "results\pops" $Dataset)
  if (-not (Test-Path -LiteralPath $popsDir)) { return $null }

  $files = Get-ChildItem -LiteralPath $popsDir -File -Filter "population_generation_*.json" -ErrorAction SilentlyContinue
  $maxId = $null
  $maxPath = $null
  foreach ($f in $files) {
    $name = $f.BaseName
    $mid = $name.Substring("population_generation_".Length)
    $gid = $null
    if ([int]::TryParse($mid, [ref]$gid)) {
      if ($maxId -eq $null -or $gid -gt $maxId) {
        $maxId = $gid
        $maxPath = $f.FullName
      }
    }
  }
  if ($maxId -eq $null) { return $null }
  return @{ Id = [int]$maxId; Path = [string]$maxPath }
}

function Invoke-EOHRun {
  param(
    [string]$Dataset,
    [string]$GroupName
  )

  $outDir = $null
  $continue = $null
  if (-not $NewRun -and $ResumeFromLatest) {
    $outDir = Get-LatestRunDir -GroupName $GroupName -Dataset $Dataset
    if ($outDir) {
      $continue = Get-LatestPopulationCheckpoint -OutDir $outDir -Dataset $Dataset
    }
  }
  if (-not $outDir) {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outDir = Join-Path $RunsRoot ("{0}\{1}\{2}_{3}" -f $GroupName, $Model, $Dataset, $timestamp)
  }

  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $logPath = Join-Path $outDir "run.log"
  New-Item -ItemType File -Force -Path $logPath | Out-Null

  $argsList = @(
    $RunEoH,
    "--dataset", $Dataset,
    "--api_endpoint", $ApiEndpoint,
    "--api_key", $ApiKey,
    "--model", $Model,
    "--num_instance", $NumInstance,
    "--pop_size", $PopSize,
    "--n_pop", $NPop,
    "--n_proc", $NProc,
    "--timeout", $Timeout,
    "--failure_stop_patience", $FailureStopPatience,
    "--output_path", $outDir
  )
  if ($NewRun) {
    $argsList += "--new_run"
  }
  elseif ($continue) {
    if ($continue.Id -ge $NPop) {
      Write-Host ("[{0}] SKIP  {1} already has generation {2} (>= target {3}) in {4}" -f (Get-Date -Format "HH:mm:ss"), $Dataset, $continue.Id, $NPop, $outDir)
      return
    }
    $argsList += @("--continue_id", $continue.Id, "--continue_path", $continue.Path)
  }

  Write-Host ("[{0}] START {1} -> {2}" -f (Get-Date -Format "HH:mm:ss"), $Dataset, $outDir)
  & python @argsList 2>&1 | Tee-Object -Append -FilePath $logPath
  if ($LASTEXITCODE -ne 0) {
    throw ("Run failed for {0} (exit {1}). See log: {2}" -f $Dataset, $LASTEXITCODE, $logPath)
  }
  Write-Host ("[{0}] DONE  {1}" -f (Get-Date -Format "HH:mm:ss"), $Dataset)
}

Invoke-EOHRun -Dataset "synthetic_sbm_1000" -GroupName "group1_sbm_er"
Invoke-EOHRun -Dataset "synthetic_er_1000" -GroupName "group1_sbm_er"
