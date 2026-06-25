[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Experiment = "quick_test",
    [string]$DataRoot = $(if ($env:LLM_STRATIFIED_DATA_ROOT) { $env:LLM_STRATIFIED_DATA_ROOT } else { "" }),
    [string]$OutputRoot = $(if ($env:LLM_STRATIFIED_OUTPUT_ROOT) { $env:LLM_STRATIFIED_OUTPUT_ROOT } else { "runs" }),
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [switch]$Wandb,
    [switch]$Cpu,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Overrides
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $DataRoot) {
    $DataRoot = Join-Path $RepoRoot "..\data"
}
$DataRootPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataRoot)
$DataRootForHydra = $DataRootPath -replace "\\", "/"
$RunStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunDir = Join-Path $OutputRoot (Join-Path "local" (Join-Path $Experiment $RunStamp))
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

if ($Cpu) {
    $env:CUDA_VISIBLE_DEVICES = ""
}
if (-not $Wandb) {
    $env:WANDB_MODE = "disabled"
}
$env:PYTHONUNBUFFERED = "1"
$env:LLM_STRATIFIED_DATA_ROOT = $DataRootPath
$env:LLM_STRATIFIED_OUTPUT_ROOT = $OutputRoot
$env:PYTHONPATH = "$RepoRoot" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

$wandbEnabled = if ($Wandb) { "true" } else { "false" }
$ArgsList = @(
    (Join-Path $RepoRoot "src/train.py"),
    "+experiment=$Experiment",
    "data.root=$DataRootForHydra",
    "output_root=$OutputRoot",
    "hydra.run.dir=$RunDir",
    "data.num_workers=0",
    "wandb.enabled=$wandbEnabled"
)

if ($Overrides) {
    $ArgsList += $Overrides
}

Write-Host "Running local experiment '$Experiment'"
Write-Host "Data root: $DataRootPath"
Write-Host "Run dir:   $RunDir"
Write-Host "Python:    $Python"

& $Python @ArgsList
exit $LASTEXITCODE
