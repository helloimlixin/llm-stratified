[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Datasets = @("stl10", "dtd", "eurosat"),
    [string[]]$Models = @("dinov3_huge", "sam_huge", "siglip2_base", "aimv2_large"),
    [string]$DataRoot = $(if ($env:LLM_STRATIFIED_DATA_ROOT) { $env:LLM_STRATIFIED_DATA_ROOT } else { "" }),
    [string]$OutputRoot = $(if ($env:LLM_STRATIFIED_OUTPUT_ROOT) { $env:LLM_STRATIFIED_OUTPUT_ROOT } else { "runs" }),
    [string]$Python = $(if ($env:PYTHON) { $env:PYTHON } else { "python" }),
    [switch]$Wandb,
    [switch]$Cpu,
    [switch]$ContinueOnError
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
if (-not $DataRoot) {
    $DataRoot = Join-Path $RepoRoot "..\data"
}
$DataRootPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataRoot)
$DataRootForHydra = $DataRootPath -replace "\\", "/"

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

$ModelConfigs = @{
    dinov3_huge = @{
        Experiment = "coco_dinov3_huge_sparse_fiber"
        Tags = "dinov3,dinov3-huge-plus"
    }
    sam_huge = @{
        Experiment = "coco_sam_fiber"
        Tags = "sam,sam-vit-huge"
    }
    siglip2_base = @{
        Experiment = "coco_siglip2_base_sparse_fiber"
        Tags = "siglip2,vision-language"
    }
    aimv2_large = @{
        Experiment = "coco_aimv2_large_sparse_fiber"
        Tags = "aimv2,autoregressive-pretraining"
    }
}

$wandbEnabled = if ($Wandb) { "true" } else { "false" }
$Summary = @()

foreach ($dataset in $Datasets) {
    $datasetKey = $dataset.ToLowerInvariant()
    foreach ($model in $Models) {
        if (-not $ModelConfigs.ContainsKey($model)) {
            throw "Unknown model '$model'. Expected one of: $($ModelConfigs.Keys -join ', ')"
        }

        $modelCfg = $ModelConfigs[$model]
        $runName = "${datasetKey}_${model}_sparse_fiber"
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $runDir = Join-Path $OutputRoot (Join-Path "local" (Join-Path $runName $stamp))
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $runDirHydra = ($ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($runDir)) -replace "\\", "/"
        $tagExpr = "[$datasetKey,$($modelCfg.Tags),cross-dataset,fiber-analysis,sparse-omp]"

        $argsList = @(
            (Join-Path $RepoRoot "src/train.py"),
            "+experiment=$($modelCfg.Experiment)",
            "data=$datasetKey",
            "data.root=$DataRootForHydra",
            "output_root=$OutputRoot",
            "hydra.run.dir=$runDirHydra",
            "data.num_workers=0",
            "wandb.enabled=$wandbEnabled",
            "wandb.name=$runName",
            "wandb.tags=$tagExpr"
        )

        Write-Host "============================================================"
        Write-Host "Running $runName"
        Write-Host "Run dir: $runDir"
        Write-Host "Experiment template: $($modelCfg.Experiment)"
        Write-Host "============================================================"

        & $Python @argsList
        $exit = $LASTEXITCODE
        $Summary += [pscustomobject]@{
            Dataset = $datasetKey
            Model = $model
            ExitCode = $exit
            RunDir = $runDir
        }
        if ($exit -ne 0) {
            Write-Host "FAILED $runName with exit code $exit"
            if (-not $ContinueOnError) {
                $Summary | Format-Table -AutoSize
                exit $exit
            }
        }
    }
}

Write-Host "============================================================"
Write-Host "Cross-dataset run summary"
Write-Host "============================================================"
$Summary | Format-Table -AutoSize
if (($Summary | Where-Object { $_.ExitCode -ne 0 }).Count -gt 0) {
    exit 1
}
