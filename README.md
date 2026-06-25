# Stratified Vision Representation Probes

This repository studies **local stratified geometry in dense vision representations**. The active experiments are no-training probes over frozen patch-token spaces from modern vision backbones, especially DINOv3, SAM, SigLIP2, AIMv2, and VAR. The goal is not to train another classifier; it is to measure where local neighborhoods behave like one smooth chart and where volume growth, corrected slope increases, or raw-patch sparse complexity reveal heterogeneous structure.

The current paper workspace lives in `docs/stratified-spaces/`. The current results index is `docs/RESULTS.md`, and the main run note is `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md`.

## Main Entry Points

- `src/train.py`: Hydra entrypoint for frozen encoder probes, local sparse dictionary probes, and legacy training-compatible configs.
- `src/training/sam_fiber_job.py`: SAM image-encoder and COCO box-prompt probe path.
- `src/volume_probe.py`: standalone no-training volume-scaling probe utilities.
- `scripts/var_generation_polysemy_probe.py`: VAR generation-side entropy/NLL probe.
- `scripts/var_generation_branch_samples.py`: matched branch-sampling follow-up for VAR anchors.
- `scripts/patch_token_distance_digest.py`: raw-patch versus token-distance agreement summaries.

## Results At A Glance

Recent work moved from small scratch-backbone trials to **frozen, image-aligned dense vision representations**. The current headline suite probes DINOv3-H+, SAM-H, SigLIP2-B, AIMv2-L, and VAR-d30 on COCO, then stress-tests DINOv3/SAM/SigLIP2/AIMv2 on STL10, DTD, and EuroSAT.

Core finding: local dimension, corrected fiber violations, same-image locality, sparse raw-patch complexity, and patch-token distance agreement are different signals. DINOv3-H+ is highly image-local; SAM-H carries a larger sparse reconstruction burden and is strongly stressed by EuroSAT; SigLIP2-B/AIMv2-L occupy higher-dimensional regimes with moderate sparse complexity; VAR-d30 is a useful generation-side control where uncertainty does not simply follow fiber violations.

| COCO model | Mean/med. dim | Fiber viol. | Same-image | Mean sparse `S_i` |
|---|---:|---:|---:|---:|
| DINOv3-H+ | 6.61 / 5.98 | 0.076 | 0.979 | 14.83 |
| SAM-H | 6.24 / 6.01 | 0.059 | 0.659 | 20.03 |
| SigLIP2-B | 11.17 / 9.30 | 0.100 | 0.806 | 14.30 |
| AIMv2-L | 12.30 / 10.29 | 0.102 | 0.746 | 12.64 |
| VAR-d30 | 36.59 / 36.38 | 0.042 | 0.143 | 13.86 |

Representative figures:

- [DINOv3-H+ fiber violations](docs/imgs/neurips_submission/coco_dinov3_huge_fiber_irregularity_heatmap.png)
- [SAM-H fiber violations](docs/imgs/neurips_submission/coco_sam_fiber_irregularity_heatmap.png)
- [DINOv3/SAM sparse residual sweep](docs/imgs/neurips_submission/coco_sparse_residual_sweep_hicap_comparison.png)
- [Patch-token distance digest](docs/imgs/neurips_submission/coco_patch_token_distance_digest.png)
- [VAR branch-sampling control](docs/imgs/neurips_submission/coco_var_generation_polysemy_branch_samples.png)

See `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md` for the compact run ledger and `docs/stratified-spaces/main.tex` for the full paper tables, captions, and visual audit.

## Installation

Install PyTorch and torchvision for your CUDA build first, then install the local dependencies:

```bash
pip install torch torchvision
pip install -r requirements-local.txt
```

Optional W&B logging:

```bash
wandb login
```

## Data Layout

Hydra configs default to `../data`, which matches a checkout beside a shared data directory:

```text
Projects/
  data/
    coco/
    dtd/
    eurosat/
    stl10_binary/
  llm-stratified/
```

Override once per shell if your datasets live elsewhere:

```bash
export LLM_STRATIFIED_DATA_ROOT=/path/to/data
export LLM_STRATIFIED_OUTPUT_ROOT=/scratch/$USER/runs/llm-stratified
```

PowerShell:

```powershell
$env:LLM_STRATIFIED_DATA_ROOT = "../data"
$env:LLM_STRATIFIED_OUTPUT_ROOT = "runs"
```

COCO is not auto-downloaded. The expected layout is:

```text
<data.root>/coco/
  train2017/
  val2017/
  annotations/instances_train2017.json
  annotations/instances_val2017.json
```

Use the helper to download or verify COCO 2017:

```bash
scripts/setup_coco2017.sh /scratch/$USER/data
scripts/setup_coco2017.sh --verify-only /scratch/$USER/data
```

## Quick Checks

Run a no-download sanity check:

```bash
python src/train.py +experiment=quick_test
```

Local wrapper, with W&B disabled by default:

```powershell
.\scripts\run_local.ps1 -Experiment quick_test
```

```bash
scripts/run_local.sh
```

## Current COCO Probes

The current paper runs use frozen representations and `training.epochs=0`. Each selected image contributes a complete patch-token grid so local dimension, fiber violations, sparse raw-patch complexity, and image-space heatmaps stay aligned.

```bash
# DINOv3-H+ frozen patch-token probe
python src/train.py +experiment=coco_dinov3_huge_sparse_fiber data.root=/scratch/$USER/data

# SAM-H image-encoder probe with COCO box-prompt mask previews
python src/train.py +experiment=coco_sam_fiber data.root=/scratch/$USER/data

# SigLIP2-B multimodal encoder probe
python src/train.py +experiment=coco_siglip2_base_sparse_fiber data.root=/scratch/$USER/data

# AIMv2-L vision encoder probe
python src/train.py +experiment=coco_aimv2_large_sparse_fiber data.root=/scratch/$USER/data

# VAR-d30 autoregressive visual-token probe
python src/train.py +experiment=coco_var_d30_sparse_fiber data.root=/scratch/$USER/data
```

For COCO debug and paper-facing runs, `data.num_workers=0` is the safest default in this repo because the dataset wrapper keeps a large annotation object in memory.

## Cross-Dataset Stress Tests

The paper also probes DINOv3-H+, SAM-H, SigLIP2-B, and AIMv2-L on STL10, DTD, and EuroSAT. The helper script records those run families and summary tables:

```powershell
.\scripts\run_cross_dataset_vision.ps1
```

Outputs are collected under `runs/local/<dataset>_<model>_sparse_fiber/` and summarized in `runs/local/cross_dataset_logs/`.

## VAR Generation-Side Probes

VAR is treated as a generation-token control rather than as a matched image encoder. The generation-side probes align final-scale tokens with entropy, observed-code NLL, and branch samples:

```bash
python scripts/var_generation_polysemy_probe.py --help
python scripts/var_generation_branch_samples.py --help
python scripts/var_generation_aggregate_from_json.py --help
```

These are documented in `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md` and integrated into the paper appendix.

## Outputs

Hydra writes runs under `${LLM_STRATIFIED_OUTPUT_ROOT:-runs}/hydra/...` unless an experiment or wrapper sets `hydra.run.dir`. Local paper runs usually live under `runs/local/<experiment>/<timestamp>/`.

Important outputs include:

- `checkpoints/fiber_analysis/`: local dimension maps, irregularity heatmaps, patch galleries, and volume-curve galleries.
- `checkpoints/fiber_history.json`: per-run fiber-bundle summaries.
- `checkpoints/sparse_probe_summary.json`: fixed-neighborhood local dictionary summaries when sparse probes are enabled.
- `token_processing_notes.md`: token collection details for SAM and frozen-token runs.
- `docs/imgs/neurips_submission/`: lightweight representative figure gallery.

## Documentation Map

- `docs/RESULTS.md`: short index of current result artifacts.
- `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md`: run ledger and interpretation notes for the current frozen COCO probes.
- `docs/stratified-spaces/main.tex`: anonymous paper source.
- `docs/stratified-spaces/draft.tex`: preprint-style draft.
- `docs/stratified-spaces/README.md`: paper bundle guide and build notes.
- `docs/stratified-spaces/framing.md`: conceptual framing for the stratified-space interpretation.

The older scratch-backbone trial reports have been removed from the active narrative. Superseded volume-probe report files now only point to the current paper artifacts.

## Tests

Unit tests avoid dataset downloads and run on CPU:

```bash
python -m unittest discover -s tests
```

## Repo Layout

```text
.
|-- configs/                   # Hydra config groups and experiment presets
|-- docs/                      # paper notes, figure assets, current result index
|-- scripts/                   # launchers and post-processing utilities
|-- src/                       # training/probe entrypoints and core modules
|-- tests/                     # CPU-friendly unit tests
`-- runs/                      # local outputs, checkpoints, plots
```
