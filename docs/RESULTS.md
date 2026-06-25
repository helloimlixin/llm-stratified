# Current Results Index

This file now points to the active result surfaces for the stratified vision representation project. The old scratch-backbone training report has been removed from the active documentation because the project has moved to frozen, image-aligned patch-token probes over modern vision systems.

## Active Results

- `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md`: current COCO run ledger, fiber summaries, sparse-probe summaries, VAR generation-side notes, and vendored figure list.
- `docs/stratified-spaces/main.tex`: paper source for the current frozen DINOv3, SAM, SigLIP2, AIMv2, and VAR analysis.
- `docs/stratified-spaces/main.pdf`: rendered paper artifact when built locally.
- `docs/imgs/neurips_submission/`: stable paper-facing PNG/PDF figures.

## Result Families

- COCO frozen encoder probes: DINOv3-H+, SAM-H, SigLIP2-B, AIMv2-L, and VAR-d30.
- Cross-dataset stress tests: STL10, DTD, and EuroSAT across DINOv3-H+, SAM-H, SigLIP2-B, and AIMv2-L.
- Local sparse dictionary probes: fixed-neighborhood OMP heatmaps, residual-threshold sweeps, and expanding-neighborhood volume curves.
- Patch-token distance diagnostics: raw RGB patch geometry versus learned token geometry.
- Generation-side VAR probes: entropy/NLL maps and matched branch-sampling controls.

## Rebuilding Or Extending

Run the relevant Hydra preset, copy paper-facing figures into `docs/imgs/neurips_submission/`, and update the run ledger in `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md`. The root `README.md` lists the current commands.
