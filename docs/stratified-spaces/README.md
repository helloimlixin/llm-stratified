# Stratified Spaces Paper Bundle

This folder contains the current paper workspace for the stratified vision representation project. The active story is the frozen-encoder analysis over DINOv3, SAM, SigLIP2, AIMv2, and VAR, with COCO as the main image-aligned benchmark and STL10/DTD/EuroSAT as stress tests.

## Contents

- `main.tex`: anonymous submission source. Full paper builds require restoring or regenerating the full figure set under `../imgs/neurips_submission/`.
- `draft.tex`: preprint-style draft with fuller run provenance.
- `outline.tex`: compact outline for the current paper narrative.
- `refs.bib`: shared bibliography.
- `neurips_2026.sty`: NeurIPS style file.
- `checklist.tex`: NeurIPS checklist included by the TeX sources.
- `framing.md`: framing note for the stratified-space interpretation.
- `patch_token_distance_appendix.tex`: raw-patch versus token-distance diagnostic appendix material.
- `residual_sweep_appendix.tex`: sparse residual-threshold appendix material.
- `../COCO_PRETRAINED_VIT_SAM_ANALYSIS.md`: run ledger and analysis note for the current COCO probes.
- `../imgs/neurips_submission/`: lightweight representative gallery by default; full paper figures are generated/restored from run artifacts when needed.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The generated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, and `.pdf` files are build artifacts and are not kept in `docs/` by default. Keep the TeX source files in git; keep full-resolution figure bundles in run artifacts unless a small representative image is needed for the README.
