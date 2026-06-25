# COCO Frozen Vision Fiber And Sparse Probe Notes

This note records the analysis-only COCO probes referenced from `docs/stratified-spaces/main.tex`. The active comparison is no-training and image-aligned: frozen patch-token clouds are probed for local dimension, corrected fiber-bundle slope-increase violations, same-image locality, raw-patch sparse complexity, and patch-token distance agreement.

## Runs

- DINOv3 ViT-H+/16: `runs/local/coco_dinov3_huge_sparse_fiber/20260509_190534`, W&B `ingbtg9a`.
- SAM ViT-Huge encoder: `runs/local/coco_sam_fiber/20260509_191042`, W&B `nsbl2nhz`.
- SigLIP2-B: `runs/local/coco_siglip2_base_sparse_fiber/20260509_203403`, W&B `rqsw6hyt`.
- AIMv2-L: `runs/local/coco_aimv2_large_sparse_fiber/20260509_203836`, W&B `xqnbkzlm`.
- VAR-d30 auxiliary generation-token probe: `runs/local/coco_var_d30_sparse_fiber/20260509_194508`, W&B `3zmdg9wf`.
- DINOv3 ViT-L/16 scale comparison: `runs/local/coco_pretrained_vit_sparse_fiber/20260509_193911`, W&B `dzwnvdkk`.
- Patch-token distance diagnostics: `runs/local/patch_token_distance/20260509_v3` and `runs/local/patch_token_distance/20260509_vision_expansion`.
- High-capacity residual sweeps: `runs/local/sparse_residual_sweep/20260509_hicap` and `runs/local/sparse_residual_sweep/20260509_vision_expansion`.
- Cross-dataset summary: `runs/local/cross_dataset_logs/cross_dataset_summary_20260511.csv`.

All COCO rows are frozen representation probes with no classifier training. DINOv3 uses `timm` checkpoints, SAM uses `facebook/sam-vit-huge`, SigLIP2 and AIMv2 use Hugging Face vision encoders, and VAR uses the FoundationVision VAR-d30 autoregressive visual-token checkpoint. COCO box prompts are used only to produce SAM mask previews for interpretation; masks are not estimator labels.

## COCO Fiber And Sparse Summary

| model | images | tokens | mean/median dim | change | fiber viol. | same-image | mean sparse complexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| DINOv3 ViT-L/16 | 21 | 4116 | 5.95 / 5.31 | 0.217 | 0.090 | 0.985 | 15.53 |
| DINOv3 ViT-H+/16 | 16 | 3136 | 6.61 / 5.98 | 0.200 | 0.076 | 0.979 | 14.83 |
| SAM ViT-Huge encoder | 16 | 3136 | 6.24 / 6.01 | 0.243 | 0.059 | 0.659 | 20.03 |
| SigLIP2-B | 16 | 3136 | 11.17 / 9.30 | 0.233 | 0.100 | 0.806 | 14.30 |
| AIMv2-L | 16 | 4096 | 12.30 / 10.29 | 0.217 | 0.102 | 0.746 | 12.64 |
| VAR-d30 | 16 | 4096 | 36.59 / 36.38 | 0.168 | 0.042 | 0.143 | 13.86 |

`change` counts any significant slope change. `fiber viol.` counts only corrected slope increases, the forbidden direction under the fiber-bundle null. Sparse complexity is the high-capacity fixed-`k=128`, `tau=0.30` OMP probe with 128 local PCA atoms and sparsity cap 64 when available.

The main pattern is that local dimension, slope-change structure, same-image locality, and sparse raw-patch reconstruction burden separate from one another. DINOv3-H+ is highly image-local; SAM-H has lower same-image locality but strong segmentation-oriented structure and a higher sparse burden; SigLIP2-B and AIMv2-L occupy higher-dimensional local regimes; VAR-d30 behaves like a different object, as expected for a generation-token state rather than a frozen encoder.

## Cross-Dataset Stress Tests

The stress-test sweep repeats the frozen-encoder pipeline on STL10, DTD, and EuroSAT for DINOv3-H+, SAM-H, SigLIP2-B, and AIMv2-L. The paper uses these runs to show that the same diagnostic suite changes with visual domain: object-centered images, texture images, and satellite scenes stress different models and different statistics.

The summary CSV is `runs/local/cross_dataset_logs/cross_dataset_summary_20260511.csv`; paper-facing figures are vendored under `docs/imgs/neurips_submission/` and `docs/stratified-spaces/imgs/neurips_submission/`.

## VAR Generation-Side Probe

VAR is included as an auxiliary control because it exposes a next-token distribution. `scripts/var_generation_polysemy_probe.py` aligns final-scale VAR patch tokens with entropy, observed-code negative log likelihood, top-1 probability, and top-2 margin. The current slice does not support a simple "fiber violation implies generative polysemy" claim: corrected fiber irregularity is essentially uncorrelated with normalized entropy and NLL, while estimated local dimension has a stronger relation to generation uncertainty.

`scripts/var_generation_branch_samples.py` performs the matched branch-sampling intervention. The focused high-entropy pair is still a negative control: branches mostly alter texture/color rather than producing clean semantic alternatives. The next stronger test is a larger matched-anchor branch run with a semantic spread metric over decoded branches.

## Vendored Paper Images

The paper-facing figure directory includes:

- `coco_dinov3_huge_*` fiber, sparse, residual, and patch-token distance figures.
- `coco_sam_*` fiber, sparse, residual, mask-preview, and patch-token distance figures.
- `coco_siglip2_base_*` fiber, sparse, residual, and distance figures.
- `coco_aimv2_large_*` fiber, sparse, residual, and distance figures.
- `coco_var_d30_*` geometry/sparse figures and `coco_var_generation_polysemy_*` controls.
- `cross_stl10_*`, `cross_dtd_*`, and `cross_eurosat_*` stress-test figures.

The source of record for exact captions, tables, and artifact provenance is `docs/stratified-spaces/main.tex`.
