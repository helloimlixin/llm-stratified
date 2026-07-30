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
- `../imgs/neurips_submission/`: lightweight representative gallery by default; full paper figures are generated/restored from run artifacts when needed.

## Run Artifacts

Keep heavy experiment outputs outside `docs/`. The current VQ-AR singular-token
claim is codebook-first: detect singular visual token IDs from the pretrained
VQ codebook embedding table, then test AR branch flatness at positions
predicting those IDs.

Current codebook-first artifacts:

- `../../runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/`: LlamaGen `c2i-B-256` VQ codebook stratification run over the normalized `16384 x 8` codebook.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_codebook_target_singular/`: broad `codebook_target_singular_any` downstream AR branch probe; mostly neutral.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_codebook_target_large_fiber/`: narrower `codebook_target_large_fiber` downstream probe; supports the uniform-polysemy direction on ranked, permuted, and local-ball metrics.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_codebook_target_large_fiber_controls/`: same COCO stress test with three random same-size target-code controls and three frequency-matched target-code controls.
- `C:/Users/hello/Projects/data/imagenet_val/`: full extracted ImageNet validation split with `imagenet_val_labels.csv` generated from the validation synset labels and canonical class-index mapping.
- `../../runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/`: correctly labeled 256-image ImageNet-val VQ-token cache.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_imagenet_val64_codebook_target_large_fiber_controls/`: CPU-scale 64-image ImageNet-val AR probe with large-fiber target codes and random/frequency controls.
- `../../runs/local/pretrained_vq_ar_random_patch_hypothesis/llamagen_c2i_B_256_imagenet_val64_large_fiber/`: random patch-embedding resampling, image-block bootstrap, and within-image permutation tests over the ImageNet-val patch records.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_rollout/`: branch-rollout visual gallery for selected singular target patches and matched regular controls.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement/`: single-token replacement ablation for the same selected anchors.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_rollout_pairs6/`: expanded suffix-rollout gallery, `6` matched pairs and `4` branches.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement_pairs12/`: expanded replacement gallery, `12` matched pairs and `6` branches.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement_pairs32_dense/`: dense replacement sweep, `32` matched pairs and `8` branches.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement_pairs32_position/`: fixed-order guardrail, `32` matched pairs and `8` branches without AR-flatness ranking.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/paired_inference_position_guardrail/`: paired bootstrap/sign-test comparison between fixed-order and flatness-ranked dense visual sweeps.
- `../../runs/local/pretrained_vq_ar_polysemy_branch_gallery/vq_ar_polysemy_more_results_summary.png`: compact result summary across the expanded visual runs.

W&B:

- Codebook stratification: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/bny85afb>
- Broad target singularity: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/74ezbarg>
- Large-fiber target singularity: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/ge76qm8j>
- Large-fiber target singularity with controls, using per-group fraction histograms: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/1869sonz>
- ImageNet-val 256-image encode: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/h3l3yziq>
- ImageNet-val 64-image AR controls: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/pnffg938>
- ImageNet-val random patch hypothesis tests: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/kpbkr8ex>

The source scripts are `../../scripts/pretrained_vq_codebook_stratification_probe.py`
and `../../scripts/pretrained_vq_ar_ks_probe.py`. Heavy JSON, neighbor arrays,
and generated figures should remain in `runs/` or W&B unless a small
representative figure is intentionally copied into `../imgs/neurips_submission/`.
The controlled COCO stress test strengthens the narrow large-fiber claim
against simple frequency confounds. The first correctly labeled ImageNet-val
replication is more conservative: it supports local-neighborhood flatness and
flattest-decile enrichment for large-fiber target codes, while full-vocabulary
ranked/permuted metrics and top-32 branch entropy are only directional or
neutral on the 64-image CPU-scale run. The random-patch follow-up makes the
local claim more robust: balanced random patch resampling, image-block
bootstrap, and within-image permutation tests support lower local-ball KS and
higher local-ball entropy for large-fiber target codes, while random and
frequency-matched target controls do not reproduce the local effect. Vision is
also better suited than language for dense sampling of this claim: every image
contributes a patch lattice, and each patch-level VQ prediction exposes nearby
codebook and top-branch alternatives. The visual branch galleries are
qualitative follow-ups that exploit this density. The expanded single-token
replacement ablation over `12` same-image pairs and `6` top-code branches gives
mean local crop diversity `0.0095` for singular anchors versus `0.0047` for
matched regular controls, with singular winning `8/12` pairs. The expanded
suffix-rollout run over `6` pairs and `4` branches gives `0.0395` versus
`0.0151`, with singular winning `5/6` pairs. The denser replacement sweep over
`32` same-image pairs and `8` branches gives `0.0070` versus `0.0024`, median
`0.0061` versus `0.0012`, with singular winning `23/32` pairs. Paired inference
supports that flatness-ranked dense effect: mean paired lift `+0.00452`,
bootstrap 95% CI `[0.00231, 0.00744]`, sign-test p=`0.010`, and paired
sign-flip p=`0.00004`. The guardrail is important: using the first eligible
codebook-singular anchors in fixed order, without AR-flatness ranking, is
neutral (`+0.00016`, CI `[-0.00089, 0.00112]`, sign-test p=`0.430`). The current
paper-facing claim should therefore be existential/subset-based: branch-flat
large-fiber singular visual tokens are polysemous, but codebook singularity
alone is not sufficient for every target token to branch visually. The
strongest rows visibly change object/material continuation at the marked patch.

Superseded VQ-AR artifacts retained for provenance after the 2026-06-27
estimator audit:

- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_paper_original_fiber/`: four-sample pre-audit run.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_16_seed20260627_paper_fiber/`: 16 generated LlamaGen-B-256 pre-audit run.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_seed20260627_paper_fiber/`: 16 COCO val2017 pre-audit stress test.

New VQ-AR paper-facing claims should not use hidden-state singular detection as
the primary definition. The original token-embedding paper operates on token
embedding geometry; the current vision analogue is therefore the VQ codebook.
Manifold p-values are two-sided sliding Welch tests, fiber p-values are
one-sided slope-increase tests, and Holm-Bonferroni correction is applied to
the p-value families.

Corrected local VQ-AR reruns live in:

- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_16_seed20260627_algorithm1_fiber_fixed/`: generated LlamaGen-B-256 samples, null support for fiber-singular uniformity.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_seed20260627_algorithm1_fiber_fixed/`: COCO stress test, evidence against fiber-singular uniformity under random class conditioning.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_16_seed20260627_algorithm1_fiber_embeddingball50/`: generated LlamaGen-B-256 fixed-volume embedding-ball radial KS test.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_seed20260627_algorithm1_fiber_embeddingball50/`: COCO fixed-volume embedding-ball radial KS stress test.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_16_seed20260627_algorithm1_fiber_radius_threshold/`: generated LlamaGen-B-256 radius-threshold embedding-ball sweep.
- `../../runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_seed20260627_algorithm1_fiber_radius_threshold/`: COCO radius-threshold embedding-ball sweep.

The `*_algorithm1_fiber_localball32/` runs are retained as superseded diagnostics only. They restricted AR probability mass to target codes observed among nearby hidden-state contexts; they do not test whether the empirical token embeddings inside a fixed-volume ball are close to locally uniform. The later `embedding_radius_*` threshold metric was also exploratory: it asked whether hidden-state neighborhoods become radially uniform at some radius. It found broad near-uniform thresholds, not a singular-specific effect, and is no longer the primary singular-token setup.

The corrected radius-threshold metrics and visualizations were also logged to W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/fprq5aag>.

The paper-facing source should cite small representative figures only after they are intentionally copied into `../imgs/neurips_submission/`.

## Build

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The generated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, and `.pdf` files are build artifacts and are not kept in `docs/` by default. Keep the TeX source files in git; keep full-resolution figure bundles in run artifacts unless a small representative image is needed for the README.
