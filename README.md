# Stratified Vision Representation Probes

This repository studies **local stratified geometry in dense vision representations**. The active experiments are no-training probes over frozen patch-token spaces from modern vision backbones, especially DINOv3, SAM, SigLIP2, AIMv2, and VAR. The goal is not to train another classifier; it is to measure where local neighborhoods behave like one smooth chart and where volume growth, corrected slope increases, or raw-patch sparse complexity reveal heterogeneous structure.

The current paper workspace lives in `docs/stratified-spaces/`. Full run ledgers and heavy visual artifacts live under `runs/` and W&B rather than inside `docs/`.

## Main Entry Points

- `src/train.py`: Hydra entrypoint for frozen encoder probes, local sparse dictionary probes, and legacy training-compatible configs.
- `src/training/sam_fiber_job.py`: SAM image-encoder and COCO box-prompt probe path.
- `src/volume_probe.py`: standalone no-training volume-scaling probe utilities.
- `scripts/var_generation_polysemy_probe.py`: VAR generation-side entropy/NLL probe.
- `scripts/var_generation_branch_samples.py`: matched branch-sampling follow-up for VAR anchors.
- `scripts/var_polysemy_nn_gallery.py`: artifact-driven nearest-neighbor gallery for singular VAR branch points.
- `scripts/patch_token_distance_digest.py`: raw-patch versus token-distance agreement summaries.
- `scripts/vision_branching_ks_probe.py`: visual branch-flattening and robust sliced-KS probe with W&B logging.
- `scripts/pretrained_var_generator.py`: real pretrained FoundationVision/VAR generator sampling with pretrained VQ-VAE and pretrained AR transformer checkpoints.
- `scripts/pretrained_var_one_sample_ks.py`: order-free KS-style uniformity probe over pretrained VAR next-scale code distributions.
- `scripts/pretrained_vq_ar_pipeline.py`: matched pretrained VQ-tokenizer plus autoregressive transformer pipeline, currently using LlamaGen.
- `scripts/prepare_imagenet_val_for_vq_ar.py`: ImageNet-val setup helper that extracts the flat validation tar and writes a canonical `path,label` CSV.
- `scripts/pretrained_vq_ar_encode_dataset.py`: real-image encoder that maps image folders through the pretrained LlamaGen VQ tokenizer for teacher-forced VQ-AR probes.
- `scripts/pretrained_vq_codebook_stratification_probe.py`: paper-style singular visual-token detector over the pretrained VQ codebook embedding table.
- `scripts/pretrained_vq_ar_ks_probe.py`: downstream AR branch-uniformity probe that can consume codebook-derived singular token IDs.
- `scripts/vq_ar_random_patch_hypothesis.py`: random patch-embedding resampling and within-image permutation tests for VQ-AR uniformity claims.
- `scripts/vq_ar_polysemy_branch_gallery.py`: visual branch-rollout gallery for singular VQ-AR target tokens versus matched regular controls.
- `scripts/pretrained_vqvae_imagegpt_pipeline.py`: toy pretrained-VQ-VAE plus small local ImageGPT-style baseline over VQ codes.

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

See `docs/stratified-spaces/main.tex` and `docs/stratified-spaces/draft.tex` for the paper tables, captions, and visual audit.

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

VAR is treated as a generation-side control rather than as a matched image encoder. Because VAR is coarse-to-fine, the generation-side probes align final-scale predicted patch codes with next-scale code entropy, observed-code NLL, and branch samples:

```bash
python scripts/var_generation_polysemy_probe.py --help
python scripts/var_generation_branch_samples.py --help
python scripts/var_generation_aggregate_from_json.py --help
python scripts/var_polysemy_nn_gallery.py --help
```

The paper-facing interpretation lives in `docs/stratified-spaces/draft.tex`; heavy run outputs stay under `runs/` and W&B.

To inspect the polysemy effect around singular branch points without reloading
VAR-d30, build an evidence sheet from the cached d30 branch JSON and embedding
pack:

```bash
python scripts/var_polysemy_nn_gallery.py \
  --run-dir runs/local/coco_var_d30_sparse_fiber/20260509_194508 \
  --out-name epoch_000_var_polysemy_nn_gallery_cross_image.png \
  --anchors 6 \
  --neighbors 6 \
  --cross-image-only \
  --wandb \
  --wandb-run-name var-d30-polysemy-nn-gallery-cross-image
```

The gallery marks the predicted fine-scale patch for each singular next-scale
code, shows cross-image nearest neighbors in d30 token-embedding space, and
stacks the existing decoded branch samples underneath so the local intervention
and downstream visual branches can be read together.

## Pretrained VAR Generator

For non-toy image generation, use a pretrained FoundationVision/VAR checkpoint. This loads the pretrained VAR VQ-VAE tokenizer and the pretrained visual autoregressive transformer, then samples ImageNet-class-conditional images without local training.

The available checkpoint ladder is:

| Checkpoint | Resolution | Reported FID | Params | Local use |
|---|---:|---:|---:|---|
| VAR-d16 | 256 | 3.55 | 310M | CPU/GPU smoke |
| VAR-d20 | 256 | 2.95 | 600M | GPU preferred |
| VAR-d24 | 256 | 2.33 | 1.0B | GPU |
| VAR-d30 | 256 | 1.97 | 2.0B | serious 256px GPU run |
| VAR-d36 | 512 | 2.63 | 2.3B | 512px GPU run |

CPU-safe smoke:

```bash
python scripts/pretrained_var_generator.py \
  --model-depth 16 \
  --samples 4 \
  --class-labels 980,437,22,562 \
  --out-dir runs/local/pretrained_var/d16_smoke \
  --wandb \
  --wandb-name pretrained-var-d16-smoke
```

Serious 256px run on a CUDA machine:

```bash
python scripts/pretrained_var_generator.py \
  --device cuda \
  --model-depth 30 \
  --samples 16 \
  --class-labels random \
  --cfg 4 \
  --top-k 900 \
  --top-p 0.95 \
  --out-dir runs/local/pretrained_var/d30_imagenet \
  --wandb \
  --wandb-name pretrained-var-d30-imagenet
```

The script writes a sample grid, individual PNGs, a JSON summary with checkpoint metadata/timings, and optional W&B media/artifacts.

## Pretrained VAR Uniformity KS Probe

To test the autoregressive uniformity claim directly, run a KS-style probe on pretrained VAR's full next-scale VQ-code distribution. The language claim is about flattening in the next-token distribution; the VAR analogue is flattening over the VQ codes for a predicted fine-scale patch in the next resolution map. The script generates or loads images, teacher-forces them through VAR, computes each final-scale patch location's 4096-way scale-conditioned code distribution, and uses two order-free flatness views: ranked-probability KS over the full vocabulary, and top-k branch KS after renormalizing the most plausible continuations. Lower top-k branch KS means the plausible next-scale code branches are closer to uniform; the existential version of the claim asks whether these unusually flat predicted patch locations are enriched for fiber singularity. The older token-ID one-sample KS is still saved as a diagnostic, but VQ code IDs are arbitrary, so the order-free statistics are primary.

CPU-safe local smoke:

```bash
python scripts/pretrained_var_one_sample_ks.py \
  --model-depth 16 \
  --samples 2 \
  --class-labels 980,437 \
  --out-dir runs/local/pretrained_var_one_sample_ks/d16_smoke \
  --wandb \
  --wandb-name pretrained-var-d16-one-sample-ks
```

Serious 256px run on CUDA:

```bash
python scripts/pretrained_var_one_sample_ks.py \
  --device cuda \
  --model-depth 30 \
  --embedding-pack runs/local/coco_var_d30_sparse_fiber/20260509_194508/checkpoints/embeddings/epoch_000.pt \
  --embedding-dataset COCO \
  --teacher-class-labels -1 \
  --ks-draws 1024 \
  --ks-permutations 64 \
  --branch-top-k 32 \
  --model-dtype float16 \
  --mmap-load \
  --out-dir runs/local/pretrained_var_one_sample_ks/d30_coco \
  --wandb \
  --wandb-name pretrained-var-d30-coco-branch-ks
```

Outputs include KS heatmaps, entropy heatmaps, fiber-irregularity heatmaps, singular-vs-regular KS histograms, scatter plots, per-token JSON, and W&B media/artifacts.

VAR-d30 is a multi-billion-parameter checkpoint; run it on CUDA. CPU `float16` can reduce memory pressure but is too slow for a credible local experiment.

## Pretrained VQ Token + Autoregressive Transformer

For a non-toy VQGAN-like autoregressive baseline, use a tokenizer and AR
transformer that were trained as a matched pair. Pretrained pixel ImageGPT is
not a drop-in model for VQGAN or ViT-VQGAN tokens because its vocabulary models
pixel/color-cluster tokens rather than learned VQ codebook IDs.

First write the compatibility report:

```bash
python scripts/pretrained_vq_ar_pipeline.py \
  --mode compatibility \
  --out-dir runs/local/pretrained_vq_ar/compatibility
```

Then run the pretrained LlamaGen class-conditional pair. This uses the released
FoundationVision/LlamaGen VQ tokenizer plus GPT-style AR checkpoint and can
auto-clone the lightweight sampler code into `external/LlamaGen/`:

```bash
python scripts/pretrained_vq_ar_pipeline.py \
  --mode llamagen-c2i \
  --profile c2i-B-256 \
  --auto-clone \
  --samples 4 \
  --class-labels 207,360,387,974 \
  --out-dir runs/local/pretrained_vq_ar/llamagen_c2i_B_256 \
  --wandb \
  --wandb-name llamagen-c2i-B-256-vq-ar-smoke
```

The smoke run writes a sample grid, generated VQ token IDs, and summary JSON.
For stronger but heavier runs, switch to `--profile c2i-L-256`,
`--profile c2i-B-384`, `--profile c2i-L-384`, or `--profile c2i-XL-384`
on CUDA. These are matched LlamaGen checkpoints; the released ImageNet class
conditional ladder reports FIDs from 5.46 for B-256 down to 2.62 for XL-384,
with larger XXL/3B checkpoints available for multi-GPU follow-up.

To run the same probe on real images rather than generated samples, first
encode a high-quality image folder through the matched VQ tokenizer:

```bash
python scripts/prepare_imagenet_val_for_vq_ar.py \
  --val-tar /path/to/ILSVRC2012_img_val.tar \
  --out-dir /path/to/imagenet_val
```

```bash
python scripts/pretrained_vq_ar_encode_dataset.py \
  --profile c2i-B-256 \
  --image-dir /path/to/imagenet-or-coco/images \
  --samples 128 \
  --label-mode labels_file \
  --labels-file /path/to/imagenet_labels.csv \
  --out-dir runs/local/pretrained_vq_ar/llamagen_c2i_B_256_dataset \
  --wandb \
  --wandb-name llamagen-c2i-B-256-dataset-encode
```

Use `--label-mode parent_index` for ImageNet-style class folders, or
`--label-mode class_labels --class-labels random` only as a stress test when no
class labels are available. The latter intentionally mismatches the
class-conditional AR model.

### Codebook-first singular visual tokens

For the singular-token claim, the paper-faithful vision analogue is the VQ
codebook embedding table: one point per visual token ID. The earlier
hidden-state and local-ball probes are retained below as exploratory
diagnostics, but the current detector first marks singular **code IDs** from
codebook geometry and only then asks whether AR predictions involving those
IDs are flatter.

Run the codebook detector:

```bash
python scripts/pretrained_vq_codebook_stratification_probe.py \
  --profile c2i-B-256 \
  --out-dir runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification \
  --small-vol-min 10 \
  --small-vol-max 50 \
  --large-vol-min 50 \
  --large-vol-max 200 \
  --window-size 5 \
  --alpha 0.001 \
  --wandb \
  --wandb-name llamagen-c2i-B-256-codebook-stratification
```

Then run the teacher-forced AR branch probe using the codebook singular IDs.
The cleaner downstream selector so far is the large-radius fiber violation:

```bash
python scripts/pretrained_vq_ar_ks_probe.py \
  --profile c2i-B-256 \
  --tokens-path runs/local/pretrained_vq_ar/llamagen_c2i_B_256_coco_val16_seed20260627/llamagen_c2i_tokens.pt \
  --class-labels 209,801,391,197,647,846,196,318,258,903,466,975,330,741,125,144 \
  --out-dir runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_codebook_target_large_fiber \
  --paper-small-vol-min 10 \
  --paper-small-vol-max 50 \
  --paper-large-vol-min 50 \
  --paper-large-vol-max 200 \
  --window-size 5 \
  --codebook-singular-codes-path runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/vq_codebook_singular_codes.json \
  --use-codebook-singular-as-active \
  --codebook-singular-source large_fiber \
  --codebook-control-source large_fiber \
  --codebook-active-position target \
  --codebook-random-controls 3 \
  --codebook-frequency-controls 3 \
  --branch-top-k 32 \
  --permuted-ks 16 \
  --permutation-reps 2000 \
  --wandb \
  --wandb-name llamagen-c2i-B-256-coco-val16-codebook-target-large-fiber-controls
```

For the correctly labeled ImageNet-val cache, pass the encoder's records file
instead of a long comma-separated class-label list:

```bash
python scripts/pretrained_vq_ar_ks_probe.py \
  --profile c2i-B-256 \
  --tokens-path runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_tokens.pt \
  --class-labels-file runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_dataset_records.json \
  --max-samples 64 \
  --out-dir runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_imagenet_val64_codebook_target_large_fiber_controls \
  --paper-small-vol-min 10 \
  --paper-small-vol-max 50 \
  --paper-large-vol-min 50 \
  --paper-large-vol-max 200 \
  --window-size 5 \
  --codebook-singular-codes-path runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/vq_codebook_singular_codes.json \
  --use-codebook-singular-as-active \
  --codebook-singular-source large_fiber \
  --codebook-control-source large_fiber \
  --codebook-active-position target \
  --codebook-random-controls 3 \
  --codebook-frequency-controls 3 \
  --branch-top-k 32 \
  --permuted-ks 16 \
  --permutation-reps 2000 \
  --wandb \
  --wandb-name llamagen-c2i-B-256-imagenet-val64-codebook-target-large-fiber-controls
```

To visualize the actual generation-side polysemy effect, reuse the ImageNet-val
tokens and AR probe records, choose singular anchors with low local-ball KS /
high local entropy, choose matched regular controls from the same image, then
force one of the model's top branch codes at the selected patch and let the
pretrained AR model roll out the remaining suffix:

```bash
python scripts/vq_ar_polysemy_branch_gallery.py \
  --profile c2i-B-256 \
  --tokens-path runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_tokens.pt \
  --records runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_imagenet_val64_codebook_target_large_fiber_controls/vq_ar_ks_tokens.json \
  --class-labels-file runs/local/pretrained_vq_ar/llamagen_c2i_B_256_imagenet_val256_seed20260628/llamagen_c2i_dataset_records.json \
  --max-samples 64 \
  --out-dir runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_rollout \
  --selector codebook_target_large_fiber \
  --pairs 3 \
  --branches 4 \
  --min-patch-id 96 \
  --max-patch-id 220 \
  --crop-context 1 \
  --rollout-suffix
```

For a cheaper ablation that only replaces the selected VQ code and decodes the
same suffix, run the same command without `--rollout-suffix`. For wider local
galleries, increase `--pairs` and `--branches`; the current expanded artifacts
use `--pairs 12 --branches 6` for replacement and `--pairs 6 --branches 4` for
suffix rollout. Vision also supports denser sampling than language here: every
image contributes a full patch lattice, and each VQ code has nearby codebook
neighbors and top-branch alternatives, so the branch intervention can be
repeated over many spatial anchors rather than a few textual contexts.

Current codebook-first results, 2026-06-28:

- LlamaGen `c2i-B-256` VQ table: `16384 x 8`, using the quantizer's L2-normalized `quantize.embedding.weight`.
- Codebook singular IDs: `7046/16384` for `singular_any`, `6883/16384` for manifold-any, `3958/16384` for fiber-any, and `3835/16384` for the large-radius fiber selector.
- The broad `codebook_target_singular_any` AR selector is neutral on the 16-image COCO val2017 cache: top-32 branch KS is `0.1532` vs `0.1514` (p=`0.436`), and branch entropy is `0.9729` vs `0.9737` (p=`0.444`).
- The narrower `codebook_target_large_fiber` selector supports the uniform-polysemy direction on the stronger order-free metrics: ranked KS `0.6679` vs `0.6811` (p=`0.0095`), permuted full-vocabulary KS `0.0335` vs `0.0360` (p=`0.0215`), local-ball KS `0.5443` vs `0.5547` (p=`0.0265`), and local-ball entropy `0.7134` vs `0.7016` (p=`0.0240`). Top-32 branch KS is in the same direction but weaker: `0.1488` vs `0.1531` (p=`0.113`).
- A control-strengthened rerun added three same-size random code-ID controls and three frequency-matched non-singular controls. The real `codebook_target_large_fiber` set kept the expected direction, while target controls mostly moved opposite: random ranked-KS deltas were `+0.0095`, `+0.0070`, `+0.0014`; frequency-matched ranked-KS deltas were `-0.0002`, `+0.0052`, `+0.0062`. Local-ball entropy was `+0.0119` for real large-fiber, but negative for all three frequency-matched target controls.
- The first correctly labeled ImageNet-val run prepared all `50,000` validation images, encoded a `256`-image cache, and ran the full AR probe on the first `64` images on CPU. The large-fiber target selector covered `3827/16384` patch positions. It supports local-neighborhood flatness: local-ball KS `0.5573` vs `0.5634` (p=`0.0050`) and local-ball entropy `0.7035` vs `0.6961` (p=`0.0045`). The flattest top-32 branch-KS decile is enriched for large-fiber targets: `26.7%` vs `23.0%` (p=`0.0015`). Full-vocabulary ranked KS is directional but not significant: `0.7121` vs `0.7159` (p=`0.138`); permuted KS and top-32 branch entropy are neutral.
- The robust random-patch follow-up samples patch embeddings instead of trusting one aggregate over imbalanced groups. With `5000` balanced resamples of `2048` singular and `2048` regular patch positions, image-block bootstraps, and within-image label permutations, the local claim holds: local-ball KS diff is `-0.0061`, balanced 95% CI `[-0.0121, 0.00004]`, image-block CI `[-0.0105, -0.0017]`, within-image p=`0.0032`; local-ball entropy diff is `+0.0074`, balanced CI `[0.00044, 0.0143]`, image-block CI `[0.00235, 0.0125]`, within-image p=`0.0030`. Frequency-matched and random target-code controls do not reproduce the local entropy/KS direction. Branch flat-tail enrichment is also robust: branch-KS flat decile singular rate `26.7%` vs `23.0%`, within-image p=`0.0008`; branch-entropy flat decile `26.4%` vs `23.0%`, p=`0.0026`.
- The visual branch galleries are qualitative generation checks rather than hypothesis tests. The wider single-token replacement ablation over `12` same-image pairs and `6` top-code branches gives higher local crop diversity for singular anchors: mean `0.0095` vs `0.0047`, median `0.0064` vs `0.0031`, with singular winning `8/12` matched pairs. Letting the pretrained AR model roll out the suffix amplifies the effect: over `6` pairs and `4` branches, mean `0.0395` vs `0.0151`, median `0.0296` vs `0.0107`, with singular winning `5/6` matched pairs. The visual pattern is what the polysemy claim predicts: singular branch patches can roll into different object/material interpretations, while matched regular patches usually preserve the same continuation.
- The densest replacement sweep so far uses `32` same-image singular/control pairs and `8` top-code branches per anchor, i.e. `64` anchors and `512` branch variants before counting originals. The singular anchors remain more polysemous: mean local crop diversity `0.0070` vs `0.0024`, median `0.0061` vs `0.0012`, with singular winning `23/32` matched pairs. This is the better vision-specific analogue of dense sampling: many local patch interventions across the image lattice, not one anecdotal continuation.
- Paired inference makes the dense visual result more solid. For the flatness-ranked dense replacement sweep, the mean paired crop-diversity lift is `+0.00452`, bootstrap 95% CI `[0.00231, 0.00744]`, sign-test p=`0.010`, and paired sign-flip p=`0.00004`. A fixed-order guardrail run using the first eligible codebook-singular anchors, without selecting for AR flatness, is neutral: mean lift `+0.00016`, bootstrap 95% CI `[-0.00089, 0.00112]`, sign-test p=`0.430`, and paired sign-flip p=`0.383`.
- Interpretation: the corrected ImageNet-val result supports a narrow local polysemy claim for a branch-flat subset of large-radius fiber-singular VQ code IDs. It does not support the stronger claim that every codebook-singular target is visually polysemous, or that singular targets are uniformly flatter under every full-vocabulary or top-branch metric.

Key artifacts:

- Codebook run: `runs/local/pretrained_vq_codebook/llamagen_c2i_B_256_codebook_stratification/`
- Codebook W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/bny85afb>
- Broad target selector W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/74ezbarg>
- Large-fiber target selector W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/ge76qm8j>
- Large-fiber target selector with random/frequency controls W&B, using per-group fraction histograms: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/1869sonz>
- ImageNet-val 256-image encode W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/h3l3yziq>
- ImageNet-val 64-image AR controls W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/pnffg938>
- ImageNet-val random patch hypothesis W&B: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/kpbkr8ex>
- ImageNet-val branch-rollout gallery: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_rollout/`
- ImageNet-val single-token replacement gallery: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement/`
- Expanded ImageNet-val branch-rollout gallery: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_rollout_pairs6/`
- Expanded ImageNet-val replacement gallery: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement_pairs12/`
- Dense ImageNet-val replacement gallery and summary: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement_pairs32_dense/`
- Fixed-order guardrail replacement gallery and summary: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/llamagen_c2i_B_256_imagenet_val64_large_fiber_replacement_pairs32_position/`
- Paired inference outputs: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/paired_inference/` and `runs/local/pretrained_vq_ar_polysemy_branch_gallery/paired_inference_position_guardrail/`
- Polysemy result summary figure: `runs/local/pretrained_vq_ar_polysemy_branch_gallery/vq_ar_polysemy_more_results_summary.png`
- Representative figures: `vq_codebook_pca_singular.png`, `vq_codebook_dimension_curves.png`, `vq_ar_ranked_ks_singular_vs_regular.png`, and `vq_ar_singular_definition_sensitivity.png`.

Protocol audit:

- The original token-embedding paper's Algorithm 1 computes two p-value families from sliding log-volume/log-radius slope windows: a two-sided manifold test for any slope change, and a one-sided fiber-bundle test for slope increases as radius grows. Both families are Holm-Bonferroni corrected.
- In the corrected vision setup, singular-token detection happens on the VQ codebook embedding table, not on AR hidden-state occurrences. Hidden-state tests are still useful as downstream diagnostics, but they are not the primary singular-token detector.
- The earlier one-sample KS radius-threshold idea asks a different question: whether empirical points inside a local ball look radially uniform at some radius. Those runs showed that a near-uniform radius threshold exists broadly, but not specifically at hidden-state singular centers.

Superseded hidden-state/radius-threshold diagnostics, 2026-06-27:

For each hidden-state token position, the old radius-threshold probe swept local embedding balls from neighbor volume 20 to 200 in steps of 5, used each boundary neighbor as `R`, estimated local intrinsic dimension from trimmed log-radius ratios, transformed inner radii by `u=(r/R)^d`, and ran a one-sample KS test against `Uniform(0,1)`. The operational threshold was the first radius that was near-uniform for two consecutive swept radii: `KS <= 0.15` and KS p-value `>= 0.05`.

| Run | Singular selector | Singular contexts | Threshold rate, singular | Threshold rate, regular | Threshold radius, singular | Threshold radius, regular | Radius p | Readout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Generated LlamaGen-B-256, 16 samples | `paper_fiber_any` | 18/4096 | 1.000 | 0.995 | 142.24 | 145.38 | 0.566 | threshold exists broadly, not singular-specific |
| COCO val2017 stress test, 16 images | `paper_fiber_any` | 17/4096 | 1.000 | 0.996 | 123.84 | 124.43 | 0.926 | threshold exists broadly, not singular-specific |

The best radius in the sweep also does not support a singular-specific effect: generated best KS is 0.0945 vs 0.0910 (p=0.105), and COCO best KS is 0.0936 vs 0.0915 (p=0.355). The fixed-volume diagnostic at volume 50 is retained as a reference only: generated KS is 0.1383 vs 0.1340 (p=0.509), and COCO KS is 0.1322 vs 0.1344 (p=0.753).

The AR branch diagnostics remain useful but answer a different question. On generated samples, top-32 branch KS is 0.1830 vs 0.1788 (p=0.827) and permuted full-vocabulary KS is 0.0505 vs 0.0494 (p=0.900). On COCO, branch KS is 0.1813 vs 0.1520 (p=0.107) and permuted KS is 0.0496 vs 0.0354 (p=0.045), which points against branch flattening under random class conditioning.

Artifacts:

- `runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_16_seed20260627_algorithm1_fiber_embeddingball50/`
- `runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_seed20260627_algorithm1_fiber_embeddingball50/`
- `runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_16_seed20260627_algorithm1_fiber_radius_threshold/`
- `runs/local/pretrained_vq_ar_ks/llamagen_c2i_B_256_coco_val16_seed20260627_algorithm1_fiber_radius_threshold/`
- W&B radius-threshold run: <https://wandb.ai/helloimlixin-rutgers/stratified-manifold-learning/runs/fprq5aag>
- Key radius-threshold figures in each directory: `vq_ar_embedding_radius_threshold_singular_vs_regular.png`, `vq_ar_embedding_radius_best_ks_singular_vs_regular.png`, and `vq_ar_embedding_radius_uniformity_curve.png`.

- `*_algorithm1_fiber_fixed/` contains the corrected Algorithm-1 AR branch-only runs before the embedding-ball geometry test was added.
- `*_algorithm1_fiber_localball32/` contains the earlier local-ball AR-probability diagnostic. It restricted the model distribution to codes observed among neighboring contexts, so it does not test the empirical embedding distribution inside the ball and should not be used as the answer to the geometry-uniformity claim.

## Pretrained VQ-VAE + Toy ImageGPT Baseline

For a minimal discrete modeling baseline only, use the pretrained FoundationVision/VAR VQ-VAE as the visual tokenizer and train a compact ImageGPT-style causal transformer over the resulting VQ code IDs. This is intentionally not the high-quality generator; use `scripts/pretrained_var_generator.py` for pretrained checkpoint sampling. A pretrained pixel-ImageGPT checkpoint is not directly plug-compatible here because its vocabulary is pixel tokens, not VQ-VAE codebook entries.

```bash
python scripts/pretrained_vqvae_imagegpt_pipeline.py \
  --image-dir docs/imgs/neurips_submission \
  --out-dir runs/local/vqvae_imagegpt/docs_gallery_smoke \
  --epochs 40 \
  --samples 8 \
  --wandb \
  --wandb-name vqvae-imagegpt-docs-gallery-smoke
```

The script writes input/reconstruction/sample grids, a loss curve, VQ tokens, the tiny GPT checkpoint, a JSON summary, and optional W&B media.

## Vision Branching KS Probe

The language-polysemy analogue is tested as branch flattening in vision: singular patches should have higher local branch entropy, lower top-branch margin, and a robust sliced-KS shift relative to regular patches. The probe logs CDFs, sliced-KS permutation histograms, patch heatmaps, top-token galleries, metrics, and output artifacts to W&B.

Quick no-Torch smoke run over an image folder:

```bash
python scripts/vision_branching_ks_probe.py \
  --image-dir docs/imgs/neurips_submission \
  --out-dir runs/local/vision_branching_ks/docs_gallery_smoke \
  --wandb \
  --wandb-name vision-branching-ks-docs-gallery-smoke
```

Saved-token run against an existing fiber artifact:

```bash
python scripts/vision_branching_ks_probe.py \
  --embeddings runs/local/coco_sam_fiber/20260509_191042/embeddings/epoch_000.pt \
  --fiber-results runs/local/coco_sam_fiber/20260509_191042/checkpoints/fiber_epoch_000.json \
  --dataset COCO \
  --out-dir runs/local/vision_branching_ks/coco_sam_20260509 \
  --wandb \
  --wandb-name vision-branching-ks-coco-sam
```

## Outputs

Hydra writes runs under `${LLM_STRATIFIED_OUTPUT_ROOT:-runs}/hydra/...` unless an experiment or wrapper sets `hydra.run.dir`. Local paper runs usually live under `runs/local/<experiment>/<timestamp>/`.

Important outputs include:

- `checkpoints/fiber_analysis/`: local dimension maps, irregularity heatmaps, patch galleries, and volume-curve galleries.
- `checkpoints/fiber_history.json`: per-run fiber-bundle summaries.
- `checkpoints/sparse_probe_summary.json`: fixed-neighborhood local dictionary summaries when sparse probes are enabled.
- `token_processing_notes.md`: token collection details for SAM and frozen-token runs.
- `docs/imgs/neurips_submission/`: lightweight representative figure gallery.

## Documentation Map

- `runs/`: local run ledgers, JSON summaries, and heavy figures.
- W&B project `stratified-manifold-learning`: synced metrics, media, and artifacts for current probes.
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
