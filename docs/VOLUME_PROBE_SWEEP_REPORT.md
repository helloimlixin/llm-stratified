# Volume Probe Sweep Report

Generated: 2026-04-11 22:18 EDT

## Inputs
- `/scratch/xl598/runs/llm-stratified/20260411_volume_probe_parallel/pixel_stratification_sweep`
- `/scratch/xl598/runs/llm-stratified/20260411_volume_probe_parallel/pretrained_pixel_sweep`
- Aggregated JSON: `/scratch/xl598/runs/llm-stratified/20260411_volume_probe_parallel/volume_probe_sweep_report.json`

## Coverage
| sweep | completed_runs | incomplete_dirs |
| --- | --- | --- |
| pixel_stratification_sweep | 10 | 0 |
| pretrained_pixel_sweep | 8 | 0 |

## Key Findings
- Overlapping raw patches increased irregularity in 1/4 matched dataset/patch pairs; the largest jump was STL10 patch 16 (0.861 -> 0.873).
- The most stratified raw-pixel configuration was STL10 patch 8 stride 8 (irregular_ratio=0.875), while the smoothest completed raw-pixel run was CIFAR10 patch 16 stride 8 (irregular_ratio=0.793).
- Across the pure pixel sweep, token irregularity was 0.052 higher than the matched raw-pixel probe on average (mean token-minus-pixel delta).
- DINO token representations were less irregular than their matched raw-pixel probes in 0/6 completed DINO runs (mean raw=0.852, mean token=0.900).
- In the multilayer DINO runs, irregularity decreased monotonically from layer 03 to 06 to last in 0/2 cases.
- Against the included untrained TinyViT baseline, DINO last-layer token irregularity was lower on 1/2 shared datasets.

## Pixel Sweep
Metric cells use `mean_dim / irregular_ratio`.

| dataset | patch | stride | raw_pixels | tokens | patch_embeddings |
| --- | --- | --- | --- | --- | --- |
| CIFAR10 | 4 | 4 | 8.18 / 0.833 | 4.57 / 0.949 | 7.98 / 0.832 |
| CIFAR10 | 8 | 4 | 12.11 / 0.823 | 7.23 / 0.872 | 11.30 / 0.813 |
| CIFAR10 | 8 | 8 | 11.36 / 0.825 | 7.23 / 0.872 | 11.30 / 0.813 |
| CIFAR10 | 16 | 8 | 15.65 / 0.793 | 14.33 / 0.810 | 14.84 / 0.797 |
| CIFAR10 | 16 | 16 | 15.23 / 0.807 | 14.33 / 0.810 | 14.84 / 0.797 |
| STL10 | 4 | 4 | 8.94 / 0.853 | 10.31 / 0.883 | 8.84 / 0.859 |
| STL10 | 8 | 4 | 14.97 / 0.873 | 8.42 / 0.944 | 12.66 / 0.868 |
| STL10 | 8 | 8 | 13.20 / 0.875 | 8.42 / 0.944 | 12.66 / 0.868 |
| STL10 | 16 | 8 | 17.60 / 0.873 | 6.11 / 0.926 | 15.41 / 0.853 |
| STL10 | 16 | 16 | 16.00 / 0.861 | 6.11 / 0.926 | 15.41 / 0.853 |

## Pretrained Sweep
Metric cells use `mean_dim / irregular_ratio`.

| dataset | variant | patch | stride | raw_pixels | patch_embeddings | tokens_or_last | layer_03 | layer_06 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR10 | dinov2_last | 14 | 7 | 4.90 / 0.838 | 10.36 / 0.807 | 5.10 / 0.906 | n/a | n/a |
| CIFAR10 | dinov2_last | 14 | 14 | 5.32 / 0.833 | 10.36 / 0.807 | 5.10 / 0.906 | n/a | n/a |
| CIFAR10 | dinov2_multilayer | 14 | 14 | 5.32 / 0.833 | 10.36 / 0.807 | 5.10 / 0.906 | 16.62 / 0.793 | 12.38 / 0.849 |
| CIFAR10 | untrained_tinyvit | 8 | 8 | 11.36 / 0.825 | 11.30 / 0.813 | 7.23 / 0.872 | n/a | n/a |
| STL10 | dinov2_last | 14 | 7 | 8.49 / 0.854 | 25.92 / 0.819 | 5.62 / 0.895 | n/a | n/a |
| STL10 | dinov2_last | 14 | 14 | 10.41 / 0.877 | 25.92 / 0.819 | 5.62 / 0.895 | n/a | n/a |
| STL10 | dinov2_multilayer | 14 | 14 | 10.41 / 0.877 | 25.92 / 0.819 | 5.62 / 0.895 | 22.57 / 0.808 | 15.58 / 0.830 |
| STL10 | untrained_tinyvit | 8 | 8 | 13.20 / 0.875 | 12.66 / 0.868 | 8.42 / 0.944 | n/a | n/a |

## Notes
- `raw_pixels` refers to the primary raw-pixel probe for that run: `patch_pixels` for non-overlapping patches or `patch_pixels_stride_<k>` when overlap was enabled.
- `tokens_or_last` maps to `tokens` for TinyViT runs and to the last-layer token representation for DINO runs.
- Comparisons across the DINO and untrained TinyViT rows are informative but not perfectly apples-to-apples because their patch grids differ.
