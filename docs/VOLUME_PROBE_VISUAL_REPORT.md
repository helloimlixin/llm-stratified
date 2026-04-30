# Volume Probe Visual Report

Generated: 2026-04-22 00:00 EDT

## Source Run
- Sweep root: `/scratch/xl598/runs/llm-stratified/20260413_154940_volume_probe_visual_sweep`
- Combined report: `/scratch/xl598/runs/llm-stratified/20260413_154940_volume_probe_visual_sweep/volume_probe_sweep_report.md`
- Coverage: `18/18` completed runs
- SLURM jobs: `51035214` (`pixel-strat-sweep`), `51035215` (`pretrained-pixel-sweep`)

## Purpose
This note condenses the completed April 13, 2026 volume-probe sweep into a paper-style figure set. The selected panels emphasize the same objects the sweep was designed to measure: local dimension distributions, irregularity concentration in a 2D projection, and local log-radius versus log-k scaling curves for representative anchors.

## Main Takeaways
- Raw pixel patches remained strongly stratified across the sweep. The strongest raw-pixel run was `STL10`, patch `8`, stride `8`, with `mean_dim = 13.20` and `irregular_ratio = 0.875`.
- TinyViT tokens were typically more irregular than matched raw pixels. In the pure pixel sweep, the token-minus-raw irregularity delta averaged `+0.050`.
- Overlap was not the main driver. For matched raw-pixel comparisons, half-stride overlap increased irregularity in only `1/4` cases. The largest increase was `STL10`, patch `16`, from `0.861` to `0.873`.
- Frozen DINO patch embeddings were smoother than DINO last-layer tokens on both shared datasets. On `STL10`, DINO patch embeddings reached `mean_dim = 25.91` with `irregular_ratio = 0.814`, while DINO last-layer tokens were `mean_dim = 5.62` with `irregular_ratio = 0.895`.
- DINO last-layer tokens were not smoother than matched raw pixels in this sweep. Across the `6` completed DINO runs, mean raw-pixel irregularity was `0.852` and mean token irregularity was `0.897`.

## Selected Metrics
| Case | Raw pixels | Tokens | Patch embeddings |
| --- | --- | --- | --- |
| STL10, patch 8, stride 8 | `13.20 / 0.875` | `8.42 / 0.944` | `12.66 / 0.868` |
| STL10, patch 16, stride 16 | `16.00 / 0.861` | `6.10 / 0.927` | `15.46 / 0.854` |
| STL10, patch 16, stride 8 | `17.60 / 0.873` | `6.10 / 0.927` | `15.46 / 0.854` |
| STL10, DINO last, stride 14 | `10.41 / 0.877` | `5.62 / 0.895` | `25.91 / 0.814` |

Metric cells use `mean_dim / irregular_ratio`.

## Figure 1: Strongest Raw Pixel Regime
`STL10`, patch `8`, stride `8` is the strongest raw-pixel configuration in the sweep. The detail panel shows a broad high-dimension raw-pixel distribution together with concentrated irregular regions in the PCA projection. The scaling panel shows that the selected anchors retain nontrivial slope changes rather than collapsing to a single smooth regime.

![Raw pixel detail, STL10 patch 8 stride 8](imgs/volume_probe_visual_report/raw_detail_stl10_ps8_stride8.png)

![Raw pixel scaling, STL10 patch 8 stride 8](imgs/volume_probe_visual_report/raw_scaling_stl10_ps8_stride8.png)

## Figure 2: TinyViT Tokens on the Same STL10 Run
On the same `STL10`, patch `8`, stride `8` run, TinyViT tokens were more irregular than the raw patches they were built from: `0.944` versus `0.875`. The detail panel reflects that compression into a lower-dimensional but more singular token geometry, and the scaling panel shows sharper regime transitions for the selected anchors.

![TinyViT token detail, STL10 patch 8 stride 8](imgs/volume_probe_visual_report/tinyvit_tokens_detail_stl10_ps8_stride8.png)

![TinyViT token scaling, STL10 patch 8 stride 8](imgs/volume_probe_visual_report/tinyvit_tokens_scaling_stl10_ps8_stride8.png)

## Figure 3: Overlap Comparison
This pair isolates the raw-pixel overlap effect on `STL10`, patch `16`. The non-overlapping baseline (`stride 16`) had `irregular_ratio = 0.861`, while the half-stride run (`stride 8`) rose modestly to `0.873`. The difference is visible, but the sweep-level result is that overlap only weakly perturbed the already-strong stratification signal.

Left: non-overlapping baseline (`stride 16`). Right: overlapping raw-pixel probe (`stride 8`).

![Raw pixel detail, STL10 patch 16 stride 16 baseline](imgs/volume_probe_visual_report/raw_detail_stl10_ps16_stride16_baseline.png)

![Raw pixel detail, STL10 patch 16 stride 8 overlap](imgs/volume_probe_visual_report/raw_detail_stl10_ps16_stride8_overlap.png)

## Figure 4: DINO Patch Embeddings Versus DINO Tokens
For `STL10` with frozen DINOv2-base at native patch size `14`, patch embeddings were visibly smoother than last-layer tokens. Quantitatively, patch embeddings had `irregular_ratio = 0.814`, while tokens had `0.895`. The detail panels show the same split: patch embeddings preserve a cleaner high-dimensional structure, whereas the final token space is more singular.

![DINO patch embedding detail, STL10](imgs/volume_probe_visual_report/dino_patch_detail_stl10.png)

![DINO patch embedding scaling, STL10](imgs/volume_probe_visual_report/dino_patch_scaling_stl10.png)

![DINO token detail, STL10](imgs/volume_probe_visual_report/dino_token_detail_stl10.png)

![DINO token scaling, STL10](imgs/volume_probe_visual_report/dino_token_scaling_stl10.png)

## Notes
- The full run root also contains `nn_irregular_*` retrieval grids for each representation; those were kept out of this condensed note to keep the figure set focused.
- Selected panels were copied into `docs/imgs/volume_probe_visual_report/` so the report stays stable even if the scratch run tree changes.
- The quantitative summary in this note comes directly from the completed report generated on 2026-04-14.
