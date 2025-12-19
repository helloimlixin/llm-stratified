# Results for Fiber Bundle Tests on TinyVIT


## Run configuration (from captured run config)

- **Script**: `tinyvit_fiber_bundle.py`
- **Dataset**: `STL10`
- **Image size**: 96
- **Patch size**: 32
- **Epochs**: 200
- **Batch size**: 64
- **Subset**:
  - train: 5000 (default; not explicitly passed in the CLI args list)
  - test: 64
- **Embed interval**: 10 (fiber/embedding visualizations saved at checkpoint epochs)
- **Max tokens (for embeddings/visuals)**: 8192

## Reproduction command

```bash
python ~/llm-stratified/tinyvit_fiber_bundle.py \
  --dataset STL10 \
  --root ~/llm-stratified/data \
  --img-size 96 \
  --patch-size 32 \
  --epochs 200 \
  --embed-interval 10 \
  --subset-test 64 \
  --batch-size 64 \
  --max-tokens 8192 \
  --outdir ~/llm-stratified/runs/fiber_test_stl10_96 \
  --wandb \
  --project tinyvit_fiber \
  --run-name fiber_test_stl10_96
```

---

## Epoch progression — training metrics

This table records metrics every **20 epochs** (plus the final epoch).

| epoch | lr | train_loss | train_acc | val_loss | val_acc |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.00012 | 2.0447 | 0.2364 | 1.9602 | 0.2344 |
| 20 | 0.000295044 | 1.4255 | 0.4671 | 1.5411 | 0.4375 |
| 40 | 0.000275471 | 1.2253 | 0.5473 | 1.5634 | 0.4062 |
| 60 | 0.000242983 | 1.0585 | 0.6140 | 1.7417 | 0.4219 |
| 80 | 0.000200924 | 0.9080 | 0.6739 | 1.9589 | 0.3906 |
| 100 | 0.000153625 | 0.7207 | 0.7446 | 1.9209 | 0.4219 |
| 120 | 0.000105952 | 0.6001 | 0.7891 | 2.0190 | 0.3906 |
| 140 | 6.28126e-05 | 0.5111 | 0.8225 | 2.0647 | 0.4375 |
| 160 | 2.86475e-05 | 0.4533 | 0.8405 | 2.1538 | 0.4375 |
| 180 | 6.97277e-06 | 0.4238 | 0.8562 | 2.1982 | 0.3906 |
| 199 | 0 | 0.4236 | 0.8490 | 2.2040 | 0.4219 |

### Notes / interpretation

- **Generalization trend**: Validation improves early (val loss drops from **1.960** @ epoch 0 to **~1.54** @ epoch 20), then val loss trends upward after ~40–60 epochs (ending at **2.204** @ epoch 199) while val accuracy largely plateaus/fluctuates (~0.39–0.44 in this 20-epoch sampling).
- **Early stopping candidate**: A reasonable early-stop window is **~20–60 epochs**. Note: the *true* best epoch may fall between 20-epoch checkpoints; consult the full `train_history.json` if you want the exact best-epoch selection.
- **Overfitting check**: Train loss steadily falls (**2.045 → 0.424**) and train acc rises (**0.236 → 0.849**) while val loss worsens, indicating overfitting past the early/mid training phase (especially after ~60 epochs).

---

## Epoch progression — fiber diagnostics (checkpoint epochs)

Fiber diagnostics are computed and saved at the same checkpoint cadence as `--embed-interval` (here: every 10 epochs, stored as epochs `0, 9, 19, ..., 199`). To match the “every 20 epochs” reporting style, we sample the closest available checkpoint epochs: `0, 19, 39, ..., 199`.

| epoch | mean_dim | median_dim | mean_neigh_dim | irregular_ratio | mean_irregularity | max_irregularity | min_pvalue |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.444 | 5.963 | 6.469 | 0.0885 | 0.2837 | 4.931 | 1.17e-05 |
| 19 | 8.914 | 8.613 | 8.959 | 0.0503 | 0.1687 | 4.401 | 3.97e-05 |
| 39 | 9.789 | 9.469 | 9.852 | 0.0677 | 0.2129 | 5.165 | 6.84e-06 |
| 59 | 10.502 | 10.145 | 10.570 | 0.0694 | 0.2252 | 4.770 | 1.7e-05 |
| 79 | 10.845 | 10.461 | 10.888 | 0.0677 | 0.2222 | 6.646 | 2.26e-07 |
| 99 | 11.247 | 11.016 | 11.298 | 0.0660 | 0.2185 | 4.638 | 2.3e-05 |
| 119 | 11.231 | 10.958 | 11.288 | 0.0590 | 0.1948 | 4.648 | 2.25e-05 |
| 139 | 11.415 | 11.220 | 11.482 | 0.0642 | 0.2073 | 4.984 | 1.04e-05 |
| 159 | 11.497 | 11.261 | 11.546 | 0.0660 | 0.2083 | 4.353 | 4.43e-05 |
| 179 | 11.591 | 11.366 | 11.633 | 0.0434 | 0.1391 | 4.582 | 2.62e-05 |
| 199 | 11.588 | 11.353 | 11.630 | 0.0677 | 0.2225 | 4.411 | 3.88e-05 |

### Notes / interpretation

- **Dimensionality growth**: `mean_dim` increases rapidly early (**6.44 → 8.91** by epoch 19) and then slows toward a plateau (**11.25 @ 99 → 11.59 @ 199**). `mean_neigh_dim` tracks closely with `mean_dim` throughout.
- **Irregularity behavior**: `irregular_ratio` starts highest at epoch 0 (**0.0885**) and is generally lower afterward, with the lowest value at epoch 179 (**0.0434**). In this run, irregularity does **not** show a sustained upward drift as training proceeds.
- **Coupling with accuracy**: Across the checkpoint epochs in this table, `mean_dim` has a **positive correlation** with `val_acc` (Pearson \(r \approx 0.69\)), while `irregular_ratio` and `mean_irregularity` are **negatively correlated** with `val_acc` (\(r \approx -0.84\) and \(r \approx -0.83\), respectively).

---

## Visualizations

Here are the visualizations of the training experiment with Fiber Bundle Tests.

### Class-wise dimension summary (selected checkpoints)

#### Epoch 000

![](imgs/class_dims/class_dims_epoch_000.png)

#### Epoch 049

![](imgs/class_dims/class_dims_epoch_049.png)

#### Epoch 099

![](imgs/class_dims/class_dims_epoch_099.png)

#### Epoch 149

![](imgs/class_dims/class_dims_epoch_149.png)

#### Epoch 199

![](imgs/class_dims/class_dims_epoch_199.png)

### Fiber bundle summary (epoch progression)

> These plots were copied into `imgs/fiber/` so the markdown renders cleanly on GitHub. We show the same checkpoints as the class-wise summaries for easier comparison.

#### Fiber / token slot counts

##### Epoch 000

![](imgs/fiber/token_slot_counts_1_0b599cfbd27b64de6295.png)

##### Epoch 049

![](imgs/fiber/token_slot_counts_60_0b599cfbd27b64de6295.png)

##### Epoch 099

![](imgs/fiber/token_slot_counts_120_0b599cfbd27b64de6295.png)

##### Epoch 149

![](imgs/fiber/token_slot_counts_180_0b599cfbd27b64de6295.png)

##### Epoch 199

![](imgs/fiber/token_slot_counts_240_0b599cfbd27b64de6295.png)

#### Fiber / patch count curve

##### Epoch 000

![](imgs/fiber/patch_count_curve_1_020eae0913c404cb4e40.png)

##### Epoch 049

![](imgs/fiber/patch_count_curve_60_622d0ae45edebb5618d3.png)

##### Epoch 099

![](imgs/fiber/patch_count_curve_120_ab38bf9950b085094894.png)

##### Epoch 149

![](imgs/fiber/patch_count_curve_180_1e5987e4bf2a22319ab6.png)

##### Epoch 199

![](imgs/fiber/patch_count_curve_240_677db3a3067f50c2f646.png)

#### Fiber / dim–radius scatter

##### Epoch 000

![](imgs/fiber/dim_radius_scatter_1_7b8863331918209b4b87.png)

##### Epoch 049

![](imgs/fiber/dim_radius_scatter_60_e18da662e662e0d284dc.png)

##### Epoch 099

![](imgs/fiber/dim_radius_scatter_120_7f1715ea2971a6ea2591.png)

##### Epoch 149

![](imgs/fiber/dim_radius_scatter_180_6c304ee05650822fb9ca.png)

##### Epoch 199

![](imgs/fiber/dim_radius_scatter_240_1d7fc831b7ee3d935a65.png)

### Fiber analysis panels (selected checkpoints)

For each checkpoint epoch, these are available under `runs/fiber_test_stl10_96/fiber_analysis/`:

- `epoch_XXX_low_dim_panel.png`, `epoch_XXX_high_dim_panel.png`
- `epoch_XXX_token_slot_counts.png`
- `epoch_XXX_patch_count.png`
- `epoch_XXX_slope_radius.png`
- token patch/radius previews and top-token examples

#### Epoch 000

| token slot counts | patch count | slope vs radius |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_000_token_slot_counts.png) | ![](imgs/fiber_analysis/epoch_000_patch_count.png) | ![](imgs/fiber_analysis/epoch_000_slope_radius.png) |

| low-dim panel | high-dim panel |
|---|---|
| ![](imgs/fiber_analysis/epoch_000_low_dim_panel.png) | ![](imgs/fiber_analysis/epoch_000_high_dim_panel.png) |

#### Epoch 099

| token slot counts | patch count | slope vs radius |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_099_token_slot_counts.png) | ![](imgs/fiber_analysis/epoch_099_patch_count.png) | ![](imgs/fiber_analysis/epoch_099_slope_radius.png) |

| low-dim panel | high-dim panel |
|---|---|
| ![](imgs/fiber_analysis/epoch_099_low_dim_panel.png) | ![](imgs/fiber_analysis/epoch_099_high_dim_panel.png) |

#### Epoch 199

| token slot counts | patch count | slope vs radius |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_199_token_slot_counts.png) | ![](imgs/fiber_analysis/epoch_199_patch_count.png) | ![](imgs/fiber_analysis/epoch_199_slope_radius.png) |

| low-dim panel | high-dim panel |
|---|---|
| ![](imgs/fiber_analysis/epoch_199_low_dim_panel.png) | ![](imgs/fiber_analysis/epoch_199_high_dim_panel.png) |

---

## Token examples (epoch progression)

These plots are already saved under `runs/fiber_test_stl10_96/fiber_analysis/` as:

- radius curves: `epoch_XXX_{low,mid,high}_dim_token_radius.png`
- patch previews: `epoch_XXX_{low,mid,high}_dim_token_patch.png`

### Epoch 000

| token_radius/low_dim_token | token_radius/mid_dim_token | token_radius/high_dim_token |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_000_low_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_000_mid_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_000_high_dim_token_radius.png) |

| token_patch/low_dim_token | token_patch/mid_dim_token | token_patch/high_dim_token |
|---|---|---|
| <img src="imgs/fiber_analysis/epoch_000_low_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_000_mid_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_000_high_dim_token_patch.png" width="320"> |

### Epoch 049

| token_radius/low_dim_token | token_radius/mid_dim_token | token_radius/high_dim_token |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_049_low_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_049_mid_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_049_high_dim_token_radius.png) |

| token_patch/low_dim_token | token_patch/mid_dim_token | token_patch/high_dim_token |
|---|---|---|
| <img src="imgs/fiber_analysis/epoch_049_low_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_049_mid_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_049_high_dim_token_patch.png" width="320"> |

### Epoch 099

| token_radius/low_dim_token | token_radius/mid_dim_token | token_radius/high_dim_token |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_099_low_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_099_mid_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_099_high_dim_token_radius.png) |

| token_patch/low_dim_token | token_patch/mid_dim_token | token_patch/high_dim_token |
|---|---|---|
| <img src="imgs/fiber_analysis/epoch_099_low_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_099_mid_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_099_high_dim_token_patch.png" width="320"> |

### Epoch 149

| token_radius/low_dim_token | token_radius/mid_dim_token | token_radius/high_dim_token |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_149_low_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_149_mid_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_149_high_dim_token_radius.png) |

| token_patch/low_dim_token | token_patch/mid_dim_token | token_patch/high_dim_token |
|---|---|---|
| <img src="imgs/fiber_analysis/epoch_149_low_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_149_mid_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_149_high_dim_token_patch.png" width="320"> |

### Epoch 199

| token_radius/low_dim_token | token_radius/mid_dim_token | token_radius/high_dim_token |
|---|---|---|
| ![](imgs/fiber_analysis/epoch_199_low_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_199_mid_dim_token_radius.png) | ![](imgs/fiber_analysis/epoch_199_high_dim_token_radius.png) |

| token_patch/low_dim_token | token_patch/mid_dim_token | token_patch/high_dim_token |
|---|---|---|
| <img src="imgs/fiber_analysis/epoch_199_low_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_199_mid_dim_token_patch.png" width="320"> | <img src="imgs/fiber_analysis/epoch_199_high_dim_token_patch.png" width="320"> |

---

## Embeddings: low/high-dim patches (epoch progression)

These are the “low/high dimension patch panels” saved as:

- `runs/fiber_test_stl10_96/fiber_analysis/epoch_XXX_low_dim_panel.png`
- `runs/fiber_test_stl10_96/fiber_analysis/epoch_XXX_high_dim_panel.png`

### Epoch 000

#### embeddings/low_dim_patches

<img src="imgs/fiber_analysis/epoch_000_low_dim_panel.png" width="1400">

#### embeddings/high_dim_patches

<img src="imgs/fiber_analysis/epoch_000_high_dim_panel.png" width="1400">

### Epoch 049

#### embeddings/low_dim_patches

<img src="imgs/fiber_analysis/epoch_049_low_dim_panel.png" width="1400">

#### embeddings/high_dim_patches

<img src="imgs/fiber_analysis/epoch_049_high_dim_panel.png" width="1400">

### Epoch 099

#### embeddings/low_dim_patches

<img src="imgs/fiber_analysis/epoch_099_low_dim_panel.png" width="1400">

#### embeddings/high_dim_patches

<img src="imgs/fiber_analysis/epoch_099_high_dim_panel.png" width="1400">

### Epoch 149

#### embeddings/low_dim_patches

<img src="imgs/fiber_analysis/epoch_149_low_dim_panel.png" width="1400">

#### embeddings/high_dim_patches

<img src="imgs/fiber_analysis/epoch_149_high_dim_panel.png" width="1400">

### Epoch 199

#### embeddings/low_dim_patches

<img src="imgs/fiber_analysis/epoch_199_low_dim_panel.png" width="1400">

#### embeddings/high_dim_patches

<img src="imgs/fiber_analysis/epoch_199_high_dim_panel.png" width="1400">

---

## Embeddings: irregular samples (epoch progression)

These images are saved under:

- `imgs/embeddings/irregular_samples_*.png`

Each epoch snapshot below shows 12 “most irregular” samples (with heatmap overlays).

### Epoch 000

| | | | |
|---|---|---|---|
| <img src="imgs/embeddings/irregular_samples_2_1728b27133f8bd2f06e5.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_f33a8414b168b7447e63.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_80e5d27f1301ce6f01ba.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_f6de53f6bbf79d2624b7.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_2_c455691d12a8e6a89590.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_51081fe164e0281e7118.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_3d2a8db80612a74d1e86.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_c9e9b17095da73e4f7fc.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_2_385749c30c3f2ceb1389.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_756cbc30a2402dd7d614.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_77d65d07ff54de13c172.png" width="260"> | <img src="imgs/embeddings/irregular_samples_2_97e20d2d34b36c283bcb.png" width="260"> |

### Epoch 049

| | | | |
|---|---|---|---|
| <img src="imgs/embeddings/irregular_samples_61_fce677509b0ea54138b8.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_048ba756127abb810125.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_78894caa5b2fe4188633.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_b2c09e6cd04ae08c0818.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_61_4ec4160a18bae8357bbe.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_492f82945f8d3bc03aac.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_e58a8c41303ac8ae224e.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_e888fbdaa5f52a606d99.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_61_38e6f48c76ef8a14b475.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_25b8de85b29abb6ae5fe.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_363093779598a1b107ea.png" width="260"> | <img src="imgs/embeddings/irregular_samples_61_7780ebd724f59d2d3754.png" width="260"> |

### Epoch 099

| | | | |
|---|---|---|---|
| <img src="imgs/embeddings/irregular_samples_121_389c6bdcb433bf59ad6e.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_d113ef62447d768130e5.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_add067fdc65540adbdbf.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_3a7dd832b29ffde969d3.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_121_2f2f94fd67affdb6bcda.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_79c256680b6597a79366.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_1d7d37b2d6e9065d8496.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_3a79d55b524db5073468.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_121_b0b8a9f0a51f34aa4577.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_7acfd97704281e76bd73.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_1b2f32458d017842efb0.png" width="260"> | <img src="imgs/embeddings/irregular_samples_121_cecbf979c5e017570487.png" width="260"> |

### Epoch 149

| | | | |
|---|---|---|---|
| <img src="imgs/embeddings/irregular_samples_181_6d0e003a2f7f7bdc1404.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_350482a5be6cf8557a6f.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_392a09fc7434e7baf066.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_47de9816361c723cae27.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_181_46134c74f7b261ae41e4.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_6cb4abb9f41ce11571e3.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_dfd00966bc58a382b3e9.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_8dc0853d188c5f1a93ea.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_181_4426555b624b3b3dabec.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_c1f7c92264b62bd1f541.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_ed479457c2f820531b65.png" width="260"> | <img src="imgs/embeddings/irregular_samples_181_84620e8b321a86af9256.png" width="260"> |

### Epoch 199

| | | | |
|---|---|---|---|
| <img src="imgs/embeddings/irregular_samples_241_88e58f05a40572feee76.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_8b5f4bf7369cf4aa6710.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_bef125e35f62ca152853.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_3075500680d1bd160616.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_241_860c5275f278f4a95f98.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_205dcee0bcbdce389067.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_ddb9681fda876f67751c.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_712664d132387b105c0f.png" width="260"> |
| <img src="imgs/embeddings/irregular_samples_241_04ab529252386e4f3177.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_56a2323f2beda2a73e0c.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_0ed7bbf7e4f0a1fd4e8e.png" width="260"> | <img src="imgs/embeddings/irregular_samples_241_24fa7e397a114f352cf9.png" width="260"> |




