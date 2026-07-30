# Documentation

The repository has two complementary manuscript bundles:

- [`stratified-spaces/`](stratified-spaces/): the larger submission manuscript
  covering frozen vision encoders, local dimension, fiber violations, sparse
  reconstruction, and generation-side controls.
- [`hypothesis-testing/`](hypothesis-testing/): the self-contained statistical
  note deriving the concentric-shell null, fitted-scale Anderson-Darling test,
  and the ImageNet, LlamaGen, and VAR experiments.

Representative paper figures live under [`imgs/neurips_submission/`](imgs/neurips_submission/).
The hypothesis-testing bundle also keeps its seven required figures locally so
it can be rebuilt from a clean checkout. Full experiment outputs, checkpoints,
and large intermediate arrays remain under `runs/` and are not duplicated into
the documentation tree.
