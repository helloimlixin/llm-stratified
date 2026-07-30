# Testing Stratified Manifold Hypothesis in Vision

This folder contains the July 30, 2026 revision of the companion statistical
note. It formalizes the relationship between the global topological hypothesis
and the stronger local smooth-volume null that can be tested from finite
samples.

## Contents

- [`Testing Stratified Manifold Hypothesis in Vision.tex`](Testing%20Stratified%20Manifold%20Hypothesis%20in%20Vision.tex): LaTeX source.
- [`Testing Stratified Manifold Hypothesis in Vision.pdf`](Testing%20Stratified%20Manifold%20Hypothesis%20in%20Vision.pdf): verified 19-page PDF.
- [`images/`](images/): the seven ImageNet, LlamaGen, and VAR figures required
  by the LaTeX source.

## Current Story

The note derives

\[
\Pr(\|X-t\|\le r\mid \|X-t\|\le r_*)=(r/r_*)^d
\]

under local intrinsic-volume uniformity, and consequently
$\log(r_*/\|X-t\|)\sim\operatorname{Exp}(d)$. A fitted-scale
Anderson-Darling statistic tests this necessary radial signature without
binning the observations.

The empirical synthesis is:

- ImageNet patch features reject the local radial-volume law at 44.52% of
  anchors.
- The normalized LlamaGen codebook rejects at 99.16%.
- The normalized VAR codebook rejects at 27.29%.
- VAR-d16 and VAR-d30 show no reliable image-level association between rejected
  target codes and generator branch uncertainty.

The supported conclusion is that vision representations exhibit
architecture- and location-dependent local measure geometry. This is compatible
with, but does not prove, a stratified manifold organization. Finite VQ
codebooks are treated as geometric proxies for latent supports rather than as
literal positive-dimensional manifolds.

## Build

All required figures are stored in the local `images/` directory. From this
directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error "Testing Stratified Manifold Hypothesis in Vision.tex"
pdflatex -interaction=nonstopmode -halt-on-error "Testing Stratified Manifold Hypothesis in Vision.tex"
```

The checked-in PDF and manuscript figures are intentional so the note can be
read and rebuilt without the full local run artifacts. Auxiliary LaTeX files
remain ignored.
