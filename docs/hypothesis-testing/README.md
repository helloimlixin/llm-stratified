# Testing Stratified Manifold Hypothesis in Vision

This folder contains the July 30, 2026 revision of the companion statistical
note. It connects a global local-Euclidean hypothesis to a finite-sample test of
local radial volume growth.

## Contents

- [`Testing Stratified Manifold Hypothesis in Vision.tex`](Testing%20Stratified%20Manifold%20Hypothesis%20in%20Vision.tex): LaTeX source.
- [`Testing Stratified Manifold Hypothesis in Vision.pdf`](Testing%20Stratified%20Manifold%20Hypothesis%20in%20Vision.pdf): rendered PDF.
- [`images/`](images/): the eight synthetic, ImageNet, LlamaGen, and VAR figures
  required by the LaTeX source.

## Test

Under local intrinsic-volume uniformity,

\[
\Pr(\lVert X-t\rVert\le r\mid \lVert X-t\rVert\le r_*)=(r/r_*)^d.
\]

The probability of shell `(r_{k-1}, r_k]` is therefore

\[
p_k(d)=\frac{r_k^d-r_{k-1}^d}{r_*^d}.
\]

The implementation fits the local dimension, constructs eight equal-null-mass
shells, and uses the multinomial deviance

\[
T_N=2N D_{\mathrm{KL}}(\widehat Q\|P).
\]

The complete fit-and-bin procedure is calibrated with 50,000 Monte Carlo null
replicates. The synthetic experiment stays near the nominal 5% size and gains
power as the neighborhood grows.

## Results

- ImageNet patch features reject at 33.04% of anchors.
- The normalized LlamaGen codebook rejects at 86.83%.
- The normalized VAR codebook rejects at 13.28%.
- VAR-d16 and VAR-d30 show no reliable image-level association between rejected
  target codes and generator uncertainty.

The supported conclusion is that vision representations have
architecture-dependent local measure geometry. This is compatible with, but
does not prove, a stratified organization.

## Build

From this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error "Testing Stratified Manifold Hypothesis in Vision.tex"
pdflatex -interaction=nonstopmode -halt-on-error "Testing Stratified Manifold Hypothesis in Vision.tex"
```

The checked-in PDF and manuscript figures allow the note to be read and rebuilt
without the larger experiment directories.
