# Stratified Pixel-Space Framing

## Project Claim

The main claim for this repo should be stated in stratified terms, not classical manifold terms:

- Natural-image data in raw pixel coordinates are continuous, but they are not well modeled as samples from one smooth manifold with uniform local dimension.
- A better working model is that image data form a **stratified space** with locally varying dimension.
- Different patch types, object boundaries, textures, flat regions, repeated structures, and compositional scenes can occupy different local strata.
- Learned visual features may make this geometry more regular, but they do not automatically collapse it to a single manifold.

In short: **continuous image signal removes discretization, not stratification**.

## Relation To The Fiber-Bundle Paper

The paper on token embeddings argues that local neighborhoods can violate both the classical manifold hypothesis and the more permissive fiber-bundle null when the local volume-vs-radius scaling changes in the wrong way. The crucial analogy is geometric, not merely discrete-versus-continuous:

- the token paper does **not** just say that token spaces are singular because tokens are discrete;
- it says that local neighborhoods in token embedding space can violate even a smooth fiber-bundle null;
- the image-space analogue is therefore **not** "pixels are continuous, so the manifold hypothesis is restored";
- instead, the claim is that continuity of the ambient coordinates does not guarantee locally uniform dimension or fiber-bundle structure for the data distribution.

In that sense, the right image-space analogue is:

- text token space is discrete and strongly singular;
- image pixel space is continuous but still geometrically heterogeneous and locally stratified;
- therefore the right test in image space is still a **stratified-manifold / fiber-bundle-style local scaling test**, not a single-manifold fit.

Short version: **continuous ambient coordinates remove discretization, not singular local geometry**.

An additional point in favor of the image-space experiment is that it is, in an important sense, \emph{finer-grained} than the token-embedding setting. The LLM paper studies a fixed discrete vocabulary embedded in latent space. Here, by contrast, we probe continuous raw patch neighborhoods and vary patch geometry across scale and overlap. So if stratification is visible already in image space, it is not a weak surrogate of the token-space result; it is evidence at an even more local geometric resolution.

## Experiment Framing

The clean experimental question is:

**Which image representation is most compatible with a stratified geometric model, and where does the classical manifold hypothesis fail?**

Recommended representation ladder:

1. `patch_pixels`
2. `patch_pixels_multiscale` over a finer patch-size grid
3. `patch_pixels_stride_*` for overlapping local continuity
4. `patch_embeddings`
5. `tokens` or intermediate token layers
6. discrete image tokens, if using the ImageGPT-style pipeline

## Better Pixel-Space Granularity

For pixel space specifically, the current grid already appears sufficient to establish the main qualitative claim: raw pixel space is stratified across multiple patch sizes and stride settings. A denser sweep would still be useful if the goal is to resolve the effect more precisely, so we can separate:

- scale effects from representation effects;
- patch-size effects from overlap effects;
- genuine local stratification from artifacts of a sparse sweep grid.

Concretely, the pixel-space sweep should be denser in both patch size and stride. A good default is:

- smaller and intermediate patch sizes instead of only a few coarse settings;
- multiple stride fractions for each valid patch size, not just full stride and half stride;
- matched token budgets across scales so fine patches are not unfairly penalized;
- per-scale summaries of `irregular_ratio`, `mean_dim`, `median_dim`, and neighborhood-dimension gaps.

That lets the paper answer a sharper follow-up question: **at what spatial scale does raw pixel space begin to look strongly stratified, and how does that transition compare with learned patch embeddings and tokens?**

## Hypotheses

Use these hypotheses explicitly in reports and writeups:

1. **Reject the classical manifold hypothesis in all spaces.**
   Even raw pixel patches should show nonuniform local dimension.
2. **Expect different degrees of stratification across representations.**
   Learned continuous features may reduce some irregularity relative to raw pixels, while late or discrete token spaces may reintroduce singular structure.
3. **Treat continuity and manifold-likeness as different properties.**
   A representation can be continuous and still strongly stratified.

## Operational Metrics

For the current repo, the most useful summary metrics are:

- `irregular_ratio`: fraction of points with significant slope-change evidence against a locally simple model
- `mean_dim` and `median_dim`: typical local dimension scale
- `mean_neighborhood_dim` or neighborhood-based dimension comparisons
- `multi_strata_ratio`: how often more than one local regime is detected
- `neighborhood_dim_gap_mean`: how sharply local dimension changes across nearby points

These should be interpreted together:

- low irregularity alone does not prove a manifold;
- high dimension alone does not imply complexity;
- the key signal is **how local dimension changes across radius and across nearby points**.

## Suggested Report Language

Use wording like this:

> We do not treat raw image patches as lying on a single smooth manifold. Instead, we model image representations as potentially stratified spaces with locally varying dimension. Our probes therefore test how strongly each representation departs from uniform manifold behavior, and whether learned features smooth or reorganize that stratification relative to raw pixels and tokenized representations.

## Minimal Experiment Section

You can drop this directly into a report:

### Stratified-Space Hypothesis

We test the hypothesis that natural-image representations are better viewed as stratified spaces than as single classical manifolds. Although raw pixel space is continuous, the image data distribution within that ambient space can still have nonuniform local dimension and singular transitions between regimes. This is the direct image-space analogue of the token-embedding result of Robinson et al.\ (2025): continuity of the coordinates does not imply manifold-like local geometry. We therefore compare raw patch pixels, learned patch embeddings, and token-space representations using the same local volume-scaling estimator.

### Evaluation Question

For each representation, we ask:

- how often do local neighborhoods exhibit significant stratification or irregular slope changes?
- how stable is local dimension across nearby points?
- does learning reduce singularity, or merely move it to a different representation layer?
- for raw pixels specifically, how does the answer change as we refine patch size and stride granularity?

### Expected Outcome

If the stratified-space view is correct, then none of the representations should be perfectly manifold-like. Instead, we should see different degrees of local regularity: early continuous features may be smoother than raw pixels in some regions, while late semantic or discrete token spaces may preserve or sharpen singular structure.
