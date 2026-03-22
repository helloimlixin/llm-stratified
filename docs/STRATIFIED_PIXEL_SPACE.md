# Stratified Pixel-Space Framing

## Project Claim

The main claim for this repo should be stated in stratified terms, not classical manifold terms:

- Natural-image data in raw pixel coordinates are continuous, but they are not well modeled as samples from one smooth manifold with uniform local dimension.
- A better working model is that image data form a **stratified space** with locally varying dimension.
- Different patch types, object boundaries, textures, flat regions, repeated structures, and compositional scenes can occupy different local strata.
- Learned visual features may make this geometry more regular, but they do not automatically collapse it to a single manifold.

In short: **continuous image signal removes discretization, not stratification**.

## Relation To The Fiber-Bundle Paper

The paper on token embeddings argues that local neighborhoods can violate both the classical manifold hypothesis and the more permissive fiber-bundle null when the local volume-vs-radius scaling changes in the wrong way. The image-space analogue is not "pixels are continuous, so the manifold hypothesis is restored." The analogue is:

- text token space is discrete and strongly singular;
- image pixel space is continuous but still geometrically heterogeneous;
- therefore the right test in image space is still a **stratified-manifold / fiber-bundle-style local scaling test**, not a single-manifold fit.

## Experiment Framing

The clean experimental question is:

**Which image representation is most compatible with a stratified geometric model, and where does the classical manifold hypothesis fail?**

Recommended representation ladder:

1. `patch_pixels`
2. `patch_pixels_stride_*` for overlapping local continuity
3. `patch_embeddings`
4. `tokens` or intermediate token layers
5. discrete image tokens, if using the ImageGPT-style pipeline

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

We test the hypothesis that natural-image representations are better viewed as stratified spaces than as single classical manifolds. Although raw pixel space is continuous, the image data distribution within that ambient space can still have nonuniform local dimension and singular transitions between regimes. We therefore compare raw patch pixels, learned patch embeddings, and token-space representations using the same local volume-scaling estimator.

### Evaluation Question

For each representation, we ask:

- how often do local neighborhoods exhibit significant stratification or irregular slope changes?
- how stable is local dimension across nearby points?
- does learning reduce singularity, or merely move it to a different representation layer?

### Expected Outcome

If the stratified-space view is correct, then none of the representations should be perfectly manifold-like. Instead, we should see different degrees of local regularity: early continuous features may be smoother than raw pixels in some regions, while late semantic or discrete token spaces may preserve or sharpen singular structure.
