# Superseded Volume Probe Sweep Report

This April 2026 sweep report has been retired from the active documentation. It was useful for developing the local volume-scaling estimator, but the current project narrative now centers on frozen, image-aligned patch-token probes over DINOv3, SAM, SigLIP2, AIMv2, and VAR.

The current empirical record is maintained in:

- `docs/RESULTS.md`
- `docs/COCO_PRETRAINED_VIT_SAM_ANALYSIS.md`
- `docs/stratified-spaces/main.tex`

The retained methodological lesson is that raw image patches and learned token spaces should be tested locally rather than summarized by one global dimension. Current runs apply that lesson to pretrained dense vision representations, project diagnostics back to image patches, and pair fiber-bundle violations with local sparse raw-patch complexity.
