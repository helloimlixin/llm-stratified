# Repository Guidelines

## Project Structure & Module Organization

- `src/` holds core training and diagnostic code.
- `train.py` is the Hydra entrypoint. Core orchestration lives in `training/*`.
- `training/sam_fiber_job.py` handles SAM image-encoder, COCO box-prompt, and mask-preview probes.
- `fiber/*` contains geometry, hypothesis tests, plotting, sparse probes, patch-token helpers, and figure I/O.
- `models.py` contains native, timm, Hugging Face vision, SAM, DINO, and VAR-compatible wrappers.
- `configs/` stores Hydra groups for data, models, training, fiber diagnostics, compute, and experiment presets.
- `tests/` contains CPU-friendly `unittest` suites.
- `docs/stratified-spaces/` contains the active paper bundle.
- `docs/imgs/neurips_submission/` hosts paper-facing figure assets.
- `runs/` is local output and should stay out of commits unless a small curated artifact is intentionally vendored.

## Build, Test, And Development Commands

- `python src/train.py +experiment=quick_test` runs a no-download Hydra sanity check.
- `python src/train.py +experiment=coco_dinov3_huge_sparse_fiber data.root=../data` runs the current DINOv3-H+ COCO probe.
- `python src/train.py +experiment=coco_sam_fiber data.root=../data` runs the current SAM-H COCO probe.
- `python src/train.py +experiment=coco_siglip2_base_sparse_fiber data.root=../data` runs the SigLIP2-B COCO probe.
- `python src/train.py +experiment=coco_aimv2_large_sparse_fiber data.root=../data` runs the AIMv2-L COCO probe.
- `python -m unittest discover -s tests` runs the unit tests.

## Coding Style & Naming Conventions

- Python 3.10+, 4-space indentation, and PEP 8 style.
- Use `snake_case` for functions and variables, `CapWords` for classes, and `ALL_CAPS` for constants.
- Keep the Hydra entrypoint thin in `train.py`; add experiment behavior under `training/`, `fiber/`, or a focused script.
- New experiment knobs should live under `configs/` and be reachable through Hydra overrides.

## Testing Guidelines

- Tests use `unittest` and avoid dataset downloads; prefer fake data for smoke coverage.
- Name new tests `tests/test_*.py` and keep them fast.
- CPU safety matters: keep worker counts low in smoke paths and avoid requiring downloaded checkpoints in unit tests.

## Documentation Guidelines

- Current result documentation should point to `docs/stratified-spaces/main.tex`, `docs/stratified-spaces/draft.tex`, W&B, and compact run summaries under `runs/`.
- When adding paper-facing figures, copy stable assets into `docs/imgs/neurips_submission/` and record run provenance.
- Do not revive superseded scratch-backbone trial reports in the active docs; keep the narrative focused on frozen, image-aligned vision representation probes.

## Commit & Pull Request Guidelines

- Existing commits use short, lowercase summaries, sometimes prefixed with `[wip]`; match that concise style.
- PRs should include the command/config used, any new results, and documentation updates when figures or metrics change.
- Keep large outputs in `runs/` and do not commit datasets or checkpoints.
