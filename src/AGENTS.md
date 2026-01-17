# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds core training/diagnostics code and entry points:
- `train.py` main training entrypoint (Hydra-configured). Core training implementation lives in `training/*`.
  - `training/loops.py` train/eval loops (also imported by unit tests)
  - `training/runner.py` training driver (DDP/Accelerate + fiber diagnostics)
  - `training/backend.py` backend init + DDP gather helpers
  - `imagegpt.py` discrete-token ImageGPT experiments
  - shared helpers in `models.py`, `data.py`, `utils.py`
- `configs/` stores Hydra config groups (`data/`, `model/`, `training/`, `fiber/`, `compute/`).
- `tests/` contains `unittest` suites (`test_*.py`).
- `docs/RESULTS.md` and `docs/imgs/` host the report and published plots.
- `scripts/` contains launch helpers (multi-GPU, SLURM). `data/` and `runs/` are local-only and gitignored.

## Build, Test, and Development Commands
- `python src/train.py data=stl10 fiber=basic data.root=./data hydra.run.dir=./runs/...` runs a fiber diagnostic training job.
- `python src/train.py +experiment=quick_test` runs a Hydra sanity check.
- `python -m unittest discover -s tests` runs CPU-only unit tests.
- `./scripts/run_multi_gpu.sh CIFAR10 training.epochs=50` launches multi-GPU training via `torchrun`.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, and PEP 8 style.
- Use `snake_case` for functions/vars, `CapWords` for classes, `ALL_CAPS` for constants (see `data.py`).
- Keep the Hydra entrypoint thin in `train.py`, model code in `models.py`/`imagegpt.py`, and training helpers under `training/`. Add experiment knobs under `configs/` and use Hydra overrides.
- New experiment knobs should be added under `configs/` and referenced via Hydra overrides.

## Testing Guidelines
- Tests use `unittest` and avoid dataset downloads; prefer `FAKEDATA` when possible.
- Name new tests `tests/test_*.py` and keep them fast.
- CI also runs a 1-epoch smoke train; ensure flags are CPU-safe (`--num-workers 0` if needed).

## Commit & Pull Request Guidelines
- Existing commits use short, lowercase summaries, sometimes prefixed with `[wip]`; match that concise style.
- PRs should include the command/config used, any new results, and updated `docs/RESULTS.md` + `docs/imgs/` when plots change.
- Keep large outputs in `runs/` and do not commit datasets or checkpoints.
