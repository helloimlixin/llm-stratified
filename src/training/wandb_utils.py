from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import wandb as wandb_module
except Exception as exc:  # pragma: no cover
    wandb_module = None
    wandb_import_error = exc
else:
    wandb_import_error = None


def ensure_wandb_dir(*, enabled: bool, output_dir: str | os.PathLike[str] | None) -> Optional[str]:
    """Default WANDB_DIR to a per-run folder under the Hydra output dir."""
    if not enabled or output_dir is None:
        return os.environ.get("WANDB_DIR")

    existing = os.environ.get("WANDB_DIR")
    if existing:
        return existing

    wandb_dir = Path(output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_DIR"] = str(wandb_dir)
    return os.environ["WANDB_DIR"]


def resolve_wandb_name(cfg: Any, *, suffix: str = "") -> str:
    """Resolve a W&B run name without letting broken interpolations abort the run."""
    configured_name = None
    try:
        configured_name = cfg.wandb.name
    except Exception as exc:
        print(f"[wandb] WARNING: could not resolve wandb.name ({exc}); falling back to an auto-generated name")

    if configured_name:
        return str(configured_name)

    data_name = getattr(getattr(cfg, "data", None), "name", "run")
    model_name = getattr(getattr(cfg, "model", None), "name", "model")
    base_name = f"{data_name}_{model_name}"

    try:
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        job_num = getattr(getattr(hydra_cfg, "job", None), "num", None)
    except Exception:
        job_num = None

    if job_num not in (None, ""):
        base_name = f"{base_name}_job_{job_num}"

    return f"{base_name}_{suffix}" if suffix else base_name


def init_wandb_run(
    *,
    enabled: bool,
    project: str,
    name: Optional[str],
    config: Optional[dict[str, Any]] = None,
    tags: Optional[Iterable[str]] = None,
    missing_message: str = "[wandb] ERROR: not installed; disabling",
    show_url: bool = False,
):
    if not enabled:
        return None
    if wandb_module is None:
        suffix = f" ({wandb_import_error})" if wandb_import_error is not None else ""
        print(f"{missing_message}{suffix}")
        return None

    try:
        if os.environ.get("WANDB_MODE", "online") == "online":
            try:
                if wandb_module.api.api_key is None:
                    print("[wandb] WARNING: Not logged in, using offline mode")
                    os.environ["WANDB_MODE"] = "offline"
            except Exception:
                pass

        wandb_module.init(project=project, name=name, config=config, tags=list(tags) if tags is not None else None)
        if show_url:
            print(f"[wandb] Initialized: {wandb_module.run.url if wandb_module.run else 'N/A'}")
    except Exception as exc:
        print(f"[wandb] ERROR: {exc}")
        return None

    return wandb_module


def finish_wandb_run(wandb=None) -> None:
    module = wandb or wandb_module
    if module is None:
        return
    try:
        if getattr(module, "run", None):
            module.finish()
    except Exception:
        pass
