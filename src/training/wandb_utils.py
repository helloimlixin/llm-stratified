from __future__ import annotations

import os
from typing import Any, Iterable, Optional

try:
    import wandb as wandb_module
except Exception as exc:  # pragma: no cover
    wandb_module = None
    wandb_import_error = exc
else:
    wandb_import_error = None


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
