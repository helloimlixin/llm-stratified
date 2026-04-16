"""Backward-compatibility shim — all functionality lives in the ``fiber`` package."""
from fiber import *  # noqa: F401, F403
from fiber import matplotlib_supports_3d as _matplotlib_supports_3d  # noqa: F401 — backward compat
