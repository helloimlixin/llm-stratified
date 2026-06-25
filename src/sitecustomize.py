"""Local Python startup fixes for script launches from `src/`.

Python 3.12's Windows platform probe can hang in `_wmi.exec_query` on some
machines. PyTorch calls `platform.system()` while importing, so disable the WMI
branch and let `platform` use its built-in non-WMI fallback.
"""

from __future__ import annotations

import os

if os.name == "nt":
    try:
        import platform

        def _skip_wmi_query(*_args, **_kwargs):
            raise OSError("disabled workspace WMI platform query")

        platform._wmi_query = _skip_wmi_query
    except Exception:
        pass
