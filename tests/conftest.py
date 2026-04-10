"""Configure import paths for extension tests."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_sys_path(path: Path) -> None:
    """Prepend one path to sys.path when it is not already present."""

    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


EXTENSION_ROOT = Path(__file__).resolve().parent.parent
HOST_ROOT = EXTENSION_ROOT.parent.parent

_ensure_sys_path(EXTENSION_ROOT)
_ensure_sys_path(HOST_ROOT)
