"""Patch the active transformers runtime for the host APIs this extension needs."""

from __future__ import annotations

from contextlib import contextmanager
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def ensure_transformers_runtime_compatibility(
    transformers_module: Any | None = None,
) -> bool:
    """Install a narrow compatibility shim for host code using old transformers APIs.

    ReForge still calls `transformers.modeling_utils.no_init_weights()` in its
    CLIP-loading path. Newer transformers builds can remove that helper while
    still being otherwise compatible with the code paths this extension needs.

    This shim restores a minimal no-op context manager only when the active
    runtime no longer exposes `no_init_weights`.
    """

    if transformers_module is None:
        import transformers as transformers_module

    modeling_utils = getattr(transformers_module, "modeling_utils", None)
    if modeling_utils is None:
        LOGGER.debug(
            "Skipping transformers compatibility check without modeling_utils."
        )
        return False

    if hasattr(modeling_utils, "no_init_weights"):
        return False

    @contextmanager
    def no_init_weights() -> Any:
        yield

    modeling_utils.no_init_weights = no_init_weights
    LOGGER.info("Installed transformers compatibility shim for no_init_weights.")
    return True
