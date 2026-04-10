"""Test the narrow transformers runtime compatibility shim."""

from __future__ import annotations

from types import SimpleNamespace

from lib_llm_sdxl_adapter.transformers_compat import (
    ensure_transformers_runtime_compatibility,
)


def test_compat_shim_is_not_applied_when_no_init_weights_exists() -> None:
    """Leave runtimes alone when they already expose the expected helper."""

    modeling_utils = SimpleNamespace(no_init_weights=lambda: None)
    transformers_module = SimpleNamespace(modeling_utils=modeling_utils)

    patched = ensure_transformers_runtime_compatibility(transformers_module)
    assert patched is False
    assert callable(modeling_utils.no_init_weights)


def test_compat_shim_installs_no_init_weights_when_missing() -> None:
    """Restore a no-op context manager when newer transformers removed it."""

    modeling_utils = SimpleNamespace()
    transformers_module = SimpleNamespace(modeling_utils=modeling_utils)

    patched = ensure_transformers_runtime_compatibility(transformers_module)
    assert patched is True
    assert hasattr(modeling_utils, "no_init_weights")
    with modeling_utils.no_init_weights():
        pass
