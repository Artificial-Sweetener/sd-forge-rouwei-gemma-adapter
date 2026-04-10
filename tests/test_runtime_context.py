"""Test runtime context ownership and disabled defaults."""

from __future__ import annotations

from lib_llm_sdxl_adapter.runtime_context import (
    build_disabled_request_config,
    clear_active_runtime_context,
    get_active_runtime_context,
    set_active_runtime_context,
)


def test_disabled_request_config_is_stable() -> None:
    """Return a fully disabled request configuration with empty fields."""

    request = build_disabled_request_config()
    assert request.enabled is False
    assert request.model_family == ""
    assert request.model_path == ""
    assert request.adapter_path == ""
    assert request.max_length == 0


def test_active_runtime_context_can_be_replaced_and_cleared() -> None:
    """Replace and clear the active runtime context explicitly."""

    request = build_disabled_request_config()
    set_active_runtime_context(request)
    assert get_active_runtime_context() is not None
    clear_active_runtime_context()
    assert get_active_runtime_context() is None
