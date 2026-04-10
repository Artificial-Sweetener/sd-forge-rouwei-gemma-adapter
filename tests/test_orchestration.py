"""Test request preparation behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lib_llm_sdxl_adapter import orchestration
from lib_llm_sdxl_adapter.runtime_context import REQUEST_CONFIG_ATTR


def test_prepare_request_rejects_non_sdxl_processing() -> None:
    """Fail early when the extension is enabled on a non-SDXL checkpoint."""

    processing = SimpleNamespace(
        sd_model=SimpleNamespace(is_sdxl=False), extra_generation_params={}
    )
    with pytest.raises(RuntimeError):
        orchestration.prepare_request(
            processing,
            enabled=True,
            model_name="model",
            adapter_name="adapter",
            preset_name="gemma",
            system_prompt="system",
            skip_first=27,
            max_length=512,
            force_reload=False,
            device="auto",
        )


def test_prepare_request_attaches_disabled_request_when_extension_is_off() -> None:
    """Attach an explicit disabled request config even when the extension is turned off."""

    processing = SimpleNamespace(
        sd_model=SimpleNamespace(is_sdxl=True), extra_generation_params={}
    )
    request = orchestration.prepare_request(
        processing,
        enabled=False,
        model_name="",
        adapter_name="",
        preset_name="gemma",
        system_prompt="",
        skip_first=27,
        max_length=512,
        force_reload=False,
        device="auto",
    )

    assert request.enabled is False
    assert getattr(processing, REQUEST_CONFIG_ATTR).enabled is False


def test_prepare_request_normalizes_t5gemma_specific_fields(monkeypatch) -> None:
    """Prepare one T5-Gemma request with family-specific field normalization."""

    processing = SimpleNamespace(
        sd_model=SimpleNamespace(is_sdxl=True), extra_generation_params={}
    )
    monkeypatch.setattr(
        orchestration,
        "resolve_model_choice",
        lambda name: SimpleNamespace(name=name, normalized_path="C:\\model"),
    )
    monkeypatch.setattr(
        orchestration,
        "resolve_adapter_choice",
        lambda name: SimpleNamespace(name=name, normalized_path="C:\\adapter"),
    )
    monkeypatch.setattr(
        orchestration.LOADER, "ensure_loaded", lambda request: SimpleNamespace()
    )

    request = orchestration.prepare_request(
        processing,
        enabled=True,
        model_name="model",
        adapter_name="adapter",
        preset_name="t5gemma",
        system_prompt="ignored",
        skip_first=27,
        max_length=1024,
        force_reload=False,
        device="auto",
    )

    assert request.enabled is True
    assert request.model_family == "t5gemma"
    assert request.system_prompt == ""
    assert request.skip_first == 0
    assert request.max_length == 1024
