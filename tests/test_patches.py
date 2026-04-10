"""Test patch dispatch and cache-signature behavior."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from lib_llm_sdxl_adapter import patches
from lib_llm_sdxl_adapter.runtime_context import (
    REQUEST_CONFIG_ATTR,
    RequestConfig,
    build_disabled_request_config,
    clear_active_runtime_context,
)


def _enabled_request() -> RequestConfig:
    return RequestConfig(
        enabled=True,
        preset_name="gemma",
        model_family="gemma",
        model_name="model",
        model_path="C:\\model",
        adapter_name="adapter",
        adapter_path="C:\\adapter",
        system_prompt="system",
        skip_first=27,
        max_length=512,
        force_reload=False,
        device="auto",
    )


def _t5_request() -> RequestConfig:
    return RequestConfig(
        enabled=True,
        preset_name="t5gemma",
        model_family="t5gemma",
        model_name="model",
        model_path="C:\\model",
        adapter_name="adapter",
        adapter_path="C:\\adapter",
        system_prompt="",
        skip_first=0,
        max_length=1024,
        force_reload=False,
        device="auto",
    )


def test_build_cache_signature_includes_extension_state() -> None:
    """Capture adapter-specific state in the cond cache suffix."""

    signature = patches.build_cache_signature(_enabled_request())
    assert signature[0] == "llm_sdxl_adapter"
    assert signature[2] is True
    assert "C:\\model" in signature
    assert "C:\\adapter" in signature


def test_dispatch_cached_params_appends_signature() -> None:
    """Append one extension-specific suffix to the host cached params tuple."""

    processing = SimpleNamespace()
    setattr(processing, REQUEST_CONFIG_ATTR, _enabled_request())

    def original(*args, **kwargs):
        return ("host",)

    result = patches.dispatch_cached_params(
        original, processing, ["prompt"], 20, [], None, False
    )
    assert result[0] == "host"
    assert result[-1][0] == "llm_sdxl_adapter"


def test_dispatch_get_learned_conditioning_passthrough_when_disabled() -> None:
    """Call the original host implementation when the extension is not active."""

    clear_active_runtime_context()

    def original(diffusion_model, batch):
        return ("original", diffusion_model, batch)

    result = patches.dispatch_get_learned_conditioning(
        original, "diffusion_model", ["prompt"]
    )
    assert result[0] == "original"


def test_dispatch_cached_params_uses_disabled_signature_when_no_request_attached() -> (
    None
):
    """Use an explicit disabled cache signature when the processing object has no request state."""

    processing = SimpleNamespace()

    def original(*args, **kwargs):
        return ("host",)

    result = patches.dispatch_cached_params(
        original, processing, ["prompt"], 20, [], None, False
    )
    assert result[-1] == patches.build_cache_signature(build_disabled_request_config())


def test_build_cache_signature_changes_for_t5gemma_max_length() -> None:
    """Include T5-Gemma-specific output-affecting state in the cache signature."""

    signature_a = patches.build_cache_signature(_t5_request())
    signature_b = patches.build_cache_signature(
        _t5_request().__class__(**{**_t5_request().__dict__, "max_length": 2048})
    )
    assert signature_a != signature_b


def test_resolve_conditioning_patch_target_prefers_basemodel(monkeypatch) -> None:
    """Use BaseModel conditioning when this ReForge build exposes it there."""

    fake_model_base_module = SimpleNamespace(
        BaseModel=type(
            "BaseModel",
            (),
            {
                "get_learned_conditioning": staticmethod(
                    lambda model, batch: ("base", batch)
                )
            },
        )
    )
    monkeypatch.setitem(
        sys.modules, "ldm_patched.modules.model_base", fake_model_base_module
    )
    monkeypatch.setitem(
        sys.modules,
        "ldm_patched.modules",
        SimpleNamespace(model_base=fake_model_base_module),
    )

    result = patches.resolve_conditioning_patch_target()
    assert result is not None
    owner, method = result
    assert owner is fake_model_base_module.BaseModel
    assert callable(method)


def test_resolve_conditioning_patch_target_returns_none_when_no_supported_target(
    monkeypatch,
) -> None:
    """Return no patch target instead of crashing when the host seam is unavailable."""

    monkeypatch.setitem(sys.modules, "ldm_patched.modules", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "ldm_patched.modules.model_base",
        SimpleNamespace(BaseModel=type("BaseModel", (), {})),
    )
    monkeypatch.setitem(sys.modules, "sgm.models", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "sgm.models.diffusion",
        SimpleNamespace(DiffusionEngine=type("DiffusionEngine", (), {})),
    )

    assert patches.resolve_conditioning_patch_target() is None
