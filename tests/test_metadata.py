"""Test infotext metadata emission."""

from __future__ import annotations

from types import SimpleNamespace

from lib_llm_sdxl_adapter.metadata import apply_request_metadata
from lib_llm_sdxl_adapter.runtime_context import RequestConfig


def test_metadata_is_only_emitted_for_enabled_requests() -> None:
    """Emit metadata only when the extension actively participates in the run."""

    processing = SimpleNamespace(extra_generation_params={})
    disabled = RequestConfig(False, "", "", "", "", "", "", "", 0, 0, False, "")
    apply_request_metadata(processing, disabled)
    assert processing.extra_generation_params == {}

    enabled = RequestConfig(
        True,
        "gemma",
        "gemma",
        "model",
        "m",
        "adapter",
        "a",
        "system",
        27,
        512,
        False,
        "auto",
    )
    apply_request_metadata(processing, enabled)
    assert processing.extra_generation_params["RouWei LLM Adapter"] == "Enabled"
    assert processing.extra_generation_params["RouWei LLM Model"] == "model"


def test_t5gemma_metadata_emits_max_length_only() -> None:
    """Emit T5-Gemma-specific metadata without Gemma-only fields."""

    processing = SimpleNamespace(extra_generation_params={})
    request = RequestConfig(
        True,
        "t5gemma",
        "t5gemma",
        "model",
        "m",
        "adapter",
        "a",
        "",
        0,
        1024,
        False,
        "auto",
    )
    apply_request_metadata(processing, request)

    assert processing.extra_generation_params["RouWei LLM Family"] == "t5gemma"
    assert processing.extra_generation_params["RouWei LLM Max Length"] == 1024
    assert "RouWei LLM Skip First" not in processing.extra_generation_params
