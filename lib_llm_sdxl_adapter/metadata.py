"""Emit reproducibility metadata for extension-enabled requests."""

from __future__ import annotations

from .model_families import GEMMA_FAMILY, T5GEMMA_FAMILY
from .runtime_context import RequestConfig


def apply_request_metadata(processing: object, request: RequestConfig) -> None:
    """Append extension metadata to the active processing object when enabled."""

    if not request.enabled:
        return

    processing.extra_generation_params.update(
        {
            "RouWei LLM Adapter": "Enabled",
            "RouWei LLM Preset": request.preset_name,
            "RouWei LLM Family": request.model_family,
            "RouWei LLM Model": request.model_name,
            "RouWei LLM Weights": request.adapter_name,
        }
    )
    if request.model_family == GEMMA_FAMILY:
        processing.extra_generation_params.update(
            {
                "RouWei LLM Skip First": request.skip_first,
                "RouWei LLM System Prompt": request.system_prompt,
            }
        )
    elif request.model_family == T5GEMMA_FAMILY:
        processing.extra_generation_params.update(
            {"RouWei LLM Max Length": request.max_length}
        )
