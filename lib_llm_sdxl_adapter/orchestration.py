"""Coordinate UI selections, validation, metadata, and request-local state."""

from __future__ import annotations

import logging

from .config import DEFAULT_MAX_LENGTH, DEFAULT_SYSTEM_PROMPT, get_preset
from .discovery import resolve_adapter_choice, resolve_model_choice
from .loader import LOADER
from .metadata import apply_request_metadata
from .model_families import GEMMA_FAMILY, T5GEMMA_FAMILY
from .runtime_context import (
    REQUEST_CONFIG_ATTR,
    RequestConfig,
    build_disabled_request_config,
    clear_active_runtime_context,
)

LOGGER = logging.getLogger(__name__)


def prepare_request(
    processing: object,
    *,
    enabled: bool,
    model_name: str,
    adapter_name: str,
    preset_name: str,
    system_prompt: str,
    skip_first: int,
    max_length: int,
    force_reload: bool,
    device: str,
) -> RequestConfig:
    """Resolve the active UI selection into one validated request configuration."""

    clear_active_runtime_context()
    request = build_disabled_request_config()
    setattr(processing, REQUEST_CONFIG_ATTR, request)

    if not enabled:
        return request

    if not getattr(processing.sd_model, "is_sdxl", False):
        raise RuntimeError(
            "Load an SDXL checkpoint before enabling RouWei LLM Adapter."
        )

    preset = get_preset(preset_name)
    model_item = resolve_model_choice(model_name)
    adapter_item = resolve_adapter_choice(adapter_name)
    request = RequestConfig(
        enabled=True,
        preset_name=preset.name,
        model_family=preset.model_family,
        model_name=model_item.name,
        model_path=model_item.normalized_path,
        adapter_name=adapter_item.name,
        adapter_path=adapter_item.normalized_path,
        system_prompt=_resolve_system_prompt(preset.model_family, system_prompt),
        skip_first=_resolve_skip_first(preset.model_family, skip_first),
        max_length=_resolve_max_length(preset.model_family, max_length),
        force_reload=bool(force_reload),
        device=device,
    )

    LOGGER.debug(
        "Prepared request for model=%s adapter=%s preset=%s",
        request.model_name,
        request.adapter_name,
        request.preset_name,
    )
    LOADER.ensure_loaded(request)
    setattr(processing, REQUEST_CONFIG_ATTR, request)
    apply_request_metadata(processing, request)
    return request


def _resolve_system_prompt(model_family: str, system_prompt: str) -> str:
    """Resolve the prompt-format state for the selected model family."""

    if model_family == GEMMA_FAMILY:
        return system_prompt or DEFAULT_SYSTEM_PROMPT
    if model_family == T5GEMMA_FAMILY:
        return ""
    raise ValueError(
        f"Unsupported model family for system prompt resolution: {model_family}"
    )


def _resolve_skip_first(model_family: str, skip_first: int) -> int:
    """Resolve the hidden-state trim policy for the selected model family."""

    if model_family == GEMMA_FAMILY:
        return max(0, int(skip_first))
    if model_family == T5GEMMA_FAMILY:
        return 0
    raise ValueError(
        f"Unsupported model family for skip_first resolution: {model_family}"
    )


def _resolve_max_length(model_family: str, max_length: int) -> int:
    """Resolve the tokenizer max-length policy for the selected model family."""

    if model_family == GEMMA_FAMILY:
        return DEFAULT_MAX_LENGTH
    if model_family == T5GEMMA_FAMILY:
        return max(8, int(max_length))
    raise ValueError(
        f"Unsupported model family for max_length resolution: {model_family}"
    )


def cleanup_request(processing: object) -> None:
    """Clear any per-request extension state left on the processing object."""

    clear_active_runtime_context()
    setattr(processing, REQUEST_CONFIG_ATTR, build_disabled_request_config())


def get_request_config(processing: object) -> RequestConfig:
    """Return the request configuration currently attached to the processing object."""

    request = getattr(processing, REQUEST_CONFIG_ATTR, None)
    if request is None:
        return build_disabled_request_config()
    return request
