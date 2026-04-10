"""Own request-local state used by the monkey-patch wrappers."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from .model_families import ModelFamily

REQUEST_CONFIG_ATTR = "_llm_sdxl_adapter_request_config"


@dataclass(frozen=True)
class RequestConfig:
    """Describe one fully resolved extension request configuration."""

    enabled: bool
    preset_name: str
    model_family: ModelFamily | str
    model_name: str
    model_path: str
    adapter_name: str
    adapter_path: str
    system_prompt: str
    skip_first: int
    max_length: int
    force_reload: bool
    device: str


@dataclass(frozen=True)
class ActiveRuntimeContext:
    """Describe the request configuration visible to patch wrappers."""

    request: RequestConfig


_ACTIVE_RUNTIME_CONTEXT: ContextVar[ActiveRuntimeContext | None] = ContextVar(
    "llm_sdxl_adapter_active_context",
    default=None,
)


def build_disabled_request_config() -> RequestConfig:
    """Return a stable disabled request configuration for non-participating runs."""

    return RequestConfig(
        enabled=False,
        preset_name="",
        model_family="",
        model_name="",
        model_path="",
        adapter_name="",
        adapter_path="",
        system_prompt="",
        skip_first=0,
        max_length=0,
        force_reload=False,
        device="",
    )


def set_active_runtime_context(request: RequestConfig) -> None:
    """Replace the currently visible runtime context."""

    _ACTIVE_RUNTIME_CONTEXT.set(ActiveRuntimeContext(request=request))


def clear_active_runtime_context() -> None:
    """Clear the currently visible runtime context."""

    _ACTIVE_RUNTIME_CONTEXT.set(None)


def get_active_runtime_context() -> ActiveRuntimeContext | None:
    """Return the currently visible runtime context if one exists."""

    return _ACTIVE_RUNTIME_CONTEXT.get()
