"""Install and remove the extension's host monkey patches."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import logging
import threading
from typing import Any, Callable

from . import EXTENSION_VERSION
from .conditioning import build_conditioning
from .orchestration import get_request_config
from .runtime_context import (
    RequestConfig,
    build_disabled_request_config,
    clear_active_runtime_context,
    get_active_runtime_context,
    set_active_runtime_context,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class HostPatchState:
    """Track installed host patches and the original callables they replaced."""

    installed: bool = False
    conditioning_owner: object | None = None
    original_get_learned_conditioning: Callable[..., Any] | None = None
    original_cached_params: Callable[..., Any] | None = None


PATCH_STATE = HostPatchState()
PATCH_LOCK = threading.RLock()


def build_cache_signature(request: RequestConfig) -> tuple[Any, ...]:
    """Build the extension-owned cache suffix appended to host cond cache keys."""

    return (
        "llm_sdxl_adapter",
        EXTENSION_VERSION,
        request.enabled,
        request.preset_name,
        request.model_family,
        request.model_path,
        request.adapter_path,
        request.system_prompt,
        request.skip_first,
        request.max_length,
    )


def dispatch_get_learned_conditioning(
    original: Callable[..., Any],
    diffusion_model: object,
    batch: Any,
) -> Any:
    """Route host conditioning requests through the extension when enabled."""

    runtime_context = get_active_runtime_context()
    if runtime_context is None or not runtime_context.request.enabled:
        return original(diffusion_model, batch)

    prompts = list(batch)
    is_negative = getattr(batch, "is_negative_prompt", False)
    LOGGER.debug(
        "Intercepting %s conditioning request with %s prompts",
        "negative" if is_negative else "positive",
        len(prompts),
    )
    return build_conditioning(diffusion_model, prompts, runtime_context.request)


def dispatch_cached_params(
    original: Callable[..., Any],
    processing: object,
    required_prompts: Any,
    steps: int,
    extra_network_data: Any,
    hires_steps: int | None = None,
    use_old_scheduling: bool = False,
) -> tuple[Any, ...]:
    """Append the extension cache signature and activate current request context."""

    base_params = original(
        processing,
        required_prompts,
        steps,
        extra_network_data,
        hires_steps,
        use_old_scheduling,
    )
    request = get_request_config(processing)
    if request.enabled:
        set_active_runtime_context(request)
    else:
        clear_active_runtime_context()
        request = build_disabled_request_config()
    return base_params + (build_cache_signature(request),)


def install_host_patches() -> None:
    """Install the approved host monkey patches exactly once."""

    with PATCH_LOCK:
        if PATCH_STATE.installed:
            return

        import modules.processing

        original_cached = modules.processing.StableDiffusionProcessing.cached_params
        conditioning_target = resolve_conditioning_patch_target()
        if conditioning_target is None:
            LOGGER.debug(
                "Conditioning patch target is not available yet; installation will retry later."
            )
            return

        conditioning_owner, original_get = conditioning_target

        @functools.wraps(original_get)
        def wrapped_get(diffusion_model: object, batch: Any) -> Any:
            return dispatch_get_learned_conditioning(
                original_get, diffusion_model, batch
            )

        @functools.wraps(original_cached)
        def wrapped_cached(
            processing: object,
            required_prompts: Any,
            steps: int,
            extra_network_data: Any,
            hires_steps: int | None = None,
            use_old_scheduling: bool = False,
        ) -> tuple[Any, ...]:
            return dispatch_cached_params(
                original_cached,
                processing,
                required_prompts,
                steps,
                extra_network_data,
                hires_steps,
                use_old_scheduling,
            )

        setattr(conditioning_owner, "get_learned_conditioning", wrapped_get)
        modules.processing.StableDiffusionProcessing.cached_params = wrapped_cached
        PATCH_STATE.installed = True
        PATCH_STATE.conditioning_owner = conditioning_owner
        PATCH_STATE.original_get_learned_conditioning = original_get
        PATCH_STATE.original_cached_params = original_cached
        LOGGER.debug(
            "Installed RouWei LLM Adapter host patches on %s.get_learned_conditioning.",
            getattr(conditioning_owner, "__name__", type(conditioning_owner).__name__),
        )


def uninstall_host_patches() -> None:
    """Restore the original host callables if the patches are currently installed."""

    with PATCH_LOCK:
        if not PATCH_STATE.installed:
            return

        import modules.processing

        if (
            PATCH_STATE.conditioning_owner is not None
            and PATCH_STATE.original_get_learned_conditioning is not None
        ):
            setattr(
                PATCH_STATE.conditioning_owner,
                "get_learned_conditioning",
                PATCH_STATE.original_get_learned_conditioning,
            )
        if PATCH_STATE.original_cached_params is not None:
            modules.processing.StableDiffusionProcessing.cached_params = (
                PATCH_STATE.original_cached_params
            )

        clear_active_runtime_context()
        PATCH_STATE.installed = False
        PATCH_STATE.conditioning_owner = None
        PATCH_STATE.original_get_learned_conditioning = None
        PATCH_STATE.original_cached_params = None
        LOGGER.debug("Restored original host callables for RouWei LLM Adapter.")


def resolve_conditioning_patch_target() -> tuple[object, Callable[..., Any]] | None:
    """Return the active host owner and conditioning callable to patch."""

    try:
        from ldm_patched.modules import model_base

        method = getattr(model_base.BaseModel, "get_learned_conditioning", None)
        if callable(method):
            return model_base.BaseModel, method
    except Exception:
        LOGGER.debug("BaseModel conditioning target is unavailable.", exc_info=True)

    try:
        import sgm.models.diffusion

        method = getattr(
            sgm.models.diffusion.DiffusionEngine, "get_learned_conditioning", None
        )
        if callable(method):
            return sgm.models.diffusion.DiffusionEngine, method
    except Exception:
        LOGGER.debug(
            "DiffusionEngine conditioning target is unavailable.", exc_info=True
        )

    return None
