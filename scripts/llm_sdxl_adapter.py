"""Provide the ReForge script entrypoint for the RouWei LLM adapter extension."""

from __future__ import annotations

import logging

from lib_llm_sdxl_adapter.transformers_compat import (
    ensure_transformers_runtime_compatibility,
)

ensure_transformers_runtime_compatibility()

import gradio as gr

from modules import script_callbacks, scripts

from lib_llm_sdxl_adapter import config, discovery, patches
from lib_llm_sdxl_adapter.orchestration import cleanup_request, prepare_request

LOGGER = logging.getLogger(__name__)


def _install_patches() -> None:
    """Install approved host patches when the extension module loads."""

    patches.install_host_patches()


def _unload_patches() -> None:
    """Restore original host callables when ReForge unloads scripts."""

    patches.uninstall_host_patches()


script_callbacks.on_script_unloaded(
    _unload_patches, name="sd-forge-rouwei-gemma-adapter"
)
script_callbacks.on_model_loaded(
    lambda _sd_model: _install_patches(), name="sd-forge-rouwei-gemma-adapter"
)
_install_patches()


def _choice_or_placeholder(choices: list[str]) -> tuple[list[str], str]:
    """Return safe dropdown choices even when nothing has been discovered yet."""

    if choices:
        return choices, choices[0]
    return [config.PLACEHOLDER_CHOICE], config.PLACEHOLDER_CHOICE


class LlmSdxlAdapterScript(scripts.Script):
    """Expose the extension UI and request preparation hooks to ReForge."""

    def title(self) -> str:
        """Return the visible title for the always-on script block."""

        return "RouWei LLM Adapter"

    def show(self, is_img2img: bool) -> object:
        """Keep the extension visible for both txt2img and img2img tabs."""

        return scripts.AlwaysVisible

    def ui(
        self,
        is_img2img: bool,
    ) -> tuple[
        gr.Checkbox,
        gr.Dropdown,
        gr.Dropdown,
        gr.Dropdown,
        gr.Textbox,
        gr.Number,
        gr.Number,
        gr.Checkbox,
        gr.Dropdown,
    ]:
        """Build the always-visible control block for the extension."""

        del is_img2img
        model_choices, default_model = _choice_or_placeholder(
            discovery.list_llm_model_names()
        )
        adapter_choices, default_adapter = _choice_or_placeholder(
            discovery.list_adapter_names()
        )
        preset_choices = config.list_preset_names()
        default_preset = (
            config.DEFAULT_PRESET_NAME
            if config.DEFAULT_PRESET_NAME in preset_choices
            else preset_choices[0]
        )

        with gr.Accordion(label=self.title(), open=False):
            enabled = gr.Checkbox(label="Enable", value=False)
            model_name = gr.Dropdown(
                label="LLM model", choices=model_choices, value=default_model
            )
            adapter_name = gr.Dropdown(
                label="Adapter", choices=adapter_choices, value=default_adapter
            )
            preset_name = gr.Dropdown(
                label="Preset", choices=preset_choices, value=default_preset
            )
            system_prompt = gr.Textbox(
                label="System prompt (Gemma)",
                value=config.DEFAULT_SYSTEM_PROMPT,
                lines=3,
            )
            skip_first = gr.Number(
                label="Skip first tokens (Gemma)",
                value=config.DEFAULT_SKIP_FIRST,
                precision=0,
            )
            max_length = gr.Number(
                label="Max length (T5-Gemma)",
                value=config.DEFAULT_MAX_LENGTH,
                precision=0,
            )
            force_reload = gr.Checkbox(label="Force reload", value=False)
            device = gr.Dropdown(
                label="Device",
                choices=list(config.SUPPORTED_DEVICE_CHOICES),
                value=config.DEFAULT_DEVICE,
            )

        return (
            enabled,
            model_name,
            adapter_name,
            preset_name,
            system_prompt,
            skip_first,
            max_length,
            force_reload,
            device,
        )

    def before_process(
        self,
        p: object,
        enabled: bool,
        model_name: str,
        adapter_name: str,
        preset_name: str,
        system_prompt: str,
        skip_first: float,
        max_length: float,
        force_reload: bool,
        device: str,
    ) -> None:
        """Resolve UI settings into a validated request configuration before generation."""

        if (
            model_name == config.PLACEHOLDER_CHOICE
            or adapter_name == config.PLACEHOLDER_CHOICE
        ):
            if enabled:
                raise RuntimeError(
                    "Select an LLM model and adapter before enabling RouWei LLM Adapter."
                )

        request = prepare_request(
            p,
            enabled=enabled,
            model_name=model_name,
            adapter_name=adapter_name,
            preset_name=preset_name,
            system_prompt=system_prompt,
            skip_first=int(skip_first),
            max_length=int(max_length),
            force_reload=force_reload,
            device=device,
        )
        LOGGER.debug("Prepared extension request; enabled=%s", request.enabled)

    def postprocess(
        self,
        p: object,
        processed: object,
        enabled: bool,
        model_name: str,
        adapter_name: str,
        preset_name: str,
        system_prompt: str,
        skip_first: float,
        max_length: float,
        force_reload: bool,
        device: str,
    ) -> None:
        """Clear request-local state after generation finishes."""

        del (
            processed,
            enabled,
            model_name,
            adapter_name,
            preset_name,
            system_prompt,
            skip_first,
            max_length,
            force_reload,
            device,
        )
        cleanup_request(p)
