"""Build ReForge-compatible conditioning from the loaded LLM and adapter."""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from .encoder import encode_prompt_batch
from .loader import LOADER
from .runtime_context import RequestConfig

LOGGER = logging.getLogger(__name__)


def build_conditioning(
    diffusion_model: object, prompts: Sequence[str], request: RequestConfig
) -> dict[str, torch.Tensor]:
    """Build one batched conditioning payload in ReForge SDXL format."""

    if not request.enabled:
        raise ValueError("Cannot build extension conditioning for a disabled request.")

    artifacts = LOADER.ensure_loaded(request)
    encoded_batch = encode_prompt_batch(
        artifacts.model,
        artifacts.tokenizer,
        list(prompts),
        request,
        device=artifacts.device,
    )

    with torch.no_grad():
        crossattn, pooled = artifacts.adapter(
            encoded_batch.hidden_states.to(device=artifacts.device),
            attention_mask=encoded_batch.attention_mask.to(device=artifacts.device),
        )

    target_device = _resolve_target_device(diffusion_model, fallback=artifacts.device)
    crossattn = crossattn.to(device=target_device, dtype=torch.float32).contiguous()
    pooled = pooled.to(device=target_device, dtype=torch.float32).contiguous()

    LOGGER.info(
        "Built conditioning for %s prompts on %s with crossattn=%s vector=%s",
        len(prompts),
        target_device,
        tuple(crossattn.shape),
        tuple(pooled.shape),
    )
    return {"crossattn": crossattn, "vector": pooled}


def _resolve_target_device(diffusion_model: object, *, fallback: str) -> str:
    """Resolve the target device for conditioning tensors."""

    forge_objects = getattr(diffusion_model, "forge_objects", None)
    if forge_objects is not None:
        clip = getattr(forge_objects, "clip", None)
        patcher = getattr(clip, "patcher", None)
        model = getattr(patcher, "model", None)
        device = getattr(model, "device", None)
        if device is not None:
            return str(device)
    return fallback
