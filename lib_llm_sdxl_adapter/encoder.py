"""Dispatch prompt encoding to the configured model-family implementation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Sequence

from .model_families import GEMMA_FAMILY, T5GEMMA_FAMILY
from .runtime_context import RequestConfig

import torch

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedBatch:
    """Carry hidden states and attention masks into the adapter layer."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


def encode_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    request: RequestConfig,
    *,
    device: str,
) -> EncodedBatch:
    """Encode one prompt batch with the configured model-family implementation."""

    if request.model_family == GEMMA_FAMILY:
        from .gemma_encoder import encode_gemma_prompt_batch

        return encode_gemma_prompt_batch(
            model, tokenizer, prompts, request, device=device
        )
    if request.model_family == T5GEMMA_FAMILY:
        from .t5gemma_encoder import encode_t5gemma_prompt_batch

        return encode_t5gemma_prompt_batch(
            model, tokenizer, prompts, request, device=device
        )
    raise ValueError(
        f"Unsupported model family for prompt encoding: {request.model_family}"
    )
