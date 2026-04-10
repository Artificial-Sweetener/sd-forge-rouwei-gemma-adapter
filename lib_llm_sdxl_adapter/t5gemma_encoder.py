"""Encode prompt batches with the T5-Gemma encoder workflow."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from .encoder import EncodedBatch
from .runtime_context import RequestConfig

LOGGER = logging.getLogger(__name__)


def encode_t5gemma_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    request: RequestConfig,
    *,
    device: str,
) -> EncodedBatch:
    """Encode one prompt batch using the upstream T5-Gemma raw-tokenizer path."""

    if not prompts:
        raise ValueError("Prompt batch cannot be empty.")

    prepared_prompts = [f"{prompt}<eos>" for prompt in prompts]
    inputs = tokenizer(
        prepared_prompts,
        return_tensors="pt",
        padding="max_length",
        max_length=request.max_length,
        truncation=True,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    final_hidden_state = outputs.last_hidden_state.to(torch.float32).contiguous()
    attention_mask = attention_mask.to(device=final_hidden_state.device).contiguous()

    LOGGER.info(
        "Encoded %s T5-Gemma prompts with hidden-state shape %s at max_length=%s",
        len(prompts),
        tuple(final_hidden_state.shape),
        request.max_length,
    )
    return EncodedBatch(hidden_states=final_hidden_state, attention_mask=attention_mask)
