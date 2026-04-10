"""Encode prompt batches with the Gemma chat-template workflow."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch

from .encoder import EncodedBatch
from .runtime_context import RequestConfig

LOGGER = logging.getLogger(__name__)


def _build_messages(
    prompt: str, system_prompt: str
) -> list[dict[str, list[dict[str, str]]]]:
    """Construct one Gemma chat-template conversation."""

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


def encode_gemma_prompt_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    request: RequestConfig,
    *,
    device: str,
) -> EncodedBatch:
    """Encode one prompt batch using the upstream Gemma chat-template path."""

    if not prompts:
        raise ValueError("Prompt batch cannot be empty.")

    conversations = [
        _build_messages(prompt, request.system_prompt) for prompt in prompts
    ]
    inputs = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        padding=True,
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        hidden_states = outputs["hidden_states"]

    final_hidden_state = hidden_states[-1]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones(
            final_hidden_state.shape[:2],
            device=final_hidden_state.device,
            dtype=torch.long,
        )

    skip_first = max(0, min(request.skip_first, final_hidden_state.shape[1]))
    final_hidden_state = (
        final_hidden_state[:, skip_first:, :].to(torch.float32).contiguous()
    )
    attention_mask = (
        attention_mask[:, skip_first:].to(device=final_hidden_state.device).contiguous()
    )

    LOGGER.info(
        "Encoded %s Gemma prompts with hidden-state shape %s after skip_first=%s",
        len(prompts),
        tuple(final_hidden_state.shape),
        request.skip_first,
    )
    return EncodedBatch(hidden_states=final_hidden_state, attention_mask=attention_mask)
