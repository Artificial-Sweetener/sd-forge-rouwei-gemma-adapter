"""Test source-parity behavior for prompt encoding."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lib_llm_sdxl_adapter.encoder import encode_prompt_batch
from lib_llm_sdxl_adapter.runtime_context import RequestConfig


class DummyBatch(dict):
    """Provide a dict-like batch object with a `.to()` method."""

    def to(self, device: str):
        return DummyBatch({key: value.to(device) for key, value in self.items()})


class DummyTokenizer:
    """Mimic the chat-template interface used by the encoder."""

    def apply_chat_template(self, conversations, **kwargs):
        del kwargs
        batch_size = len(conversations)
        input_ids = torch.arange(batch_size * 5, dtype=torch.long).reshape(
            batch_size, 5
        )
        attention_mask = torch.ones_like(input_ids)
        return DummyBatch({"input_ids": input_ids, "attention_mask": attention_mask})


class DummyModel:
    """Return predictable hidden states for skip-first validation."""

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, token_count = input_ids.shape
        hidden_states = torch.arange(
            batch_size * token_count * 3, dtype=torch.float32
        ).reshape(batch_size, token_count, 3)
        return SimpleNamespace(hidden_states=[hidden_states, hidden_states + 1000])


class DummyT5Tokenizer:
    """Record one T5-Gemma tokenizer call and return a predictable batch."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompts, **kwargs):
        self.calls.append({"prompts": list(prompts), **kwargs})
        batch_size = len(prompts)
        input_ids = torch.arange(batch_size * 4, dtype=torch.long).reshape(
            batch_size, 4
        )
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class DummyT5Model:
    """Return predictable last hidden states for T5-Gemma validation."""

    def __call__(self, **kwargs):
        input_ids = kwargs["input_ids"]
        batch_size, token_count = input_ids.shape
        hidden_states = torch.arange(
            batch_size * token_count * 5, dtype=torch.float32
        ).reshape(batch_size, token_count, 5)
        return SimpleNamespace(last_hidden_state=hidden_states + 500)


def _gemma_request() -> RequestConfig:
    """Build one enabled Gemma request for encoder tests."""

    return RequestConfig(
        enabled=True,
        preset_name="gemma",
        model_family="gemma",
        model_name="model",
        model_path="m",
        adapter_name="adapter",
        adapter_path="a",
        system_prompt="system",
        skip_first=2,
        max_length=512,
        force_reload=False,
        device="cpu",
    )


def _t5gemma_request() -> RequestConfig:
    """Build one enabled T5-Gemma request for encoder tests."""

    return RequestConfig(
        enabled=True,
        preset_name="t5gemma",
        model_family="t5gemma",
        model_name="model",
        model_path="m",
        adapter_name="adapter",
        adapter_path="a",
        system_prompt="",
        skip_first=0,
        max_length=512,
        force_reload=False,
        device="cpu",
    )


def test_encode_prompt_batch_applies_skip_first_after_hidden_state_extraction() -> None:
    """Mirror the source behavior of trimming leading template tokens after encoding."""

    encoded = encode_prompt_batch(
        DummyModel(),
        DummyTokenizer(),
        ["a prompt", "another prompt"],
        _gemma_request(),
        device="cpu",
    )

    assert encoded.hidden_states.shape == (2, 3, 3)
    assert encoded.attention_mask.shape == (2, 3)
    assert torch.equal(
        encoded.hidden_states[0, 0], torch.tensor([1006.0, 1007.0, 1008.0])
    )


def test_encode_prompt_batch_uses_t5gemma_eos_and_max_length_behavior() -> None:
    """Mirror the upstream T5-Gemma tokenizer and hidden-state behavior."""

    tokenizer = DummyT5Tokenizer()
    encoded = encode_prompt_batch(
        DummyT5Model(),
        tokenizer,
        ["first prompt", "second prompt"],
        _t5gemma_request(),
        device="cpu",
    )

    assert encoded.hidden_states.shape == (2, 4, 5)
    assert encoded.attention_mask.shape == (2, 4)
    assert tokenizer.calls[0]["prompts"] == ["first prompt<eos>", "second prompt<eos>"]
    assert tokenizer.calls[0]["padding"] == "max_length"
    assert tokenizer.calls[0]["max_length"] == 512
    assert tokenizer.calls[0]["truncation"] is True
    assert torch.equal(
        encoded.hidden_states[0, 0], torch.tensor([500.0, 501.0, 502.0, 503.0, 504.0])
    )
