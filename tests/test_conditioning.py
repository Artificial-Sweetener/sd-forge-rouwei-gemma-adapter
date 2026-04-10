"""Test conditioning packaging for the ReForge host format."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from lib_llm_sdxl_adapter.conditioning import build_conditioning
from lib_llm_sdxl_adapter.runtime_context import RequestConfig


class DummyAdapter:
    """Return deterministic sequence and pooled outputs."""

    def __call__(self, hidden_states, attention_mask=None):
        del attention_mask
        batch, *_ = hidden_states.shape
        return (
            torch.ones(batch, 308, 2048, dtype=torch.float32),
            torch.ones(batch, 1280, dtype=torch.float32),
        )


def test_build_conditioning_returns_crossattn_and_vector(monkeypatch) -> None:
    """Map the source sequence/pooled outputs into the ReForge host fields."""

    request = RequestConfig(
        True,
        "gemma",
        "gemma",
        "model",
        "m",
        "adapter",
        "a",
        "system",
        27,
        512,
        False,
        "cpu",
    )
    artifacts = SimpleNamespace(
        model=object(),
        tokenizer=object(),
        adapter=DummyAdapter(),
        device="cpu",
    )

    monkeypatch.setattr(
        "lib_llm_sdxl_adapter.conditioning.LOADER.ensure_loaded", lambda req: artifacts
    )
    monkeypatch.setattr(
        "lib_llm_sdxl_adapter.conditioning.encode_prompt_batch",
        lambda *args, **kwargs: SimpleNamespace(
            hidden_states=torch.ones(2, 10, 1152, dtype=torch.float32),
            attention_mask=torch.ones(2, 10, dtype=torch.long),
        ),
    )

    diffusion_model = SimpleNamespace(
        forge_objects=SimpleNamespace(
            clip=SimpleNamespace(
                patcher=SimpleNamespace(model=SimpleNamespace(device="cpu"))
            )
        )
    )
    result = build_conditioning(diffusion_model, ["prompt 1", "prompt 2"], request)

    assert set(result.keys()) == {"crossattn", "vector"}
    assert result["crossattn"].shape == (2, 308, 2048)
    assert result["vector"].shape == (2, 1280)
