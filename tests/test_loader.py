"""Test family-aware loader dispatch and cache-key behavior."""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from lib_llm_sdxl_adapter.loader import LoaderService
import lib_llm_sdxl_adapter.loader as loader_module
from lib_llm_sdxl_adapter.runtime_context import RequestConfig


def _request(model_family: str, preset_name: str) -> RequestConfig:
    """Build one enabled request for loader tests."""

    return RequestConfig(
        enabled=True,
        preset_name=preset_name,
        model_family=model_family,
        model_name="model",
        model_path="C:\\model",
        adapter_name="adapter",
        adapter_path="C:\\adapter",
        system_prompt="system",
        skip_first=27,
        max_length=512,
        force_reload=False,
        device="auto",
    )


def test_loader_cache_key_distinguishes_model_family() -> None:
    """Keep cached runtime assets separated across model families."""

    gemma_key = LoaderService._build_cache_key(_request("gemma", "gemma"))
    t5_key = LoaderService._build_cache_key(_request("t5gemma", "t5gemma"))
    assert gemma_key != t5_key


def test_loader_dispatches_to_gemma_loader(tmp_path: Path, monkeypatch) -> None:
    """Select the Gemma loader path for Gemma-family requests."""

    model_path = tmp_path / "gemma-model"
    model_path.mkdir()
    monkeypatch.setattr(
        LoaderService,
        "_load_gemma_model_and_tokenizer",
        staticmethod(
            lambda model_path, device, dtype: ("gemma_model", "gemma_tokenizer")
        ),
    )
    monkeypatch.setattr(
        LoaderService,
        "_load_t5gemma_model_and_tokenizer",
        staticmethod(lambda model_path, device, dtype: ("t5_model", "t5_tokenizer")),
    )

    result = LoaderService._load_model_and_tokenizer(
        str(model_path), "gemma", "cpu", torch.float32
    )
    assert result == ("gemma_model", "gemma_tokenizer")


def test_loader_dispatches_to_t5gemma_loader(tmp_path: Path, monkeypatch) -> None:
    """Select the T5-Gemma loader path for T5-family requests."""

    model_path = tmp_path / "t5gemma-model"
    model_path.mkdir()
    monkeypatch.setattr(
        LoaderService,
        "_load_gemma_model_and_tokenizer",
        staticmethod(
            lambda model_path, device, dtype: ("gemma_model", "gemma_tokenizer")
        ),
    )
    monkeypatch.setattr(
        LoaderService,
        "_load_t5gemma_model_and_tokenizer",
        staticmethod(lambda model_path, device, dtype: ("t5_model", "t5_tokenizer")),
    )

    result = LoaderService._load_model_and_tokenizer(
        str(model_path), "t5gemma", "cpu", torch.float32
    )
    assert result == ("t5_model", "t5_tokenizer")


def test_loader_wraps_unknown_gemma_architecture_with_host_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explain ReForge host pinning when Gemma model support is too old."""

    class FailingAutoModelForCausalLM:
        """Stand in for transformers AutoModelForCausalLM."""

        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError(
                "The checkpoint you are trying to load has model type "
                "`gemma3_text` but Transformers does not recognize this architecture."
            )

    class DummyAutoTokenizer:
        """Stand in for transformers AutoTokenizer."""

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return "tokenizer"

    import builtins
    import types

    real_import = builtins.__import__
    fake_transformers = types.SimpleNamespace(
        AutoModelForCausalLM=FailingAutoModelForCausalLM,
        AutoTokenizer=DummyAutoTokenizer,
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers":
            return fake_transformers
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(
        loader_module, "get_installed_transformers_version", lambda: "4.48.1"
    )
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as exc_info:
        LoaderService._load_gemma_model_and_tokenizer("C:\\model", "cpu", torch.float32)

    message = str(exc_info.value)
    assert "transformers='4.48.1'" in message
    assert "--skip-install" in message
    assert "--skip-prepare-environment" in message


def test_infer_adapter_preset_name_detects_t5gemma_from_projection_shape() -> None:
    """Infer the T5 preset from the adapter projection input dimension."""

    state_dict = {
        "seq_projection.weight": torch.zeros((2048, 2304)),
    }

    assert LoaderService._infer_adapter_preset_name(state_dict) == "t5gemma"


def test_infer_adapter_preset_name_detects_gemma_from_projection_shape() -> None:
    """Infer the Gemma preset from the adapter projection input dimension."""

    state_dict = {
        "seq_projection.weight": torch.zeros((2048, 1152)),
    }

    assert LoaderService._infer_adapter_preset_name(state_dict) == "gemma"


def test_wrap_adapter_preset_error_explains_t5gemma_mismatch() -> None:
    """Explain when a T5 adapter is loaded with the Gemma preset."""

    preset = loader_module.get_preset("gemma")
    state_dict = {
        "seq_projection.weight": torch.zeros((2048, 2304)),
    }
    error = RuntimeError("Error(s) in loading state_dict for LLMToSDXLAdapter:")

    wrapped = LoaderService._wrap_adapter_preset_error(
        error,
        state_dict,
        preset,
        "C:\\models\\llm_adapters\\rouweiT5Gemma_v02.safetensors",
    )

    message = str(wrapped)
    assert "does not match the selected preset 'gemma'" in message
    assert "looks like a 't5gemma' adapter" in message
    assert "matching encoder model" in message


def test_t5gemma_loader_moves_model_to_device_without_device_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load T5-Gemma without relying on accelerate-only device_map behavior."""

    captured_kwargs: dict[str, object] = {}
    model_moves: list[str] = []

    class DummyModel:
        """Record device movement and eval calls from the loader."""

        def to(self, device: str) -> "DummyModel":
            model_moves.append(device)
            return self

        def eval(self) -> None:
            return None

    class DummyT5GemmaEncoderModel:
        """Stand in for transformers T5GemmaEncoderModel."""

        @staticmethod
        def from_pretrained(model_path: str, **kwargs):
            captured_kwargs.update(kwargs)
            return DummyModel()

    class DummyAutoTokenizer:
        """Stand in for transformers AutoTokenizer."""

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return "tokenizer"

    import builtins
    import types

    real_import = builtins.__import__
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=DummyAutoTokenizer,
        T5GemmaEncoderModel=DummyT5GemmaEncoderModel,
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "transformers":
            return fake_transformers
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    model, tokenizer = LoaderService._load_t5gemma_model_and_tokenizer(
        "C:\\model", "cuda:0", torch.bfloat16
    )

    assert tokenizer == "tokenizer"
    assert isinstance(model, DummyModel)
    assert captured_kwargs == {
        "torch_dtype": torch.bfloat16,
        "is_encoder_decoder": False,
    }
    assert model_moves == ["cuda:0"]
