"""Test filesystem discovery behavior."""

from __future__ import annotations

from pathlib import Path

from lib_llm_sdxl_adapter import discovery


def test_valid_model_directory_requires_marker_file(
    tmp_path: Path, monkeypatch
) -> None:
    """Only directories with expected Hugging Face markers should be returned."""

    models_root = tmp_path / "models"
    llm_root = models_root / "llm"
    adapter_root = models_root / "llm_adapters"
    good_model = llm_root / "gemma-3-1b-it"
    bad_model = llm_root / "not_a_model"
    good_model.mkdir(parents=True)
    bad_model.mkdir(parents=True)
    (good_model / "config.json").write_text("{}", encoding="utf-8")
    adapter_root.mkdir(parents=True)
    (adapter_root / "rouwei.safetensors").write_text("weights", encoding="utf-8")

    monkeypatch.setattr(discovery, "_get_models_root", lambda: models_root)

    models = discovery.discover_llm_models()
    adapters = discovery.discover_adapters()

    assert [item.name for item in models] == ["gemma-3-1b-it"]
    assert [item.name for item in adapters] == ["rouwei.safetensors"]
