"""Test preset and default configuration behavior."""

from __future__ import annotations

from lib_llm_sdxl_adapter import config


def test_gemma_preset_matches_source_values() -> None:
    """Keep the v1 Gemma preset aligned with the source adapter preset."""

    preset = config.get_preset("gemma")
    assert preset.model_family == "gemma"
    assert preset.llm_dim == 1152
    assert preset.sdxl_seq_dim == 2048
    assert preset.sdxl_pooled_dim == 1280
    assert preset.max_input_len == 512
    assert preset.target_seq_len == 308
    assert preset.n_wide_blocks == 2
    assert preset.n_narrow_blocks == 3
    assert preset.num_heads == 16
    assert preset.dropout == 0.1


def test_default_skip_first_matches_source_default() -> None:
    """Keep the default skip_first value aligned with the source workflow."""

    assert config.DEFAULT_SKIP_FIRST == 27


def test_t5gemma_preset_matches_source_values() -> None:
    """Keep the T5-Gemma preset aligned with the upstream adapter preset."""

    preset = config.get_preset("t5gemma")
    assert preset.model_family == "t5gemma"
    assert preset.llm_dim == 2304
    assert preset.sdxl_seq_dim == 2048
    assert preset.sdxl_pooled_dim == 1280
    assert preset.max_input_len == 512
    assert preset.target_seq_len == 308
    assert preset.n_wide_blocks == 3
    assert preset.n_narrow_blocks == 3
    assert preset.num_heads == 16
    assert preset.dropout == 0.0


def test_default_max_length_matches_source_default() -> None:
    """Keep the T5-Gemma max-length default aligned with the upstream workflow."""

    assert config.DEFAULT_MAX_LENGTH == 512


def test_default_preset_is_t5gemma() -> None:
    """Keep the UI default aligned with the current recommended adapter path."""

    assert config.DEFAULT_PRESET_NAME == "t5gemma"
