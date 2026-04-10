"""Define presets and defaults for the supported RouWei LLM adapter workflows."""

from __future__ import annotations

from dataclasses import dataclass

from .model_families import GEMMA_FAMILY, ModelFamily, T5GEMMA_FAMILY

DEFAULT_SYSTEM_PROMPT = (
    "You are expert in understanding of user prompts for image generations. "
    "Create an image according to the prompt from user."
)
DEFAULT_SKIP_FIRST = 27
DEFAULT_MAX_LENGTH = 512
DEFAULT_DEVICE = "auto"
DEFAULT_PRESET_NAME = "t5gemma"
SUPPORTED_DEVICE_CHOICES = ("auto", "cuda:0", "cuda:1", "cpu")
PLACEHOLDER_CHOICE = "<none found>"


@dataclass(frozen=True)
class AdapterPreset:
    """Describe one supported model-family and adapter-architecture preset."""

    name: str
    model_family: ModelFamily
    llm_dim: int
    sdxl_seq_dim: int
    sdxl_pooled_dim: int
    max_input_len: int
    target_seq_len: int
    n_wide_blocks: int
    n_narrow_blocks: int
    num_heads: int
    dropout: float


ROUWEI_GEMMA_PRESET = AdapterPreset(
    name="gemma",
    model_family=GEMMA_FAMILY,
    llm_dim=1152,
    sdxl_seq_dim=2048,
    sdxl_pooled_dim=1280,
    max_input_len=512,
    target_seq_len=308,
    n_wide_blocks=2,
    n_narrow_blocks=3,
    num_heads=16,
    dropout=0.1,
)

ROUWEI_T5GEMMA_PRESET = AdapterPreset(
    name="t5gemma",
    model_family=T5GEMMA_FAMILY,
    llm_dim=2304,
    sdxl_seq_dim=2048,
    sdxl_pooled_dim=1280,
    max_input_len=512,
    target_seq_len=308,
    n_wide_blocks=3,
    n_narrow_blocks=3,
    num_heads=16,
    dropout=0.0,
)

PRESETS = {
    ROUWEI_GEMMA_PRESET.name: ROUWEI_GEMMA_PRESET,
    ROUWEI_T5GEMMA_PRESET.name: ROUWEI_T5GEMMA_PRESET,
}


def get_preset(name: str) -> AdapterPreset:
    """Return the adapter preset for the supplied preset name."""

    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported adapter preset: {name}") from exc


def list_preset_names() -> list[str]:
    """Return the supported preset names in UI order."""

    return list(PRESETS.keys())
