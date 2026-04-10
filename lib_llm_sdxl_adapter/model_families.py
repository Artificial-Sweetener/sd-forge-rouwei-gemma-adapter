"""Define supported LLM model-family identifiers for the extension."""

from __future__ import annotations

from typing import Literal

ModelFamily = Literal["gemma", "t5gemma"]

GEMMA_FAMILY: ModelFamily = "gemma"
T5GEMMA_FAMILY: ModelFamily = "t5gemma"
SUPPORTED_MODEL_FAMILIES: tuple[ModelFamily, ...] = (GEMMA_FAMILY, T5GEMMA_FAMILY)
