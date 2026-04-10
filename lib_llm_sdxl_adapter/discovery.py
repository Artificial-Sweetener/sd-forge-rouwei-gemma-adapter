"""Discover LLM model directories and adapter files for the extension."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)
MODEL_MARKER_FILES = ("config.json", "model.safetensors", "pytorch_model.bin")
ADAPTER_SUFFIXES = (".safetensors",)


@dataclass(frozen=True)
class DiscoveryItem:
    """Represent one discovered filesystem-backed model choice."""

    name: str
    path: Path

    @property
    def normalized_path(self) -> str:
        """Return a stable resolved path string for cache keys and metadata."""

        return str(self.path.resolve())


def _get_models_root() -> Path:
    """Return the ReForge models root directory."""

    from modules.paths_internal import models_path

    return Path(models_path)


def get_llm_root() -> Path:
    """Return the root directory used for Hugging Face LLM model folders."""

    return _get_models_root() / "llm"


def get_adapter_root() -> Path:
    """Return the root directory used for adapter weight files."""

    return _get_models_root() / "llm_adapters"


def _is_valid_model_directory(candidate: Path) -> bool:
    """Return whether the path looks like a usable Hugging Face model directory."""

    if not candidate.is_dir():
        return False

    existing_names = {entry.name for entry in candidate.iterdir()}
    return any(marker in existing_names for marker in MODEL_MARKER_FILES)


def discover_llm_models() -> list[DiscoveryItem]:
    """Return discovered Hugging Face model directories in stable name order."""

    root = get_llm_root()
    if not root.exists():
        LOGGER.info("LLM model root does not exist: %s", root)
        return []

    return [
        DiscoveryItem(name=path.name, path=path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if _is_valid_model_directory(path)
    ]


def discover_adapters() -> list[DiscoveryItem]:
    """Return discovered adapter weight files in stable name order."""

    root = get_adapter_root()
    if not root.exists():
        LOGGER.info("Adapter root does not exist: %s", root)
        return []

    return [
        DiscoveryItem(name=path.name, path=path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.lower())
        if path.is_file() and path.suffix.lower() in ADAPTER_SUFFIXES
    ]


def list_llm_model_names() -> list[str]:
    """Return the UI-facing LLM model names."""

    return [item.name for item in discover_llm_models()]


def list_adapter_names() -> list[str]:
    """Return the UI-facing adapter names."""

    return [item.name for item in discover_adapters()]


def resolve_model_choice(name: str) -> DiscoveryItem:
    """Return the discovered model item for a UI-selected model name."""

    for item in discover_llm_models():
        if item.name == name:
            return item
    raise ValueError(f"LLM model not found: {name}")


def resolve_adapter_choice(name: str) -> DiscoveryItem:
    """Return the discovered adapter item for a UI-selected adapter name."""

    for item in discover_adapters():
        if item.name == name:
            return item
    raise ValueError(f"Adapter not found: {name}")
