"""Load and cache the LLM model, tokenizer, and adapter for the active request."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import importlib.metadata
import logging
from pathlib import Path
import threading
from typing import Any

import torch

from .adapter_model import LLMToSDXLAdapter
from .config import AdapterPreset, get_preset
from .model_families import GEMMA_FAMILY, T5GEMMA_FAMILY
from .runtime_context import RequestConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedArtifacts:
    """Carry the loaded runtime objects needed for adapter conditioning."""

    model: Any
    tokenizer: Any
    adapter: LLMToSDXLAdapter
    preset: AdapterPreset
    device: str
    dtype: torch.dtype
    model_path: str
    adapter_path: str


class LoaderService:
    """Own all loading, caching, and cleanup for extension runtime assets."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._artifacts: LoadedArtifacts | None = None
        self._cache_key: tuple[str, ...] | None = None

    def ensure_loaded(self, request: RequestConfig) -> LoadedArtifacts:
        """Load or reuse the configured model/tokenizer/adapter triple."""

        if not request.enabled:
            raise ValueError("Cannot load artifacts for a disabled request.")

        cache_key = self._build_cache_key(request)
        with self._lock:
            if (
                not request.force_reload
                and self._artifacts is not None
                and cache_key == self._cache_key
            ):
                LOGGER.info(
                    "Reusing cached LLM adapter artifacts for %s / %s",
                    request.model_name,
                    request.adapter_name,
                )
                return self._artifacts

            self._clear_locked()
            preset = get_preset(request.preset_name)
            device = self._resolve_device(request.device)
            dtype = self._resolve_dtype(device)
            model, tokenizer = self._load_model_and_tokenizer(
                request.model_path, request.model_family, device, dtype
            )
            adapter = self._load_adapter(request.adapter_path, preset, device)
            artifacts = LoadedArtifacts(
                model=model,
                tokenizer=tokenizer,
                adapter=adapter,
                preset=preset,
                device=device,
                dtype=dtype,
                model_path=request.model_path,
                adapter_path=request.adapter_path,
            )
            self._artifacts = artifacts
            self._cache_key = cache_key
            return artifacts

    def clear(self) -> None:
        """Clear any loaded artifacts and release GPU memory where possible."""

        with self._lock:
            self._clear_locked()

    def _clear_locked(self) -> None:
        """Clear loaded artifacts while the service lock is already held."""

        if self._artifacts is None:
            return

        LOGGER.info("Clearing cached LLM adapter artifacts.")
        del self._artifacts
        self._artifacts = None
        self._cache_key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _build_cache_key(request: RequestConfig) -> tuple[str, ...]:
        """Build the loader-owned cache key for reusable runtime assets."""

        return (
            request.model_path,
            request.adapter_path,
            request.preset_name,
            request.model_family,
            request.device,
        )

    @staticmethod
    def _resolve_device(device_choice: str) -> str:
        """Resolve the requested device string into a concrete runtime device."""

        if device_choice == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return device_choice

    @staticmethod
    def _resolve_dtype(device: str) -> torch.dtype:
        """Resolve the LLM load dtype for the selected device."""

        if device.startswith("cuda"):
            if (
                hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            ):
                return torch.bfloat16
            return torch.float16
        return torch.float32

    @staticmethod
    def _load_model_and_tokenizer(
        model_path: str,
        model_family: str,
        device: str,
        dtype: torch.dtype,
    ) -> tuple[Any, Any]:
        """Load the requested model-family and tokenizer pair."""

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"LLM model path does not exist: {model_path}")

        LOGGER.info(
            "Loading %s model from %s on %s with dtype=%s",
            model_family,
            model_path,
            device,
            dtype,
        )
        if model_family == GEMMA_FAMILY:
            return LoaderService._load_gemma_model_and_tokenizer(
                model_path, device, dtype
            )
        if model_family == T5GEMMA_FAMILY:
            return LoaderService._load_t5gemma_model_and_tokenizer(
                model_path, device, dtype
            )
        raise ValueError(f"Unsupported model family for loading: {model_family}")

    @staticmethod
    def _load_gemma_model_and_tokenizer(
        model_path: str, device: str, dtype: torch.dtype
    ) -> tuple[Any, Any]:
        """Load one Gemma causal-LM model and tokenizer pair."""

        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        except ValueError as exc:
            raise LoaderService._wrap_host_transformers_version_error(
                exc,
                model_family=GEMMA_FAMILY,
                feature_name="Gemma-family causal model loading",
            ) from exc
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    @staticmethod
    def _load_t5gemma_model_and_tokenizer(
        model_path: str, device: str, dtype: torch.dtype
    ) -> tuple[Any, Any]:
        """Load one T5-Gemma encoder model and tokenizer pair."""

        try:
            from transformers import AutoTokenizer, T5GemmaEncoderModel
        except (ImportError, AttributeError) as exc:
            installed_version = get_installed_transformers_version()
            raise RuntimeError(
                "The T5-Gemma preset requires a host transformers build with "
                f"T5GemmaEncoderModel. This environment currently has "
                f"transformers={installed_version!r}. Upgrade the ReForge venv "
                "and launch with `--skip-install` or `--skip-prepare-environment`."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "is_encoder_decoder": False,
        }

        model = T5GemmaEncoderModel.from_pretrained(
            model_path,
            **load_kwargs,
        )
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return model, tokenizer

    @staticmethod
    def _load_adapter(
        adapter_path: str, preset: AdapterPreset, device: str
    ) -> LLMToSDXLAdapter:
        """Load the adapter weights into a preset-shaped adapter network."""

        from safetensors.torch import load_file

        path = Path(adapter_path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

        LOGGER.info("Loading adapter weights from %s on %s", adapter_path, device)
        adapter = LLMToSDXLAdapter(
            llm_dim=preset.llm_dim,
            sdxl_seq_dim=preset.sdxl_seq_dim,
            sdxl_pooled_dim=preset.sdxl_pooled_dim,
            max_input_len=preset.max_input_len,
            target_seq_len=preset.target_seq_len,
            n_wide_blocks=preset.n_wide_blocks,
            n_narrow_blocks=preset.n_narrow_blocks,
            num_heads=preset.num_heads,
            dropout=preset.dropout,
        )
        state_dict = load_file(adapter_path)
        try:
            adapter.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise LoaderService._wrap_adapter_preset_error(
                exc, state_dict, preset, adapter_path
            ) from exc
        adapter.to(device)
        adapter.eval()
        return adapter

    @staticmethod
    def _infer_adapter_preset_name(state_dict: dict[str, Any]) -> str | None:
        """Infer the matching preset name from adapter tensor shapes when possible."""

        seq_projection_weight = state_dict.get("seq_projection.weight")
        if seq_projection_weight is not None and hasattr(seq_projection_weight, "shape"):
            if len(seq_projection_weight.shape) == 2:
                llm_dim = int(seq_projection_weight.shape[1])
                if llm_dim == 2304:
                    return T5GEMMA_FAMILY
                if llm_dim == 1152:
                    return GEMMA_FAMILY

        if any(key.startswith("wide_attention_blocks.2.") for key in state_dict):
            return T5GEMMA_FAMILY

        return None

    @staticmethod
    def _wrap_adapter_preset_error(
        exc: RuntimeError,
        state_dict: dict[str, Any],
        preset: AdapterPreset,
        adapter_path: str,
    ) -> RuntimeError:
        """Return a clearer preset mismatch error for adapter state-dict failures."""

        message = str(exc)
        if "Error(s) in loading state_dict for LLMToSDXLAdapter" not in message:
            return exc

        inferred_preset_name = LoaderService._infer_adapter_preset_name(state_dict)
        adapter_name = Path(adapter_path).name

        if inferred_preset_name is not None and inferred_preset_name != preset.name:
            return RuntimeError(
                f"Adapter '{adapter_name}' does not match the selected preset "
                f"'{preset.name}'. It looks like a '{inferred_preset_name}' adapter. "
                f"Select the '{inferred_preset_name}' preset and the matching "
                "encoder model, then try again."
            )

        return RuntimeError(
            f"Adapter '{adapter_name}' could not be loaded with preset "
            f"'{preset.name}'. Check that the selected adapter matches the preset "
            "and encoder family."
        )

    @staticmethod
    def _wrap_host_transformers_version_error(
        exc: ValueError,
        *,
        model_family: str,
        feature_name: str,
    ) -> RuntimeError | ValueError:
        """Return a clearer host-version error for known transformers failures."""

        message = str(exc)
        if "does not recognize this architecture" not in message:
            return exc

        installed_version = get_installed_transformers_version()
        return RuntimeError(
            f"{feature_name} requires a newer host transformers build. This "
            f"environment currently has transformers={installed_version!r}. "
            "Upgrade the ReForge venv and launch with `--skip-install` or "
            "`--skip-prepare-environment`."
        )


def get_installed_transformers_version() -> str:
    """Return the installed transformers version for runtime error reporting."""

    try:
        return importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


LOADER = LoaderService()
