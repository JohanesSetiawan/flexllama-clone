"""
Application Configuration Module

This module provides Pydantic-based configuration models for the router model.
All configuration is validated and type-checked at load time.

Configuration Sections:
    - ApiConfig: API server settings (host, port, CORS)
    - SystemConfig: System settings (VRAM, queue, timeouts)
    - ModelParams: Per-model llama-server parameters
    - ModelConfig: Model file path and parameters
    - AppConfig: Root configuration container

Environment Variables:
    - LLAMA_SERVER_PATH: Override llama-server binary path
    - BASE_MODELS_PATH: Override base models directory

Usage:
    from app_refactor.core.config import load_config
    
    config = load_config("config.json")
    print(config.api.host, config.api.port)
    print(config.models.keys())
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """
    Rate limiting configuration.

    When present, enables request rate limiting using Redis backend.
    If not configured, rate limiting is disabled.
    """

    requests_per_minute: int = Field(
        default=60,
        ge=1,
        le=10000,
        description="Maximum requests per minute"
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for rate limit storage"
    )


class ApiConfig(BaseModel):
    """
    API server configuration.

    Attributes:
        host: IP address to bind (0.0.0.0 for all interfaces)
        port: Port number (1024-65535)
        cors_origins: List of allowed CORS origins
        rate_limit: Optional rate limiting config (None = disabled)
    """

    host: str = Field(
        default="0.0.0.0",
        description="IP address for API server binding"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Port for API server (1024-65535)"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000"],
        description="Allowed origins for CORS"
    )
    rate_limit: Optional[RateLimitConfig] = Field(
        default=None,
        description="Rate limiting config (None = disabled)"
    )


class SystemConfig(BaseModel):
    """
    System configuration for the router model.

    Controls llama-server behavior, VRAM management, queue system,
    and other performance parameters.
    """

    # Idle timeout settings
    enable_idle_timeout: bool = Field(
        default=True,
        description="Enable auto-unload of idle models"
    )
    idle_timeout_sec: int = Field(
        default=300,
        ge=60,
        le=86400,
        description="Idle time (seconds) before model unload"
    )

    # Server paths
    llama_server_path: str = Field(
        default=os.getenv("LLAMA_SERVER_PATH", ""),
        description="Absolute path to llama-server binary"
    )
    base_models_path: str = Field(
        default=os.getenv("BASE_MODELS_PATH", ""),
        description="Base directory for model files"
    )

    # Model limits
    max_concurrent_models: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent models in VRAM"
    )

    # Timeout settings
    request_timeout_sec: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Request timeout for llama-server (seconds)"
    )
    timeout_warmup_sec: int = Field(
        default=180,
        ge=120,
        le=3600,
        description="Timeout for model warmup during load"
    )
    wait_ready_sec: int = Field(
        default=120,
        ge=120,
        le=3600,
        description="Wait time for runner ready status"
    )

    # Preload settings
    preload_models: List[str] = Field(
        default=[],
        description="Model aliases to preload at startup"
    )
    preload_delay_sec: int = Field(
        default=5,
        ge=1,
        le=120,
        description="Delay between preloading models"
    )

    # VRAM management
    min_vram_required: int = Field(
        default=500,
        ge=200,
        le=750,
        description="Minimum VRAM (MB) required before loading new model"
    )
    vram_multiplier: float = Field(
        default=1.1,
        ge=1.0,
        le=3.0,
        description="Multiplier for VRAM estimation"
    )
    keep_warm_models: int = Field(
        default=2,
        ge=0,
        description="Number of models to keep warm (not unloaded)"
    )

    # GPU settings
    gpu_devices: List[int] = Field(
        default=[0],
        description="GPU device indices to use"
    )

    # Performance settings
    parallel_requests: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Parallel request slots per model"
    )
    cpu_threads: int = Field(
        default=8,
        ge=1,
        le=64,
        description="CPU threads for non-GPU operations"
    )
    use_mmap: bool = Field(
        default=True,
        description="Use memory mapping for model loading"
    )
    flash_attention: str = Field(
        default="on",
        description="Flash Attention mode: 'on' or 'off'"
    )

    # Queue settings
    max_queue_size_per_model: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum queue size per model"
    )
    queue_timeout_sec: int = Field(
        default=300,
        ge=30,
        le=600,
        description="Timeout for requests in queue"
    )
    queue_processor_idle_sec: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Idle time before queue processor stops"
    )

    # HTTP client settings
    http_max_keepalive: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Max keepalive connections"
    )
    http_max_connections: int = Field(
        default=200,
        ge=20,
        le=1000,
        description="Max total HTTP connections"
    )

    # Retry settings
    model_load_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Max retries for failed model load"
    )

    @property
    def calculated_max_queue_size(self) -> int:
        """Calculate optimal queue size based on timeout and expected latency."""
        return max(20, int(self.queue_timeout_sec / 10) * 2)

    @field_validator("llama_server_path")
    @classmethod
    def validate_llama_server_path(cls, v: str) -> str:
        """Validate llama-server binary path exists and is executable."""
        # Environment variable takes precedence
        env_path = os.getenv("LLAMA_SERVER_PATH", "")
        if env_path:
            v = env_path

        if not v:
            raise ValueError(
                "llama_server_path must be set, or use LLAMA_SERVER_PATH env var"
            )

        path = Path(v)
        if not path.exists():
            raise ValueError(f"llama-server not found at: {v}")
        if not path.is_file():
            raise ValueError(f"llama_server_path is not a file: {v}")
        if not os.access(path, os.X_OK):
            raise ValueError(f"llama-server is not executable: {v}")

        return v

    @field_validator("base_models_path")
    @classmethod
    def validate_base_models_path(cls, v: str) -> str:
        """Validate base models directory if specified."""
        # Environment variable takes precedence
        env_path = os.getenv("BASE_MODELS_PATH", "")
        if env_path:
            v = env_path

        # Optional - if empty, all model_path must be absolute
        if v:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"base_models_path not found: {v}")
            if not path.is_dir():
                raise ValueError(f"base_models_path is not a directory: {v}")

        return v


class ModelParams(BaseModel):
    """
    Per-model parameters for llama-server.

    These parameters are passed to llama-server when the model is loaded.
    Each model can have different configuration according to its needs.
    """

    n_gpu_layers: int = Field(
        default=99,
        ge=-1,
        description="Number of model layers to load on GPU"
    )
    n_ctx: int = Field(
        default=4096,
        ge=512,
        le=131072,
        description="Context window size (tokens)"
    )
    n_batch: int = Field(
        default=256,
        ge=128,
        le=512,
        description="Batch size for prompt processing"
    )
    rope_freq_base: Optional[int] = Field(
        default=None,
        ge=0,
        description="RoPE frequency base for extended context"
    )
    embedding: bool = Field(
        default=False,
        description="Enable embedding mode"
    )
    chat_template: Optional[str] = Field(
        default=None,
        description="Override chat template"
    )
    parallel_override: Optional[int] = Field(
        default=None,
        ge=1,
        le=32,
        description="Override parallel requests for this model"
    )
    batch_override: Optional[int] = Field(
        default=None,
        ge=128,
        le=4096,
        description="Override batch size for this model"
    )
    type_k: Optional[str] = Field(
        default="f16",
        description="KV cache key type (f16, q8_0, q4_0, etc.)"
    )
    type_v: Optional[str] = Field(
        default="f16",
        description="KV cache value type (f16, q8_0, q4_0, etc.)"
    )

    @field_validator("type_k", "type_v")
    @classmethod
    def validate_cache_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate KV cache type is supported."""
        if v is None or v == "":
            return "f16"

        valid_types = [
            "f16", "f32", "bf16", "q8_0",
            "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"
        ]
        if v not in valid_types:
            raise ValueError(
                f"Cache type must be one of: {', '.join(valid_types)}"
            )
        return v


class ModelConfig(BaseModel):
    """
    Configuration for a single model.

    Each model has its own file path and specific parameters.
    Models are identified by their alias (key in models dict).

    Attributes:
        model_path: Path to .gguf model file
        params: Model-specific parameters
    """

    model_path: str = Field(
        ...,
        description="Path to model .gguf file"
    )
    params: ModelParams = Field(
        default_factory=ModelParams,
        description="Model-specific parameters"
    )

    # Internal field for resolved absolute path
    _resolved_path: Optional[str] = None

    def resolve_path(self, base_models_path: str) -> str:
        """
        Resolve model_path to absolute path.

        Args:
            base_models_path: Base directory for relative paths

        Returns:
            Absolute path to model file

        Raises:
            ValueError: If path cannot be resolved or file not found
        """
        path = Path(self.model_path)

        if path.is_absolute():
            resolved = path
        else:
            if not base_models_path:
                raise ValueError(
                    f"model_path '{self.model_path}' is relative, "
                    f"but base_models_path is not set. "
                    f"Set base_models_path or use absolute path."
                )
            resolved = Path(base_models_path) / self.model_path

        # Validate resolved path
        if not resolved.exists():
            raise ValueError(f"Model file not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"Model path is not a file: {resolved}")
        if not str(resolved).endswith(".gguf"):
            raise ValueError(f"Model file must be .gguf format: {resolved}")

        self._resolved_path = str(resolved)
        return self._resolved_path

    def get_resolved_path(self) -> str:
        """
        Get the resolved absolute path.

        Returns:
            Absolute path to model file

        Raises:
            RuntimeError: If resolve_path() was not called
        """
        if self._resolved_path is None:
            raise RuntimeError(
                "resolve_path() must be called first. "
                "This should happen automatically when loading config."
            )
        return self._resolved_path


class AppConfig(BaseModel):
    """
    Root application configuration.

    Contains all configuration sections: api, system, and models.

    Attributes:
        api: API server configuration
        system: System configuration
        models: Dictionary of model configurations (alias -> config)
    """

    api: ApiConfig
    system: SystemConfig
    models: Dict[str, ModelConfig]

    @field_validator("models")
    @classmethod
    def validate_models_not_empty(
        cls, v: Dict[str, ModelConfig]
    ) -> Dict[str, ModelConfig]:
        """Ensure at least one model is defined."""
        if not v:
            raise ValueError("At least one model must be defined")
        return v

    @model_validator(mode="after")
    def resolve_all_model_paths(self) -> "AppConfig":
        """
        Resolve all model paths to absolute paths after config is loaded.

        This allows model_path to be relative to base_models_path.
        """
        base_path = self.system.base_models_path

        for model_alias, model_conf in self.models.items():
            try:
                model_conf.resolve_path(base_path)
            except ValueError as e:
                raise ValueError(f"Error in model '{model_alias}': {e}")

        return self


def load_config(path: str) -> AppConfig:
    """
    Load and validate configuration from JSON file.

    Args:
        path: Path to config.json file

    Returns:
        Validated AppConfig instance

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If JSON is invalid or validation fails
        RuntimeError: If other error occurs
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = AppConfig(**data)

        logger.info(f"Configuration loaded from '{path}'")
        logger.info(f"Available models: {list(config.models.keys())}")

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: '{path}'")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}")
    except ValueError as e:
        raise ValueError(f"Config validation failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Error reading config: {e}")
