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
    - LLAMA_SERVER_PATH: Override llama-server binary path (takes priority over config file)
                         Docker: automatically set to /app/llama-server
                         Local dev: reads from config file
    - BASE_MODELS_PATH: Override base models directory (takes priority over config file)
    - CONFIG_PATH: Override config file path (default: config.json or konfig.json)

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


class RedisConfig(BaseModel):
    """
    Redis configuration for caching and queue management.

    Enables semantic caching to reduce GPU load and queue TTL for SLA compliance.
    If not configured, caching is disabled and in-memory queue is used.
    """

    # Connection settings
    url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )

    # Cache settings
    enable_cache: bool = Field(
        default=True,
        description="Enable semantic response caching"
    )
    cache_ttl_sec: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache TTL in seconds (default: 1 hour)"
    )
    cache_prefix: str = Field(
        default="router:cache:",
        description="Redis key prefix for cached responses"
    )

    # Queue settings
    enable_redis_queue: bool = Field(
        default=False,
        description="Use Redis-backed queue instead of in-memory"
    )
    queue_ttl_sec: int = Field(
        default=90,
        ge=30,
        le=300,
        description="Max time request can wait in queue (default: 90s for 1.5min SLA)"
    )
    queue_prefix: str = Field(
        default="router:queue:",
        description="Redis key prefix for queue entries"
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
        default="",
        description="Absolute path to llama-server binary"
    )
    base_models_path: str = Field(
        default="",
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

    # Default parallel requests (used if not specified in model flags)
    default_parallel: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Default parallel request slots (override per-model with --parallel flag)"
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
        """
        Validate llama-server binary path exists and is executable.

        Priority order:
        1. LLAMA_SERVER_PATH environment variable (Docker/override)
        2. Value from config file (local development)
        """
        # Environment variable takes precedence (for Docker auto-switch)
        env_path = os.getenv("LLAMA_SERVER_PATH")
        if env_path:
            logger.info(f"Using llama-server from ENV: {env_path}")
            v = env_path
        elif v:
            logger.info(f"Using llama-server from config: {v}")

        if not v:
            raise ValueError(
                "llama_server_path must be set in config, or use LLAMA_SERVER_PATH env var"
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
        """
        Validate base models directory if specified.

        Priority order:
        1. BASE_MODELS_PATH environment variable (Docker/override)
        2. Value from config file (local development)
        """
        # Environment variable takes precedence
        env_path = os.getenv("BASE_MODELS_PATH")
        if env_path:
            logger.info(f"Using base models path from ENV: {env_path}")
            v = env_path
        elif v:
            logger.info(f"Using base models path from config: {v}")

        # Optional - if empty, all model_path must be absolute
        if v:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"base_models_path not found: {v}")
            if not path.is_dir():
                raise ValueError(f"base_models_path is not a directory: {v}")

        return v


class ModelConfig(BaseModel):
    """
    Configuration for a single model.

    Uses flexible flags array that gets passed directly to llama-server.
    This allows full customization without codebase changes.

    Attributes:
        model_path: Path to .gguf model file
        flags: CLI flags to pass to llama-server
    """

    model_path: str = Field(
        ...,
        description="Path to model .gguf file"
    )
    flags: List[str] = Field(
        default_factory=list,
        description="CLI flags for llama-server (e.g., ['--ngl', '99', '--ctx-size', '4096'])"
    )

    # Internal field for resolved absolute path
    _resolved_path: Optional[str] = None

    def get_flag_value(self, flag_name: str, default: str = None) -> Optional[str]:
        """
        Get value for a specific flag from flags list.

        Args:
            flag_name: Flag name (e.g., '--ctx-size', '-c')
            default: Default value if flag not found

        Returns:
            Flag value or default
        """
        try:
            idx = self.flags.index(flag_name)
            if idx + 1 < len(self.flags):
                return self.flags[idx + 1]
        except ValueError:
            pass
        return default

    def get_ctx_size(self) -> int:
        """Get context size from flags (for VRAM estimation)."""
        for flag in ["--ctx-size", "-c"]:
            val = self.get_flag_value(flag)
            if val:
                return int(val)
        return 4096  # Default

    def get_n_gpu_layers(self) -> int:
        """Get GPU layers from flags (for VRAM estimation)."""
        for flag in ["--n-gpu-layers", "--ngl", "-ngl"]:
            val = self.get_flag_value(flag)
            if val:
                return int(val)
        return 99  # Default (all layers)

    def is_embedding_mode(self) -> bool:
        """Check if embedding mode is enabled."""
        return "--embedding" in self.flags

    def get_parallel(self, system_default: int = 4) -> int:
        """Get parallel requests setting."""
        val = self.get_flag_value("--parallel")
        if val:
            return int(val)
        return system_default

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
    redis: Optional[RedisConfig] = Field(
        default=None,
        description="Redis config for caching/queuing (None = disabled)"
    )
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


def load_config(path: Optional[str] = None) -> AppConfig:
    """
    Load and validate configuration from JSON file.

    Args:
        path: Path to config.json file. If None, uses CONFIG_PATH env var
              or defaults to 'config.json' in working directory.

    Returns:
        Validated AppConfig instance

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If JSON is invalid or validation fails
        RuntimeError: If other error occurs
    """
    # Fallback to env var or default
    if path is None:
        path = os.getenv("CONFIG_PATH", "config.json")

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
