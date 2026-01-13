"""
Application Dependencies Module

This module provides dependency injection for FastAPI endpoints.
Dependencies are initialized at startup and injected into route handlers.

Components:
    - Container: Holds all initialized dependencies
    - Getter functions: FastAPI Depends() compatible functions

Usage:
    from app_refactor.lifecycle.dependencies import (
        get_manager,
        get_queue_manager,
        get_http_client
    )
    
    @router.post("/v1/chat/completions")
    async def chat(
        request: Request,
        manager: ModelManager = Depends(get_manager)
    ):
        ...
"""

import httpx
import asyncio
import logging
from typing import Optional, Any
from dataclasses import dataclass, field

from ..core.config import AppConfig
from ..core.manager import ModelManager
from ..core.queue import QueueManager
from ..services.warmup_service import WarmupService
from ..services.telemetry_service import TelemetryService
from ..services.health_service import HealthService
from ..core.model_status import ModelStatusTracker
from ..services.metrics_service import MetricsService


logger = logging.getLogger(__name__)


@dataclass
class AppContainer:
    """
    Container for application dependencies.

    Holds all initialized components that are injected into route handlers.
    Provides centralized access and cleanup.

    Attributes:
        config: Application configuration
        manager: Model lifecycle manager
        queue_manager: Request queue manager
        warmup_manager: Model preload manager
        telemetry: Request telemetry collector
        health_monitor: Background health monitor
        status_tracker: Model status tracker
        prometheus_collector: Prometheus metrics collector
        http_client: Async HTTP client for proxying
        shutdown_event: Event signaling shutdown
        gpu_handle: NVML GPU handle
    """
    config: Optional[AppConfig] = None
    manager: Optional[ModelManager] = None
    queue_manager: Optional[QueueManager] = None
    warmup_service: Optional[WarmupService] = None
    telemetry: Optional[TelemetryService] = None
    health_service: Optional[HealthService] = None
    status_tracker: Optional[ModelStatusTracker] = None
    metrics_service: Optional[MetricsService] = None
    http_client: Optional[httpx.AsyncClient] = None
    shutdown_event: asyncio.Event = field(default_factory=asyncio.Event)
    gpu_handle: Optional[Any] = None
    background_tasks: list = field(default_factory=list)
    active_requests: int = 0


# Global container instance
_container: Optional[AppContainer] = None


def get_container() -> AppContainer:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = AppContainer()
    return _container


def set_container(container: AppContainer) -> None:
    """Set the global container instance."""
    global _container
    _container = container


# =============================================================================
# FastAPI Dependency Functions
# =============================================================================

def get_config() -> Optional[AppConfig]:
    """Get application configuration."""
    return get_container().config


def get_manager() -> Optional[ModelManager]:
    """Get model manager instance."""
    return get_container().manager


def get_queue_manager() -> Optional[QueueManager]:
    """Get queue manager instance."""
    return get_container().queue_manager


def get_warmup_service() -> Optional[WarmupService]:
    """Get warmup service instance."""
    return get_container().warmup_service


def get_telemetry() -> Optional[TelemetryService]:
    """Get telemetry collector instance."""
    return get_container().telemetry


def get_health_service() -> Optional[HealthService]:
    """Get health service instance."""
    return get_container().health_service


def get_status_tracker() -> Optional[ModelStatusTracker]:
    """Get status tracker instance."""
    return get_container().status_tracker


def get_metrics_service() -> Optional[MetricsService]:
    """Get MetricsService instance."""
    return get_container().metrics_service


def get_http_client() -> Optional[httpx.AsyncClient]:
    """Get HTTP client instance."""
    return get_container().http_client


def get_shutdown_event() -> asyncio.Event:
    """Get shutdown event."""
    return get_container().shutdown_event


def is_shutting_down() -> bool:
    """Check if application is shutting down."""
    return get_container().shutdown_event.is_set()


def get_active_requests() -> int:
    """Get current active request count."""
    return get_container().active_requests


def increment_active_requests() -> int:
    """Increment active request count and return new value."""
    container = get_container()
    container.active_requests += 1
    return container.active_requests


def decrement_active_requests() -> int:
    """Decrement active request count and return new value."""
    container = get_container()
    container.active_requests = max(0, container.active_requests - 1)
    return container.active_requests
