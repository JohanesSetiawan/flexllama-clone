"""
Lifecycle Package

This package provides application lifecycle management including:
- Dependency injection (dependencies.py)
- Startup initialization (startup.py)  
- Graceful shutdown (shutdown.py)
"""

from .dependencies import (
    AppContainer,
    get_container,
    set_container,
    get_config,
    get_manager,
    get_queue_manager,
    get_warmup_manager,
    get_telemetry,
    get_health_monitor,
    get_status_tracker,
    get_prometheus,
    get_http_client,
    get_shutdown_event,
    is_shutting_down,
    get_active_requests,
    increment_active_requests,
    decrement_active_requests,
)

from .startup import startup_handler
from .shutdown import shutdown_handler

__all__ = [
    # Container
    "AppContainer",
    "get_container",
    "set_container",

    # Dependency getters
    "get_config",
    "get_manager",
    "get_queue_manager",
    "get_warmup_manager",
    "get_telemetry",
    "get_health_monitor",
    "get_status_tracker",
    "get_prometheus",
    "get_http_client",
    "get_shutdown_event",
    "is_shutting_down",
    "get_active_requests",
    "increment_active_requests",
    "decrement_active_requests",

    # Lifecycle handlers
    "startup_handler",
    "shutdown_handler",
]
