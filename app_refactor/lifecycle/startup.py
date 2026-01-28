"""
Application Startup Module

This module handles FastAPI application startup initialization.
All components are initialized in a specific order to ensure dependencies
are available when needed.

Initialization Order:
    1. Status tracker (for early status updates)
    2. Configuration loading
    3. HTTP client
    4. Model manager (VRAM tracker starts here)
    5. Queue manager
    6. Warmup manager
    7. Telemetry collector
    8. GPU monitoring (NVML)
    9. Health monitor
    10. Prometheus metrics
    11. Start warmup (preload models)
    12. Start health monitoring
    13. Start background tasks

Usage:
    from app_refactor.lifecycle.startup import startup_handler
    
    @app.on_event("startup")
    async def on_startup():
        await startup_handler()
"""

import os
import asyncio
import logging
import httpx
import pynvml
from pathlib import Path

from .dependencies import get_container, AppContainer
from ..core.config import load_config
from ..core.manager import ModelManager
from ..core.queue import QueueManager
from ..services.warmup_service import WarmupService
from ..services.telemetry_service import TelemetryService
from ..services.health_service import HealthService
from ..services.cache_service import init_cache_service
from ..services.redis_queue_service import init_redis_queue_service
from ..core.model_status import init_status_tracker
from ..services.metrics_service import init_metrics_service
from ..core.logging_server import setup_logging


logger = logging.getLogger(__name__)


# Config path from environment or default
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = os.getenv("CONFIG_PATH", str(PROJECT_ROOT / "config.json"))


async def startup_handler() -> None:
    """
    Initialize all application components at startup.

    This function is called by FastAPI's on_event("startup") handler.
    Components are initialized in order of their dependencies.
    """
    container = get_container()

    try:
        # Step 1: Initialize status tracker first for early status updates
        logger.info("Initializing ModelStatusTracker")
        container.status_tracker = init_status_tracker()
        await container.status_tracker.set_server_status("initializing")

        # Step 2: Load configuration
        logger.info(f"Loading config from: {CONFIG_PATH}")
        container.config = load_config(CONFIG_PATH)

        # Initialize status for all configured models
        await container.status_tracker.initialize_from_config(
            list(container.config.models.keys())
        )

        # Step 2.5: Initialize Redis cache service
        if container.config.redis:
            logger.info("Initializing Redis Cache Service")
            container.cache_service = await init_cache_service(container.config.redis)
        else:
            logger.info("Redis caching disabled (no redis config)")
            container.cache_service = None

        # Step 2.6: Initialize Redis queue service
        if container.config.redis and container.config.redis.enable_redis_queue:
            logger.info("Initializing Redis Queue Service")
            container.redis_queue_service = await init_redis_queue_service(
                container.config.redis
            )
        else:
            logger.info("Redis queue disabled (using in-memory queue)")
            container.redis_queue_service = None

        # Step 3: Initialize HTTP client
        logger.info("Initializing HTTP client")
        container.http_client = _create_http_client(container.config)

        # Step 4: Initialize Model Manager
        logger.info("Initializing ModelManager")
        container.manager = ModelManager(
            container.config,
            container.shutdown_event
        )

        # Step 5: Initialize Queue Manager
        logger.info("Initializing QueueManager")
        container.queue_manager = QueueManager(
            container.config,
            redis_queue_service=container.redis_queue_service
        )

        # Step 6: Initialize Warmup Service
        logger.info("Initializing WarmupService")
        container.warmup_service = WarmupService(
            manager=container.manager,
            config=container.config,
            shutdown_event=container.shutdown_event
        )

        # Step 7: Initialize Telemetry Collector
        logger.info("Initializing TelemetryService")
        container.telemetry = TelemetryService()

        # Step 8: Initialize GPU monitoring
        logger.info("Initializing GPU monitoring")
        container.gpu_handle = _init_gpu_monitoring()

        # Step 9: Initialize Health Service
        logger.info("Initializing HealthService")
        container.health_service = HealthService(
            manager=container.manager,
            check_interval_sec=30
        )

        # Step 10: Initialize Metrics Service
        logger.info("Initializing MetricsService")
        container.metrics_service = init_metrics_service(
            gpu_device_index=container.config.system.gpu_devices[0],
            max_concurrent_models=container.config.system.max_concurrent_models
        )

        # Step 11: Start warmup service (preload models)
        logger.info("Starting warmup service")
        await container.warmup_service.start()

        # Step 12: Start health monitoring
        logger.info("Starting health monitoring")
        container.health_service.start()

        # Step 13: Start background tasks
        logger.info("Starting background tasks")
        await _start_background_tasks(container)

        # Mark server as ready
        await container.status_tracker.set_server_status("ready")
        logger.info("Server startup complete!")

    except Exception as e:
        logger.exception(f"FATAL: Server initialization failed: {e}")

        # Cleanup on failure
        await _emergency_cleanup(container)
        raise


def _create_http_client(config) -> httpx.AsyncClient:
    """Create configured HTTP client for proxying requests."""
    limits = httpx.Limits(
        max_keepalive_connections=config.system.http_max_keepalive,
        max_connections=config.system.http_max_connections,
        keepalive_expiry=60.0
    )

    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=2.0,  # Local connections are fast
            read=config.system.request_timeout_sec * 2,
            write=5.0,
            pool=5.0
        ),
        limits=limits,
        http2=False  # HTTP/1.1 is faster for local llama.cpp
    )


def _init_gpu_monitoring():
    """Initialize NVML for GPU monitoring."""
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        logger.info(f"Connected to GPU: {gpu_name}")
        return gpu_handle
    except Exception as e:
        logger.warning(f"Failed to initialize GPU monitoring: {e}")
        return None


async def _start_background_tasks(container: AppContainer) -> None:
    """Start all background tasks."""
    from ..tasks.status_sync import sync_model_statuses

    # Status sync task
    status_sync_task = asyncio.create_task(
        sync_model_statuses(
            container.manager,
            container.status_tracker,
            container.shutdown_event
        )
    )
    container.background_tasks.append(status_sync_task)

    logger.info(f"Started {len(container.background_tasks)} background tasks")


async def _emergency_cleanup(container: AppContainer) -> None:
    """Emergency cleanup on startup failure."""
    logger.info("Performing emergency cleanup...")

    # Stop manager (will stop runners and VRAM service)
    if container.manager:
        try:
            await container.manager.stop_all_runners()
            logger.info("Stopped all runners during emergency cleanup")
        except Exception as e:
            logger.warning(f"Error stopping runners: {e}")

        # Stop VRAM service monitoring
        if hasattr(container.manager, 'vram_service') and container.manager.vram_service:
            try:
                container.manager.vram_service.stop_monitoring()
                logger.info("Stopped VRAM monitoring during emergency cleanup")
            except Exception as e:
                logger.warning(f"Error stopping VRAM monitoring: {e}")

    # Shutdown GPU/NVML
    if container.gpu_handle:
        try:
            pynvml.nvmlShutdown()
            logger.info("NVML shutdown during emergency cleanup")
        except Exception:
            pass

    # Close HTTP client
    if container.http_client:
        try:
            await container.http_client.aclose()
            logger.info("HTTP client closed during emergency cleanup")
        except Exception:
            pass

    logger.info("Emergency cleanup complete")
