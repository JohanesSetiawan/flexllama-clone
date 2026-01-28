"""
Main Application Module

This module defines the FastAPI application factory and configuration.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import AppConfig
from .routes import api_router
from .lifecycle.startup import startup_handler
from .lifecycle.shutdown import shutdown_handler
from .lifecycle.dependencies import get_container, set_container, AppContainer

from .middlewares.limit_request import RequestSizeLimitMiddleware
from .middlewares.request_tracker import RequestTrackerMiddleware
from .middlewares.telemetry_middleware import TelemetryMiddleware
from .middlewares.metrics_middleware import MetricsMiddleware
from .middlewares.auth import AuthMiddleware
from .middlewares.rate_limit import setup_rate_limiting


logger = logging.getLogger(__name__)


def create_app(config: AppConfig) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Application configuration

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="Router Model API",
        version="2.0.0",
        description="High-performance LLM Router and Gateway",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Initialize Dependency Container
    container = AppContainer()
    container.config = config
    set_container(container)

    # Add Middlewares (Order matters!)
    # Execution order: last added -> first executed

    # 1. CORS (First to handle pre-flight)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Authentication (Reject unauthorized early)
    app.add_middleware(AuthMiddleware)

    # 3. Rate Limiting (Only if configured)
    setup_rate_limiting(app, config.api.rate_limit)

    # 4. Request Size Limit (Early rejection)
    app.add_middleware(
        RequestSizeLimitMiddleware,
        max_size=10 * 1024 * 1024  # 10MB
    )

    # 3. Telemetry (Detailed tracking)
    app.add_middleware(TelemetryMiddleware)

    # 4. Metrics (Simple counters)
    app.add_middleware(MetricsMiddleware)

    # 5. Request Tracker (Active request count for graceful shutdown)
    app.add_middleware(RequestTrackerMiddleware)

    # Include Routes
    app.include_router(api_router)

    # Lifecycle Events
    @app.on_event("startup")
    async def on_startup():
        await startup_handler()

    @app.on_event("shutdown")
    async def on_shutdown():
        await shutdown_handler()

    logger.info("FastAPI application created")
    return app
