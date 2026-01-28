"""
Routes Configuration

This module aggregates all controller routers into a single API router.
"""

from fastapi import APIRouter
from .controllers import (
    health_router,
    status_router,
    vram_router,
    model_router,
    inference_router,
    metrics_router
)

# Main API Router
api_router = APIRouter()

# System endpoints (no prefix)
api_router.include_router(health_router)  # /health, /ready, /live
api_router.include_router(vram_router)    # /vram

# /metrics, /metrics/stream, /metrics/report
api_router.include_router(metrics_router)

# Status endpoints (no prefix, but includes v1 routes)
api_router.include_router(status_router)  # /status, /status/{alias}

# OpenAI-compatible endpoints (with /v1 prefix)
api_router.include_router(model_router, prefix="/v1")      # /v1/models, etc.

# /v1/chat/completions, etc.
api_router.include_router(inference_router, prefix="/v1")
