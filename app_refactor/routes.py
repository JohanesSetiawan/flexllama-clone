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
    inference_router
)

# Main API Router
api_router = APIRouter()

# Register controllers
api_router.include_router(health_router)
api_router.include_router(status_router)
api_router.include_router(vram_router)
api_router.include_router(model_router)
api_router.include_router(inference_router)
