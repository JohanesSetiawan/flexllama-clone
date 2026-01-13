"""
Model Controller Module

This module defines endpoints for managing model lifecycle (load, eject, list).
"""

from fastapi import APIRouter, Depends, HTTPException, status, Body
from ..lifecycle.dependencies import get_manager
from ..core.manager import ModelManager
from ..schemas.requests import (
    ModelLoadRequest,
    ModelEjectRequest,
    ModelsListResponse,
    ModelInfo,
    ErrorResponse
)

router = APIRouter(tags=["Models"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List available models"
)
async def list_models(
    manager: ModelManager = Depends(get_manager)
):
    """
    List all models configured in the system.

    Returns:
        ModelsListResponse: List of available models and their status.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    # Convert manager model info to schema
    # Logic to fetch models list from config + tracked status
    # Assuming manager has access to config

    # Manager doesn't expose a 'list_all_models' method directly that returns Pydantic
    # We construct it here.

    models_data = []

    # Iterate over config models
    for alias, model_config in manager.config.models.items():
        # Check status
        runner = manager.active_runners.get(alias)
        model_status = "stopped"
        port = None

        if runner:
            model_status = runner.status
            port = runner.port

        # VRAM usage
        vram_mb = 0.0
        if alias in manager.vram_service.model_tracks:
            vram_mb = manager.vram_service.model_tracks[alias].current_vram_used_mb

        models_data.append(ModelInfo(
            id=alias,
            created=0,  # Placeholder
            status=model_status,
            port=port,
            vram_mb=vram_mb
        ))

    return ModelsListResponse(data=models_data)


@router.post(
    "/models/load",
    summary="Load a model",
    responses={
        200: {"description": "Model loaded successfully"},
        400: {"model": ErrorResponse},
        503: {"model": ErrorResponse}
    }
)
async def load_model(
    request: ModelLoadRequest = Body(...),
    manager: ModelManager = Depends(get_manager)
):
    """
    Load a model into memory.

    Args:
        request: Load request with model alias.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    try:
        await manager.load_model(request.model)
        return {"status": "success", "message": f"Model '{request.model}' loaded"}
    except Exception as e:
        # Manager usually logs error. We catch specifics in a real app.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post(
    "/models/eject",
    summary="Eject a model",
    responses={
        200: {"description": "Model ejected successfully"},
        404: {"model": ErrorResponse}
    }
)
async def eject_model(
    request: ModelEjectRequest = Body(...),
    manager: ModelManager = Depends(get_manager)
):
    """
    Eject a model from memory.

    Args:
        request: Eject request with model alias.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    success = await manager.eject_model(request.model)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{request.model}' not running"
        )

    return {"status": "success", "message": f"Model '{request.model}' ejected"}
