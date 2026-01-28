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


@router.get(
    "/models/{model_alias}/status",
    summary="Get model loading status"
)
async def get_model_loading_status(
    model_alias: str,
    manager: ModelManager = Depends(get_manager)
):
    """
    Get detailed loading status for a specific model.

    Args:
        model_alias: The model alias to check.

    Returns:
        Model loading status with progress details.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    try:
        status_info = await manager.get_model_status(model_alias)
        return {"model": model_alias, **status_info}
    except LookupError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.post(
    "/models/{model_alias}/reset",
    summary="Reset failed model status"
)
async def reset_model_failure(
    model_alias: str,
    manager: ModelManager = Depends(get_manager)
):
    """
    Reset failed model status to allow retry.

    Useful when you've fixed configuration and want to retry loading.

    Args:
        model_alias: The model alias to reset.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    async with manager.lock:
        if model_alias in manager.failed_models:
            failed_info = manager.failed_models[model_alias]
            del manager.failed_models[model_alias]

            return {
                "status": "success",
                "model": model_alias,
                "message": f"Model failure status cleared. Had {failed_info['attempts']} failed attempts.",
                "previous_error": failed_info['error']
            }
        else:
            return {
                "status": "not_found",
                "model": model_alias,
                "message": f"Model '{model_alias}' has no failure record."
            }


@router.get(
    "/models/failed",
    summary="List failed models"
)
async def get_failed_models(
    manager: ModelManager = Depends(get_manager)
):
    """
    Get list of models that have failed to start.

    Returns:
        List of failed models with error details.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    async with manager.lock:
        if not manager.failed_models:
            return {
                "failed_models": [],
                "message": "No failed models"
            }

        return {
            "failed_models": [
                {
                    "model": alias,
                    "attempts": info["attempts"],
                    "error": info["error"][:200]
                }
                for alias, info in manager.failed_models.items()
            ]
        }


@router.get(
    "/queue/stats",
    summary="Get queue statistics"
)
async def get_queue_stats(
    manager: ModelManager = Depends(get_manager)
):
    """
    Get detailed statistics for all model queues.

    Returns:
        Queue length, total processed, rejection count, and processing status.
    """
    from ..lifecycle.dependencies import get_queue_manager
    queue_manager = get_queue_manager()

    if not queue_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Queue manager not available"
        )

    stats = queue_manager.get_all_stats()

    # Add summary
    total_queued = sum(q.get("queue_length", 0) for q in stats.values())
    total_processing = sum(q.get("current_processing", 0)
                           for q in stats.values())
    total_processed = sum(q.get("total_processed", 0) for q in stats.values())
    total_rejected = sum(q.get("total_rejected", 0) for q in stats.values())

    return {
        "summary": {
            "total_queued": total_queued,
            "total_processing": total_processing,
            "total_processed": total_processed,
            "total_rejected": total_rejected
        },
        "per_model": stats
    }
