"""
Status Controller Module

This module defines endpoints for detailed system and model status monitoring.
"""

from typing import Dict
from fastapi import APIRouter, Depends, HTTPException, status, Path, Request
from ..lifecycle.dependencies import get_status_tracker, get_manager
from ..core.model_status import ModelStatusTracker
from ..core.manager import ModelManager
from ..schemas.requests import FullStatusResponse, ModelStatusInfo

router = APIRouter(tags=["Status"])


@router.get(
    "/status",
    response_model=FullStatusResponse,
    summary="Get full system status"
)
async def get_full_status(
    status_tracker: ModelStatusTracker = Depends(get_status_tracker)
):
    """
    Get detailed status of the server and all models.

    Returns:
        FullStatusResponse: Comprehensive status report.
    """
    if not status_tracker:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Status tracker not available"
        )

    return await status_tracker.get_full_status()


@router.get(
    "/status/{model_alias}",
    response_model=ModelStatusInfo,
    summary="Get specific model status"
)
async def get_model_status(
    model_alias: str = Path(..., description="Model alias"),
    manager: ModelManager = Depends(get_manager)
):
    """
    Get status for a specific model.

    Args:
        model_alias: Name of the model to check.

    Returns:
        ModelStatusInfo: Detailed status of the model.
    """
    if not manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model manager not available"
        )

    model_status = await manager.get_model_status(model_alias)
    if not model_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_alias}' not found or not managed"
        )

    # Map dictionary to Pydantic model
    return ModelStatusInfo(**model_status)


@router.get(
    "/v1/models/status/stream",
    summary="Stream model status updates via SSE"
)
async def stream_model_status(
    request: Request,
    status_tracker: ModelStatusTracker = Depends(get_status_tracker)
):
    """
    Server-Sent Events endpoint for real-time model status updates.

    Streams model status changes every 2 seconds.

    Usage:
        curl -N http://localhost:8000/v1/models/status/stream
    """
    import json
    import asyncio
    from fastapi.responses import StreamingResponse
    from ..lifecycle.dependencies import get_shutdown_event

    shutdown_event = get_shutdown_event()

    if not status_tracker:
        return {"error": "Status tracker not available"}

    async def event_generator():
        try:
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    break

                # Check shutdown
                if shutdown_event and shutdown_event.is_set():
                    yield 'event: shutdown\ndata: {"message": "Server shutting down"}\n\n'
                    break

                # Get current status
                try:
                    full_status = await status_tracker.get_full_status()
                    yield f"event: status\ndata: {json.dumps(full_status.dict() if hasattr(full_status, 'dict') else full_status)}\n\n"
                except Exception as e:
                    yield f'event: error\ndata: {{"error": "{str(e)}"}}\n\n'

                await asyncio.sleep(2)

        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
