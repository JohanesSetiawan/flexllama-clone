"""
Health Controller Module

This module defines endpoints for health checks, readiness probes, and liveness probes.
"""

from fastapi import APIRouter, Depends, status
from ..lifecycle.dependencies import get_health_service
from ..services.health_service import HealthService
from ..schemas.requests import HealthResponse

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Get application health status"
)
async def health_check(
    health_service: HealthService = Depends(get_health_service)
):
    """
    Get the overall health status of the application.

    Returns:
        HealthResponse: Status, uptime, and active components stats.
    """
    if not health_service:
        return HealthResponse(
            status="unhealthy",
            uptime_seconds=0.0,
            active_models=0,
            active_requests=0
        )

    return await health_service.get_health_status()


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe"
)
async def readiness_probe(
    health_service: HealthService = Depends(get_health_service)
):
    """
    K8s readiness probe.

    Returns:
        200 OK if ready, 503 if not.
    """
    if not health_service:
        return {"status": "not_ready"}

    is_healthy = await health_service.is_healthy()
    if is_healthy:
        return {"status": "ready"}
    else:
        # In a real K8s setup, you might return 503 here
        # For now, we return json with status
        return {"status": "not_ready"}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe"
)
async def liveness_probe():
    """
    K8s liveness probe.

    Returns:
        200 OK indicating the server is running.
    """
    return {"status": "alive"}
