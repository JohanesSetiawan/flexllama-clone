"""
VRAM Controller Module

This module defines endpoints for VRAM usage monitoring and reporting.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from ..lifecycle.dependencies import get_vram_service
from ..services.vram_service import VRAMService
from ..schemas.requests import VRAMReportResponse

router = APIRouter(tags=["VRAM"])


@router.get(
    "/vram",
    response_model=VRAMReportResponse,
    summary="Get VRAM usage report"
)
async def get_vram_report(
    vram_service: VRAMService = Depends(get_vram_service)
):
    """
    Get detailed VRAM usage report for the GPU and models.

    Returns:
        VRAMReportResponse: Detailed VRAM breakdown.
    """
    if not vram_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VRAM service not available"
        )

    return await vram_service.get_vram_report()
