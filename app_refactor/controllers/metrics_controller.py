"""
Metrics Controller Module

This module defines endpoints for Prometheus metrics and real-time metrics streaming.
"""

import time
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request, Depends, Response
from fastapi.responses import JSONResponse, StreamingResponse

from ..lifecycle.dependencies import (
    get_metrics_service,
    get_manager,
    get_queue_manager,
    get_shutdown_event
)
from ..services.metrics_service import MetricsService


router = APIRouter(tags=["Metrics"])


@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint"
)
async def get_prometheus_metrics(
    metrics_service: MetricsService = Depends(get_metrics_service)
):
    """
    Prometheus exposition format metrics endpoint.

    This endpoint is designed to be scraped by Prometheus server.
    Returns all metrics in Prometheus text format.

    Usage:
        - Configure Prometheus to scrape this endpoint
        - Default scrape interval: 15s
    """
    if not metrics_service:
        return Response(
            content="# Prometheus collector not initialized\n",
            media_type="text/plain",
            status_code=503
        )

    content = metrics_service.get_prometheus_metrics()
    return Response(
        content=content,
        media_type=metrics_service.get_content_type()
    )


@router.get(
    "/metrics/stream",
    summary="Real-time metrics streaming via SSE"
)
async def stream_metrics(
    request: Request,
    metrics_service: MetricsService = Depends(get_metrics_service),
    manager=Depends(get_manager),
    queue_manager=Depends(get_queue_manager),
    shutdown_event=Depends(get_shutdown_event)
):
    """
    Server-Sent Events (SSE) endpoint for real-time metrics streaming.

    Streams metrics updates every 2 seconds with per-model breakdown.

    Event types:
        - metrics: Real-time metrics snapshot
        - heartbeat: Keep-alive every 30 seconds

    Usage:
        curl -N http://localhost:8000/metrics/stream
    """
    if not metrics_service:
        return JSONResponse(
            {"error": "Prometheus collector not initialized"},
            status_code=503
        )

    async def event_generator():
        heartbeat_interval = 30
        metrics_interval = 2
        last_heartbeat = time.time()

        try:
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    break

                # Check shutdown
                if shutdown_event and shutdown_event.is_set():
                    yield f'event: shutdown\ndata: {{"message": "Server shutting down"}}\n\n'
                    break

                # Send metrics
                try:
                    snapshot = await metrics_service.get_realtime_snapshot(
                        manager=manager,
                        queue_manager=queue_manager
                    )
                    yield f"event: metrics\ndata: {json.dumps(snapshot)}\n\n"
                except Exception as e:
                    error_msg = str(e)
                    yield f'event: error\ndata: {{"error": "{error_msg}"}}\n\n'

                # Send heartbeat if needed
                current_time = time.time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    yield f'event: heartbeat\ndata: {{"timestamp": "{datetime.now().isoformat()}"}}\n\n'
                    last_heartbeat = current_time

                await asyncio.sleep(metrics_interval)

        except asyncio.CancelledError:
            pass
        except Exception:
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


@router.get(
    "/metrics/report",
    summary="Get 5-minute aggregated metrics report"
)
async def get_metrics_report(
    metrics_service: MetricsService = Depends(get_metrics_service),
    manager=Depends(get_manager),
    queue_manager=Depends(get_queue_manager)
):
    """
    Get detailed 5-minute aggregated metrics report.

    Returns comprehensive metrics including:
        - Server status and uptime
        - GPU VRAM usage (MB and GB)
        - Per-model detailed breakdown:
            - Request counts (total, success, errors)
            - Latency statistics (avg, min, max, p95)
            - Queue statistics
            - Token throughput
        - Aggregated totals

    Usage:
        curl http://localhost:8000/metrics/report
    """
    if not metrics_service:
        return JSONResponse(
            {"error": "Prometheus collector not initialized"},
            status_code=503
        )

    try:
        report = await metrics_service.get_5min_report(
            manager=manager,
            queue_manager=queue_manager
        )
        return report
    except Exception as e:
        return JSONResponse(
            {"error": f"Failed to generate report: {str(e)}"},
            status_code=500
        )
