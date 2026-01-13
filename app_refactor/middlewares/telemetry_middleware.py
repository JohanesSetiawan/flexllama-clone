"""
Telemetry Middleware

This middleware collects detailed telemetry and Prometheus metrics for requests.
It integrates with both the TelemetryCollector and PrometheusMetricsCollector.

Features:
    - Request ID generation and tracking
    - Duration measurement
    - Token counting from responses
    - Queue wait time tracking
    - Prometheus metrics recording
    - Skip list for monitoring endpoints

Usage:
    from app_refactor.middlewares.telemetry_middleware import TelemetryMiddleware
    
    app.add_middleware(TelemetryMiddleware)
"""

import time
import uuid
import logging
from typing import Set
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ..lifecycle.dependencies import get_telemetry, get_metrics_service
from ..services.telemetry_service import RequestMetrics


logger = logging.getLogger(__name__)


# Endpoints to skip for telemetry (monitoring endpoints)
SKIP_ENDPOINTS: Set[str] = {
    '/health',
    '/metrics',
    '/metrics/stream',
    '/metrics/report',
    '/v1/telemetry/summary',
    '/vram',
    '/v1/health/models',
    '/status',
    '/status/stream',
    '/favicon.ico',
}


class TelemetryMiddleware(BaseHTTPMiddleware):
    """
    Middleware for detailed request telemetry and Prometheus metrics.

    Attaches to request.state:
    - request_id: Unique request identifier
    - start_time: Request start timestamp
    - tokens_generated: Token count (set by handlers)
    - model_alias: Model being used (set by handlers)
    - queue_time: Time spent in queue (set by handlers)
    """

    async def dispatch(self, request: Request, call_next):
        """Process request with telemetry tracking."""
        # Generate request ID and set start time
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Initialize request state
        request.state.request_id = request_id
        request.state.start_time = start_time
        request.state.tokens_generated = 0
        request.state.model_alias = None
        request.state.queue_time = 0.0
        request.state.processing_time = 0.0

        # Skip monitoring endpoints
        if request.url.path in SKIP_ENDPOINTS:
            return await call_next(request)

        try:
            response = await call_next(request)

            # Record successful request
            await self._record_request(
                request=request,
                request_id=request_id,
                start_time=start_time,
                response=response,
                error=None
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as e:
            # Record failed request
            await self._record_request(
                request=request,
                request_id=request_id,
                start_time=start_time,
                response=None,
                error=str(e)
            )
            raise

    async def _record_request(
        self,
        request: Request,
        request_id: str,
        start_time: float,
        response,
        error: str = None
    ) -> None:
        """Record request metrics to telemetry and Prometheus."""
        end_time = time.time()
        duration = end_time - start_time

        # Get values from request state
        model_alias = getattr(request.state, 'model_alias', None) or "unknown"
        tokens = getattr(request.state, 'tokens_generated', 0)
        queue_time = getattr(request.state, 'queue_time', 0.0)
        processing_time = getattr(request.state, 'processing_time', 0.0)

        # Determine status
        if error:
            status = "error"
            status_code = 500
        elif response:
            status_code = response.status_code
            status = "success" if status_code < 400 else "error"
        else:
            status = "error"
            status_code = 500

        # Record to TelemetryCollector
        telemetry = get_telemetry()
        if telemetry:
            metrics_data = RequestMetrics(
                request_id=request_id,
                model_alias=model_alias,
                endpoint=request.url.path,
                start_time=start_time,
                end_time=end_time,
                status_code=status_code,
                queue_time=queue_time,
                processing_time=processing_time,
                tokens_generated=tokens,
                error=error
            )
            await telemetry.record_request(metrics_data)

        # Record to Prometheus
        metrics_service = get_metrics_service()
        if metrics_service and model_alias != "unknown":
            await metrics_service.record_request_end(
                model=model_alias,
                endpoint=request.url.path,
                duration_seconds=duration,
                status=status,
                tokens=tokens,
                queue_wait_seconds=queue_time,
                status_code=status_code
            )
