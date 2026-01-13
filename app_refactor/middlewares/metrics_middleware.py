"""
Metrics Middleware

This middleware collects basic request metrics (legacy format).
For Prometheus metrics, use the TelemetryMiddleware instead.

Features:
    - Counts total requests per endpoint
    - Counts success/failure per endpoint
    - Records request duration

Usage:
    from app_refactor.middlewares.metrics_middleware import MetricsMiddleware
    
    app.add_middleware(MetricsMiddleware)
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from ..utils.legacy_metrics import metrics


logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting basic request metrics.

    Tracks:
    - Total requests per endpoint
    - Success/failure counts
    - Request duration
    """

    async def dispatch(self, request: Request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()
        endpoint = request.url.path

        try:
            response = await call_next(request)

            # Record metrics
            metrics["requests_total"][endpoint] += 1

            if response.status_code < 400:
                metrics["requests_success"][endpoint] += 1
            else:
                metrics["requests_failed"][endpoint] += 1

            # Record duration
            duration = time.time() - start_time
            self._record_duration(endpoint, duration)

            return response

        except Exception as e:
            # Record failure
            metrics["requests_total"][endpoint] += 1
            metrics["requests_failed"][endpoint] += 1
            raise

    def _record_duration(self, endpoint: str, duration: float) -> None:
        """Record request duration with pruning."""
        durations = metrics["request_duration_seconds"][endpoint]
        durations.append(duration)

        # Keep only last 1000 entries
        max_entries = 1000
        if len(durations) > max_entries:
            metrics["request_duration_seconds"][endpoint] = durations[-max_entries:]
