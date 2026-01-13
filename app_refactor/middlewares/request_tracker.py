"""
Request Tracker Middleware

This middleware tracks active requests and rejects new requests during shutdown.

Features:
    - Counts active requests
    - Rejects requests during shutdown
    - Thread-safe counter using async lock

Usage:
    from app_refactor.middlewares.request_tracker import RequestTrackerMiddleware
    
    app.add_middleware(RequestTrackerMiddleware)
"""

import asyncio
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastapi import status

from ..lifecycle.dependencies import (
    get_container,
    is_shutting_down,
    increment_active_requests,
    decrement_active_requests
)


logger = logging.getLogger(__name__)


class RequestTrackerMiddleware(BaseHTTPMiddleware):
    """
    Middleware to track active requests and handle shutdown gracefully.

    During shutdown:
    - Rejects new incoming requests with 503
    - Existing requests are allowed to complete

    Attributes:
        _lock: Async lock for thread-safe counter updates
    """

    def __init__(self, app):
        super().__init__(app)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        """Process request with tracking."""
        # Reject requests during shutdown
        if is_shutting_down():
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"detail": "Server is shutting down"}
            )

        # Increment active requests
        async with self._lock:
            increment_active_requests()

        try:
            response = await call_next(request)
            return response
        finally:
            # Decrement active requests
            async with self._lock:
                decrement_active_requests()
