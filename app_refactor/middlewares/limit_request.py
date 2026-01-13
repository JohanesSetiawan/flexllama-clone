"""
Request Size Limiting Middleware

This module provides middleware to limit request body size.
Useful for preventing large requests that could cause memory issues
or denial of service attacks.

Features:
    - Limits based on Content-Length header
    - Only applies to POST, PUT, PATCH methods
    - Configurable max size (default 10MB)
    - Returns HTTP 413 if request is too large

Usage:
    from app_refactor.middlewares.limit_request import RequestSizeLimitMiddleware
    
    # In FastAPI app
    app.add_middleware(RequestSizeLimitMiddleware, max_size=10 * 1024 * 1024)

Note:
    Default limit of 10MB is sufficient for most LLM requests. For use cases
    with very long inputs (e.g., document processing), this can be increased.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


# Default maximum request size: 10MB
DEFAULT_MAX_SIZE_BYTES = 10 * 1024 * 1024


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware that limits the maximum size of request bodies.

    This middleware checks the Content-Length header for POST, PUT, and PATCH
    requests and rejects those that exceed the configured maximum size.

    Attributes:
        max_size: Maximum allowed request body size in bytes
    """

    def __init__(self, app: ASGIApp, max_size: int = DEFAULT_MAX_SIZE_BYTES):
        """
        Initialize the middleware.

        Args:
            app: The ASGI application to wrap
            max_size: Maximum request body size in bytes (default: 10MB)
        """
        super().__init__(app)
        self.max_size = max_size

    async def dispatch(self, request: Request, call_next):
        """
        Process the request and check body size.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware or route handler

        Returns:
            HTTP 413 response if request is too large, otherwise the normal response
        """
        # Only check body size for methods that have request bodies
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")

            if content_length:
                try:
                    size = int(content_length)
                    if size > self.max_size:
                        max_size_mb = self.max_size / (1024 * 1024)
                        return JSONResponse(
                            status_code=413,
                            content={
                                "error": {
                                    "message": (
                                        f"Request body too large. "
                                        f"Maximum size is {max_size_mb:.1f}MB"
                                    ),
                                    "type": "request_entity_too_large",
                                    "code": "body_too_large"
                                }
                            }
                        )
                except ValueError:
                    # Invalid content-length header, let the request through
                    pass

        response = await call_next(request)
        return response
