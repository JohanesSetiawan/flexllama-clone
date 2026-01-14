"""
Authentication Middleware

Validates API Key via Authorization Bearer header.
Rejects unauthorized requests with 401 status.
"""

import os
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


logger = logging.getLogger(__name__)


# Paths that bypass authentication
PUBLIC_PATHS = frozenset({
    "/docs",
    "/redoc",
    "/openapi.json",
})


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Bearer token authentication middleware.

    Validates Authorization header against API_KEY from environment.
    Public paths (docs, health) bypass authentication.
    """

    def __init__(self, app, api_key: str | None = None):
        """
        Initialize auth middleware.

        Args:
            app: ASGI application
            api_key: API key to validate against (defaults to API_KEY env var)
        """
        super().__init__(app)
        self.api_key = api_key or os.getenv("API_KEY", "")

        if not self.api_key:
            logger.warning("API_KEY not set. All requests will be rejected.")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> JSONResponse:
        """Process request and validate authentication."""

        # Skip auth for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Validate Bearer token
        if not self._validate_token(request):
            return self._unauthorized_response()

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        return path in PUBLIC_PATHS

    def _validate_token(self, request: Request) -> bool:
        """
        Validate Authorization Bearer token.

        Returns:
            True if token is valid, False otherwise
        """
        if not self.api_key:
            return False

        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return False

        token = auth_header[7:]  # Strip "Bearer " prefix
        return token == self.api_key

    def _unauthorized_response(self) -> JSONResponse:
        """Return 401 Unauthorized response."""
        return JSONResponse(
            status_code=401,
            content={
                "error": "unauthorized",
                "message": "Invalid or missing API key"
            }
        )
