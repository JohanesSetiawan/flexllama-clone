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
    Non-existent routes bypass auth to return proper 404.
    """

    def __init__(self, app, api_key: str | None = None):
        """
        Initialize auth middleware.

        Args:
            app: ASGI application
            api_key: API key to validate against (defaults to API_KEY env var)
        """
        super().__init__(app)
        self.app = app
        self.api_key = api_key or os.getenv("API_KEY", "")
        self.auth_enabled = bool(self.api_key)

        if not self.auth_enabled:
            logger.info("API_KEY not set. Authentication DISABLED.")
        else:
            logger.info("API_KEY configured. Authentication ENABLED.")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> JSONResponse:
        """Process request and validate authentication."""

        # Skip auth entirely if not configured
        if not self.auth_enabled:
            return await call_next(request)

        # Skip auth for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Skip auth for non-existent routes (let them return 404)
        if not self._route_exists(request):
            return await call_next(request)

        # Validate Bearer token
        if not self._validate_token(request):
            return self._unauthorized_response()

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no auth required)."""
        return path in PUBLIC_PATHS

    def _route_exists(self, request: Request) -> bool:
        """Check if the requested route exists in the app."""
        try:
            # Get the router from app
            app = request.app
            for route in app.routes:
                match, _ = route.matches(request.scope)
                if match.value > 0:  # Match.FULL or Match.PARTIAL
                    return True
            return False
        except Exception:
            # If check fails, assume route exists and require auth
            return True

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
