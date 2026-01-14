"""
Rate Limiting Middleware

Provides configurable rate limiting using SlowAPI with Redis backend.
Only active when rate_limit config is present.
"""

import logging
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..core.config import RateLimitConfig


logger = logging.getLogger(__name__)


def create_limiter(config: Optional[RateLimitConfig]) -> Optional[Limiter]:
    """
    Create rate limiter instance if config is present.

    Args:
        config: Rate limit configuration (None = disabled)

    Returns:
        Limiter instance or None if disabled
    """
    if config is None:
        logger.info("Rate limiting disabled (no config)")
        return None

    limiter = Limiter(
        key_func=get_remote_address,
        storage_uri=config.redis_url,
        default_limits=[f"{config.requests_per_minute}/minute"]
    )

    logger.info(
        f"Rate limiting enabled: {config.requests_per_minute}/min "
        f"(Redis: {config.redis_url})"
    )

    return limiter


def rate_limit_exceeded_handler(
    request: Request,
    exc: RateLimitExceeded
) -> JSONResponse:
    """
    Handle rate limit exceeded exception.

    Returns:
        JSON response with 429 status and retry info
    """
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": str(exc.detail)
        }
    )


def setup_rate_limiting(app, config: Optional[RateLimitConfig]) -> None:
    """
    Configure rate limiting on FastAPI app.

    Args:
        app: FastAPI application instance
        config: Rate limit configuration (None = skip setup)
    """
    if config is None:
        return

    limiter = create_limiter(config)
    if limiter is None:
        return

    # Store limiter in app state for route decorators
    app.state.limiter = limiter

    # Add middleware
    app.add_middleware(SlowAPIMiddleware)

    # Add exception handler
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
