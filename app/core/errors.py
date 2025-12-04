from typing import Optional
from fastapi import HTTPException


class OpenAIError(HTTPException):
    """Base class untuk OpenAI-compatible errors."""

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        param: Optional[str] = None,
        code: Optional[str] = None
    ):
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code

        detail = {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code
            }
        }

        super().__init__(status_code=status_code, detail=detail)


class InvalidRequestError(OpenAIError):
    """400 - Invalid request."""

    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(
            status_code=400,
            message=message,
            error_type="invalid_request_error",
            param=param
        )


class AuthenticationError(OpenAIError):
    """401 - Invalid authentication."""

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(
            status_code=401,
            message=message,
            error_type="authentication_error"
        )


class NotFoundError(OpenAIError):
    """404 - Resource not found."""

    def __init__(self, message: str, resource: str = "model"):
        super().__init__(
            status_code=404,
            message=message,
            error_type="not_found_error",
            param=resource
        )


class RateLimitError(OpenAIError):
    """429 - Rate limit exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            message=message,
            error_type="rate_limit_error"
        )


class ServerError(OpenAIError):
    """500 - Internal server error."""

    def __init__(self, message: str = "Internal server error"):
        super().__init__(
            status_code=500,
            message=message,
            error_type="server_error"
        )


class ServiceUnavailableError(OpenAIError):
    """503 - Service unavailable."""

    def __init__(self, message: str):
        super().__init__(
            status_code=503,
            message=message,
            error_type="service_unavailable_error"
        )


class InsufficientVRAMError(Exception):
    """Raised when there's not enough VRAM to load a model."""

    def __init__(
        self,
        model_alias: str,
        required_mb: float,
        available_mb: float,
        loaded_models: list = None
    ):
        self.model_alias = model_alias
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.loaded_models = loaded_models or []

        message = (
            f"Insufficient VRAM to load model '{model_alias}'. "
            f"Required: ~{required_mb:.0f} MB, Available: {available_mb:.0f} MB."
        )
        if loaded_models:
            message += f" Currently loaded: {loaded_models}"

        super().__init__(message)


class QueueFullError(Exception):
    """Raised when the request queue is full."""

    def __init__(self, model_alias: str, queue_size: int, max_size: int):
        self.model_alias = model_alias
        self.queue_size = queue_size
        self.max_size = max_size

        message = f"Queue full for model '{model_alias}' ({queue_size}/{max_size})"
        super().__init__(message)
