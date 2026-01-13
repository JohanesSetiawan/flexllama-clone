"""
OpenAI-Compatible Error Handling Module

This module provides exception classes compatible with OpenAI API error format.
All errors follow the same response structure as OpenAI for seamless integration
with existing clients.

Error Response Format:
    {
        "error": {
            "message": "Human-readable error message",
            "type": "error_type",
            "param": "problematic parameter (optional)",
            "code": "specific error code (optional)"
        }
    }

Available Exception Classes:
    - OpenAIError: Base class for all OpenAI-compatible errors
    - InvalidRequestError (400): Invalid request or parameter
    - AuthenticationError (401): Invalid or missing API key
    - NotFoundError (404): Model or resource not found
    - RateLimitError (429): Rate limit exceeded
    - ServerError (500): Internal server error
    - ServiceUnavailableError (503): Service unavailable (model loading, etc.)
    - InsufficientVRAMError: Not enough VRAM to load model
    - QueueFullError: Request queue is full

Usage:
    from app_refactor.core.errors import NotFoundError, InsufficientVRAMError
    
    # Raise OpenAI-compatible error
    raise NotFoundError(f"Model '{model_alias}' not found")
    
    # Raise VRAM error with details
    raise InsufficientVRAMError(
        model_alias="qwen3-8b",
        required_mb=8000,
        available_mb=4000,
        loaded_models=["gemma3-4b"]
    )
"""

from typing import Optional, List
from fastapi import HTTPException


class OpenAIError(HTTPException):
    """
    Base class for all OpenAI-compatible HTTP errors.

    Follows OpenAI API error response format so clients already integrated
    with OpenAI can handle errors in the same way.

    Attributes:
        status_code: HTTP status code (400, 401, 404, 429, 500, 503)
        message: Human-readable error message
        error_type: Error type per OpenAI format (e.g., "invalid_request_error")
        param: Parameter that caused the error (optional)
        code: Specific error code (optional)
    """

    def __init__(
        self,
        status_code: int,
        message: str,
        error_type: str,
        param: Optional[str] = None,
        code: Optional[str] = None
    ):
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
    """
    400 - Invalid request error.

    Raised when the request is malformed or contains invalid parameters.
    """

    def __init__(self, message: str, param: Optional[str] = None):
        super().__init__(
            status_code=400,
            message=message,
            error_type="invalid_request_error",
            param=param
        )


class AuthenticationError(OpenAIError):
    """
    401 - Authentication error.

    Raised when API key is invalid or missing.
    """

    def __init__(self, message: str = "Invalid API key"):
        super().__init__(
            status_code=401,
            message=message,
            error_type="authentication_error"
        )


class NotFoundError(OpenAIError):
    """
    404 - Resource not found error.

    Raised when the requested model or resource does not exist.
    """

    def __init__(self, message: str, resource: str = "model"):
        super().__init__(
            status_code=404,
            message=message,
            error_type="not_found_error",
            param=resource
        )


class RateLimitError(OpenAIError):
    """
    429 - Rate limit error.

    Raised when the client has exceeded the allowed request rate.
    """

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            status_code=429,
            message=message,
            error_type="rate_limit_error"
        )


class ServerError(OpenAIError):
    """
    500 - Internal server error.

    Raised when an unexpected error occurs on the server.
    """

    def __init__(self, message: str = "Internal server error"):
        super().__init__(
            status_code=500,
            message=message,
            error_type="server_error"
        )


class ServiceUnavailableError(OpenAIError):
    """
    503 - Service unavailable error.

    Raised when the service is temporarily unavailable (e.g., model loading).
    """

    def __init__(self, message: str):
        super().__init__(
            status_code=503,
            message=message,
            error_type="service_unavailable_error"
        )


class InsufficientVRAMError(Exception):
    """
    Raised when there is not enough VRAM to load a model.

    This is a domain-specific error that should be caught and converted
    to an appropriate HTTP response by the controller layer.

    Attributes:
        model_alias: The model that could not be loaded
        required_mb: Estimated VRAM required in MB
        available_mb: Currently available VRAM in MB
        loaded_models: List of currently loaded model aliases
    """

    def __init__(
        self,
        model_alias: str,
        required_mb: float,
        available_mb: float,
        loaded_models: Optional[List[str]] = None
    ):
        self.model_alias = model_alias
        self.required_mb = required_mb
        self.available_mb = available_mb
        self.loaded_models = loaded_models or []

        message = (
            f"Insufficient VRAM to load model '{model_alias}'. "
            f"Required: ~{required_mb:.0f} MB, Available: {available_mb:.0f} MB."
        )
        if self.loaded_models:
            message += f" Currently loaded: {self.loaded_models}"

        super().__init__(message)


class QueueFullError(Exception):
    """
    Raised when the request queue for a model is full.

    This indicates the system is under heavy load and cannot accept
    more requests for the specified model.

    Attributes:
        model_alias: The model whose queue is full
        queue_size: Current queue size
        max_size: Maximum allowed queue size
    """

    def __init__(self, model_alias: str, queue_size: int, max_size: int):
        self.model_alias = model_alias
        self.queue_size = queue_size
        self.max_size = max_size

        message = (
            f"Queue full for model '{model_alias}' "
            f"({queue_size}/{max_size}). Try again later."
        )
        super().__init__(message)


class ModelStartupError(Exception):
    """
    Raised when a model fails to start.

    This can occur due to configuration issues, file not found,
    or llama-server process failure.

    Attributes:
        model_alias: The model that failed to start
        reason: Detailed reason for the failure
        is_retriable: Whether the error might resolve on retry
    """

    def __init__(
        self,
        model_alias: str,
        reason: str,
        is_retriable: bool = True
    ):
        self.model_alias = model_alias
        self.reason = reason
        self.is_retriable = is_retriable

        message = f"Failed to start model '{model_alias}': {reason}"
        super().__init__(message)


class ConfigurationError(Exception):
    """
    Raised when there is a configuration error.

    This typically indicates a problem with config.json that
    prevents the application from starting correctly.
    """

    def __init__(self, message: str):
        super().__init__(f"Configuration error: {message}")
