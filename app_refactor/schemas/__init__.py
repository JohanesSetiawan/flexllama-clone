"""
Schemas Package

This package provides Pydantic models for request and response validation.
All schemas are compatible with OpenAI API format.
"""

from .requests import (
    # Chat Completion
    ChatMessage,
    ChatCompletionRequest,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatCompletionResponse,

    # Embeddings
    EmbeddingRequest,
    EmbeddingData,
    EmbeddingResponse,

    # Model Management
    ModelEjectRequest,
    ModelLoadRequest,
    ModelInfo,
    ModelsListResponse,

    # Health and Status
    HealthResponse,
    ModelStatusInfo,
    ServerStatus,
    StatusSummary,
    FullStatusResponse,

    # VRAM
    GPUInfo,
    ModelVRAMInfo,
    VRAMReportResponse,

    # Error
    ErrorDetail,
    ErrorResponse,
)

__all__ = [
    # Chat Completion
    "ChatMessage",
    "ChatCompletionRequest",
    "ChatCompletionChoice",
    "ChatCompletionUsage",
    "ChatCompletionResponse",

    # Embeddings
    "EmbeddingRequest",
    "EmbeddingData",
    "EmbeddingResponse",

    # Model Management
    "ModelEjectRequest",
    "ModelLoadRequest",
    "ModelInfo",
    "ModelsListResponse",

    # Health and Status
    "HealthResponse",
    "ModelStatusInfo",
    "ServerStatus",
    "StatusSummary",
    "FullStatusResponse",

    # VRAM
    "GPUInfo",
    "ModelVRAMInfo",
    "VRAMReportResponse",

    # Error
    "ErrorDetail",
    "ErrorResponse",
]
