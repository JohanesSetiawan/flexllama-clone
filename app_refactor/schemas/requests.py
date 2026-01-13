"""
Request and Response Schemas Module

This module defines Pydantic models for API request and response validation.
All schemas follow OpenAI-compatible formats for seamless integration.

Schema Categories:
    - Chat Completion: Request/response for /v1/chat/completions
    - Embeddings: Request/response for /v1/embeddings
    - Model Management: Request/response for model operations
    - Health/Status: Response schemas for monitoring endpoints

Usage:
    from app_refactor.schemas.requests import (
        ChatCompletionRequest,
        EmbeddingRequest,
        ModelEjectRequest
    )
    
    # FastAPI will auto-validate
    @router.post("/v1/chat/completions")
    async def chat(request: ChatCompletionRequest):
        ...
"""

from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Chat Completion Schemas
# =============================================================================

class ChatMessage(BaseModel):
    """A single message in a chat conversation."""
    role: Literal["system", "user", "assistant", "tool"] = Field(
        ...,
        description="Role of the message sender"
    )
    content: Optional[str] = Field(
        None,
        description="Content of the message"
    )
    name: Optional[str] = Field(
        None,
        description="Optional name for the message sender"
    )
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Tool calls made by the assistant"
    )
    tool_call_id: Optional[str] = Field(
        None,
        description="ID of the tool call this message is responding to"
    )


class ChatCompletionRequest(BaseModel):
    """
    Request schema for /v1/chat/completions endpoint.

    Compatible with OpenAI's chat completion API.
    """
    model: str = Field(
        ...,
        description="Model alias to use for completion"
    )
    messages: List[ChatMessage] = Field(
        ...,
        description="List of messages in the conversation"
    )
    temperature: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0-2)"
    )
    top_p: Optional[float] = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        description="Maximum tokens to generate"
    )
    stream: Optional[bool] = Field(
        False,
        description="Whether to stream the response"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        None,
        description="Stop sequences"
    )
    presence_penalty: Optional[float] = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Presence penalty for repetition"
    )
    frequency_penalty: Optional[float] = Field(
        0.0,
        ge=-2.0,
        le=2.0,
        description="Frequency penalty for repetition"
    )
    n: Optional[int] = Field(
        1,
        ge=1,
        le=10,
        description="Number of completions to generate"
    )
    user: Optional[str] = Field(
        None,
        description="Unique identifier for the end-user"
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="List of tools available to the model"
    )
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None,
        description="Control tool usage behavior"
    )

    # Priority for queue management (router-specific)
    priority: Optional[str] = Field(
        "normal",
        description="Request priority: high, normal, low"
    )


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion response."""
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = None


class ChatCompletionUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """Response schema for /v1/chat/completions endpoint."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[ChatCompletionUsage] = None


# =============================================================================
# Embedding Schemas
# =============================================================================

class EmbeddingRequest(BaseModel):
    """
    Request schema for /v1/embeddings endpoint.

    Compatible with OpenAI's embeddings API.
    """
    model: str = Field(
        ...,
        description="Model alias to use for embeddings"
    )
    input: Union[str, List[str]] = Field(
        ...,
        description="Text(s) to generate embeddings for"
    )
    encoding_format: Optional[str] = Field(
        "float",
        description="Encoding format: float or base64"
    )

    @field_validator("input")
    @classmethod
    def validate_input(cls, v):
        """Ensure input is not empty."""
        if isinstance(v, str) and not v.strip():
            raise ValueError("Input text cannot be empty")
        if isinstance(v, list) and (not v or all(not s.strip() for s in v if isinstance(s, str))):
            raise ValueError("Input list cannot be empty")
        return v


class EmbeddingData(BaseModel):
    """A single embedding result."""
    object: str = "embedding"
    index: int
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    """Response schema for /v1/embeddings endpoint."""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: ChatCompletionUsage


# =============================================================================
# Model Management Schemas
# =============================================================================

class ModelEjectRequest(BaseModel):
    """Request to eject a model from memory."""
    model: str = Field(
        ...,
        description="Model alias to eject"
    )


class ModelLoadRequest(BaseModel):
    """Request to load a model into memory."""
    model: str = Field(
        ...,
        description="Model alias to load"
    )


class ModelInfo(BaseModel):
    """Information about a single model."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"
    status: Optional[str] = None
    port: Optional[int] = None
    vram_mb: Optional[float] = None


class ModelsListResponse(BaseModel):
    """Response for /v1/models endpoint."""
    object: str = "list"
    data: List[ModelInfo]


# =============================================================================
# Health and Status Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        ...,
        description="Server status: healthy, degraded, unhealthy"
    )
    uptime_seconds: Optional[float] = None
    active_models: Optional[int] = None
    active_requests: Optional[int] = None


class ModelStatusInfo(BaseModel):
    """Status information for a single model."""
    alias: str
    status: str
    port: Optional[int] = None
    started_at: Optional[str] = None
    last_used_at: Optional[str] = None
    load_progress: Optional[float] = None
    error_message: Optional[str] = None
    vram_used_mb: Optional[float] = None
    updated_at: Optional[str] = None


class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    started_at: Optional[str] = None
    updated_at: Optional[str] = None


class StatusSummary(BaseModel):
    """Summary of model statuses."""
    total: int
    by_status: Dict[str, int]


class FullStatusResponse(BaseModel):
    """Complete status response with all models."""
    server: ServerStatus
    models: Dict[str, ModelStatusInfo]
    summary: StatusSummary


# =============================================================================
# VRAM Schemas
# =============================================================================

class GPUInfo(BaseModel):
    """GPU information."""
    total_mb: float
    total_gb: float
    used_mb: float
    used_gb: float
    free_mb: float
    free_gb: float
    usage_percentage: float
    baseline_used_mb: Optional[float] = None


class ModelVRAMInfo(BaseModel):
    """VRAM information for a single model."""
    model_alias: str
    port: int
    status: str
    vram_used_mb: float
    vram_used_gb: float
    average_usage_mb: float
    load_duration_sec: Optional[float] = None
    vram_percentage: Optional[float] = None


class VRAMReportResponse(BaseModel):
    """VRAM usage report response."""
    gpu_info: GPUInfo
    tracked_models_count: int
    loaded_models_count: int
    total_allocated_by_models_mb: float
    total_allocated_by_models_gb: float
    models: List[ModelVRAMInfo]
    can_load_more: bool
    estimated_free_for_new_model_mb: float
    estimated_free_for_new_model_gb: float
    status: str


# =============================================================================
# Error Schemas
# =============================================================================

class ErrorDetail(BaseModel):
    """Error detail in OpenAI format."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response in OpenAI format."""
    error: ErrorDetail
