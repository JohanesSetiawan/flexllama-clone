"""
Inference Controller Module

This module defines the main endpoints for LLM inference (chat completions and embeddings).
"""

from fastapi import APIRouter, Depends, Request, Response, status
from ..lifecycle.dependencies import get_proxy_service, get_embeddings_service
from ..services.proxy_service import ProxyService
from ..services.embeddings_service import EmbeddingsService
from ..schemas.requests import (
    ChatCompletionRequest,
    EmbeddingRequest,
    ChatCompletionResponse,
    EmbeddingResponse
)

router = APIRouter(tags=["Inference"])


@router.post(
    "/chat/completions",
    # Proxy service returns Response object directly (streaming or JSON)
    response_model=None,
    summary="Create chat completion"
)
async def chat_completions(
    request: Request,
    proxy_service: ProxyService = Depends(get_proxy_service)
):
    """
    Generate a chat completion response.

    This endpoint proxies the request to the appropriate model runner.
    """
    if not proxy_service:
        return Response(
            content="Proxy service unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    # We pass the raw request to proxy service to preserve headers/body stream if needed
    # But proxy service implementation takes 'request: Request'.
    return await proxy_service.proxy_chat_completion(request)


@router.post(
    "/embeddings",
    response_model=EmbeddingResponse,
    summary="Create embeddings"
)
async def create_embeddings(
    request: Request,
    embeddings_service: EmbeddingsService = Depends(get_embeddings_service)
):
    """
    Generate embeddings for input text.
    """
    if not embeddings_service:
        return Response(
            content="Embeddings service unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE
        )

    return await embeddings_service.generate_embeddings(request)
