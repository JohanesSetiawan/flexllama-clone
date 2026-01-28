"""
Embeddings Service Module

This module provides the embeddings service for generating text embeddings
using llama-server embedding endpoints.

Features:
    - OpenAI-compatible embeddings API format
    - Batch processing for multiple inputs
    - Token usage estimation

Usage:
    from app_refactor.services.embeddings_service import EmbeddingsService
    
    service = EmbeddingsService(manager, http_client, config, warmup_manager)
    response = await service.generate_embeddings(request)
"""

import time
import logging
from typing import List, Union, Dict, Any, TYPE_CHECKING

import httpx
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from ..core.config import AppConfig
from .warmup_service import WarmupService
from .cache_service import CacheService, get_cache_service

if TYPE_CHECKING:
    from ..core.manager import ModelManager


logger = logging.getLogger(__name__)


class EmbeddingsService:
    """
    Service for generating text embeddings.

    Handles:
    - Model validation (embedding support)
    - Batch embedding generation
    - OpenAI-compatible response format

    Attributes:
        manager: ModelManager instance
        http_client: HTTPX async client
        config: Application configuration
        warmup_manager: Optional warmup manager
    """

    def __init__(
        self,
        manager: "ModelManager",
        http_client: httpx.AsyncClient,
        config: AppConfig,
        warmup_service: Union[WarmupService, None] = None,
        cache_service: Union[CacheService, None] = None
    ):
        self.manager = manager
        self.http_client = http_client
        self.config = config
        self.warmup_service = warmup_service
        self.cache_service = cache_service or get_cache_service()

    async def generate_embeddings(
        self,
        request: Request
    ) -> JSONResponse:
        """
        Generate embeddings for input text(s).

        Args:
            request: FastAPI request with embedding body

        Returns:
            JSONResponse with OpenAI-compatible embeddings
        """
        try:
            # Parse request
            body = await request.json()
            model_alias = body.get("model")

            if not model_alias:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Field 'model' is required in request body"
                )

            # Validate model alias format
            if not self._validate_alias(model_alias):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Model alias must contain only alphanumeric, dash, or underscore"
                )

            # Set model alias for telemetry
            request.state.model_alias = model_alias

            # Verify model supports embeddings
            model_config = self.config.models.get(model_alias)
            if not model_config:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model '{model_alias}' not found in config"
                )

            if not model_config.is_embedding_mode():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{model_alias}' does not support embeddings. "
                    f"Add '--embedding' to flags in config."
                )

            # Get input text(s)
            input_data = body.get("input")
            if not input_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Field 'input' is required for embeddings"
                )

            # Normalize input to list
            inputs = self._normalize_input(input_data)

            # Record for warmup
            if self.warmup_service:
                self.warmup_service.record_request(model_alias)

            # Get runner
            queue_start = time.time()
            runner = await self.manager.get_runner_for_request(model_alias)

            # Generate embeddings (with caching)
            embeddings, total_tokens = await self._generate_batch(
                runner=runner,
                inputs=inputs,
                model_alias=model_alias
            )

            # Record timing
            request.state.queue_time = time.time() - queue_start
            request.state.tokens_generated = 0  # Embeddings don't generate tokens

            # Return OpenAI-compatible format
            return JSONResponse(content={
                "object": "list",
                "data": embeddings,
                "model": model_alias,
                "usage": {
                    "prompt_tokens": total_tokens,
                    "total_tokens": total_tokens
                }
            })

        except HTTPException:
            raise
        except LookupError as e:
            logger.error(f"Model not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except httpx.ConnectError as e:
            logger.warning(f"Connection error for {model_alias}: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Runner for '{model_alias}' not available"
            )
        except httpx.TimeoutException as e:
            logger.error(f"Timeout for {model_alias}: {e}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timeout for model '{model_alias}'"
            )
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(e)
            )
        except Exception as e:
            logger.exception("Unexpected error in generate_embeddings")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error occurred"
            )

    def _validate_alias(self, alias: str) -> bool:
        """Validate model alias format."""
        return all(c.isalnum() or c in ('-', '_') for c in alias)

    def _normalize_input(
        self,
        input_data: Union[str, List[str]]
    ) -> List[str]:
        """Normalize input to list of strings."""
        if isinstance(input_data, str):
            return [input_data]
        elif isinstance(input_data, list):
            return input_data
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Field 'input' must be a string or list of strings"
            )

    async def _generate_batch(
        self,
        runner,
        inputs: List[str],
        model_alias: str = ""
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Generate embeddings for a batch of inputs with caching.

        Args:
            runner: RunnerProcess instance
            inputs: List of text strings
            model_alias: Model identifier for cache key

        Returns:
            Tuple of (embeddings list, total tokens)
        """
        internal_url = f"{runner.url}/embedding"
        all_embeddings = []
        total_tokens = 0

        for idx, text in enumerate(inputs):
            # Check cache first
            cache_key_body = {"input": text, "model": model_alias}
            cached = None
            if self.cache_service:
                cached = await self.cache_service.get(model_alias, cache_key_body)

            if cached and "embedding" in cached:
                # Cache HIT
                logger.debug(f"[Embedding] Cache HIT for text index {idx}")
                all_embeddings.append({
                    "object": "embedding",
                    "embedding": cached["embedding"],
                    "index": idx
                })
                total_tokens += len(text) // 4
                continue

            # Cache MISS - compute embedding
            embed_body = {"content": text}
            req = self.http_client.build_request(
                method="POST",
                url=internal_url,
                json=embed_body,
                headers={"Content-Type": "application/json"}
            )

            response = await self.http_client.send(req)

            if response.status_code != 200:
                error_detail = response.text
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Embedding request failed: {error_detail}"
                )

            result = response.json()

            # Parse response format
            if isinstance(result, list):
                embedding = result
            elif isinstance(result, dict):
                embedding = result.get("embedding", [])
            else:
                logger.error(f"Unexpected embedding response: {type(result)}")
                embedding = []

            # Store in cache
            if self.cache_service and embedding:
                await self.cache_service.set(
                    model_alias,
                    cache_key_body,
                    {"embedding": embedding}
                )

            all_embeddings.append({
                "object": "embedding",
                "embedding": embedding,
                "index": idx
            })

            total_tokens += len(text) // 4

        return all_embeddings, total_tokens
