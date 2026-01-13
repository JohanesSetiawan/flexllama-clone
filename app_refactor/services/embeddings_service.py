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
from typing import List, Union, Dict, Any

import httpx
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse

from ..core.config import AppConfig
from ..core.manager import ModelManager
from .warmup_service import WarmupService


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
        manager: ModelManager,
        http_client: httpx.AsyncClient,
        config: AppConfig,
        warmup_service: Union[WarmupService, None] = None
    ):
        self.manager = manager
        self.http_client = http_client
        self.config = config
        self.warmup_service = warmup_service

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

            if not model_config.params.embedding:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Model '{model_alias}' does not support embeddings. "
                    f"Set 'embedding: true' in config."
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

            # Generate embeddings
            embeddings, total_tokens = await self._generate_batch(
                runner=runner,
                inputs=inputs
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
        inputs: List[str]
    ) -> tuple[List[Dict[str, Any]], int]:
        """
        Generate embeddings for a batch of inputs.

        Args:
            runner: RunnerProcess instance
            inputs: List of text strings

        Returns:
            Tuple of (embeddings list, total tokens)
        """
        internal_url = f"{runner.url}/embedding"
        all_embeddings = []
        total_tokens = 0

        for idx, text in enumerate(inputs):
            # llama-server expects {"content": "text"}
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

            # llama-server returns list directly or dict with "embedding"
            if isinstance(result, list):
                embedding = result
            elif isinstance(result, dict):
                embedding = result.get("embedding", [])
            else:
                logger.error(
                    f"Unexpected embedding response format: {type(result)}"
                )
                embedding = []

            all_embeddings.append({
                "object": "embedding",
                "embedding": embedding,
                "index": idx
            })

            # Estimate tokens (~1 token per 4 chars)
            total_tokens += len(text) // 4

        return all_embeddings, total_tokens
