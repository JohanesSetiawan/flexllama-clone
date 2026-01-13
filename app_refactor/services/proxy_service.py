"""
Proxy Service Module

This module provides the core request proxying logic for forwarding
requests to llama-server instances via the queue system.

Components:
    - ProxyService: Main service for request proxying
    - Queue processing logic
    - Response handling (streaming and non-streaming)

Usage:
    from app_refactor.services.proxy_service import ProxyService
    
    service = ProxyService(manager, queue_manager, http_client, config)
    response = await service.proxy_chat_completion(request)
"""

import json
import time
import heapq
import asyncio
import logging
from typing import Dict, Any, Optional, List, AsyncGenerator

import httpx
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from ..core.config import AppConfig
from ..core.manager import ModelManager
from ..core.queue import QueueManager, QueuedRequest, RequestPriority, ModelRequestQueue
from ..core.errors import InsufficientVRAMError
from ..services.metrics_service import get_metrics_service, MetricsService
from ..services.warmup_service import WarmupService


logger = logging.getLogger(__name__)


class ProxyService:
    """
    Service for proxying requests to llama-server instances.

    Handles:
    - Request queuing with priority
    - Queue processing with retry logic
    - Streaming and non-streaming responses
    - Error handling and VRAM errors

    Attributes:
        manager: ModelManager instance
        queue_manager: QueueManager instance
        http_client: HTTPX async client
        config: Application configuration
        warmup_service: Optional warmup service for tracking
    """

    def __init__(
        self,
        manager: ModelManager,
        queue_manager: QueueManager,
        http_client: httpx.AsyncClient,
        config: AppConfig,
        warmup_service: Optional[WarmupService] = None
    ):
        self.manager = manager
        self.queue_manager = queue_manager
        self.http_client = http_client
        self.config = config
        self.warmup_service = warmup_service

    async def proxy_chat_completion(
        self,
        request: Request,
        endpoint: str = "/v1/chat/completions"
    ) -> JSONResponse | StreamingResponse:
        """
        Proxy a chat completion request through the queue system.

        Args:
            request: FastAPI request
            endpoint: Backend endpoint path

        Returns:
            JSONResponse or StreamingResponse
        """
        try:
            # Parse request body
            body = await request.json()
            model_alias = body.get("model")

            if not model_alias:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Field 'model' is required in request body"
                )

            # Set model alias for telemetry
            request.state.model_alias = model_alias

            # Determine streaming mode
            is_streaming = body.get("stream", False)

            # Get priority from header
            priority = self._get_priority(request)

            # Record request for warmup tracking
            if self.warmup_service:
                self.warmup_service.record_request(model_alias)

            # Get queue for model
            queue = await self.queue_manager.get_queue(model_alias)

            # Get request ID from middleware
            request_id = getattr(request.state, 'request_id', 'unknown')

            logger.info(
                f"[Proxy] Enqueuing request {request_id} for {model_alias} "
                f"(priority: {priority.name}, streaming: {is_streaming})"
            )

            # Process through queue
            queue_start_time = time.time()

            try:
                queue_timeout = self.config.system.queue_timeout_sec * 3

                result = await self._process_request_via_queue(
                    queue=queue,
                    request_id=request_id,
                    model_alias=model_alias,
                    body=body,
                    priority=priority,
                    endpoint=endpoint,
                    timeout=queue_timeout
                )

                # Record queue time
                request.state.queue_time = time.time() - queue_start_time

                # Handle response
                return self._create_response(result, request)

            except TimeoutError as e:
                return self._handle_timeout_error(e, request_id, queue_start_time)
            except InsufficientVRAMError as e:
                return self._handle_vram_error(e, model_alias)
            except RuntimeError as e:
                return self._handle_runtime_error(e, model_alias)

        except HTTPException:
            raise
        except LookupError as e:
            logger.error(f"Model not found: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(e)
            )
        except Exception as e:
            logger.exception("Unexpected error in proxy_chat_completion")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error occurred"
            )

    def _get_priority(self, request: Request) -> RequestPriority:
        """Get request priority from header."""
        priority_header = request.headers.get(
            "X-Request-Priority", "normal"
        ).lower()

        priority_map = {
            "high": RequestPriority.HIGH,
            "normal": RequestPriority.NORMAL,
            "low": RequestPriority.LOW
        }
        return priority_map.get(priority_header, RequestPriority.NORMAL)

    def _create_response(
        self,
        result: Dict[str, Any],
        request: Request
    ) -> JSONResponse | StreamingResponse:
        """Create appropriate response from queue result."""
        if result["type"] == "stream":
            return self._create_streaming_response(result)
        else:
            request.state.tokens_generated = result.get("tokens", 0)
            return JSONResponse(
                content=result["data"],
                status_code=result["status_code"]
            )

    def _create_streaming_response(
        self,
        result: Dict[str, Any]
    ) -> StreamingResponse:
        """Create streaming response from chunks."""
        async def stream_generator():
            try:
                for chunk in result["chunks"]:
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream generator: {e}")
                error_chunk = 'data: {"error": "Stream error"}\n\n'
                yield error_chunk.encode()

        return StreamingResponse(
            stream_generator(),
            status_code=result["status_code"],
            media_type="text/event-stream"
        )

    def _handle_timeout_error(
        self,
        error: TimeoutError,
        request_id: str,
        queue_start_time: float
    ):
        """Handle queue timeout error."""
        wait_time = time.time() - queue_start_time
        logger.error(
            f"[Proxy] Request {request_id} timeout after {wait_time:.1f}s: {error}"
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Request timeout in queue after {wait_time:.1f}s"
        )

    def _handle_vram_error(
        self,
        error: InsufficientVRAMError,
        model_alias: str
    ):
        """Handle insufficient VRAM error."""
        logger.warning(
            f"[Proxy] Insufficient VRAM for {model_alias}: "
            f"need {error.required_mb:.0f} MB, have {error.available_mb:.0f} MB"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": str(error),
                    "type": "insufficient_vram_error",
                    "code": "vram_exhausted",
                    "model": error.model_alias,
                    "required_mb": round(error.required_mb),
                    "available_mb": round(error.available_mb),
                    "loaded_models": error.loaded_models
                }
            }
        )

    def _handle_runtime_error(self, error: RuntimeError, model_alias: str):
        """Handle runtime error (queue full, etc)."""
        error_msg = str(error)
        if "queue" in error_msg.lower():
            logger.warning(f"[Proxy] Queue full for {model_alias}: {error}")
        else:
            logger.warning(f"[Proxy] Runtime error for {model_alias}: {error}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error)
        )

    async def _process_request_via_queue(
        self,
        queue: ModelRequestQueue,
        request_id: str,
        model_alias: str,
        body: Dict[str, Any],
        priority: RequestPriority,
        endpoint: str,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Process request via queue system.

        Steps:
        1. Enqueue request
        2. Wait for queue processor
        3. Return result
        """
        if timeout is None:
            timeout = self.config.system.queue_timeout_sec * 3

        # Create future for response
        response_future: asyncio.Future = asyncio.Future()

        # Create queued request
        queued_req = QueuedRequest(
            priority=priority.value,
            timestamp=time.time(),
            request_id=request_id,
            model_alias=model_alias,
            body=body,
            response_future=response_future
        )

        # Enqueue
        async with queue.lock:
            if len(queue.queue) >= queue.max_queue_size:
                queue.total_rejected += 1
                prom_collector = get_metrics_service()
                if prom_collector:
                    prom_collector.record_queue_rejected(model_alias)
                raise RuntimeError(
                    f"Queue for model '{model_alias}' is full ({queue.max_queue_size})"
                )

            heapq.heappush(queue.queue, queued_req)
            queue.total_requests += 1

            # Update Prometheus
            prom_collector = get_metrics_service()
            if prom_collector:
                prom_collector.update_queue_depth(
                    model_alias, len(queue.queue))

            queue.queue_not_empty.set()
            should_start_processor = not queue.processing

        # Start processor if needed
        if should_start_processor:
            logger.info(f"[Queue] Starting processor for {model_alias}")
            asyncio.create_task(
                self._queue_processor(model_alias, queue, endpoint)
            )

        # Wait for result
        try:
            result = await asyncio.wait_for(response_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            # Remove from queue on timeout
            async with queue.lock:
                try:
                    queue.queue.remove(queued_req)
                except ValueError:
                    pass
            raise TimeoutError(f"Request timeout after {timeout}s in queue")

    async def _queue_processor(
        self,
        model_alias: str,
        queue: ModelRequestQueue,
        endpoint: str
    ) -> None:
        """Background task to process queue for a model."""
        async with queue.lock:
            if queue.processing:
                return
            queue.processing = True

        logger.info(f"[Queue] Starting processor for model '{model_alias}'")
        idle_timeout = self.config.system.queue_processor_idle_sec

        try:
            while True:
                queued_req = await queue.dequeue()

                if queued_req is None:
                    queue.queue_not_empty.clear()
                    try:
                        await asyncio.wait_for(
                            queue.queue_not_empty.wait(),
                            timeout=idle_timeout
                        )
                        continue
                    except asyncio.TimeoutError:
                        # Check if warm model
                        if self.warmup_manager and self.warmup_manager.is_model_warm(model_alias):
                            continue
                        else:
                            logger.info(
                                f"[Queue] Processor idle for {model_alias}, stopping"
                            )
                            break

                # Process request
                async with queue.lock:
                    queue.current_processing += 1

                try:
                    await self._process_queued_request(queued_req, model_alias, endpoint, queue)
                except Exception as e:
                    logger.exception(
                        f"[Queue] Error processing {queued_req.request_id}: {e}"
                    )
                    if not queued_req.response_future.done():
                        queued_req.response_future.set_exception(e)
                finally:
                    async with queue.lock:
                        queue.current_processing -= 1

        except asyncio.CancelledError:
            logger.info(f"[Queue] Processor cancelled for {model_alias}")
            raise
        finally:
            async with queue.lock:
                queue.processing = False
            logger.info(f"[Queue] Processor stopped for '{model_alias}'")

    async def _process_queued_request(
        self,
        queued_req: QueuedRequest,
        model_alias: str,
        endpoint: str,
        queue: ModelRequestQueue
    ) -> None:
        """Process a single queued request."""
        process_start = time.time()

        logger.info(
            f"[Queue] Processing {queued_req.request_id} for {model_alias}"
        )

        # Get runner
        runner = await self.manager.get_runner_for_request(model_alias)

        # Execute request
        result = await self._execute_request(
            body=queued_req.body,
            request_id=queued_req.request_id,
            model_alias=model_alias,
            runner=runner,
            endpoint=endpoint
        )

        # Set result
        if not queued_req.response_future.done():
            queued_req.response_future.set_result(result)

        async with queue.lock:
            queue.total_processed += 1

        total_time = time.time() - process_start
        queue_wait = process_start - queued_req.timestamp

        logger.info(
            f"[Queue] Request {queued_req.request_id} completed: "
            f"queue_wait={queue_wait:.3f}s, total={total_time:.3f}s"
        )

    async def _execute_request(
        self,
        body: Dict[str, Any],
        request_id: str,
        model_alias: str,
        runner,
        endpoint: str
    ) -> Dict[str, Any]:
        """Execute request against llama-server with retry logic."""
        max_retries = 2
        retry_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                internal_url = f"{runner.url}{endpoint}"
                req = self.http_client.build_request(
                    method="POST",
                    url=internal_url,
                    json=body,
                    headers={"Content-Type": "application/json"}
                )

                response = await self.http_client.send(req, stream=True)
                is_streaming = body.get("stream", False)

                if is_streaming:
                    return await self._handle_streaming_response(response)
                else:
                    return await self._handle_json_response(response, model_alias)

            except (httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if attempt < max_retries:
                    logger.warning(
                        f"[Queue] Request {request_id} attempt {attempt + 1} "
                        f"failed: {e}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    logger.error(
                        f"[Queue] Request {request_id} failed after {max_retries + 1} attempts"
                    )
                    raise

    async def _handle_streaming_response(
        self,
        response: httpx.Response
    ) -> Dict[str, Any]:
        """Handle streaming response from llama-server."""
        chunks = []
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
        await response.aclose()

        return {
            "type": "stream",
            "chunks": chunks,
            "status_code": response.status_code
        }

    async def _handle_json_response(
        self,
        response: httpx.Response,
        model_alias: str
    ) -> Dict[str, Any]:
        """Handle JSON response from llama-server."""
        content = await response.aread()
        await response.aclose()

        response_data = json.loads(content.decode('utf-8'))

        # Check for context shift
        context_shifted = False
        if 'timings' in response_data:
            timings = response_data['timings']
            if timings.get('context_shift', 0) > 0:
                context_shifted = True
                logger.warning(
                    f"[{model_alias}] Context shift detected! "
                    f"Consider using shorter conversations."
                )
                if 'metadata' not in response_data:
                    response_data['metadata'] = {}
                response_data['metadata']['context_shifted'] = True

        # Extract tokens
        tokens = 0
        if 'usage' in response_data:
            tokens = response_data['usage'].get('completion_tokens', 0)
        elif 'choices' in response_data and response_data['choices']:
            content_text = response_data['choices'][0].get(
                'message', {}
            ).get('content', '')
            tokens = len(content_text) // 4

        return {
            "type": "json",
            "data": response_data,
            "tokens": tokens,
            "status_code": response.status_code,
            "context_shifted": context_shifted
        }
