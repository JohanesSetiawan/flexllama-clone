"""
Request Queue Management Module

This module provides a priority-based request queue system with backpressure
for handling LLM model requests efficiently.

Components:
    - RequestPriority: Enum for request priority levels (HIGH, NORMAL, LOW)
    - QueuedRequest: Dataclass for storing requests in the queue
    - ModelRequestQueue: Per-model queue with priority heap and backpressure
    - QueueManager: Manager for all queues, one queue per model

Features:
    - Priority-based ordering: HIGH requests are processed first
    - Backpressure: Queue has max size to prevent memory exhaustion
    - Timeout handling: Requests that wait too long will timeout
    - Heap-based: O(log n) insertion and extraction

Request Flow:
    1. Request enters via enqueue() with priority
    2. Queue processor retrieves request via dequeue() (priority order)
    3. Request is processed and result returned via Future
    4. If queue is full, request is rejected with error

Priority Levels:
    - HIGH (1): High priority requests (VIP users, urgent tasks)
    - NORMAL (2): Normal requests (default)
    - LOW (3): Low priority requests (batch processing, background tasks)

Usage:
    queue_manager = QueueManager(config)
    queue = await queue_manager.get_queue("qwen3-8b")
    
    # Enqueue with priority
    result = await queue.enqueue(
        request_id="req-123",
        body={"prompt": "..."},
        priority=RequestPriority.HIGH,
        timeout=300
    )
"""

import time
import heapq
import asyncio
import logging
from enum import Enum
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


class RequestPriority(Enum):
    """
    Priority levels for queued requests.

    Lower numeric value = higher priority.
    """
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass(order=True)
class QueuedRequest:
    """
    Request stored in the queue with priority ordering.

    Uses (priority, timestamp) as sort key for heap operations.
    Lower priority value = higher priority (HIGH=1 > NORMAL=2 > LOW=3).

    Attributes:
        priority: Numeric priority value (lower = higher priority)
        timestamp: Request creation time for FIFO within same priority
        request_id: Unique identifier for the request
        model_alias: Target model for the request
        body: Request payload
        response_future: Future to set result when processing completes
    """
    priority: int
    timestamp: float
    request_id: str = field(compare=False)
    model_alias: str = field(compare=False)
    body: Dict[str, Any] = field(compare=False)
    response_future: asyncio.Future = field(compare=False)

    def __post_init__(self):
        """Initialize sort key for heap comparison."""
        self.sort_key = (self.priority, self.timestamp)


class ModelRequestQueue:
    """
    Queue for a single model with priority heap and backpressure.

    Optimized with heapq for O(log n) insertion instead of O(n) deque insertion.

    Attributes:
        model_alias: The model this queue serves
        max_queue_size: Maximum number of pending requests
        processing: Whether queue processor is running
        total_requests: Total requests ever enqueued
        total_processed: Total requests successfully processed
        total_rejected: Total requests rejected due to full queue
        current_processing: Number of requests currently being processed
    """

    def __init__(self, model_alias: str, max_queue_size: int = 100):
        """
        Initialize a model request queue.

        Args:
            model_alias: Alias of the model this queue serves
            max_queue_size: Maximum pending requests before rejection
        """
        self.model_alias = model_alias
        self.max_queue_size = max_queue_size

        # Internal heap for priority ordering
        self._heap: List[QueuedRequest] = []

        # Processing state
        self.processing = False
        self.lock = asyncio.Lock()
        self.queue_not_empty = asyncio.Event()

        # Metrics
        self.total_requests = 0
        self.total_processed = 0
        self.total_rejected = 0
        self.current_processing = 0

    @property
    def queue(self) -> List[QueuedRequest]:
        """Compatibility property for existing code that accesses .queue"""
        return self._heap

    def __len__(self) -> int:
        """Return current queue length."""
        return len(self._heap)

    async def enqueue(
        self,
        request_id: str,
        body: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout: float = 600
    ) -> Dict[str, Any]:
        """
        Add request to queue and wait for result.

        Args:
            request_id: Unique identifier for the request
            body: Request payload
            priority: Request priority level
            timeout: Maximum time to wait for result in seconds

        Returns:
            Response data from the model

        Raises:
            RuntimeError: If queue is full
            TimeoutError: If request times out in queue
        """
        enqueue_time = time.time()

        async with self.lock:
            # Check queue capacity
            if len(self._heap) >= self.max_queue_size:
                self.total_rejected += 1
                raise RuntimeError(
                    f"Queue for model '{self.model_alias}' is full "
                    f"({self.max_queue_size}). Try again later."
                )

            # Create queued request
            response_future: asyncio.Future = asyncio.Future()
            queued_req = QueuedRequest(
                priority=priority.value,
                timestamp=time.time(),
                request_id=request_id,
                model_alias=self.model_alias,
                body=body,
                response_future=response_future
            )

            # O(log n) heap insertion
            heapq.heappush(self._heap, queued_req)

            self.total_requests += 1
            self.queue_not_empty.set()

        # Wait for response with timeout
        try:
            result = await asyncio.wait_for(response_future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            wait_time = time.time() - enqueue_time
            logger.error(
                f"[{self.model_alias}] Request {request_id} timeout after "
                f"{wait_time:.1f}s (queue_length={len(self._heap)}, "
                f"processing={self.current_processing})"
            )
            # Remove from queue on timeout - O(n) but rare
            await self._remove_request(queued_req)
            raise TimeoutError(f"Request timeout after {timeout}s in queue")

    async def _remove_request(self, request: QueuedRequest) -> None:
        """Remove a specific request from the queue."""
        async with self.lock:
            try:
                self._heap.remove(request)
                heapq.heapify(self._heap)  # Restore heap property
            except ValueError:
                pass  # Already processed

    async def dequeue(self) -> Optional[QueuedRequest]:
        """
        Get next request from queue (highest priority first).

        Returns:
            The next request to process, or None if queue is empty
        """
        async with self.lock:
            if self._heap:
                # O(log n) heap pop
                return heapq.heappop(self._heap)
            return None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.

        Returns:
            Dictionary containing queue metrics
        """
        return {
            "queue_length": len(self._heap),
            "total_requests": self.total_requests,
            "total_processed": self.total_processed,
            "total_rejected": self.total_rejected,
            "current_processing": self.current_processing,
            "processing": self.processing
        }


class QueueManager:
    """
    Manager for all model request queues.

    Maintains one queue per model, creating queues on demand.
    Supports Redis-backed queue when enabled, with in-memory fallback.

    Attributes:
        config: Application configuration
        queues: Dictionary of model alias to in-memory queue
        redis_queue: Optional Redis queue service
    """

    def __init__(self, config, redis_queue_service=None):
        """
        Initialize the queue manager.

        Args:
            config: Application configuration with queue settings
            redis_queue_service: Optional Redis queue service for persistence
        """
        self.config = config
        self.queues: Dict[str, ModelRequestQueue] = {}
        self.lock = asyncio.Lock()
        self.redis_queue = redis_queue_service

    @property
    def use_redis(self) -> bool:
        """Check if Redis queue is enabled and connected."""
        return self.redis_queue is not None and self.redis_queue.connected

    async def get_queue(self, model_alias: str) -> ModelRequestQueue:
        """
        Get or create queue for a model.

        Args:
            model_alias: Alias of the model

        Returns:
            The queue for the specified model
        """
        async with self.lock:
            if model_alias not in self.queues:
                max_size = self.config.system.max_queue_size_per_model
                self.queues[model_alias] = ModelRequestQueue(
                    model_alias=model_alias,
                    max_queue_size=max_size
                )
            return self.queues[model_alias]

    async def enqueue_redis(
        self,
        model_alias: str,
        request_id: str,
        body: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL
    ) -> bool:
        """
        Enqueue request to Redis (if enabled).

        Args:
            model_alias: Target model
            request_id: Unique request ID
            body: Request payload
            priority: Request priority

        Returns:
            True if enqueued to Redis, False if using in-memory
        """
        if not self.use_redis:
            return False

        max_size = self.config.system.max_queue_size_per_model
        await self.redis_queue.enqueue(
            model_alias=model_alias,
            request_id=request_id,
            body=body,
            priority=priority.value,
            max_queue_size=max_size
        )
        return True

    async def dequeue_redis(self, model_alias: str) -> Optional[Dict[str, Any]]:
        """
        Dequeue request from Redis (if enabled).

        Args:
            model_alias: Target model

        Returns:
            Request data dict, or None if empty/not using Redis
        """
        if not self.use_redis:
            return None

        return await self.redis_queue.dequeue(model_alias)

    async def set_result_redis(
        self,
        request_id: str,
        result: Dict[str, Any]
    ) -> bool:
        """Store result in Redis for polling."""
        if not self.use_redis:
            return False

        return await self.redis_queue.set_result(request_id, result)

    async def get_result_redis(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get result from Redis."""
        if not self.use_redis:
            return None

        return await self.redis_queue.get_result(request_id)

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all queues.

        Returns:
            Dictionary mapping model alias to queue statistics
        """
        stats = {
            alias: queue.get_stats()
            for alias, queue in self.queues.items()
        }
        stats["_redis_enabled"] = self.use_redis
        return stats

    def get_total_pending(self) -> int:
        """
        Get total pending requests across all queues.

        Returns:
            Total number of pending requests
        """
        return sum(len(queue) for queue in self.queues.values())
