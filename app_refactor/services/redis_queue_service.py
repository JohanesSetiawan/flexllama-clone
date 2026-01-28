"""
Redis Queue Service Module

Provides Redis-backed persistent queue using Sorted Sets for priority ordering.
Requests survive server restarts and can be shared across workers.

Features:
    - Priority ordering via Sorted Set scores
    - TTL enforcement (default 90s) for SLA compliance
    - Result storage for async polling pattern
    - Graceful fallback when Redis unavailable

Usage:
    queue_service = RedisQueueService(redis_config)
    await queue_service.connect()
    
    # Enqueue request
    await queue_service.enqueue(model_alias, request_id, body, priority)
    
    # Dequeue and process
    request = await queue_service.dequeue(model_alias)
    # ... process ...
    await queue_service.set_result(request_id, result)
"""

import json
import time
import logging
from typing import Optional, Dict, Any

import redis.asyncio as redis

from ..core.config import RedisConfig


logger = logging.getLogger(__name__)


class RedisQueueService:
    """
    Redis-backed queue service using Sorted Sets.

    Sorted Set score format: priority * 1e12 + timestamp
    This ensures priority ordering first, then FIFO within same priority.

    Attributes:
        config: Redis configuration
        client: Redis async client
        connected: Connection status
        queue_ttl: Max seconds request can wait in queue
    """

    # Priority multiplier to ensure priority takes precedence over timestamp
    PRIORITY_MULTIPLIER = 1e12

    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize Redis queue service.

        Args:
            config: Redis configuration. If None, service is disabled.
        """
        self.config = config
        self.client: Optional[redis.Redis] = None
        self.connected = False
        self.queue_ttl = config.queue_ttl_sec if config else 90

    async def connect(self) -> bool:
        """
        Establish Redis connection.

        Returns:
            True if connected, False otherwise.
        """
        if not self.config or not self.config.enable_redis_queue:
            logger.info("Redis queue disabled in config")
            return False

        try:
            self.client = redis.from_url(
                self.config.url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            self.connected = True
            logger.info(f"Redis queue connected: {self.config.url}")
            return True

        except Exception as e:
            logger.warning(f"Redis queue connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("Redis queue disconnected")

    def _queue_key(self, model_alias: str) -> str:
        """Generate Redis key for model queue."""
        prefix = self.config.queue_prefix if self.config else "router:queue:"
        return f"{prefix}{model_alias}"

    def _result_key(self, request_id: str) -> str:
        """Generate Redis key for request result."""
        prefix = self.config.queue_prefix if self.config else "router:queue:"
        return f"{prefix}result:{request_id}"

    def _calculate_score(self, priority: int, timestamp: float) -> float:
        """
        Calculate Sorted Set score for priority ordering.

        Score = priority * 1e12 + timestamp
        Lower score = higher priority (processed first)

        Args:
            priority: Priority value (1=HIGH, 2=NORMAL, 3=LOW)
            timestamp: Unix timestamp

        Returns:
            Score for ZADD
        """
        return priority * self.PRIORITY_MULTIPLIER + timestamp

    async def enqueue(
        self,
        model_alias: str,
        request_id: str,
        body: Dict[str, Any],
        priority: int = 2,
        max_queue_size: int = 500
    ) -> bool:
        """
        Add request to Redis queue.

        Args:
            model_alias: Target model
            request_id: Unique request identifier
            body: Request payload
            priority: Priority level (1=HIGH, 2=NORMAL, 3=LOW)
            max_queue_size: Max queue size before rejection

        Returns:
            True if enqueued, False if queue full or error

        Raises:
            RuntimeError: If queue is full
        """
        if not self.connected or not self.client:
            raise RuntimeError("Redis queue not connected")

        queue_key = self._queue_key(model_alias)
        timestamp = time.time()

        try:
            # Check queue size
            current_size = await self.client.zcard(queue_key)
            if current_size >= max_queue_size:
                raise RuntimeError(
                    f"Queue for '{model_alias}' is full ({max_queue_size})"
                )

            # Prepare request data
            request_data = {
                "request_id": request_id,
                "model_alias": model_alias,
                "body": body,
                "priority": priority,
                "enqueue_time": timestamp
            }

            score = self._calculate_score(priority, timestamp)
            member = json.dumps(request_data, ensure_ascii=False)

            # Add to sorted set
            await self.client.zadd(queue_key, {member: score})

            logger.debug(
                f"[Redis Queue] Enqueued {request_id} for {model_alias} "
                f"(priority={priority}, score={score:.0f})"
            )
            return True

        except RuntimeError:
            raise
        except Exception as e:
            logger.error(f"Redis enqueue error: {e}")
            raise RuntimeError(f"Failed to enqueue: {e}")

    async def dequeue(self, model_alias: str) -> Optional[Dict[str, Any]]:
        """
        Get next request from queue (highest priority first).

        Also enforces TTL - expired requests are discarded.

        Args:
            model_alias: Target model queue

        Returns:
            Request data dict, or None if queue empty
        """
        if not self.connected or not self.client:
            return None

        queue_key = self._queue_key(model_alias)

        try:
            # Pop lowest score (highest priority)
            result = await self.client.zpopmin(queue_key, count=1)

            if not result:
                return None

            member, score = result[0]
            request_data = json.loads(member)

            # Check TTL
            enqueue_time = request_data.get("enqueue_time", 0)
            age = time.time() - enqueue_time

            if age > self.queue_ttl:
                # Request expired - store timeout result
                request_id = request_data.get("request_id", "unknown")
                logger.warning(
                    f"[Redis Queue] Request {request_id} expired "
                    f"after {age:.1f}s (TTL={self.queue_ttl}s)"
                )
                await self.set_result(request_id, {
                    "error": "timeout",
                    "detail": f"Request expired after {age:.1f}s in queue"
                })
                # Recursively get next valid request
                return await self.dequeue(model_alias)

            logger.debug(
                f"[Redis Queue] Dequeued {request_data.get('request_id')} "
                f"(age={age:.1f}s)"
            )
            return request_data

        except Exception as e:
            logger.error(f"Redis dequeue error: {e}")
            return None

    async def set_result(
        self,
        request_id: str,
        result: Dict[str, Any],
        ttl: int = 300
    ) -> bool:
        """
        Store result for polling.

        Args:
            request_id: Request identifier
            result: Result data
            ttl: Result TTL in seconds (default 5 min)

        Returns:
            True if stored successfully
        """
        if not self.connected or not self.client:
            return False

        try:
            result_key = self._result_key(request_id)
            json_data = json.dumps(result, ensure_ascii=False)
            await self.client.setex(result_key, ttl, json_data)
            return True

        except Exception as e:
            logger.error(f"Redis set_result error: {e}")
            return False

    async def get_result(
        self,
        request_id: str,
        delete: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Poll for request result.

        Args:
            request_id: Request identifier
            delete: Whether to delete result after retrieval

        Returns:
            Result data, or None if not ready
        """
        if not self.connected or not self.client:
            return None

        try:
            result_key = self._result_key(request_id)

            if delete:
                # Atomic get and delete
                json_data = await self.client.getdel(result_key)
            else:
                json_data = await self.client.get(result_key)

            if json_data:
                return json.loads(json_data)
            return None

        except Exception as e:
            logger.error(f"Redis get_result error: {e}")
            return None

    async def get_queue_size(self, model_alias: str) -> int:
        """Get current queue size for a model."""
        if not self.connected or not self.client:
            return 0

        try:
            queue_key = self._queue_key(model_alias)
            return await self.client.zcard(queue_key)
        except Exception:
            return 0

    async def get_all_queue_sizes(self) -> Dict[str, int]:
        """Get queue sizes for all models."""
        if not self.connected or not self.client:
            return {}

        try:
            prefix = self.config.queue_prefix if self.config else "router:queue:"
            pattern = f"{prefix}*"
            result = {}

            async for key in self.client.scan_iter(match=pattern):
                # Skip result keys
                if ":result:" in key:
                    continue
                model = key.replace(prefix, "")
                size = await self.client.zcard(key)
                result[model] = size

            return result
        except Exception:
            return {}


# Module-level singleton
_redis_queue_service: Optional[RedisQueueService] = None


def get_redis_queue_service() -> Optional[RedisQueueService]:
    """Get global Redis queue service instance."""
    return _redis_queue_service


async def init_redis_queue_service(
    config: Optional[RedisConfig]
) -> RedisQueueService:
    """
    Initialize global Redis queue service.

    Args:
        config: Redis configuration

    Returns:
        Initialized RedisQueueService
    """
    global _redis_queue_service
    _redis_queue_service = RedisQueueService(config)
    await _redis_queue_service.connect()
    return _redis_queue_service
