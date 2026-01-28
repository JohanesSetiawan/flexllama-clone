"""
Redis Cache Service Module

Provides semantic caching for LLM responses using Redis.
Reduces GPU load by returning cached responses for identical prompts.

Features:
    - SHA256-based cache key generation
    - Configurable TTL (default: 1 hour)
    - Graceful degradation on Redis errors
    - Non-streaming response caching only

Usage:
    cache_service = CacheService(redis_config)
    await cache_service.connect()

    # Check cache
    cached = await cache_service.get(model_alias, request_body)
    if cached:
        return cached

    # Store after generation
    await cache_service.set(model_alias, request_body, response_data)
"""

import json
import hashlib
import logging
from typing import Optional, Dict, Any

import redis.asyncio as redis

from ..core.config import RedisConfig


logger = logging.getLogger(__name__)


class CacheService:
    """
    Semantic cache service using Redis.

    Caches non-streaming LLM responses based on request hash.
    Streaming responses are not cached due to their nature.

    Attributes:
        config: Redis configuration
        client: Redis async client instance
        connected: Connection status flag
    """

    def __init__(self, config: Optional[RedisConfig] = None):
        """
        Initialize cache service.

        Args:
            config: Redis configuration. If None, caching is disabled.
        """
        self.config = config
        self.client: Optional[redis.Redis] = None
        self.connected = False

    async def connect(self) -> bool:
        """
        Establish Redis connection.

        Returns:
            True if connected successfully, False otherwise.
        """
        if not self.config or not self.config.enable_cache:
            logger.info("Cache disabled: no Redis config or caching disabled")
            return False

        try:
            self.client = redis.from_url(
                self.config.url,
                encoding="utf-8",
                decode_responses=True
            )
            # Test connection
            await self.client.ping()
            self.connected = True
            logger.info(f"Cache service connected to Redis: {self.config.url}")
            return True

        except Exception as e:
            logger.warning(
                f"Cache connection failed: {e}. Continuing without cache.")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.connected = False
            logger.info("Cache service disconnected")

    def _generate_cache_key(
        self,
        model_alias: str,
        request_body: Dict[str, Any]
    ) -> str:
        """
        Generate deterministic cache key from request.

        Creates SHA256 hash of normalized request body.
        Excludes volatile fields (stream, request_id) from hash.

        Args:
            model_alias: Target model identifier
            request_body: Request payload

        Returns:
            Cache key string in format: {prefix}{model}:{hash}
        """
        # Copy and normalize request body
        normalized = request_body.copy()

        # Remove fields that should not affect cache key
        volatile_fields = ["stream", "request_id", "user"]
        for field in volatile_fields:
            normalized.pop(field, None)

        # Create deterministic JSON string
        json_str = json.dumps(normalized, sort_keys=True, ensure_ascii=True)

        # Generate hash
        content_hash = hashlib.sha256(json_str.encode()).hexdigest()[:16]

        prefix = self.config.cache_prefix if self.config else "router:cache:"
        return f"{prefix}{model_alias}:{content_hash}"

    async def get(
        self,
        model_alias: str,
        request_body: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response if exists.

        Args:
            model_alias: Target model identifier
            request_body: Request payload for key generation

        Returns:
            Cached response dict if hit, None if miss or error.
        """
        if not self.connected or not self.client:
            return None

        # Skip cache for streaming requests
        if request_body.get("stream", False):
            return None

        try:
            cache_key = self._generate_cache_key(model_alias, request_body)
            cached_data = await self.client.get(cache_key)

            if cached_data:
                logger.debug(f"Cache HIT: {cache_key}")
                return json.loads(cached_data)

            logger.debug(f"Cache MISS: {cache_key}")
            return None

        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    async def set(
        self,
        model_alias: str,
        request_body: Dict[str, Any],
        response_data: Dict[str, Any]
    ) -> bool:
        """
        Store response in cache.

        Args:
            model_alias: Target model identifier
            request_body: Original request for key generation
            response_data: Response to cache

        Returns:
            True if stored successfully, False otherwise.
        """
        if not self.connected or not self.client:
            return False

        # Do not cache streaming responses
        if request_body.get("stream", False):
            return False

        try:
            cache_key = self._generate_cache_key(model_alias, request_body)
            json_data = json.dumps(response_data, ensure_ascii=False)
            ttl = self.config.cache_ttl_sec if self.config else 3600

            await self.client.setex(cache_key, ttl, json_data)
            logger.debug(f"Cache SET: {cache_key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False

    async def invalidate(self, model_alias: str) -> int:
        """
        Invalidate all cache entries for a model.

        Useful when model is updated or reloaded.

        Args:
            model_alias: Model to invalidate cache for

        Returns:
            Number of keys deleted.
        """
        if not self.connected or not self.client:
            return 0

        try:
            prefix = self.config.cache_prefix if self.config else "router:cache:"
            pattern = f"{prefix}{model_alias}:*"
            keys = []

            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(
                    f"Invalidated {deleted} cache entries for {model_alias}")
                return deleted

            return 0

        except Exception as e:
            logger.warning(f"Cache invalidate error: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache service status.

        Returns:
            Dictionary with connection status and config info.
        """
        return {
            "connected": self.connected,
            "enabled": self.config.enable_cache if self.config else False,
            "ttl_sec": self.config.cache_ttl_sec if self.config else 0,
            "url": self.config.url if self.config else None
        }


# Module-level singleton
_cache_service: Optional[CacheService] = None


def get_cache_service() -> Optional[CacheService]:
    """Get the global cache service instance."""
    return _cache_service


async def init_cache_service(config: Optional[RedisConfig]) -> CacheService:
    """
    Initialize the global cache service.

    Args:
        config: Redis configuration

    Returns:
        Initialized CacheService instance
    """
    global _cache_service
    _cache_service = CacheService(config)
    await _cache_service.connect()
    return _cache_service
