"""
Model Warmup and Pre-loading Module

This module provides a pre-loading system for models at startup and a
keep-warm strategy to maintain popular models loaded in VRAM.

Components:
    - ModelWarmupManager: Manager for preloading and warmup strategy

Features:
    - Preload models at startup (configurable via preload_models)
    - Keep-warm strategy for popular models based on usage statistics
    - VRAM-aware loading (skip models if VRAM is insufficient)
    - Popularity tracking based on request count
    - Recent activity tracking to prevent idle preload

Preload Modes:
    - ["*"]: Load all models in config
    - ["model1", "model2"]: Load specific models
    - []: Don't preload anything

Warmup Strategy:
    - Every 5 minutes, check top N popular models (keep_warm_models)
    - If model not running but recently active, preload
    - If model running, update last_used_time to prevent idle timeout
    - Skip models that failed to load due to VRAM constraints

Usage:
    warmup_manager = ModelWarmupManager(manager, config, shutdown_event)
    await warmup_manager.start()

    # Record request for popularity tracking
    warmup_manager.record_request("qwen3-8b")

    # Check if model is warm
    is_warm = warmup_manager.is_model_warm("qwen3-8b")

    # Stop
    await warmup_manager.stop()

Configuration (via system config):
    - preload_models: List of models to preload
    - preload_delay_sec: Delay between model loads
    - keep_warm_models: Number of popular models to keep warm
    - timeout_warmup_sec: Timeout for model loading
    - wait_ready_sec: Max wait for model ready status
"""

import time
import asyncio
import logging
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict


logger = logging.getLogger(__name__)


# Constants
WARM_CHECK_INTERVAL_SEC = 300  # Check every 5 minutes
RECENT_ACTIVITY_WINDOW_SEC = 600  # 10 minutes
RELOAD_TIMEOUT_SEC = 120.0


class WarmupService:
    """
    Manage model pre-loading and keep-warm strategy.

    Handles:
    - Preload models at startup
    - Keep popular models warm (prevent idle timeout)
    - Track model popularity based on request count

    Attributes:
        manager: ModelManager instance
        config: Application configuration
        shutdown_event: Event to signal shutdown
        request_counts: Request count per model for popularity
        last_request_time: Last request timestamp per model
        vram_failed_models: Models that failed to load due to VRAM
    """

    def __init__(self, manager, config, shutdown_event: asyncio.Event):
        """
        Initialize the warmup manager.

        Args:
            manager: ModelManager instance
            config: Application configuration
            shutdown_event: Event signaling shutdown
        """
        self.manager = manager
        self.config = config
        self.shutdown_event = shutdown_event

        # Usage statistics
        self.request_counts: Dict[str, int] = defaultdict(int)
        self.last_request_time: Dict[str, float] = {}

        # Track models that failed due to VRAM (don't retry these)
        self.vram_failed_models: Set[str] = set()

        # Background tasks
        self.preload_task: Optional[asyncio.Task] = None
        self.warmup_task: Optional[asyncio.Task] = None

        # Activity window for warm maintenance
        self.recent_activity_window = RECENT_ACTIVITY_WINDOW_SEC

    def _resolve_preload_models(self) -> List[str]:
        """
        Resolve preload_models configuration to list of model aliases.

        Supports:
        - ["*"]: Load all models in config
        - ["model1", "model2", ...]: Load specific models
        - []: Don't preload anything

        Returns:
            List of model aliases to preload
        """
        preload_config = self.config.system.preload_models

        if not preload_config:
            return []

        # Check for wildcard ["*"]
        if len(preload_config) == 1 and preload_config[0] == "*":
            all_models = list(self.config.models.keys())
            logger.info(
                f"[Preload] Wildcard '*' detected. "
                f"Will preload ALL {len(all_models)} models: {all_models}"
            )
            return all_models

        # Validate specific model aliases
        valid_models = []
        for model_alias in preload_config:
            if model_alias == "*":
                logger.warning(
                    "[Preload] Wildcard '*' should be used alone. Ignoring."
                )
                continue

            if model_alias in self.config.models:
                valid_models.append(model_alias)
            else:
                logger.warning(
                    f"[Preload] Model '{model_alias}' not in config. Skipping."
                )

        return valid_models

    async def _load_single_model(
        self,
        model_alias: str,
        index: int,
        total: int
    ) -> bool:
        """
        Load a single model with detailed logging and error handling.

        Args:
            model_alias: Model to load
            index: Current index (for logging)
            total: Total models to load (for logging)

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(
                f"[Preload] [{index}/{total}] Loading model: {model_alias}"
            )

            start_time = time.time()

            # Get runner with timeout
            runner = await asyncio.wait_for(
                self.manager.get_runner_for_request(model_alias),
                timeout=self.config.system.timeout_warmup_sec
            )

            # Wait for ready status
            success = await self._wait_for_ready(
                runner,
                model_alias,
                index,
                total
            )

            if success:
                load_time = time.time() - start_time
                self._log_load_success(
                    model_alias, runner, load_time, index, total)

            return success

        except asyncio.TimeoutError:
            logger.error(
                f"[Preload] [{index}/{total}] TIMEOUT | "
                f"Model '{model_alias}' exceeded "
                f"{self.config.system.timeout_warmup_sec}s"
            )
            return False
        except Exception as e:
            logger.error(
                f"[Preload] [{index}/{total}] ERROR | "
                f"Model '{model_alias}': {e}"
            )
            return False

    async def _wait_for_ready(
        self,
        runner,
        model_alias: str,
        index: int,
        total: int
    ) -> bool:
        """Wait for runner to reach ready status."""
        max_wait = self.config.system.wait_ready_sec
        wait_start = time.time()

        while runner.status not in ("ready", "crashed", "stopped"):
            if time.time() - wait_start > max_wait:
                logger.warning(
                    f"[Preload] [{index}/{total}] Model '{model_alias}' "
                    f"stuck at '{runner.status}' after {max_wait}s"
                )
                return False

            if self.shutdown_event.is_set():
                logger.info(
                    "[Preload] Shutdown detected while waiting for ready"
                )
                return False

            await asyncio.sleep(1)

        if runner.status != "ready":
            logger.error(
                f"[Preload] [{index}/{total}] FAILED | "
                f"Model '{model_alias}' ended with status '{runner.status}'"
            )
            return False

        return True

    def _log_load_success(
        self,
        model_alias: str,
        runner,
        load_time: float,
        index: int,
        total: int
    ) -> None:
        """Log successful model load with VRAM info."""
        vram_report = self.manager.vram_service.get_vram_report()
        logger.info(
            f"[Preload] [{index}/{total}] SUCCESS | "
            f"'{model_alias}' loaded at {runner.url} (took {load_time:.1f}s) | "
            f"VRAM: {vram_report['gpu_info']['used_gb']:.2f} GB used, "
            f"{vram_report['gpu_info']['free_gb']:.2f} GB free"
        )

    async def preload_models_serial(self) -> None:
        """Preload models one at a time serially."""
        models_to_preload = self._resolve_preload_models()

        if not models_to_preload:
            logger.info("[Preload] No models configured for preloading")
            return

        total_models = len(models_to_preload)
        preload_delay = self.config.system.preload_delay_sec

        logger.info(
            f"[Preload] Starting SERIAL preload for {total_models} models: "
            f"{models_to_preload}"
        )

        # Warning if preloading more than max_concurrent
        if total_models > self.config.system.max_concurrent_models:
            logger.warning(
                f"[Preload] Preload count ({total_models}) > "
                f"max_concurrent_models ({self.config.system.max_concurrent_models})"
            )

        successful = 0
        failed = 0

        for idx, model_alias in enumerate(models_to_preload, 1):
            if self.shutdown_event.is_set():
                logger.info("[Preload] Shutdown detected. Stopping preload.")
                return

            result = await self._load_single_model(model_alias, idx, total_models)

            if result:
                successful += 1
            else:
                failed += 1

            # Delay before next model (except last)
            if idx < total_models and preload_delay > 0:
                await self._delay_with_shutdown_check(preload_delay)

        self._log_preload_summary(total_models, successful, failed)

    async def preload_models_queued(self, max_parallel: int = 2) -> None:
        """
        Preload models with queued loading.

        Due to load_lock in VRAMTracker, models load sequentially even when
        multiple tasks are created. This prevents VRAM race conditions.
        """
        models_to_preload = self._resolve_preload_models()

        if not models_to_preload:
            logger.info("[Preload] No models configured for preloading")
            return

        total_models = len(models_to_preload)
        preload_delay = self.config.system.preload_delay_sec

        logger.info(
            f"[Preload] Starting QUEUED preload for {total_models} models: "
            f"{models_to_preload}"
        )

        successful = 0
        failed = 0
        skipped = 0

        for batch_start in range(0, total_models, max_parallel):
            if self.shutdown_event.is_set():
                logger.info("[Preload] Shutdown detected. Stopping preload.")
                return

            batch_end = min(batch_start + max_parallel, total_models)
            batch = models_to_preload[batch_start:batch_end]

            logger.info(
                f"[Preload] Batch {batch_start // max_parallel + 1}: "
                f"Queuing {len(batch)} models: {batch}"
            )

            # Create and await batch tasks
            results = await self._process_batch(batch, batch_start, total_models)

            # Process results
            for model_alias, result in zip(batch, results):
                if isinstance(result, Exception):
                    if self._is_vram_error(result):
                        self.vram_failed_models.add(model_alias)
                        skipped += 1
                    else:
                        failed += 1
                elif result:
                    successful += 1
                else:
                    failed += 1

            # Delay before next batch
            if batch_end < total_models and preload_delay > 0:
                await self._delay_with_shutdown_check(preload_delay)

        self._log_preload_summary(total_models, successful, failed, skipped)

    async def _process_batch(
        self,
        batch: List[str],
        batch_start: int,
        total: int
    ) -> List[Any]:
        """Process a batch of models concurrently."""
        tasks = [
            asyncio.create_task(
                self._load_single_model(
                    model_alias,
                    batch_start + i + 1,
                    total
                )
            )
            for i, model_alias in enumerate(batch)
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def _is_vram_error(self, error: Exception) -> bool:
        """Check if error is VRAM related."""
        error_str = str(error).lower()
        return "vram" in error_str or "insufficient" in error_str

    async def _delay_with_shutdown_check(self, delay_sec: int) -> None:
        """Wait with periodic shutdown checks."""
        logger.info(f"[Preload] Waiting {delay_sec}s before next...")
        for _ in range(delay_sec):
            if self.shutdown_event.is_set():
                logger.info("[Preload] Shutdown detected during delay")
                return
            await asyncio.sleep(1)

    def _log_preload_summary(
        self,
        total: int,
        successful: int,
        failed: int,
        skipped: int = 0
    ) -> None:
        """Log preload completion summary."""
        logger.info(
            f"[Preload] COMPLETE | Total: {total} | Success: {successful} | "
            f"Failed: {failed} | Skipped: {skipped}"
        )

        vram_report = self.manager.vram_service.get_vram_report()
        logger.info(
            f"[Preload] VRAM STATUS | "
            f"Used: {vram_report['gpu_info']['used_gb']:.2f} GB | "
            f"Free: {vram_report['gpu_info']['free_gb']:.2f} GB | "
            f"Status: {vram_report['status']}"
        )

    async def maintain_warm_models(self) -> None:
        """Background task to keep popular models warm."""
        try:
            while not self.shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self.shutdown_event.wait(),
                        timeout=WARM_CHECK_INTERVAL_SEC
                    )
                    break
                except asyncio.TimeoutError:
                    pass

                await self._perform_warm_maintenance()

        except asyncio.CancelledError:
            logger.info("Warm maintenance task cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error in maintain_warm_models: {e}")

    async def _perform_warm_maintenance(self) -> None:
        """Perform one cycle of warm model maintenance."""
        keep_warm_count = self.config.system.keep_warm_models
        if keep_warm_count == 0:
            return

        popular_models = self.get_popular_models(top_n=keep_warm_count)
        if not popular_models:
            return

        # Filter recently active models that haven't failed due to VRAM
        active_models = [
            m for m in popular_models
            if self.is_recently_active(m) and m not in self.vram_failed_models
        ]

        if not active_models:
            return

        logger.info(f"Maintaining warm models: {active_models}")

        for model_alias in active_models:
            if self.shutdown_event.is_set():
                return

            await self._maintain_single_model(model_alias)

    async def _maintain_single_model(self, model_alias: str) -> None:
        """Maintain warmth for a single model."""
        try:
            # Check state inside lock
            need_reload, need_preload = await self._check_model_state(model_alias)

            # Perform loading outside of lock to prevent deadlock
            if need_reload:
                await self._reload_model(model_alias, "dead runner")
            elif need_preload:
                await self._reload_model(model_alias, "popular model")

        except asyncio.TimeoutError:
            logger.error(f"Timeout maintaining warm model '{model_alias}'")
        except Exception as e:
            logger.error(f"Failed to maintain warm model '{model_alias}': {e}")

    async def _check_model_state(
        self,
        model_alias: str
    ) -> tuple[bool, bool]:
        """Check if model needs reload or preload."""
        need_reload = False
        need_preload = False

        async with self.manager.lock:
            if model_alias in self.manager.active_runners:
                runner = self.manager.active_runners[model_alias]
                if runner.is_alive():
                    # Keep warm by updating last_used_time
                    runner.last_used_time = time.time()
                else:
                    need_reload = True
            else:
                # Check if worth preloading
                last_time = self.last_request_time.get(model_alias, 0)
                time_since = time.time() - last_time
                if time_since < self.config.system.idle_timeout_sec:
                    need_preload = True

        return need_reload, need_preload

    async def _reload_model(self, model_alias: str, reason: str) -> None:
        """Reload a model with timeout."""
        logger.info(f"Preloading {reason}: {model_alias}")
        try:
            await asyncio.wait_for(
                self.manager.get_runner_for_request(model_alias),
                timeout=RELOAD_TIMEOUT_SEC
            )
            logger.info(f"Successfully preloaded '{model_alias}'")
        except asyncio.TimeoutError:
            logger.error(
                f"Timeout preloading '{model_alias}'. Skipping this cycle."
            )

    async def preload_models(self) -> None:
        """Preload models using appropriate strategy."""
        max_concurrent = self.config.system.max_concurrent_models

        # Use queued loading when multiple concurrent models are allowed
        if max_concurrent > 1:
            batch_size = min(2, max_concurrent)
            logger.info(
                f"[Preload] Using QUEUED loading (batch size {batch_size})")
            await self.preload_models_queued(max_parallel=batch_size)
        else:
            logger.info("[Preload] Using SERIAL loading")
            await self.preload_models_serial()

    async def start(self) -> None:
        """Start warmup manager tasks."""
        try:
            await self.preload_models()
        except asyncio.CancelledError:
            logger.info("Preload cancelled during startup")
            return

        self.warmup_task = asyncio.create_task(self.maintain_warm_models())
        logger.info("Model warmup manager started")

    async def stop(self) -> None:
        """Stop warmup manager."""
        logger.info("Stopping warmup manager...")

        if self.warmup_task and not self.warmup_task.done():
            self.warmup_task.cancel()
            try:
                await asyncio.wait_for(self.warmup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        logger.info("Warmup manager stopped")

    def record_request(self, model_alias: str) -> None:
        """Record that a model was requested (for popularity tracking)."""
        self.request_counts[model_alias] += 1
        self.last_request_time[model_alias] = time.time()

    def clear_vram_failed(self, model_alias: Optional[str] = None) -> None:
        """
        Clear VRAM failed status for a model (or all models).

        Call this when VRAM becomes available (e.g., after model eject).
        """
        if model_alias:
            self.vram_failed_models.discard(model_alias)
            logger.info(
                f"[Warmup] Cleared VRAM-failed status for '{model_alias}'")
        else:
            self.vram_failed_models.clear()
            logger.info("[Warmup] Cleared all VRAM-failed statuses")

    def get_popular_models(self, top_n: int = 5) -> List[str]:
        """Get top N most popular models based on request count."""
        sorted_models = sorted(
            self.request_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [model for model, _ in sorted_models[:top_n]]

    def is_recently_active(self, model_alias: str) -> bool:
        """
        Check if model was recently active (request within activity window).

        This prevents preloading models that have already timed out.
        """
        if model_alias not in self.last_request_time:
            return False

        time_since = time.time() - self.last_request_time[model_alias]
        return time_since < self.recent_activity_window

    def is_model_warm(self, model_alias: str) -> bool:
        """
        Check if model is in the warm models list.

        Returns:
            True if model is a warm model (top N popular), False otherwise
        """
        keep_warm_count = self.config.system.keep_warm_models
        if keep_warm_count <= 0:
            return False

        popular_models = self.get_popular_models(top_n=keep_warm_count)
        return model_alias in popular_models

    def get_popularity_stats(self) -> Dict[str, Any]:
        """Get popularity statistics for all models."""
        return {
            "request_counts": dict(self.request_counts),
            "last_request_times": {
                alias: time.time() - ts
                for alias, ts in self.last_request_time.items()
            },
            "vram_failed_models": list(self.vram_failed_models),
            "popular_models": self.get_popular_models(
                top_n=self.config.system.keep_warm_models
            )
        }
