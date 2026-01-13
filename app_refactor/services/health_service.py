"""
Model Health Monitoring Module

This module provides a health monitoring system for active models.
Health checks are performed periodically to detect problematic models
and perform auto-recovery if needed.

Components:
    - HealthCheckResult: Result from a single health check
    - ModelHealth: Health status tracking for one model
    - HealthMonitor: Background monitor for all active models

Health Status:
    - healthy: Model responding normally
    - degraded: Model starting to have issues (2+ consecutive failures)
    - down: Model not responding (5+ consecutive failures, will be restarted)

Features:
    - Periodic health checks to /health endpoint of llama-server
    - Response time tracking for performance monitoring
    - Consecutive failure counting for status determination
    - Auto-restart for models that are down
    - History tracking for uptime calculation

Monitoring Flow:
    1. Monitor loop runs every check_interval_sec (default 30s)
    2. For each active model, send GET /health to runner
    3. Record result (is_healthy, response_time, error)
    4. Update consecutive_failures counter
    5. If model is down (5+ failures), trigger restart

Usage:
    health_monitor = HealthMonitor(manager, check_interval_sec=30)
    health_monitor.start()
    
    # Get health status for all models
    health_status = health_monitor.get_all_health()
    
    # Stop monitoring
    await health_monitor.stop()
"""

import time
import httpx
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


# Constants for health status thresholds
DEGRADED_THRESHOLD = 2  # Consecutive failures before degraded status
DOWN_THRESHOLD = 5      # Consecutive failures before down status
HEALTH_CHECK_TIMEOUT = 5.0  # Timeout for health check request (seconds)


@dataclass
class HealthCheckResult:
    """
    Result from a single health check.

    Attributes:
        timestamp: When the check was performed
        model_alias: Model that was checked
        is_healthy: Whether the check passed
        response_time_ms: Response time in milliseconds
        error: Error message if check failed
    """
    timestamp: datetime
    model_alias: str
    is_healthy: bool
    response_time_ms: float
    error: str = ""


@dataclass
class ModelHealth:
    """
    Track health status for a single model.

    Maintains history of health checks and calculates statistics.

    Attributes:
        model_alias: Model identifier
        consecutive_failures: Number of consecutive failed checks
        last_check: Timestamp of last health check
        history: List of recent health check results
        max_history: Maximum number of results to keep
    """
    model_alias: str
    consecutive_failures: int = 0
    last_check: datetime = field(default_factory=datetime.now)
    history: List[HealthCheckResult] = field(default_factory=list)
    max_history: int = 100

    def record_check(self, result: HealthCheckResult) -> None:
        """
        Record a health check result.

        Updates consecutive failure count and maintains history.

        Args:
            result: The health check result to record
        """
        self.last_check = result.timestamp

        if result.is_healthy:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        # Keep limited history
        self.history.append(result)
        if len(self.history) > self.max_history:
            self.history.pop(0)

    @property
    def is_degraded(self) -> bool:
        """Check if model is in degraded state."""
        return self.consecutive_failures >= DEGRADED_THRESHOLD

    @property
    def is_down(self) -> bool:
        """Check if model is completely down."""
        return self.consecutive_failures >= DOWN_THRESHOLD

    @property
    def status(self) -> str:
        """Get current health status string."""
        if self.is_down:
            return "down"
        elif self.is_degraded:
            return "degraded"
        return "healthy"

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from health history.

        Returns:
            Dictionary containing uptime percentage, consecutive failures,
            average response time, last check time, and status.
        """
        if not self.history:
            return {}

        # Calculate stats from last 20 checks
        recent = self.history[-20:]
        healthy_count = sum(1 for h in recent if h.is_healthy)
        response_times = [h.response_time_ms for h in recent if h.is_healthy]

        stats: Dict[str, Any] = {
            "uptime_percentage": f"{(healthy_count / len(recent) * 100):.1f}%",
            "consecutive_failures": self.consecutive_failures,
            "last_check": self.last_check.isoformat(),
            "status": self.status
        }

        if response_times:
            avg_time = sum(response_times) / len(response_times)
            stats["avg_response_time_ms"] = f"{avg_time:.1f}"
        else:
            stats["avg_response_time_ms"] = "N/A"

        return stats


class HealthService:
    """
    Monitor health of active models.

    Runs a background task that periodically checks the health of all
    active model runners and takes action (restart) if models are down.

    Attributes:
        manager: ModelManager instance for accessing runners
        check_interval_sec: Interval between health checks
        model_health: Dictionary of model health trackers
        running: Whether the monitor is currently running
    """

    def __init__(self, manager, check_interval_sec: int = 30):
        """
        Initialize the health monitor.

        Args:
            manager: ModelManager instance
            check_interval_sec: Seconds between health checks (default: 30)
        """
        self.manager = manager
        self.check_interval_sec = check_interval_sec
        self.model_health: Dict[str, ModelHealth] = {}
        self.lock = asyncio.Lock()
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None

    async def check_model_health(
        self,
        model_alias: str,
        runner
    ) -> HealthCheckResult:
        """
        Perform health check for a single model.

        Args:
            model_alias: Model identifier
            runner: Runner process instance

        Returns:
            HealthCheckResult with check outcome
        """
        start_time = time.time()

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{runner.url}/health",
                    timeout=HEALTH_CHECK_TIMEOUT
                )

                response_time_ms = (time.time() - start_time) * 1000
                is_healthy = response.status_code == 200

                return HealthCheckResult(
                    timestamp=datetime.now(),
                    model_alias=model_alias,
                    is_healthy=is_healthy,
                    response_time_ms=response_time_ms,
                    error="" if is_healthy else f"Status: {response.status_code}"
                )

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                timestamp=datetime.now(),
                model_alias=model_alias,
                is_healthy=False,
                response_time_ms=response_time_ms,
                error=str(e)
            )

    async def _get_active_models(self) -> Dict[str, Any]:
        """Get dictionary of active, ready models."""
        async with self.manager.lock:
            return {
                alias: runner
                for alias, runner in self.manager.active_runners.items()
                if runner.is_alive() and runner.status == "ready"
            }

    async def _restart_model(self, alias: str) -> bool:
        """
        Attempt to restart a model.

        Args:
            alias: Model alias to restart

        Returns:
            True if restart successful, False otherwise
        """
        try:
            logger.info(f"Attempting restart for '{alias}'")
            await self.manager.eject_model(alias)
            await asyncio.sleep(2)
            await self.manager.get_runner_for_request(alias)
            logger.info(f"Model '{alias}' successfully restarted")
            return True
        except Exception as e:
            logger.error(f"Failed to restart model '{alias}': {e}")
            return False

    async def monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        logger.info("Health monitor started")

        while self.running:
            try:
                active_models = await self._get_active_models()
                models_to_restart: List[str] = []

                # Check health for each model
                for alias, runner in active_models.items():
                    result = await self.check_model_health(alias, runner)

                    async with self.lock:
                        if alias not in self.model_health:
                            self.model_health[alias] = ModelHealth(
                                model_alias=alias
                            )

                        health = self.model_health[alias]
                        health.record_check(result)

                        # Log if there are issues
                        if health.is_degraded and not health.is_down:
                            logger.warning(
                                f"Model '{alias}' is degraded. "
                                f"Consecutive failures: {health.consecutive_failures}"
                            )

                        if health.is_down:
                            logger.error(
                                f"Model '{alias}' is DOWN. Will attempt restart."
                            )
                            models_to_restart.append(alias)

                # Perform restarts outside of lock
                for alias in models_to_restart:
                    await self._restart_model(alias)

                # Clean up tracking for inactive models
                await self._cleanup_inactive_models(set(active_models.keys()))

            except Exception as e:
                logger.exception(f"Error in health monitor loop: {e}")

            # Wait for next check
            await asyncio.sleep(self.check_interval_sec)

        logger.info("Health monitor stopped")

    async def _cleanup_inactive_models(self, active_aliases: set) -> None:
        """Remove health tracking for models that are no longer active."""
        async with self.lock:
            inactive = set(self.model_health.keys()) - active_aliases
            for alias in inactive:
                del self.model_health[alias]

    def start(self) -> None:
        """Start health monitoring."""
        if not self.running:
            self.running = True
            self.monitor_task = asyncio.create_task(self.monitor_loop())
            logger.info("Health monitor task created")

    async def stop(self) -> None:
        """Stop health monitoring."""
        if self.running:
            self.running = False
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            logger.info("Health monitor stopped")

    def get_all_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Get health status for all monitored models.

        Returns:
            Dictionary mapping model alias to health statistics
        """
        return {
            alias: health.get_stats()
            for alias, health in self.model_health.items()
        }

    def get_model_health(self, model_alias: str) -> Optional[Dict[str, Any]]:
        """
        Get health status for a specific model.

        Args:
            model_alias: Model to get health for

        Returns:
            Health statistics or None if model not tracked
        """
        health = self.model_health.get(model_alias)
        return health.get_stats() if health else None
