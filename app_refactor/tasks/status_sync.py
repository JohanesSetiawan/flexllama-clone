"""
Status Synchronization Task

This module provides background tasks for synchronizing model statuses
between the ModelManager and ModelStatusTracker.

Tasks:
    - sync_model_statuses: Periodic sync of model statuses

Usage:
    task = asyncio.create_task(
        sync_model_statuses(manager, status_tracker, shutdown_event)
    )
"""

import asyncio
import logging
from typing import Optional

from ..core.model_status import ModelStatus, ModelStatusTracker
from ..core.manager import ModelManager
from ..services.vram_service import VRAMService


logger = logging.getLogger(__name__)


# Sync interval in seconds
SYNC_INTERVAL_SEC = 5


async def sync_model_statuses(
    manager: Optional[ModelManager],
    status_tracker: Optional[ModelStatusTracker],
    shutdown_event: asyncio.Event
) -> None:
    """
    Background task to sync model statuses from manager to tracker.

    This ensures the status tracker reflects the actual state of
    model runners, updating status, port, and VRAM information.

    Args:
        manager: ModelManager instance
        status_tracker: ModelStatusTracker instance
        shutdown_event: Event signaling shutdown
    """
    logger.info("Model status sync task started")

    try:
        while not shutdown_event.is_set():
            try:
                # Wait for shutdown or timeout
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=SYNC_INTERVAL_SEC
                )
                break  # Shutdown signaled
            except asyncio.TimeoutError:
                pass  # Normal timeout, continue sync

            # Perform sync
            await _sync_statuses(manager, status_tracker)

    except asyncio.CancelledError:
        logger.info("Model status sync task cancelled")
        raise
    except Exception as e:
        logger.exception(f"Error in model status sync task: {e}")
    finally:
        logger.info("Model status sync task stopped")


async def _sync_statuses(
    manager: Optional[ModelManager],
    status_tracker: Optional[ModelStatusTracker]
) -> None:
    """Sync statuses from manager to tracker."""
    if not manager or not status_tracker:
        return

    try:
        # Get all configured models
        all_models = set(manager.config.models.keys())

        # Get currently active models
        async with manager.lock:
            active_models = set(manager.active_runners.keys())

            for alias in all_models:
                if alias in manager.active_runners:
                    runner = manager.active_runners[alias]

                    # Map runner status to ModelStatus
                    status = _map_runner_status(runner.status)

                    # Get VRAM usage if available
                    vram_mb = None
                    if alias in manager.vram_service.model_tracks:
                        track = manager.vram_service.model_tracks[alias]
                        vram_mb = track.current_vram_used_mb

                    # Update tracker
                    await status_tracker.update_status(
                        alias=alias,
                        status=status,
                        port=runner.port if runner.is_alive() else None,
                        vram_used_mb=vram_mb
                    )
                else:
                    # Model not active
                    current = await status_tracker.get_status(alias)
                    if current and current.status not in (
                        ModelStatus.OFF,
                        ModelStatus.FAILED,
                        ModelStatus.CRASHED
                    ):
                        await status_tracker.update_status(
                            alias=alias,
                            status=ModelStatus.OFF
                        )

    except Exception as e:
        logger.error(f"Error syncing model statuses: {e}")


def _map_runner_status(runner_status: str) -> ModelStatus:
    """Map runner status string to ModelStatus enum."""
    status_map = {
        "stopped": ModelStatus.OFF,
        "starting": ModelStatus.STARTING,
        "loading": ModelStatus.LOADING,
        "ready": ModelStatus.READY,
        "crashed": ModelStatus.CRASHED,
    }
    return status_map.get(runner_status, ModelStatus.UNKNOWN)
