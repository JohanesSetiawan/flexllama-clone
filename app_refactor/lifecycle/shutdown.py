"""
Application Shutdown Module

This module handles graceful FastAPI application shutdown.
All components are stopped in reverse order of their startup.

Shutdown Order:
    1. Update status tracker to "shutting_down"
    2. Signal shutdown event
    3. Cancel all background tasks
    4. Stop health monitor
    5. Wait for active requests (with timeout)
    6. Stop all model runners
    7. Close HTTP client
    8. Shutdown NVML/GPU monitoring

Usage:
    from app_refactor.lifecycle.shutdown import shutdown_handler
    
    @app.on_event("shutdown")
    async def on_shutdown():
        await shutdown_handler()
"""

import asyncio
import logging
import pynvml
import time

from .dependencies import get_container


logger = logging.getLogger(__name__)


# Timeout constants
TASK_CANCEL_TIMEOUT = 2.0
HEALTH_MONITOR_TIMEOUT = 5.0
REQUEST_DRAIN_TIMEOUT = 10
RUNNER_STOP_TIMEOUT = 15.0
HTTP_CLIENT_TIMEOUT = 5.0


async def shutdown_handler() -> None:
    """
    Gracefully shutdown all application components.

    This function is called by FastAPI's on_event("shutdown") handler.
    Components are stopped in reverse order of their startup.
    """
    container = get_container()

    logger.info("Application shutdown initiated")

    # Step 1: Update status tracker
    await _update_status_shutting_down(container)

    # Step 2: Signal shutdown event
    container.shutdown_event.set()

    # Step 3: Cancel background tasks
    await _cancel_background_tasks(container)

    # Step 4: Stop health service
    await _stop_health_service(container)

    # Step 5: Wait for active requests to drain
    await _wait_for_active_requests(container)

    # Step 6: Stop all model runners
    await _stop_all_runners(container)

    # Step 7: Close HTTP client
    await _close_http_client(container)

    # Step 8: Shutdown GPU monitoring
    _shutdown_gpu(container)

    logger.info("Application shutdown complete")


async def _update_status_shutting_down(container) -> None:
    """Update status tracker to shutting_down."""
    if container.status_tracker:
        try:
            await container.status_tracker.set_server_status("shutting_down")
        except Exception as e:
            logger.warning(f"Failed to update status tracker: {e}")


async def _cancel_background_tasks(container) -> None:
    """Cancel all background tasks."""
    task_count = len(container.background_tasks)
    if not task_count:
        return

    logger.info(f"Cancelling {task_count} background tasks")

    for task in container.background_tasks:
        if task.done():
            continue

        task.cancel()
        try:
            await asyncio.wait_for(task, timeout=TASK_CANCEL_TIMEOUT)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception as e:
            logger.warning(f"Error cancelling task: {e}")


async def _stop_health_service(container) -> None:
    """Stop the health service."""
    if not container.health_service:
        return

    logger.info("Stopping health service")
    try:
        await asyncio.wait_for(
            container.health_service.stop(),
            timeout=HEALTH_MONITOR_TIMEOUT
        )
        logger.info("Health service stopped")
    except asyncio.TimeoutError:
        logger.warning("Health service stop timeout")
    except Exception as e:
        logger.warning(f"Error stopping health service: {e}")


async def _wait_for_active_requests(container) -> None:
    """Wait for active requests to complete with timeout."""
    start_time = time.time()

    while container.active_requests > 0:
        elapsed = time.time() - start_time
        if elapsed >= REQUEST_DRAIN_TIMEOUT:
            logger.warning(
                f"Shutdown timeout reached. "
                f"Force closing with {container.active_requests} "
                f"requests still active."
            )
            break

        logger.info(
            f"Waiting for {container.active_requests} "
            f"active requests to complete..."
        )
        await asyncio.sleep(1)


async def _stop_all_runners(container) -> None:
    """Stop all model runners."""
    if not container.manager:
        return

    logger.info("Stopping all model runners")

    try:
        await asyncio.wait_for(
            container.manager.stop_all_runners(),
            timeout=RUNNER_STOP_TIMEOUT
        )
        logger.info("All runners stopped")
    except asyncio.TimeoutError:
        logger.error("Timeout stopping runners. Force killing...")
        await _force_kill_runners(container)


async def _force_kill_runners(container) -> None:
    """Force kill all remaining runners."""
    if not container.manager:
        return

    async with container.manager.lock:
        for runner in container.manager.active_runners.values():
            if runner.process:
                try:
                    runner.process.kill()
                    logger.warning(f"Force killed runner: {runner.alias}")
                except Exception as e:
                    logger.error(f"Error killing runner {runner.alias}: {e}")


async def _close_http_client(container) -> None:
    """Close the HTTP client."""
    if not container.http_client:
        return

    logger.info("Closing HTTP client")

    try:
        await asyncio.wait_for(
            container.http_client.aclose(),
            timeout=HTTP_CLIENT_TIMEOUT
        )
        logger.info("HTTP client closed")
    except asyncio.TimeoutError:
        logger.warning("HTTP client close timeout")
    except Exception as e:
        logger.warning(f"Error closing HTTP client: {e}")


def _shutdown_gpu(container) -> None:
    """Shutdown NVML GPU monitoring."""
    if not container.gpu_handle:
        return

    try:
        pynvml.nvmlShutdown()
        logger.info("NVML shutdown complete")
    except Exception as e:
        logger.error(f"NVML shutdown error: {e}")
