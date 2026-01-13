"""
Model Status Tracking Module

This module provides a real-time status tracking system for all models
with SSE (Server-Sent Events) support for live updates to frontend.

Components:
    - ModelStatus: Enum for model status (OFF, STARTING, LOADING, READY, etc.)
    - ModelStatusInfo: Dataclass with complete model status information
    - ModelStatusTracker: Main tracker with SSE support and file persistence

Status Flow:
    OFF -> STARTING -> LOADING -> READY/LOADED
                    -> CRASHED (if failed)
                    -> FAILED (if config error)
    READY -> STOPPING -> OFF

Status Definitions:
    - OFF: Model not active, no runner process
    - STARTING: llama-server subprocess is being spawned
    - LOADING: Model is being loaded into VRAM
    - READY/LOADED: Model ready to accept requests
    - STANDBY: Model idle but still in memory (deprecated)
    - STOPPING: Model is being stopped
    - CRASHED: Model crashed unexpectedly
    - FAILED: Model failed to start (configuration/VRAM error)

Features:
    - Real-time status tracking per model
    - SSE broadcasting for live updates
    - File persistence for access before FastAPI is ready
    - VRAM usage tracking
    - Timestamp tracking (started_at, last_used_at, updated_at)

Usage:
    tracker = init_status_tracker()
    await tracker.initialize_from_config(["qwen3-8b", "gemma3-4b"])
    
    # Update status
    await tracker.update_status("qwen3-8b", ModelStatus.LOADING)
    await tracker.update_status("qwen3-8b", ModelStatus.READY, port=8085)
    
    # Get status
    status = await tracker.get_status("qwen3-8b")
    all_statuses = await tracker.get_full_status()
    
    # SSE subscription
    queue = await tracker.subscribe()
    # ... consume updates from queue ...
    await tracker.unsubscribe(queue)
"""

import json
import asyncio
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """
    Enum for model status.

    Inherits from str to allow JSON serialization and string comparison.
    """
    OFF = "off"           # Not active, no runner
    STARTING = "starting"  # Subprocess being spawned
    LOADING = "loading"   # Loading into VRAM
    READY = "ready"       # Ready for requests
    LOADED = "loaded"     # Alias for READY
    STANDBY = "standby"   # Idle but in memory (deprecated)
    STOPPING = "stopping"  # Being stopped
    CRASHED = "crashed"   # Crashed unexpectedly
    FAILED = "failed"     # Failed to start
    UNKNOWN = "unknown"   # Unknown state


@dataclass
class ModelStatusInfo:
    """
    Complete status information for a model.

    Attributes:
        alias: Model identifier
        status: Current model status
        port: Port number if running
        started_at: Timestamp when model started
        last_used_at: Timestamp of last request
        load_progress: Loading progress (0-100%)
        error_message: Error message if failed/crashed
        vram_used_mb: VRAM usage in megabytes
        updated_at: Timestamp of last status update
    """
    alias: str
    status: ModelStatus
    port: Optional[int] = None
    started_at: Optional[str] = None
    last_used_at: Optional[str] = None
    load_progress: Optional[float] = None
    error_message: Optional[str] = None
    vram_used_mb: Optional[float] = None
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of status info
        """
        return {
            "alias": self.alias,
            "status": self.status.value if isinstance(self.status, ModelStatus) else self.status,
            "port": self.port,
            "started_at": self.started_at,
            "last_used_at": self.last_used_at,
            "load_progress": self.load_progress,
            "error_message": self.error_message,
            "vram_used_mb": self.vram_used_mb,
            "updated_at": self.updated_at
        }


class ModelStatusTracker:
    """
    Tracker for status of all models.

    Provides real-time status tracking with SSE broadcasting and
    file persistence for access before FastAPI is fully initialized.

    Attributes:
        statuses: Dictionary of model alias to status info
        status_file: Path to status persistence file
        server_status: Current server status string
        server_started_at: Server start timestamp
    """

    def __init__(self, status_file_path: Optional[str] = None):
        """
        Initialize the status tracker.

        Args:
            status_file_path: Path to status persistence file.
                If None, uses default at logs/model_status.json
        """
        self.statuses: Dict[str, ModelStatusInfo] = {}
        self.lock = asyncio.Lock()

        # SSE subscribers
        self._subscribers: Set[asyncio.Queue] = set()
        self._subscribers_lock = asyncio.Lock()

        # Status file path
        if status_file_path:
            self.status_file = Path(status_file_path)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.status_file = project_root / "logs" / "model_status.json"

        # Ensure parent directory exists
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Server status
        self.server_status = "initializing"
        self.server_started_at: Optional[str] = None

        logger.info(
            f"ModelStatusTracker initialized. Status file: {self.status_file}")

    async def initialize_from_config(self, model_aliases: List[str]) -> None:
        """
        Initialize status for all models from config.

        All models start with OFF status.

        Args:
            model_aliases: List of model aliases from config
        """
        async with self.lock:
            for alias in model_aliases:
                if alias not in self.statuses:
                    self.statuses[alias] = ModelStatusInfo(
                        alias=alias,
                        status=ModelStatus.OFF
                    )

            await self._save_to_file_unsafe()

        logger.info(f"Initialized status for {len(model_aliases)} models")

    async def update_status(
        self,
        alias: str,
        status: ModelStatus,
        port: Optional[int] = None,
        error_message: Optional[str] = None,
        load_progress: Optional[float] = None,
        vram_used_mb: Optional[float] = None
    ) -> None:
        """
        Update model status.

        Args:
            alias: Model alias
            status: New status
            port: Port if running
            error_message: Error message if failed/crashed
            load_progress: Loading progress (0-100%)
            vram_used_mb: VRAM usage in MB
        """
        async with self.lock:
            now = datetime.now().isoformat()

            if alias in self.statuses:
                info = self.statuses[alias]
                info.status = status
                info.updated_at = now

                if port is not None:
                    info.port = port
                if error_message is not None:
                    info.error_message = error_message
                if load_progress is not None:
                    info.load_progress = load_progress
                if vram_used_mb is not None:
                    info.vram_used_mb = vram_used_mb

                # Update timestamps based on status
                self._update_timestamps(info, status, now)
            else:
                # Create new entry
                self.statuses[alias] = ModelStatusInfo(
                    alias=alias,
                    status=status,
                    port=port,
                    error_message=error_message,
                    load_progress=load_progress,
                    vram_used_mb=vram_used_mb,
                    started_at=now if status == ModelStatus.STARTING else None,
                    updated_at=now
                )

            await self._save_to_file_unsafe()

        # Broadcast to subscribers (outside lock)
        await self._broadcast_update(alias)

        logger.debug(f"Model '{alias}' status updated to: {status.value}")

    def _update_timestamps(
        self,
        info: ModelStatusInfo,
        status: ModelStatus,
        now: str
    ) -> None:
        """Update timestamps based on status change."""
        if status == ModelStatus.STARTING:
            info.started_at = now
            info.error_message = None
            info.load_progress = 0
        elif status in (ModelStatus.READY, ModelStatus.LOADED):
            info.last_used_at = now
            info.load_progress = 100
        elif status == ModelStatus.OFF:
            info.port = None
            info.load_progress = None
            info.vram_used_mb = None

    async def update_last_used(self, alias: str) -> None:
        """Update last_used_at timestamp for a model."""
        async with self.lock:
            if alias in self.statuses:
                now = datetime.now().isoformat()
                self.statuses[alias].last_used_at = now
                self.statuses[alias].updated_at = now

    async def update_vram(self, alias: str, vram_used_mb: float) -> None:
        """Update VRAM usage for a model."""
        async with self.lock:
            if alias in self.statuses:
                self.statuses[alias].vram_used_mb = vram_used_mb
                self.statuses[alias].updated_at = datetime.now().isoformat()

    async def get_status(self, alias: str) -> Optional[ModelStatusInfo]:
        """Get status for a single model."""
        async with self.lock:
            return self.statuses.get(alias)

    async def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all models."""
        async with self.lock:
            return {
                alias: info.to_dict()
                for alias, info in self.statuses.items()
            }

    async def get_full_status(self) -> Dict[str, Any]:
        """
        Get full status including server info.

        Returns:
            Dictionary with server status, all model statuses, and summary
        """
        async with self.lock:
            models = {
                alias: info.to_dict()
                for alias, info in self.statuses.items()
            }

            # Count by status
            status_counts: Dict[str, int] = {}
            for info in self.statuses.values():
                status_val = (
                    info.status.value
                    if isinstance(info.status, ModelStatus)
                    else info.status
                )
                status_counts[status_val] = status_counts.get(
                    status_val, 0) + 1

            return {
                "server": {
                    "status": self.server_status,
                    "started_at": self.server_started_at,
                    "updated_at": datetime.now().isoformat()
                },
                "models": models,
                "summary": {
                    "total": len(self.statuses),
                    "by_status": status_counts
                }
            }

    async def set_server_status(self, status: str) -> None:
        """Set server status."""
        self.server_status = status
        if status == "ready":
            self.server_started_at = datetime.now().isoformat()

        async with self.lock:
            await self._save_to_file_unsafe()

        await self._broadcast_server_update()

    async def _save_to_file_unsafe(self) -> None:
        """
        Save status to file. MUST be called within lock.

        Enables status access before FastAPI is ready.
        Uses atomic write (temp file + rename) for safety.
        """
        try:
            data = {
                "server": {
                    "status": self.server_status,
                    "started_at": self.server_started_at,
                    "updated_at": datetime.now().isoformat()
                },
                "models": {
                    alias: info.to_dict()
                    for alias, info in self.statuses.items()
                }
            }

            # Write atomically
            temp_file = self.status_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Rename (atomic on POSIX)
            temp_file.rename(self.status_file)

        except Exception as e:
            logger.error(f"Failed to save status file: {e}")

    @classmethod
    def read_status_file(cls, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Read status from file.

        Static method that can be called without an instance.
        Useful for accessing status before FastAPI is ready.

        Args:
            file_path: Path to status file. If None, uses default.

        Returns:
            Dictionary with status info, or empty dict if file not found.
        """
        if file_path:
            status_file = Path(file_path)
        else:
            project_root = Path(__file__).parent.parent.parent
            status_file = project_root / "logs" / "model_status.json"

        try:
            if status_file.exists():
                with open(status_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to read status file: {e}")
            return {}

    # --- SSE Subscription Methods ---

    async def subscribe(self) -> asyncio.Queue:
        """
        Subscribe to status updates.

        Returns:
            Queue that will receive status updates
        """
        queue: asyncio.Queue = asyncio.Queue()

        async with self._subscribers_lock:
            self._subscribers.add(queue)

        logger.debug(f"New SSE subscriber. Total: {len(self._subscribers)}")
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unsubscribe from updates."""
        async with self._subscribers_lock:
            self._subscribers.discard(queue)

        logger.debug(
            f"SSE subscriber removed. Total: {len(self._subscribers)}")

    async def _broadcast_update(self, alias: str) -> None:
        """Broadcast model status update to all subscribers."""
        async with self.lock:
            if alias not in self.statuses:
                return

            data = {
                "type": "model_update",
                "data": self.statuses[alias].to_dict()
            }

        await self._send_to_subscribers(data)

    async def _broadcast_server_update(self) -> None:
        """Broadcast server status update."""
        data = {
            "type": "server_update",
            "data": {
                "status": self.server_status,
                "started_at": self.server_started_at,
                "updated_at": datetime.now().isoformat()
            }
        }

        await self._send_to_subscribers(data)

    async def _send_to_subscribers(self, data: Dict[str, Any]) -> None:
        """Send data to all subscribers."""
        async with self._subscribers_lock:
            dead_subscribers: List[asyncio.Queue] = []

            for queue in self._subscribers:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    logger.warning(
                        "SSE subscriber queue full, dropping message")
                except Exception as e:
                    logger.error(f"Error sending to subscriber: {e}")
                    dead_subscribers.append(queue)

            # Remove dead subscribers
            for queue in dead_subscribers:
                self._subscribers.discard(queue)


# Global instance - initialized at startup
_status_tracker: Optional[ModelStatusTracker] = None


def get_status_tracker() -> Optional[ModelStatusTracker]:
    """Get the global status tracker instance."""
    return _status_tracker


def init_status_tracker(
    status_file_path: Optional[str] = None
) -> ModelStatusTracker:
    """
    Initialize the global status tracker.

    Args:
        status_file_path: Optional path to status file

    Returns:
        The initialized ModelStatusTracker instance
    """
    global _status_tracker
    _status_tracker = ModelStatusTracker(status_file_path)
    return _status_tracker
