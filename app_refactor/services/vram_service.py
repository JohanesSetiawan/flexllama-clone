"""
GPU VRAM Tracking Module

This module provides an accurate VRAM tracking system for GPU to prevent
Out of Memory (OOM) errors when loading multiple models.

Components:
    - ModelVRAMSnapshot: Snapshot of VRAM usage at a point in time
    - ModelVRAMTracking: Complete VRAM tracking for a single model
    - VRAMTracker: Main tracker with NVML integration

Features:
    - Real-time VRAM monitoring via NVIDIA NVML
    - Per-model VRAM usage tracking (before/after load)
    - Sequential loading with load_lock to prevent race conditions
    - Automatic VRAM estimation to reject models if insufficient
    - Background monitoring with periodic snapshots
    - Baseline tracking for system overhead

Tracking Flow:
    1. track_model_load_start(): Acquire load_lock, record VRAM before load
    2. Model loading process (llama-server start)
    3. track_model_load_complete(): Record VRAM after load, calculate delta, release lock
    4. track_model_eject(): Remove tracking when model is unloaded

VRAM Status:
    - healthy: Free VRAM >= 2x min_required
    - warning: Free VRAM >= min_required
    - critical: Free VRAM < min_required

Usage:
    vram_tracker = VRAMTracker(gpu_device_index=0, min_vram_required=500)
    vram_tracker.start_monitoring()
    
    # Check before loading model
    can_load, available_mb, message = vram_tracker.can_load_model(
        estimated_vram_mb=4000,
        safety_buffer_mb=200
    )
    
    if can_load:
        await vram_tracker.track_model_load_start("qwen3-8b", port=8085)
        # ... load model ...
        await vram_tracker.track_model_load_complete("qwen3-8b")
    
    # Get report
    report = vram_tracker.get_vram_report()

Note:
    Requires pynvml (NVIDIA Management Library) to be installed.
    Only supports NVIDIA GPUs.
"""

import pynvml
import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


@dataclass
class ModelVRAMSnapshot:
    """
    Snapshot of VRAM usage for a model at a point in time.

    Attributes:
        model_alias: Model identifier
        timestamp: When the snapshot was taken
        vram_used_mb: VRAM used by this model (MB)
        total_vram_used_mb: Total GPU VRAM used (MB)
        port: Model runner port
        status: Model status at snapshot time
    """
    model_alias: str
    timestamp: datetime
    vram_used_mb: float
    total_vram_used_mb: float
    port: int
    status: str


@dataclass
class ModelVRAMTracking:
    """
    Complete VRAM tracking for a single model.

    Tracks VRAM usage before and after model load, maintains
    a history of snapshots for average usage calculation.

    Attributes:
        model_alias: Model identifier
        port: Runner port
        vram_before_load_used_mb: Total GPU VRAM used before load
        vram_after_load_used_mb: Total GPU VRAM used after load
        current_vram_used_mb: Current VRAM used by this model
        load_start_time: When model loading started
        load_end_time: When model loading completed
        status: Current status (loading, loaded, failed, ejected)
        snapshots: History of VRAM snapshots
        max_snapshots: Maximum snapshots to keep
    """
    model_alias: str
    port: int
    vram_before_load_used_mb: float
    vram_after_load_used_mb: float = 0.0
    current_vram_used_mb: float = 0.0
    load_start_time: Optional[datetime] = None
    load_end_time: Optional[datetime] = None
    status: str = "loading"
    snapshots: List[ModelVRAMSnapshot] = field(default_factory=list)
    max_snapshots: int = 10

    def add_snapshot(self, snapshot: ModelVRAMSnapshot) -> None:
        """Add a snapshot to history, removing oldest if at capacity."""
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

    def get_average_usage_mb(self) -> float:
        """Get average VRAM usage from snapshots."""
        if not self.snapshots:
            return self.current_vram_used_mb

        usages = [s.vram_used_mb for s in self.snapshots]
        return sum(usages) / len(usages)

    def get_load_duration_sec(self) -> Optional[float]:
        """Get duration of model loading in seconds."""
        if self.load_start_time and self.load_end_time:
            return (self.load_end_time - self.load_start_time).total_seconds()
        return None


class VRAMService:
    """
    Track VRAM usage per model dynamically with accurate measurement.

    Uses NVIDIA NVML to monitor GPU memory and tracks per-model
    VRAM consumption by measuring before/after load deltas.

    Attributes:
        gpu_device_index: GPU device to monitor
        min_vram_required: Minimum VRAM for health status (MB)
        model_tracks: Dictionary of model tracking data
        baseline_vram_used_mb: System VRAM overhead before any models
        initial_free_vram_mb: Free VRAM at initialization
    """

    def __init__(
        self,
        gpu_device_index: int = 0,
        min_vram_required: int = 500
    ):
        """
        Initialize the VRAM tracker.

        Args:
            gpu_device_index: GPU device index to monitor
            min_vram_required: Minimum free VRAM for healthy status (MB)
        """
        self.gpu_device_index = gpu_device_index
        self.gpu_handle = None
        self.min_vram_required = min_vram_required

        # Track models: {model_alias: ModelVRAMTracking}
        self.model_tracks: Dict[str, ModelVRAMTracking] = {}
        self.lock = asyncio.Lock()

        # Lock for sequential model loading
        self.load_lock = asyncio.Lock()
        self.currently_loading: Optional[str] = None

        # Background monitoring
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval_sec = 5

        # Baseline VRAM (system overhead)
        self.baseline_vram_used_mb = 0.0
        self.initial_free_vram_mb = 0.0

        # Initialize NVML
        self._init_nvml()

    def _init_nvml(self) -> None:
        """Initialize NVML and get GPU handle."""
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(
                self.gpu_device_index
            )
            gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)

            # Record baseline VRAM (system overhead before loading any model)
            vram_info = self.get_current_vram_info()
            self.baseline_vram_used_mb = vram_info["used_mb"]
            self.initial_free_vram_mb = vram_info["free_mb"]

            logger.info(
                f"[VRAM Tracker] Initialized for GPU {self.gpu_device_index}: "
                f"{gpu_name} | Baseline: {self.baseline_vram_used_mb:.0f} MB | "
                f"Free: {self.initial_free_vram_mb:.0f} MB"
            )
        except Exception as e:
            logger.error(f"[VRAM Tracker] Failed to initialize NVML: {e}")
            raise

    def get_current_vram_info(self) -> Dict[str, float]:
        """
        Get current VRAM info from GPU.

        Returns:
            Dictionary with total_mb, used_mb, free_mb
        """
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                "total_mb": mem_info.total / (1024 ** 2),
                "used_mb": mem_info.used / (1024 ** 2),
                "free_mb": mem_info.free / (1024 ** 2)
            }
        except Exception as e:
            logger.error(f"[VRAM Tracker] Failed to get VRAM info: {e}")
            return {"total_mb": 0, "used_mb": 0, "free_mb": 0}

    async def track_model_load_start(
        self,
        model_alias: str,
        port: int
    ) -> None:
        """
        Start tracking a model load.

        Acquires load_lock to ensure sequential loading.

        Args:
            model_alias: Model identifier
            port: Runner port
        """
        await self.load_lock.acquire()
        self.currently_loading = model_alias

        async with self.lock:
            vram_info = self.get_current_vram_info()

            self.model_tracks[model_alias] = ModelVRAMTracking(
                model_alias=model_alias,
                port=port,
                vram_before_load_used_mb=vram_info["used_mb"],
                load_start_time=datetime.now(),
                status="loading"
            )

            logger.info(
                f"[VRAM Tracker] Started tracking '{model_alias}' | "
                f"VRAM before: {vram_info['used_mb']:.0f} MB | "
                f"Free: {vram_info['free_mb']:.0f} MB"
            )

    async def track_model_load_complete(self, model_alias: str) -> None:
        """
        Mark model load as complete and release load_lock.

        Calculates VRAM consumed by comparing before/after measurements.

        Args:
            model_alias: Model identifier
        """
        async with self.lock:
            if model_alias not in self.model_tracks:
                logger.warning(
                    f"[VRAM Tracker] Model '{model_alias}' not in tracking"
                )
                self._release_load_lock(model_alias)
                return

            track = self.model_tracks[model_alias]
            vram_info = self.get_current_vram_info()

            # Calculate VRAM consumed: after - before
            track.vram_after_load_used_mb = vram_info["used_mb"]
            vram_consumed = track.vram_after_load_used_mb - track.vram_before_load_used_mb

            # Update tracking
            track.current_vram_used_mb = max(0, vram_consumed)
            track.load_end_time = datetime.now()
            track.status = "loaded"

            # Add initial snapshot
            self._add_snapshot(track, vram_info)

            load_duration = track.get_load_duration_sec()
            logger.info(
                f"[VRAM Tracker] '{model_alias}' loaded | "
                f"VRAM: {track.current_vram_used_mb:.0f} MB | "
                f"Load time: {load_duration:.1f}s | "
                f"Total used: {vram_info['used_mb']:.0f} MB"
            )

        self._release_load_lock(model_alias)

    async def track_model_load_failed(
        self,
        model_alias: str,
        error: str
    ) -> None:
        """
        Mark model load as failed and release load_lock.

        Args:
            model_alias: Model identifier
            error: Error message
        """
        async with self.lock:
            if model_alias in self.model_tracks:
                logger.warning(
                    f"[VRAM Tracker] '{model_alias}' failed to load: {error}"
                )
                del self.model_tracks[model_alias]

        self._release_load_lock(model_alias)

    async def track_model_eject(self, model_alias: str) -> None:
        """
        Remove tracking for an ejected model.

        Args:
            model_alias: Model identifier
        """
        async with self.lock:
            if model_alias in self.model_tracks:
                track = self.model_tracks[model_alias]
                logger.info(
                    f"[VRAM Tracker] '{model_alias}' ejected | "
                    f"Was using: {track.current_vram_used_mb:.0f} MB"
                )
                del self.model_tracks[model_alias]
            else:
                logger.debug(
                    f"[VRAM Tracker] Model '{model_alias}' not in tracking"
                )

    def _release_load_lock(self, model_alias: str) -> None:
        """Release load_lock if held for this model."""
        if self.currently_loading == model_alias:
            self.currently_loading = None
            if self.load_lock.locked():
                self.load_lock.release()
                logger.debug(
                    f"[VRAM Tracker] load_lock released for '{model_alias}'"
                )

    def _add_snapshot(
        self,
        track: ModelVRAMTracking,
        vram_info: Dict[str, float]
    ) -> None:
        """Add a VRAM snapshot to tracking."""
        snapshot = ModelVRAMSnapshot(
            model_alias=track.model_alias,
            timestamp=datetime.now(),
            vram_used_mb=track.current_vram_used_mb,
            total_vram_used_mb=vram_info["used_mb"],
            port=track.port,
            status=track.status
        )
        track.add_snapshot(snapshot)

    def get_available_vram_mb(self) -> float:
        """Get actual available VRAM in MB from GPU."""
        vram_info = self.get_current_vram_info()
        return vram_info["free_mb"]

    def can_load_model(
        self,
        estimated_vram_mb: float,
        safety_buffer_mb: float = 200
    ) -> Tuple[bool, float, str]:
        """
        Check if there's enough VRAM to load a new model.

        Args:
            estimated_vram_mb: Estimated VRAM needed for the model
            safety_buffer_mb: Extra buffer to prevent GPU OOM (default 200MB)

        Returns:
            Tuple of (can_load, available_mb, message)
        """
        available_mb = self.get_available_vram_mb()
        required_mb = estimated_vram_mb + safety_buffer_mb

        can_load = available_mb >= required_mb

        if can_load:
            message = (
                f"VRAM OK: {available_mb:.0f} MB available, "
                f"need {required_mb:.0f} MB"
            )
        else:
            message = (
                f"VRAM insufficient: {available_mb:.0f} MB available, "
                f"need {required_mb:.0f} MB"
            )

        return can_load, available_mb, message

    async def update_all_tracks(self) -> None:
        """Update snapshots for all tracked models."""
        async with self.lock:
            if not self.model_tracks:
                return

            vram_info = self.get_current_vram_info()

            for track in self.model_tracks.values():
                if track.status == "loaded":
                    self._add_snapshot(track, vram_info)

    async def monitor_loop(self) -> None:
        """Background monitoring loop."""
        logger.info("[VRAM Tracker] Monitor started")

        while self.monitoring:
            try:
                await self.update_all_tracks()
                await asyncio.sleep(self.monitor_interval_sec)
            except asyncio.CancelledError:
                logger.info("[VRAM Tracker] Monitor cancelled")
                break
            except Exception as e:
                logger.exception(f"[VRAM Tracker] Monitor error: {e}")
                await asyncio.sleep(self.monitor_interval_sec)

        logger.info("[VRAM Tracker] Monitor stopped")

    def start_monitoring(self) -> None:
        """Start background VRAM monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_task = asyncio.create_task(self.monitor_loop())
            logger.info("[VRAM Tracker] Monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background VRAM monitoring."""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_task:
                self.monitor_task.cancel()
                try:
                    await self.monitor_task
                except asyncio.CancelledError:
                    pass
            logger.info("[VRAM Tracker] Monitoring stopped")

    def get_vram_report(self) -> Dict[str, Any]:
        """
        Get comprehensive VRAM usage report.

        Returns:
            Dictionary with GPU info, model details, and status
        """
        vram_info = self.get_current_vram_info()

        # Calculate total allocated by tracked models
        total_allocated = sum(
            track.current_vram_used_mb
            for track in self.model_tracks.values()
            if track.status == "loaded"
        )

        # Build model details
        models_info = self._build_models_info(vram_info)

        # Calculate status
        estimated_free = vram_info["free_mb"]
        can_load_more = estimated_free >= self.min_vram_required

        return {
            "gpu_info": {
                "total_mb": round(vram_info["total_mb"], 2),
                "total_gb": round(vram_info["total_mb"] / 1024, 2),
                "used_mb": round(vram_info["used_mb"], 2),
                "used_gb": round(vram_info["used_mb"] / 1024, 2),
                "free_mb": round(vram_info["free_mb"], 2),
                "free_gb": round(vram_info["free_mb"] / 1024, 2),
                "usage_percentage": round(
                    (vram_info["used_mb"] / vram_info["total_mb"]) * 100, 2
                ) if vram_info["total_mb"] > 0 else 0,
                "baseline_used_mb": round(self.baseline_vram_used_mb, 2)
            },
            "tracked_models_count": len(self.model_tracks),
            "loaded_models_count": len([
                t for t in self.model_tracks.values() if t.status == "loaded"
            ]),
            "total_allocated_by_models_mb": round(total_allocated, 2),
            "total_allocated_by_models_gb": round(total_allocated / 1024, 2),
            "models": models_info,
            "can_load_more": can_load_more,
            "estimated_free_for_new_model_mb": round(estimated_free, 2),
            "estimated_free_for_new_model_gb": round(estimated_free / 1024, 2),
            "status": self._get_vram_status(
                vram_info["free_mb"],
                self.min_vram_required
            )
        }

    def _build_models_info(
        self,
        vram_info: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Build list of model info dictionaries."""
        models_info = []

        for alias, track in self.model_tracks.items():
            model_info: Dict[str, Any] = {
                "model_alias": alias,
                "port": track.port,
                "status": track.status,
                "vram_used_mb": round(track.current_vram_used_mb, 2),
                "vram_used_gb": round(track.current_vram_used_mb / 1024, 2),
                "average_usage_mb": round(track.get_average_usage_mb(), 2),
                "load_duration_sec": track.get_load_duration_sec()
            }

            # Add percentage
            if vram_info["total_mb"] > 0:
                model_info["vram_percentage"] = round(
                    (track.current_vram_used_mb / vram_info["total_mb"]) * 100,
                    2
                )

            models_info.append(model_info)

        # Sort by VRAM usage (descending)
        models_info.sort(key=lambda x: x["vram_used_mb"], reverse=True)
        return models_info

    def _get_vram_status(self, free_mb: float, min_required: float) -> str:
        """Determine VRAM health status."""
        if free_mb >= min_required * 2:
            return "healthy"
        elif free_mb >= min_required:
            return "warning"
        else:
            return "critical"

    def shutdown(self) -> None:
        """Shutdown NVML."""
        try:
            pynvml.nvmlShutdown()
            logger.info("[VRAM Tracker] NVML shutdown complete")
        except Exception as e:
            logger.error(f"[VRAM Tracker] NVML shutdown error: {e}")
